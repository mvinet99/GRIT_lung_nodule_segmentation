#!/usr/bin/env python
"""
Step 4 - Inference on an external / held-out cohort given nodule CENTROID COORDINATES
(no masks needed), e.g. temporal_nodule_volume_analysis.csv:

    pid, study_yr, nodule_id, image_path, coordX, coordY, coordZ, [any other columns...]

Pipeline (identical preprocessing / model / post-processing to training + step 3):
  1. CT -> LPS -> 1 x 1 x 1 mm -> HU clip [-1000, 400].
  2. Centroid (coordX/Y/Z) -> voxel on the 1 mm grid -> 64 x 64 x 64 crop around it.
  3. 2D V-Net on all 64 axial slices (batched across nodules / scans on the GPU).
  4. Post-processing at 1 mm: threshold -> hole filling -> 3D connected components ->
     remove tiny / single-slice blobs -> keep the component nearest the given centroid
     (--keep center, default; use --keep all to keep everything).
  5. Outputs (in --out_dir):
       pred_nifti/<uid>_pred.nii.gz        mask on the ORIGINAL CT grid (overlays the input CT)
       pred_nifti/<scan>_pred_all.nii.gz   union of all nodules of that CT
       crops_64/<uid>_img.nii.gz           reconstructed 64x64x64 1 mm CT volume (64 slices stacked)
       crops_64/<uid>_pred.nii.gz          64x64x64 predicted mask
       crops_64/<uid>_prob.nii.gz          64x64x64 probability map
       crops_128/<uid>_*.nii.gz            128x128x128 versions for nodules that needed the expanded FOV
       png/<uid>/zNNN.png                  CT | prediction  (+ GT panels if masks given)
       metrics_per_nodule.csv              all input columns + volume, diameters, QC flags
       coord_check.csv                     coordinate-convention check (see below)

Adaptive field of view (default on)
  If a nodule's final mask comes within --border_margin voxels (default 3) of any face of the
  64^3 crop, that nodule alone is re-run with a 128 x 128 x 128 crop (--expand_roi / --expand_depth)
  centred on its pass-1 mask; the component nearest the GIVEN centroid is still the one kept.
  Its pred_nifti mask, PNGs and CSV row are replaced; the 64^3 crop stays in crops_64/ and the
  expanded one is written to crops_128/. CSV columns: touches_border, expanded_fov,
  touches_border_after_expansion, crop_size, crop_dir. Disable with --no_adaptive_fov.

Coordinates
  coordX/Y/Z are assumed to be PHYSICAL mm. Their convention matters:
    lps   - DEFAULT. DICOM patient coordinates (ImagePositionPatient) and ITK / SimpleITK
            world coordinates (LUNA16-style annotations). SimpleITK reads the NIfTI CTs in LPS,
            so LPS coordinates map straight onto the images.
    ras   - nibabel / NIfTI-affine / 3D Slicer / ITK-SNAP-cursor coordinates (x, y sign-flipped)
    voxel - voxel indices (i, j, k) of the ORIGINAL image
    auto  - pick whichever convention fits the data best
  Before the full run, --probe_scans CTs are checked: the script stops if another convention
  puts clearly more centroids on nodule-like tissue than the chosen one (see coord_check.csv).

Pre-cropped nodule cubes saved as .pt (raw HU)
  Any image_path ending in .pt / .pth is read as ONE pre-cropped nodule cube instead of a CT:
    * a tensor / numpy array (D, H, W) = (axial slices, rows, cols), or a dict holding it under
      one of the keys image / img / ct / volume / vol / data / x / cube (first array otherwise);
      leading singleton dims (1, 1, D, H, W) are squeezed. --pt_axes xyz if stored as (W, H, D).
    * voxel spacing from --pt_spacing x,y,z (default 1,1,1 mm); resampled to 1 mm if different.
    * the nodule is assumed to be at the cube centre (--pt_center cube), or the coordX/Y/Z
      columns give its voxel position (x, y, z) inside the cube (--pt_center coords).
  The cube then goes through the identical crop / model / post-processing / adaptive-FOV steps
  (the crop is padded with -1000 HU outside the cube; the FOV can only grow if the cube is
  larger than 64^3). Extra output: pred_pt/<uid>_pred.pt = uint8 mask with the cube's shape.
  pred_nifti/<uid>_pred.nii.gz is on the cube's own grid (origin 0, given spacing).
  Feed cubes with --pt_dir FOLDER (all *.pt / *.pth in it), or list them in --csv image_path
  (CSV and CT rows can be mixed; cube rows need no coordinates).

Usage:
  CUDA_VISIBLE_DEVICES=0 python 04_infer_test_cohort.py \
      --csv temporal_nodule_volume_analysis_09032026.csv \
      --ckpt run/best.pt --out_dir run/pred_nlst

  # pre-cropped .pt cubes
  CUDA_VISIBLE_DEVICES=0 python 04_infer_test_cohort.py --pt_dir /path/to/cubes \
      --ckpt run/best.pt --out_dir run/pred_cubes
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from tqdm import tqdm

from common import (HU_MAX, HU_MIN, TARGET_SPACING, crop_start, crop_to_sitk, crop_with_pad,
                    load_ct_1mm, read_path_table, resample_mask_to_ref, resample_to_spacing)
from infer_lib import (array_to_orig_image, crop_to_original, geom_of, geom_stub, header_geom,
                       load_model, make_pools, mask_center_phys, phys_to_ref_zyx, postprocess,
                       run_batched_pipeline, save_slice_pngs, scaled_batch, seg_metrics,
                       size_measurements, touches_border, union_from_files)

SYSTEMS = ("lps", "ras", "voxel")
PT_EXT = (".pt", ".pth")
PT_KEYS = ("image", "img", "ct", "volume", "vol", "data", "x", "cube")


# ----------------------------------------------------------------------------
# Pre-cropped .pt nodule cubes
# ----------------------------------------------------------------------------
def is_pt(path):
    return str(path).lower().endswith(PT_EXT)


def read_pt_cube(path, axes="zyx"):
    """.pt file -> float32 numpy (D, H, W) in HU."""
    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001  (e.g. numpy arrays inside -> needs full unpickling; own files only)
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        key = next((k for k in PT_KEYS if k in obj), None)
        if key is None:
            key = next((k for k, v in obj.items() if isinstance(v, (torch.Tensor, np.ndarray))), None)
        if key is None:
            raise ValueError(f"no tensor found in dict keys {list(obj)}")
        obj = obj[key]
    arr = obj.detach().cpu().numpy() if isinstance(obj, torch.Tensor) else np.asarray(obj)
    arr = np.squeeze(arr).astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3D cube, got shape {arr.shape}")
    if axes == "xyz":
        arr = arr.transpose(2, 1, 0)
    return arr


def load_pt_1mm(path, cfg):
    """Cube -> (ref_1mm sitk image, clipped HU zyx array, cube geometry) - same contract as
    load_ct_1mm + header_geom, so the rest of the pipeline treats the cube like a small CT."""
    arr = read_pt_cube(path, cfg["pt_axes"])
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(tuple(cfg["pt_spacing"]))  # origin (0,0,0), identity direction
    og = geom_of(img)
    if np.allclose(cfg["pt_spacing"], TARGET_SPACING):
        ref = img
    else:
        ref = resample_to_spacing(img, TARGET_SPACING, sitk.sitkLinear, HU_MIN)
    hu = np.clip(sitk.GetArrayFromImage(ref), HU_MIN, HU_MAX).astype(np.float32)
    return ref, hu, og, arr.shape


# ----------------------------------------------------------------------------
# Coordinates
# ----------------------------------------------------------------------------
def to_lps_point(xyz, system, orig_geom):
    x, y, z = (float(v) for v in xyz)
    if system == "lps":
        return (x, y, z)
    if system == "ras":
        return (-x, -y, z)
    if system == "voxel":
        return geom_stub(orig_geom).TransformContinuousIndexToPhysicalPoint((x, y, z))
    raise ValueError(system)


def probe_scan(job):
    """For every nodule of one CT and every convention: is the point inside the volume,
    and what is the mean HU in a ~3 mm neighbourhood (nodule ~ > -600 HU, lung ~ -850 HU)?"""
    scan_id, image_path, rows, cfg = job
    out = []
    try:
        img = sitk.ReadImage(str(image_path), sitk.sitkInt16)
        arr = sitk.GetArrayFromImage(img)
        og = geom_of(img)
    except Exception as e:  # noqa: BLE001
        return [dict(scan_id=scan_id, error=repr(e))]
    sp = np.array(img.GetSpacing())
    rad = np.maximum(np.round(1.5 / sp).astype(int), 1)  # ~3 mm cube, per axis (x, y, z)
    for r in rows:
        rec = dict(uid=r["_uid"], scan_id=scan_id)
        for s in SYSTEMS:
            idx = img.TransformPhysicalPointToContinuousIndex(
                to_lps_point([r[c] for c in cfg["coord_cols"]], s, og))
            i, j, k = (int(round(v)) for v in idx)
            inside = (0 <= i < arr.shape[2]) and (0 <= j < arr.shape[1]) and (0 <= k < arr.shape[0])
            hu = np.nan
            if inside:
                blk = arr[max(k - rad[2], 0):k + rad[2] + 1, max(j - rad[1], 0):j + rad[1] + 1,
                          max(i - rad[0], 0):i + rad[0] + 1]
                hu = float(blk.mean())
            rec[f"{s}_inside"] = inside
            rec[f"{s}_hu"] = hu
            rec[f"{s}_hit"] = bool(inside and hu > -600)
        out.append(rec)
    return out


def check_coord_system(jobs, cfg, n_scans, workers, out_dir, requested):
    """Probe a sample of CTs with every convention. requested='auto' -> return the best one.
    Otherwise keep `requested`, but abort if another convention fits the data clearly better."""
    rng = np.random.default_rng(0)
    sample = [jobs[i] for i in sorted(rng.choice(len(jobs), min(n_scans, len(jobs)), replace=False))]
    probe_jobs = [(j[0], j[1], j[2], cfg) for j in sample]
    import multiprocessing
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        recs = [r for rs in tqdm(pool.imap_unordered(probe_scan, probe_jobs), total=len(probe_jobs),
                                 desc="coord check") for r in rs]
    df = pd.DataFrame([r for r in recs if "error" not in r])
    df.to_csv(out_dir / "coord_check.csv", index=False)
    print(f"\nCoordinate-convention check on {len(probe_jobs)} CTs / {len(df)} nodules:")
    print(f"  {'system':6s}  {'inside volume':>13s}  {'median HU @ point':>17s}  {'on dense tissue':>15s}")
    scores = {}
    for s in SYSTEMS:
        inside = df[f"{s}_inside"].mean()
        hit = df[f"{s}_hit"].mean()
        med = df[f"{s}_hu"].median()
        scores[s] = hit
        print(f"  {s:6s}  {inside:13.0%}  {med:17.0f}  {hit:15.0%}")
    best = max(SYSTEMS, key=lambda s: (scores[s], s == "lps"))
    use = best if requested == "auto" else requested
    if scores[best] < 0.6:
        print("  !! WARNING: no convention puts most centroids on nodule-like tissue. "
              "Check coord columns / units before trusting the results.")
    if requested != "auto" and scores[best] - scores[requested] >= 0.25:
        raise SystemExit(
            f"\n  !! '{requested}' puts {scores[requested]:.0%} of sampled centroids on nodule-like tissue, "
            f"but '{best}' puts {scores[best]:.0%}.\n"
            f"     The coordinates were probably not written in {requested.upper()}. Stopping so you don't "
            f"segment the wrong locations.\n     Re-run with --coord_system {best}, or with "
            f"--skip_coord_check if you are sure (details in coord_check.csv).")
    print(f"  -> using '{use}'" + ("" if requested == "auto" else " (requested; consistent with the data)"))
    print()
    return use


# ----------------------------------------------------------------------------
# Stage 1 (CPU workers): load + preprocess one CT, cut a 64^3 crop per nodule
#   job = (scan_id, image_path, rows, cfg, extra); extra = {} in pass 1,
#   {"size": (D,H,W), "redo": {uid: crop_center_phys}} in pass 2 (bigger crops, those nodules only)
# ----------------------------------------------------------------------------
def load_scan(job):
    scan_id, image_path, rows, cfg, extra = job
    redo = extra.get("redo")
    size = tuple(extra.get("size", cfg["size"]))
    pt = is_pt(image_path)
    try:
        if pt:
            ref, hu, og, cube_shape = load_pt_1mm(image_path, cfg)
        else:
            ref, hu = load_ct_1mm(image_path)
            og = header_geom(image_path)
    except Exception as e:  # noqa: BLE001
        return dict(scan_id=scan_id, error=f"{image_path}: {e!r}")

    gts = {}
    if cfg["mask_col"]:
        for r in rows:
            mp = str(r.get(cfg["mask_col"], ""))
            if mp and mp.lower() != "nan" and mp not in gts and Path(mp).exists():
                gts[mp] = resample_mask_to_ref(mp, ref)
    union_gt = None
    if gts:
        union_gt = np.zeros(hu.shape, np.uint8)
        for m in gts.values():
            union_gt |= m

    ref_geom = geom_of(ref)
    nodules, warnings = [], []
    for r in rows:
        uid = r["_uid"]
        if redo is not None and uid not in redo:
            continue
        if pt:  # nodule position inside the cube: its centre, or coordX/Y/Z as cube voxel (x, y, z)
            if cfg["pt_center"] == "coords":
                vox = [float(r[c]) for c in cfg["coord_cols"]]
            else:
                vox = [(cube_shape[2] - 1) / 2, (cube_shape[1] - 1) / 2, (cube_shape[0] - 1) / 2]
            p_lps = geom_stub(og).TransformContinuousIndexToPhysicalPoint(vox)
        else:
            p_lps = to_lps_point([r[c] for c in cfg["coord_cols"]], cfg["coord_system"], og)
        idx = ref.TransformPhysicalPointToContinuousIndex(p_lps)
        center = np.array([idx[2], idx[1], idx[0]])          # (z, y, x) on the 1 mm grid
        inside = bool(np.all(center >= -0.5) and np.all(center <= np.array(hu.shape) - 0.5))
        if not inside and redo is None:
            warnings.append(f"[warn] {uid}: centroid lies OUTSIDE the "
                            f"{'cube' if pt else 'CT volume'} ({cfg['coord_system']})")
        # pass 1: crop around the given centroid; pass 2: around the pass-1 mask
        crop_center = center if redo is None else phys_to_ref_zyx(redo[uid], ref_geom)
        start = crop_start(crop_center, size)
        ci = np.clip(np.round(center).astype(int), 0, np.array(hu.shape) - 1)
        nodules.append(dict(
            uid=uid, row={k: v for k, v in r.items() if not k.startswith("_")},
            start=start, center_crop=center - np.array(start), center_in_bounds=inside,
            hu_at_center=float(hu[tuple(ci)]),
            img=crop_with_pad(hu, start, size, HU_MIN).astype(np.int16),
            gt=crop_with_pad(union_gt, start, size, 0) if union_gt is not None else None))
    # a crop can only gain real anatomy from a larger FOV if the source is bigger than the crop
    can_expand = (not pt) or any(h > c for h, c in zip(hu.shape, cfg["size"]))
    return dict(scan_id=scan_id, image_path=image_path, ref_geom=ref_geom, orig_geom=og,
                nodules=nodules, warnings=warnings, all_uids=[r["_uid"] for r in rows],
                pass2=redo is not None, is_pt=pt, can_expand=can_expand)


# ----------------------------------------------------------------------------
# Stage 3 (CPU workers): post-process + write everything for one CT
# ----------------------------------------------------------------------------
def finish_scan(scan, probs, cfg):
    out_dir = Path(cfg["out_dir"])
    og, rg = scan["orig_geom"], scan["ref_geom"]
    stub = geom_stub(rg)
    vox_vol_orig = float(np.prod(og["spacing"]))
    pass2 = scan["pass2"]
    rows, scan_pred = [], None
    for nod, prob in zip(scan["nodules"], probs):
        uid, start, c = nod["uid"], nod["start"], nod["center_crop"]
        pred = postprocess(prob, cfg["thr"], cfg["min_voxels"], cfg["min_slices"], cfg["keep"],
                           fill_holes=cfg["fill_holes"], center=c)

        # (1) mask in ORIGINAL CT space
        p_orig = crop_to_original(pred, rg, start, og)
        scan_pred = p_orig if scan_pred is None else (scan_pred | p_orig)
        sitk.WriteImage(array_to_orig_image(p_orig, og), str(out_dir / "pred_nifti" / f"{uid}_pred.nii.gz"))
        if scan["is_pt"]:  # same shape / axis order as the input cube
            m = p_orig.astype(np.uint8)
            if cfg["pt_axes"] == "xyz":
                m = m.transpose(2, 1, 0)
            torch.save(torch.from_numpy(np.ascontiguousarray(m)), out_dir / "pred_pt" / f"{uid}_pred.pt")

        # (2) reconstructed nodule space at 1 mm (64^3, or 128^3 when the FOV was expanded),
        #     real-world geometry -> overlays the CT in ITK-SNAP
        cd = out_dir / f"crops_{prob.shape[1]}"
        sitk.WriteImage(crop_to_sitk(nod["img"], stub, start), str(cd / f"{uid}_img.nii.gz"))
        sitk.WriteImage(crop_to_sitk(pred.astype(np.uint8), stub, start), str(cd / f"{uid}_pred.nii.gz"))
        sitk.WriteImage(crop_to_sitk(prob.astype(np.float32), stub, start), str(cd / f"{uid}_prob.nii.gz"))
        if nod["gt"] is not None:
            sitk.WriteImage(crop_to_sitk(nod["gt"], stub, start), str(cd / f"{uid}_gt.nii.gz"))

        # (3) PNGs
        if cfg["save_png"]:
            if pass2:
                shutil.rmtree(out_dir / "png" / uid, ignore_errors=True)  # drop the 64^3 PNGs
            save_slice_pngs(nod["img"], pred, nod["gt"], out_dir / "png", uid, cfg["png_slices"],
                            center=c)

        # measurements + QC
        rec = dict(nod["row"])
        rec.update(uid=uid, scan_id=scan["scan_id"], input_type="pt_cube" if scan["is_pt"] else "ct",
                   coord_system=f"cube_{cfg['pt_center']}" if scan["is_pt"] else cfg["coord_system"],
                   center_in_bounds=nod["center_in_bounds"], hu_at_center=nod["hu_at_center"],
                   grit_threshold_pct=nod.get("grit_pct"))
        border = touches_border(pred, cfg["border_margin"])
        rec.update(crop_size="x".join(str(v) for v in prob.shape), crop_dir=cd.name)
        if pass2:
            rec.update(touches_border=True, expanded_fov=True, touches_border_after_expansion=border)
        else:
            rec.update(touches_border=border, expanded_fov=False, touches_border_after_expansion=np.nan)
            if border and cfg["adaptive"] and scan["can_expand"]:
                rec["_redo"] = mask_center_phys(pred, start, rg)
            elif border and cfg["adaptive"]:
                rec["fov_limited_by_cube"] = True  # cube no bigger than the crop: nothing to expand into
        rec.update(size_measurements(pred))
        rec["pred_vol_mm3_orig_space"] = float(p_orig.sum() * vox_vol_orig)
        ci = tuple(np.clip(np.round(c).astype(int), 0, np.array(prob.shape) - 1))
        rec["prob_at_center"] = float(prob[ci])
        rec["max_prob"] = float(prob.max())
        if pred.any():
            d = np.linalg.norm(np.argwhere(pred) - c, axis=1).min()
            rec["center_to_pred_mm"] = float(d)
        else:
            rec["center_to_pred_mm"] = float("nan")
        rec["empty_prediction"] = not pred.any()
        rec["center_in_pred"] = bool(pred.any() and rec["center_to_pred_mm"] <= 2.0)
        if nod["gt"] is not None:
            m = seg_metrics(pred, nod["gt"])
            m.pop("pred_vol_mm3")
            rec.update(m)
        rows.append(rec)

    if scan_pred is not None and not scan["is_pt"]:  # one nodule per cube -> no union file
        if pass2:  # rebuild the scan union from ALL per-nodule files (pass-1 + replaced ones)
            scan_pred = union_from_files(out_dir / "pred_nifti", scan["all_uids"], og)
        sitk.WriteImage(array_to_orig_image(scan_pred, og),
                        str(out_dir / "pred_nifti" / f"{scan['scan_id']}_pred_all.nii.gz"))
    return rows, None


# ----------------------------------------------------------------------------
def nodule_stem(path):
    n = Path(path).name
    for ext in PT_EXT + (".nii.gz", ".nii"):
        if n.lower().endswith(ext):
            return n[: -len(ext)]
    return Path(path).stem


def make_ids(df, id_cols):
    short = {"study_yr": "T", "nodule_id": "n"}

    def fmt(c, v):
        v = str(int(v)) if isinstance(v, (float, np.floating)) and float(v).is_integer() else str(v)
        return f"{short.get(c, '')}{v}" if c != id_cols[0] else v
    uids = df.apply(lambda r: "_".join(fmt(c, r[c]) for c in id_cols), axis=1)
    dup = uids.duplicated(keep=False)
    uids[dup] = uids[dup] + "_r" + df.index[dup].astype(str)
    return uids.str.replace(r"[^\w.\-]", "_", regex=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="", help="cohort table (.csv or .xlsx); image_path may be .nii.gz or .pt")
    ap.add_argument("--pt_dir", default="", help="folder of pre-cropped .pt nodule cubes (raw HU)")
    ap.add_argument("--pt_spacing", default="1,1,1", help="voxel spacing of the .pt cubes, x,y,z in mm")
    ap.add_argument("--pt_axes", choices=["zyx", "xyz"], default="zyx",
                    help="axis order of the .pt arrays: zyx = (slices, rows, cols) [default]")
    ap.add_argument("--pt_center", choices=["cube", "coords"], default="cube",
                    help="nodule at the cube centre, or at coordX/Y/Z given as cube voxel (x, y, z)")
    ap.add_argument("--ckpt", required=True, help="trained model, e.g. run/best.pt")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--coord_cols", default="coordX,coordY,coordZ")
    ap.add_argument("--coord_system", choices=("auto",) + SYSTEMS, default="lps",
                    help="lps (default: DICOM / ITK / SimpleITK world mm), ras, voxel, or auto")
    ap.add_argument("--probe_scans", type=int, default=30,
                    help="CTs sampled to verify the coordinate convention before the full run")
    ap.add_argument("--skip_coord_check", action="store_true",
                    help="don't verify the coordinate convention (not recommended)")
    ap.add_argument("--id_cols", default="", help="columns forming the nodule id "
                    "(default: pid,study_yr,nodule_id - whichever exist)")
    ap.add_argument("--mask_col", default="", help="optional GT mask column -> Dice etc. + GT panels")
    ap.add_argument("--roi", type=int, default=64)
    ap.add_argument("--depth", type=int, default=64)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--min_voxels", type=int, default=8)
    ap.add_argument("--min_slices", type=int, default=2)
    ap.add_argument("--keep", choices=["center", "all", "largest"], default="center",
                    help="center = only the component nearest the given centroid (best for volumetry)")
    ap.add_argument("--no_fill_holes", action="store_true")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--no_adaptive_fov", action="store_true",
                    help="disable the second, larger-crop pass for nodules touching the crop border")
    ap.add_argument("--border_margin", type=int, default=3,
                    help="a mask within this many voxels of a crop face counts as touching the border")
    ap.add_argument("--expand_roi", type=int, default=128, help="in-plane size of the expanded crop")
    ap.add_argument("--expand_depth", type=int, default=128, help="slices in the expanded crop")
    ap.add_argument("--no_png", action="store_true")
    ap.add_argument("--png_slices", choices=["union", "all"], default="union",
                    help="union = slices with a prediction (centre slices if empty); all = all 64")
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--load_workers", type=int, default=8)
    ap.add_argument("--save_workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N CTs only")
    args = ap.parse_args()
    assert args.roi % 16 == 0 and args.expand_roi % 16 == 0

    out_dir = Path(args.out_dir)
    if not args.csv and not args.pt_dir:
        raise SystemExit("give --csv and/or --pt_dir")
    for d in ("pred_nifti", f"crops_{args.roi}", f"crops_{args.expand_roi}", "png", "pred_pt"):
        (out_dir / d).mkdir(parents=True, exist_ok=True)

    coord_cols = [c.strip() for c in args.coord_cols.split(",")]
    frames = []
    if args.csv:
        frames.append(read_path_table(args.csv))
    if args.pt_dir:
        files = sorted(p for p in Path(args.pt_dir).iterdir() if is_pt(p))
        if not files:
            raise SystemExit(f"no .pt / .pth files in {args.pt_dir}")
        frames.append(pd.DataFrame({"file": [nodule_stem(p) for p in files],
                                    "image_path": [str(p) for p in files]}))
    df = pd.concat(frames, ignore_index=True)
    pt_rows = df["image_path"].map(is_pt)
    ct_rows = ~pt_rows
    need_coords = ct_rows | (pt_rows & (args.pt_center == "coords"))
    missing = [c for c in coord_cols if c not in df.columns]
    if missing and need_coords.any():
        raise SystemExit(f"coordinate columns {missing} not found; columns are {list(df.columns)}")
    if need_coords.any():
        bad = need_coords & df[coord_cols].isna().any(axis=1)
        if bad.any():
            print(f"[warn] dropping {bad.sum()} rows with missing coordinates")
            df = df[~bad]
    df = df.reset_index(drop=True)
    pt_rows = df["image_path"].map(is_pt)
    id_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()] or \
        [c for c in ("pid", "study_yr", "nodule_id") if c in df.columns]
    if id_cols:
        df["_uid"] = make_ids(df, id_cols)
    else:
        df["_uid"] = ""
    no_id = df["_uid"].eq("") | (df[id_cols].isna().any(axis=1) if id_cols else True)
    df.loc[no_id, "_uid"] = df.loc[no_id, "image_path"].map(nodule_stem)  # .pt rows without ids
    dup = df["_uid"].duplicated(keep=False)
    df.loc[dup, "_uid"] = df.loc[dup, "_uid"] + "_r" + df.index[dup].astype(str)
    df["_uid"] = df["_uid"].str.replace(r"[^\w.\-]", "_", regex=True)
    print(f"inputs: {int((~pt_rows).sum())} CT nodule rows, {int(pt_rows.sum())} .pt cubes")
    mask_col = args.mask_col if args.mask_col in df.columns else ""

    # one job per CT
    jobs = []
    for k, (ip, g) in enumerate(df.groupby("image_path", sort=False)):
        first = g.iloc[0]
        if is_pt(ip) or not id_cols:
            scan_id = first["_uid"]
        else:
            scan_id = "_".join(str(u) for u in first["_uid"].split("_")[: max(1, len(id_cols) - 1)])
        jobs.append([scan_id, ip, g.to_dict("records")])
    seen = {}
    for j in jobs:  # make scan ids unique
        seen[j[0]] = seen.get(j[0], 0) + 1
        if seen[j[0]] > 1:
            j[0] = f"{j[0]}_s{seen[j[0]] - 1}"
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs)} images (CTs / cubes), {sum(len(j[2]) for j in jobs)} nodules")

    cfg = dict(out_dir=str(out_dir), size=[args.depth, args.roi, args.roi], coord_cols=coord_cols,
               coord_system=args.coord_system, mask_col=mask_col, thr=args.thr,
               min_voxels=args.min_voxels, min_slices=args.min_slices, keep=args.keep,
               fill_holes=not args.no_fill_holes, save_png=not args.no_png, png_slices=args.png_slices,
               adaptive=not args.no_adaptive_fov, border_margin=args.border_margin,
               pt_spacing=[float(v) for v in args.pt_spacing.split(",")], pt_axes=args.pt_axes,
               pt_center=args.pt_center)
    assert len(cfg["pt_spacing"]) == 3, "--pt_spacing expects x,y,z"
    size1 = cfg["size"]
    size2 = [args.expand_depth, args.expand_roi, args.expand_roi]
    ct_jobs = [j for j in jobs if not is_pt(j[1])]
    if ct_jobs and (not args.skip_coord_check or args.coord_system == "auto"):
        cfg["coord_system"] = check_coord_system(ct_jobs, cfg, args.probe_scans, args.load_workers,
                                                 out_dir, args.coord_system)
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(vars(args) | {"coord_system_used": cfg["coord_system"], "id_cols": id_cols}, f, indent=2)

    # ---- pass 1: every nodule at the standard crop size
    full_jobs = [(j[0], j[1], j[2], cfg, {}) for j in jobs]
    load_pool, save_pool = make_pools(args.load_workers, args.save_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    model = load_model(args.ckpt, device)
    rows, _ = run_batched_pipeline(full_jobs, load_scan, finish_scan, cfg, model, device,
                                   load_pool, save_pool, args.batch_size, args.tta)

    # ---- pass 2: only nodules whose mask touched the crop border, re-run with a bigger crop
    redo_by_scan = {}
    for r in rows:
        if "_redo" in r:
            redo_by_scan.setdefault(r["scan_id"], {})[r["uid"]] = r.pop("_redo")
    n_redo = sum(len(v) for v in redo_by_scan.values())
    if n_redo:
        print(f"\n[adaptive FOV] {n_redo} / {len(rows)} nodules touch the crop border "
              f"-> re-running them at {'x'.join(map(str, size2))}")
        jobs2 = [(j[0], j[1], j[2], cfg, {"size": size2, "redo": redo_by_scan[j[0]]})
                 for j in jobs if j[0] in redo_by_scan]
        lp2, sp2 = make_pools(args.load_workers, args.save_workers)
        rows2, _ = run_batched_pipeline(jobs2, load_scan, finish_scan, cfg, model, device, lp2, sp2,
                                        scaled_batch(args.batch_size, size1, size2), args.tta)
        new = {r["uid"]: r for r in rows2}
        rows = [new.get(r["uid"], r) for r in rows]
        still = sum(bool(r.get("touches_border_after_expansion")) for r in rows2)
        print(f"[adaptive FOV] done; {still} still touch the border at the expanded size")
    elif not cfg["adaptive"]:
        n_touch = sum(bool(r.get("touches_border")) for r in rows)
        print(f"\n[adaptive FOV] disabled; {n_touch} / {len(rows)} nodules touch the crop border "
              f"(see touches_border in the CSV)")
    else:
        print("\n[adaptive FOV] no nodule touched the crop border")

    res = pd.DataFrame(rows)
    expected = [r["_uid"] for j in jobs for r in j[2]]
    failed = sorted(set(expected) - set(res["uid"] if len(res) else []))
    if failed:
        pd.DataFrame({"uid": failed}).to_csv(out_dir / "failed_nodules.csv", index=False)
    if len(res):
        res = res.set_index("uid").loc[[u for u in expected if u not in failed]].reset_index()
        res.to_csv(out_dir / "metrics_per_nodule.csv", index=False)
        print("\n=== summary ===")
        print(f"nodules processed: {len(res)}   failed: {len(failed)}")
        print(f"empty predictions: {int(res.empty_prediction.sum())}   "
              f"centroid not inside prediction (>2 mm): {int((~res.center_in_pred).sum())}   "
              f"centroid outside CT: {int((~res.center_in_bounds).sum())}")
        v = res.loc[~res.empty_prediction, "pred_vol_mm3"]
        if len(v):
            print(f"predicted volume (mm^3): median {v.median():.0f}  IQR {v.quantile(.25):.0f}-{v.quantile(.75):.0f}")
        if "fov_limited_by_cube" in res:
            print(f"cubes whose mask touches the cube border (cannot expand): "
                  f"{int(res.fov_limited_by_cube.fillna(False).astype(bool).sum())}")
        print(f"expanded FOV: {int(res.expanded_fov.sum())}   "
              f"still touching border after expansion: {int((res.touches_border_after_expansion == True).sum())}")  # noqa: E712
        if "dice" in res:
            print(f"Dice vs GT: mean {res.dice.mean():.4f}  median {res.dice.median():.4f}")
    print(f"outputs -> {out_dir}")


if __name__ == "__main__":
    main()
