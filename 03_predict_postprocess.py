#!/usr/bin/env python
"""
Step 3 - Inference + post-processing (+ metrics if ground truth is available).

Input spreadsheet: same format as LIDC_paths.xlsx (pid, image_path, mask_path).
  * mask_path optional - used for (a) locating the nodule and (b) metrics.
  * Instead of a mask you can give the crop centre in physical LPS mm via columns
    center_x, center_y, center_z (e.g. from a detector).

Per row (nodule):
  1. Same preprocessing as training: LPS, 1 mm iso, HU clip [-1000, 400].
  2. Crop a (--depth) x (--roi) x (--roi) volume around the centre
     (optionally displaced by --offset_mm / --random_offset_mm to test off-centre nodules).
     --roi can be larger than 64 (any multiple of 16) - the network is fully convolutional.
  3. Predict every axial slice with the 2D V-Net (optional flip TTA) -> probability volume.
  4. Post-process in 3D at 1 mm:
       threshold -> per-slice hole filling -> 3D connected components ->
       drop components < --min_voxels mm^3 or spanning < --min_slices slices ->
       keep all / largest / component nearest the crop centre (--keep).
  5. Resample the crop prediction onto the ORIGINAL CT grid
     (original spacing + orientation), saved as NIfTI aligned with the input CT.
  6. Metrics (1 mm crop space, vs. union of all GT nodules of that scan):
     Dice, IoU, precision, recall, HD95, volumes, and slice-level false positives
     on GT-empty slices. A scan-level Dice in original space is also written.

Usage:
  python 03_predict_postprocess.py --xlsx test_paths.xlsx --ckpt runs/vnet2d_lidc/final.pt \
      --out_dir runs/vnet2d_lidc/pred_test [--random_offset_mm 15] [--save_crops]
"""

import argparse
import math
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

from common import (HU_MIN, centroid_zyx, crop_start, crop_to_sitk, crop_with_pad,
                    load_ct_1mm, read_path_table, resample_mask_to_ref)
from infer_lib import (array_to_orig_image, crop_to_original, geom_of, geom_stub, load_model,
                       make_pools, parse_offset, postprocess, run_batched_pipeline,
                       save_slice_pngs, seg_metrics)


# ----------------------------------------------------------------------------
# Stage 1 (CPU workers): load + preprocess one scan, cut every nodule crop
# ----------------------------------------------------------------------------
def load_scan(job):
    pid, s_idx, image_path, rows, cfg = job
    scan_id = f"{pid}_s{s_idx}"
    size = tuple(cfg["size"])
    try:
        ref, hu = load_ct_1mm(image_path)
        rd = sitk.ImageFileReader()
        rd.SetFileName(str(image_path))
        rd.ReadImageInformation()  # header only
        orig_geom = dict(size=list(rd.GetSize()), origin=list(rd.GetOrigin()),
                         spacing=list(rd.GetSpacing()), direction=list(rd.GetDirection()))
    except Exception as e:  # noqa: BLE001
        return dict(scan_id=scan_id, error=f"{image_path}: {e!r}")

    gts = {}
    if cfg["has_mask"]:
        for r in rows:
            mp = str(r.get("mask_path", ""))
            if mp and mp.lower() != "nan" and mp not in gts and Path(mp).exists():
                gts[mp] = resample_mask_to_ref(mp, ref)
    union_gt = np.zeros(hu.shape, np.uint8)
    for m in gts.values():
        union_gt |= m

    nodules, warnings = [], []
    for r in rows:
        mp = str(r.get("mask_path", "")) if cfg["has_mask"] else ""
        name = Path(mp).name.split(".")[0] if mp in gts else f"row{r['_row']}"
        uid = f"{scan_id}_{name}"
        if cfg["has_ctr"] and pd.notna(r.get("center_x")):
            idx = ref.TransformPhysicalPointToContinuousIndex(
                [float(r["center_x"]), float(r["center_y"]), float(r["center_z"])])
            center = np.array([idx[2], idx[1], idx[0]])
        elif mp in gts and gts[mp].any():
            center = centroid_zyx(gts[mp])
        else:
            warnings.append(f"[skip] {uid}: no centre (no mask and no center_x/y/z)")
            continue
        off = np.array(cfg["fixed_off"], dtype=float)
        if cfg["random_offset_mm"] > 0:  # deterministic per nodule (independent of worker order)
            rng = np.random.default_rng([cfg["seed"], zlib.crc32(uid.encode())])
            th = rng.uniform(0, 2 * math.pi)
            rr = rng.uniform(0, cfg["random_offset_mm"])
            off += np.array([0.0, rr * math.sin(th), rr * math.cos(th)])
        start = crop_start(center + off, size)
        nodules.append(dict(
            uid=uid, mask_path=mp, start=start, off=off,
            center_crop=center - np.array(start),          # true nodule centre (GRIT descriptor)
            img=crop_with_pad(hu, start, size, HU_MIN).astype(np.int16),
            gt=crop_with_pad(union_gt, start, size, 0) if gts else None))
    return dict(scan_id=scan_id, pid=pid, image_path=image_path, ref_geom=geom_of(ref),
                orig_geom=orig_geom, gt_paths=list(gts), nodules=nodules, warnings=warnings)


# ----------------------------------------------------------------------------
# Stage 3 (CPU workers): post-process, metrics, NIfTI + PNG writing for one scan
# ----------------------------------------------------------------------------
def finish_scan(scan, probs, cfg):
    out_dir = Path(cfg["out_dir"])
    og, rg = scan["orig_geom"], scan["ref_geom"]
    rows = []
    scan_pred = None
    for nod, prob in zip(scan["nodules"], probs):
        uid, start, gt_c = nod["uid"], nod["start"], nod["gt"]
        pred = postprocess(prob, cfg["thr"], cfg["min_voxels"], cfg["min_slices"], cfg["keep"],
                           fill_holes=cfg["fill_holes"])
        p_orig = crop_to_original(pred, rg, start, og)
        scan_pred = p_orig if scan_pred is None else (scan_pred | p_orig)
        sitk.WriteImage(array_to_orig_image(p_orig, og), str(out_dir / "pred_nifti" / f"{uid}_pred.nii.gz"))

        rec = dict(uid=uid, pid=scan["pid"], scan_id=scan["scan_id"], image_path=scan["image_path"],
                   mask_path=nod["mask_path"], offset_z=nod["off"][0], offset_y=nod["off"][1],
                   offset_x=nod["off"][2], max_prob=float(prob.max()),
                   grit_threshold_pct=nod.get("grit_pct"))
        if gt_c is not None:
            rec.update(seg_metrics(pred, gt_c))
        rows.append(rec)

        if cfg["save_png"]:
            save_slice_pngs(nod["img"], pred, gt_c, out_dir / "png", uid, cfg["png_slices"])
        if cfg["save_crops"]:
            cd, stub = out_dir / "crops_1mm", geom_stub(rg)
            sitk.WriteImage(crop_to_sitk(nod["img"], stub, start), str(cd / f"{uid}_img.nii.gz"))
            sitk.WriteImage(crop_to_sitk(prob.astype(np.float32), stub, start), str(cd / f"{uid}_prob.nii.gz"))
            sitk.WriteImage(crop_to_sitk(pred, stub, start), str(cd / f"{uid}_pred.nii.gz"))
            if gt_c is not None:
                sitk.WriteImage(crop_to_sitk(gt_c, stub, start), str(cd / f"{uid}_gt.nii.gz"))

    scan_row = None
    if scan_pred is not None:
        sitk.WriteImage(array_to_orig_image(scan_pred, og),
                        str(out_dir / "pred_nifti" / f"{scan['scan_id']}_pred_all.nii.gz"))
        if scan["gt_paths"]:
            gg = np.zeros(scan_pred.shape, bool)
            for mp in scan["gt_paths"]:
                m = sitk.ReadImage(str(mp), sitk.sitkFloat32)
                m = sitk.Resample(sitk.Cast(m > 0, sitk.sitkFloat32), og["size"], sitk.Transform(),
                                  sitk.sitkNearestNeighbor, og["origin"], og["spacing"], og["direction"],
                                  0.0, sitk.sitkFloat32)
                gg |= sitk.GetArrayFromImage(m) > 0.5
            tp = (scan_pred & gg).sum()
            scan_row = dict(scan_id=scan["scan_id"], pid=scan["pid"], image_path=scan["image_path"],
                            dice_original_space=2 * tp / max(scan_pred.sum() + gg.sum(), 1))
    return rows, scan_row


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--roi", type=int, default=64, help="in-plane crop size (multiple of 16)")
    ap.add_argument("--depth", type=int, default=64, help="number of axial slices in the crop")
    ap.add_argument("--offset_mm", default="", help="fixed crop-centre offset 'dz,dy,dx' (mm)")
    ap.add_argument("--random_offset_mm", type=float, default=0.0,
                    help="random in-plane offset of the crop centre, magnitude U(0, R) mm, random direction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--min_voxels", type=int, default=8, help="min component size (mm^3 at 1 mm iso)")
    ap.add_argument("--min_slices", type=int, default=2, help="min z-extent of a component")
    ap.add_argument("--keep", choices=["all", "largest", "center"], default="all")
    ap.add_argument("--no_fill_holes", action="store_true")
    ap.add_argument("--tta", action="store_true", help="flip test-time augmentation")
    ap.add_argument("--save_crops", action="store_true", help="also save 1 mm crop img/prob/pred/gt NIfTIs")
    ap.add_argument("--save_png", action="store_true",
                    help="save per-slice PNGs (CT | prediction | GT | overlay) to <out_dir>/png/<nodule>/")
    ap.add_argument("--png_slices", choices=["union", "gt", "all"], default="union",
                    help="union = slices with GT or prediction (incl. false positives); "
                         "gt = GT slices only; all = every slice of the crop")
    # throughput
    ap.add_argument("--batch_size", type=int, default=1024,
                    help="slices per GPU forward pass (slices from many nodules/scans are mixed)")
    ap.add_argument("--load_workers", type=int, default=8, help="CPU workers loading/resampling CTs")
    ap.add_argument("--save_workers", type=int, default=8,
                    help="CPU workers for post-processing + writing NIfTI/PNG")
    args = ap.parse_args()
    assert args.roi % 16 == 0, "--roi must be a multiple of 16"

    out_dir = Path(args.out_dir)
    (out_dir / "pred_nifti").mkdir(parents=True, exist_ok=True)
    if args.save_crops:
        (out_dir / "crops_1mm").mkdir(parents=True, exist_ok=True)

    df = read_path_table(args.xlsx)
    df["_row"] = df.index
    has_mask = "mask_path" in df.columns
    has_ctr = all(c in df.columns for c in ("center_x", "center_y", "center_z"))
    cfg = dict(out_dir=str(out_dir), size=[args.depth, args.roi, args.roi],
               fixed_off=parse_offset(args.offset_mm).tolist(), random_offset_mm=args.random_offset_mm,
               seed=args.seed, has_mask=has_mask, has_ctr=has_ctr, thr=args.thr,
               min_voxels=args.min_voxels, min_slices=args.min_slices, keep=args.keep,
               fill_holes=not args.no_fill_holes, save_png=args.save_png,
               png_slices=args.png_slices, save_crops=args.save_crops)
    jobs = [(pid, s_idx, ip, g.to_dict("records"), cfg)
            for pid, gp in df.groupby("pid", sort=True)
            for s_idx, (ip, g) in enumerate(gp.groupby("image_path", sort=True))]
    n_nod = len(df)
    print(f"{len(jobs)} scans, {n_nod} nodules | GPU batch {args.batch_size} slices | "
          f"{args.load_workers} load workers, {args.save_workers} save workers")

    # pools are created BEFORE CUDA is initialised (workers never touch the GPU)
    load_pool, save_pool = make_pools(args.load_workers, args.save_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    model = load_model(args.ckpt, device)
    rows, scan_rows = run_batched_pipeline(jobs, load_scan, finish_scan, cfg, model, device,
                                           load_pool, save_pool, args.batch_size, args.tta)

    res = pd.DataFrame(rows).sort_values("uid") if rows else pd.DataFrame()
    res.to_csv(out_dir / "metrics_per_nodule.csv", index=False)
    if scan_rows:
        pd.DataFrame(scan_rows).sort_values("scan_id").to_csv(out_dir / "metrics_per_scan.csv", index=False)
    if len(res) and "dice" in res:
        print("\n=== summary (1 mm crop space) ===")
        for c in ["dice", "iou", "precision", "recall", "hd95_mm"]:
            print(f"{c:>10s}: mean {res[c].mean():.4f}  median {res[c].median():.4f}")
        fp_rate = res["n_fp_slices_on_empty"].sum() / max(res["n_gt_empty_slices"].sum(), 1)
        print(f"FP rate on GT-empty slices: {fp_rate:.4f}")
        if scan_rows:
            print(f"scan-level Dice (original space): {pd.DataFrame(scan_rows)['dice_original_space'].mean():.4f}")
    print(f"outputs -> {out_dir}")


if __name__ == "__main__":
    main()
