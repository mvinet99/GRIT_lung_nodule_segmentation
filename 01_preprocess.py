#!/usr/bin/env python
"""
Step 1 - Preprocess LIDC for 2D nodule segmentation.

For every CT listed in LIDC_paths.xlsx:
  1. Reorient to LPS and resample to 1 x 1 x 1 mm (linear for CT, linear+0.5 threshold for masks).
  2. Clip HU to [-1000, 400].
  3. Build a UNION mask of all nodules of that scan (so a neighbouring nodule that
     falls inside another nodule's crop is labelled as foreground, not background).
  4. For every nodule (row), compute its centroid from its own mask and crop a
     64 (z) x 64 x 64 cube around it.  The cube is stored with extra in-plane
     context (default 128 x 128) so the training "shift" augmentation can move
     the nodule off-centre using REAL anatomy instead of padding.
     -> the central 64 x 64 x 64 region is exactly the crop you asked for.
  5. Optionally sample random nodule-free crops from inside the lungs
     (extra hard negatives: vessels, airways, pleura).
  6. Write a slice index:
       label 1  = axial slice whose central 64x64 window contains nodule pixels
       label 0  = axial slice whose whole (context) plane is empty
       (slices with nodule pixels only in the outer context ring are skipped)

Outputs (in --out_dir):
  crops/<crop_id>_img.npy   int16  (64, C, C)  clipped HU
  crops/<crop_id>_msk.npy   uint8  (64, C, C)  union nodule mask
  qc_nifti/<crop_id>_{img,msk}.nii.gz          central 64^3 crop with real geometry (for ITK-SNAP)
  crops_manifest.csv                            one row per crop
  slices_index.csv                              one row per usable axial slice
  preprocess_errors.csv                         anything that failed

Usage:
  python 01_preprocess.py --xlsx LIDC_paths.xlsx --out_dir /radraid/.../LIDC_2d_prep --workers 16
"""
import argparse
import hashlib
import os
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

from common import (CROP_SIZE, HU_MIN, centroid_zyx, crop_start, crop_to_sitk,
                    crop_with_pad, load_ct_1mm, read_path_table,
                    resample_mask_to_ref, rough_lung_mask)


def nodule_stem(mask_path):
    name = Path(mask_path).name
    for ext in (".nii.gz", ".nii", ".nrrd", ".mha"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return Path(mask_path).stem


def slice_labels(msk, ctx, patch=CROP_SIZE):
    """Return list of (z, label) for one crop. See module docstring."""
    c0 = (ctx - patch) // 2
    out = []
    for z in range(msk.shape[0]):
        if msk[z, c0:c0 + patch, c0:c0 + patch].any():
            out.append((z, 1))
        elif not msk[z].any():
            out.append((z, 0))
    return out


def save_crop(out_dir, crop_id, img_c, msk_c, ref_img, start, ctx, save_qc):
    np.save(out_dir / "crops" / f"{crop_id}_img.npy", img_c.astype(np.int16))
    np.save(out_dir / "crops" / f"{crop_id}_msk.npy", msk_c.astype(np.uint8))
    if save_qc:
        c0 = (ctx - CROP_SIZE) // 2
        sl = (slice(None), slice(c0, c0 + CROP_SIZE), slice(c0, c0 + CROP_SIZE))
        qc_start = [start[0], start[1] + c0, start[2] + c0]
        sitk.WriteImage(crop_to_sitk(img_c[sl].astype(np.int16), ref_img, qc_start),
                        str(out_dir / "qc_nifti" / f"{crop_id}_img.nii.gz"))
        sitk.WriteImage(crop_to_sitk(msk_c[sl].astype(np.uint8), ref_img, qc_start),
                        str(out_dir / "qc_nifti" / f"{crop_id}_msk.nii.gz"))


def process_scan(job):
    scan_id, pid, image_path, nodules, cfg = job
    out_dir = Path(cfg["out_dir"])
    ctx = cfg["context"]
    size = (CROP_SIZE, ctx, ctx)
    manifest, slices, errors = [], [], []
    try:
        ref, hu = load_ct_1mm(image_path)

        # --- resample every nodule mask of this scan onto the 1 mm grid
        masks = []
        for mask_path in nodules:
            try:
                m = resample_mask_to_ref(mask_path, ref)
                if not m.any():
                    errors.append(dict(pid=pid, image_path=image_path, mask_path=mask_path,
                                       error="mask empty after resampling"))
                    continue
                masks.append((mask_path, m))
            except Exception as e:  # noqa: BLE001
                errors.append(dict(pid=pid, image_path=image_path, mask_path=mask_path, error=repr(e)))
        union = np.zeros(hu.shape, dtype=np.uint8)
        for _, m in masks:
            union |= m

        # --- one crop per nodule, centred on that nodule's centroid
        for mask_path, m in masks:
            cz, cy, cx = centroid_zyx(m)
            start = crop_start((cz, cy, cx), size)
            img_c = crop_with_pad(hu, start, size, HU_MIN)
            msk_c = crop_with_pad(union, start, size, 0)
            crop_id = f"{scan_id}_{nodule_stem(mask_path)}"
            save_crop(out_dir, crop_id, img_c, msk_c, ref, start, ctx, cfg["save_qc"])
            lab = slice_labels(msk_c, ctx)
            phys = ref.TransformContinuousIndexToPhysicalPoint([float(cx), float(cy), float(cz)])
            manifest.append(dict(
                crop_id=crop_id, pid=pid, scan_id=scan_id, kind="nodule",
                image_path=image_path, mask_path=mask_path,
                center_z=cz, center_y=cy, center_x=cx,
                center_phys_x=phys[0], center_phys_y=phys[1], center_phys_z=phys[2],
                start_z=start[0], start_y=start[1], start_x=start[2], context=ctx,
                nodule_volume_mm3=int(m.sum()),
                n_pos_slices=sum(1 for _, l in lab if l == 1),
                n_neg_slices=sum(1 for _, l in lab if l == 0)))
            slices += [dict(crop_id=crop_id, pid=pid, scan_id=scan_id, kind="nodule", z=z, label=l)
                       for z, l in lab]

        # --- random nodule-free background crops inside the lungs
        n_bg = cfg["bg_crops_per_scan"]
        if n_bg > 0:
            seed = int(hashlib.md5(image_path.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            lung = rough_lung_mask(hu)
            cand = np.argwhere(lung)
            if len(cand) == 0:  # fallback: any low-density voxel in the central region
                zz, yy, xx = hu.shape
                sub = hu[zz // 5: 4 * zz // 5, yy // 5: 4 * yy // 5, xx // 5: 4 * xx // 5] < -500
                cand = np.argwhere(sub) + np.array([zz // 5, yy // 5, xx // 5])
            made = 0
            for _ in range(200):
                if made >= n_bg or len(cand) == 0:
                    break
                c = cand[rng.integers(len(cand))]
                start = crop_start(c, size)
                if crop_with_pad(union, start, size, 0).any():
                    continue  # touches an annotated nodule
                img_c = crop_with_pad(hu, start, size, HU_MIN)
                msk_c = np.zeros(size, dtype=np.uint8)
                crop_id = f"{scan_id}_bg{made + 1}"
                save_crop(out_dir, crop_id, img_c, msk_c, ref, start, ctx, cfg["save_qc"])
                phys = ref.TransformContinuousIndexToPhysicalPoint([float(c[2]), float(c[1]), float(c[0])])
                manifest.append(dict(
                    crop_id=crop_id, pid=pid, scan_id=scan_id, kind="background",
                    image_path=image_path, mask_path="",
                    center_z=c[0], center_y=c[1], center_x=c[2],
                    center_phys_x=phys[0], center_phys_y=phys[1], center_phys_z=phys[2],
                    start_z=start[0], start_y=start[1], start_x=start[2], context=ctx,
                    nodule_volume_mm3=0, n_pos_slices=0, n_neg_slices=CROP_SIZE))
                slices += [dict(crop_id=crop_id, pid=pid, scan_id=scan_id, kind="background", z=z, label=0)
                           for z in range(CROP_SIZE)]
                made += 1
    except Exception:  # noqa: BLE001
        errors.append(dict(pid=pid, image_path=image_path, mask_path="", error=traceback.format_exc()))
    return manifest, slices, errors


def init_worker():
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xlsx", required=True, help="LIDC_paths.xlsx (pid, image_path, mask_path)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--context", type=int, default=128,
                    help="in-plane size stored around each 64^3 crop (>=64). Extra context lets the "
                         "shift augmentation use real anatomy. 64 = store exactly 64^3 (shift then pads).")
    ap.add_argument("--bg_crops_per_scan", type=int, default=1,
                    help="random nodule-free lung crops per scan (extra negatives). 0 to disable.")
    ap.add_argument("--no_qc_nifti", action="store_true", help="skip writing QC NIfTI crops")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--limit", type=int, default=0, help="debug: only process first N scans")
    args = ap.parse_args()
    assert args.context >= CROP_SIZE and args.context % 2 == 0

    out_dir = Path(args.out_dir)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    (out_dir / "qc_nifti").mkdir(parents=True, exist_ok=True)

    df = read_path_table(args.xlsx)
    if "mask_path" not in df.columns:
        raise ValueError("Training spreadsheet needs a mask_path column")
    print(f"{len(df)} nodule rows, {df['image_path'].nunique()} scans, {df['pid'].nunique()} patients")

    cfg = dict(out_dir=str(out_dir), context=args.context,
               bg_crops_per_scan=args.bg_crops_per_scan, save_qc=not args.no_qc_nifti)

    # group rows by scan (a patient can have >1 CT series)
    jobs = []
    for pid, g_pid in df.groupby("pid", sort=True):
        for s_idx, (image_path, g) in enumerate(g_pid.groupby("image_path", sort=True)):
            scan_id = f"{pid}_s{s_idx}"
            jobs.append((scan_id, pid, image_path, list(dict.fromkeys(g["mask_path"])), cfg))
    if args.limit:
        jobs = jobs[: args.limit]

    manifest, slices, errors = [], [], []
    with Pool(args.workers, initializer=init_worker) as pool:
        for m, s, e in tqdm(pool.imap_unordered(process_scan, jobs), total=len(jobs), desc="scans"):
            manifest += m
            slices += s
            errors += e

    man = pd.DataFrame(manifest).sort_values("crop_id")
    sl = pd.DataFrame(slices).sort_values(["crop_id", "z"])
    man.to_csv(out_dir / "crops_manifest.csv", index=False)
    sl.to_csv(out_dir / "slices_index.csv", index=False)
    pd.DataFrame(errors, columns=["pid", "image_path", "mask_path", "error"]).to_csv(
        out_dir / "preprocess_errors.csv", index=False)

    n_nod = (man["kind"] == "nodule").sum() if len(man) else 0
    print(f"\nnodule crops: {n_nod}   background crops: {len(man) - n_nod}")
    if len(sl):
        print(f"positive slices: {(sl.label == 1).sum()}   "
              f"negative slices (nodule crops): {((sl.label == 0) & (sl.kind == 'nodule')).sum()}   "
              f"negative slices (background crops): {(sl.kind == 'background').sum()}")
    print(f"errors: {len(errors)}  (see preprocess_errors.csv)")
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
