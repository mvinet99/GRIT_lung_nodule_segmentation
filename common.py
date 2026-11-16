"""Shared helpers: I/O, orientation, resampling, HU clipping, cropping.

Everything that must be identical between preprocessing (training) and
inference lives here so the two can never drift apart.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

# ----------------------------------------------------------------------------
# Global constants
# ----------------------------------------------------------------------------
HU_MIN, HU_MAX = -1000.0, 400.0
TARGET_SPACING = (1.0, 1.0, 1.0)      # (x, y, z) in mm, SimpleITK order
ORIENTATION = "LPS"                    # after reorientation, numpy axis 0 = axial (z)
CROP_SIZE = 64                         # 64 x 64 x 64 crop around each nodule


# ----------------------------------------------------------------------------
# Spreadsheet
# ----------------------------------------------------------------------------
def read_path_table(path):
    """Read LIDC_paths.xlsx (or .csv). Needs columns pid, image_path, (mask_path)."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]  # drop index col
    df.columns = [str(c).strip() for c in df.columns]
    for col in ("pid", "image_path"):
        if col not in df.columns:
            raise ValueError(f"Spreadsheet is missing required column '{col}'")
    for col in ("pid", "image_path", "mask_path"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Image I/O + resampling
# ----------------------------------------------------------------------------
def read_oriented(path, pixel_type=sitk.sitkFloat32):
    """Read a NIfTI and reorient so that numpy axis 0 is the axial (S-I) axis."""
    img = sitk.ReadImage(str(path), pixel_type)
    return sitk.DICOMOrient(img, ORIENTATION)


def resample_to_spacing(img, spacing=TARGET_SPACING, interp=sitk.sitkLinear, default=HU_MIN):
    in_sp = np.array(img.GetSpacing(), dtype=float)
    in_sz = np.array(img.GetSize(), dtype=float)
    out_sz = np.maximum(np.round(in_sz * in_sp / np.array(spacing)), 1).astype(int)
    f = sitk.ResampleImageFilter()
    f.SetOutputSpacing(tuple(float(s) for s in spacing))
    f.SetSize([int(s) for s in out_sz])
    f.SetOutputOrigin(img.GetOrigin())
    f.SetOutputDirection(img.GetDirection())
    f.SetInterpolator(interp)
    f.SetDefaultPixelValue(float(default))
    f.SetOutputPixelType(sitk.sitkFloat32)
    return f.Execute(img)


def load_ct_1mm(image_path):
    """Load CT -> LPS -> 1 mm isotropic -> HU clipped. Returns (sitk_img_1mm, np zyx float32)."""
    img = read_oriented(image_path)
    img = resample_to_spacing(img, TARGET_SPACING, sitk.sitkLinear, HU_MIN)
    arr = np.clip(sitk.GetArrayFromImage(img), HU_MIN, HU_MAX).astype(np.float32)
    return img, arr


def resample_mask_to_ref(mask_path, ref_img):
    """Resample a binary mask onto the (1 mm) reference grid.

    Uses linear interpolation of the binary mask followed by a 0.5 threshold
    (smoother than nearest neighbour). Falls back to nearest neighbour for
    tiny nodules that would otherwise vanish.
    """
    m = sitk.ReadImage(str(mask_path), sitk.sitkFloat32)
    m = sitk.Cast(m > 0, sitk.sitkFloat32)
    out = sitk.Resample(m, ref_img, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(out) >= 0.5
    if not arr.any():
        out = sitk.Resample(m, ref_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0, sitk.sitkFloat32)
        arr = sitk.GetArrayFromImage(out) > 0.5
    return arr.astype(np.uint8)


def normalize_hu(arr):
    """Clipped HU -> [0, 1]."""
    arr = np.clip(arr.astype(np.float32), HU_MIN, HU_MAX)
    return (arr - HU_MIN) / (HU_MAX - HU_MIN)


# ----------------------------------------------------------------------------
# Cropping
# ----------------------------------------------------------------------------
def crop_with_pad(arr, start, size, pad_value):
    """Crop arr[start : start+size] (any ndim) with constant padding where out of bounds."""
    start = [int(s) for s in start]
    size = [int(s) for s in size]
    out = np.full(size, pad_value, dtype=arr.dtype)
    src, dst = [], []
    for s, n, dim in zip(start, size, arr.shape):
        a0, a1 = max(s, 0), min(s + n, dim)
        if a1 <= a0:
            return out
        src.append(slice(a0, a1))
        dst.append(slice(a0 - s, a1 - s))
    out[tuple(dst)] = arr[tuple(src)]
    return out


def paste_into(dest, patch, start):
    """Inverse of crop_with_pad: write patch into dest at start (clipping to bounds)."""
    src, dst = [], []
    for s, n, dim in zip(start, patch.shape, dest.shape):
        a0, a1 = max(int(s), 0), min(int(s) + n, dim)
        if a1 <= a0:
            return dest
        dst.append(slice(a0, a1))
        src.append(slice(a0 - int(s), a1 - int(s)))
    dest[tuple(dst)] = patch[tuple(src)]
    return dest


def crop_start(center_zyx, size_zyx):
    return [int(round(c)) - s // 2 for c, s in zip(center_zyx, size_zyx)]


def centroid_zyx(mask):
    return np.array(ndimage.center_of_mass(mask > 0), dtype=float)


def crop_to_sitk(arr_zyx, ref_img, start_zyx):
    """Wrap a crop as a SimpleITK image with correct physical geometry (for QC in ITK-SNAP)."""
    out = sitk.GetImageFromArray(arr_zyx)
    origin = ref_img.TransformContinuousIndexToPhysicalPoint(
        [float(start_zyx[2]), float(start_zyx[1]), float(start_zyx[0])])
    out.SetOrigin(origin)
    out.SetSpacing(ref_img.GetSpacing())
    out.SetDirection(ref_img.GetDirection())
    return out


# ----------------------------------------------------------------------------
# Rough lung mask (used to sample random nodule-free "background" crops)
# ----------------------------------------------------------------------------
def rough_lung_mask(hu_zyx, thr=-320.0, min_vol=50_000):
    """Classic threshold + connected-component lung mask. Rough but adequate for
    choosing random background crop centres inside the lungs."""
    air = hu_zyx < thr
    lab, n = ndimage.label(air)
    if n == 0:
        return np.zeros_like(air)
    border = np.unique(np.concatenate([
        lab[:, 0, :].ravel(), lab[:, -1, :].ravel(),
        lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    sizes = np.bincount(lab.ravel())
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes[1:] >= min_vol
    keep[border] = False
    keep[0] = False
    lung = keep[lab]
    if lung.any():
        lung = ndimage.binary_erosion(lung, iterations=3)
    return lung
