"""Shared inference machinery used by 03_predict_postprocess.py (LIDC val) and
04_infer_test_cohort.py (external cohort with centroid coordinates).

  * model loading + batched slice prediction
  * 3D post-processing, metrics, nodule size measurements
  * PNG visualisation
  * geometry helpers (1 mm crop <-> original CT grid)
  * run_batched_pipeline: parallel CT loading -> big mixed GPU batches -> parallel saving
"""
import multiprocessing
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage
from scipy.spatial.distance import pdist
from tqdm import tqdm

from common import CROP_SIZE, crop_to_sitk, normalize_hu
from grit import GritSelector, descriptor_from_crop, make_channels
from vnet2d import VNet2D


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    model = VNet2D(a.get("in_channels", 1), 1, norm=a.get("norm", "batch"), dropout=a.get("dropout", 0.2))
    sd = {k.replace("module.", "", 1): v for k, v in ck["model"].items()}
    model.load_state_dict(sd)
    # GRIT checkpoints carry the K-means centres fitted on their training nodules
    model.grit = GritSelector.from_dict(a["grit_selector"]) if a.get("grit_selector") else None
    return model.to(device).eval()


def grit_level(model, img_hu_crop, center_in_crop):
    """GRIT threshold % for one nodule crop (None for 1-channel models)."""
    if getattr(model, "grit", None) is None:
        return None
    return model.grit.level_of(descriptor_from_crop(normalize_hu(img_hu_crop), center_in_crop))


@torch.no_grad()
def predict_volume(model, img_zyx, device, batch=64, tta=False, grit_pct=None):
    """img_zyx: normalised [0,1] (N, H, W) stack of axial slices -> probabilities (N, H, W).
    GRIT models (model.grit set) also need grit_pct: threshold % per slice (scalar or length N);
    the 3 channels are built per slice exactly as in training."""
    out = np.zeros(img_zyx.shape, dtype=np.float32)
    use_grit = getattr(model, "grit", None) is not None
    if use_grit:
        if grit_pct is None:
            raise ValueError("this is a GRIT model: pass grit_pct (see grit_level())")
        grit_pct = np.broadcast_to(np.asarray(grit_pct, dtype=np.float64), (img_zyx.shape[0],))
    for z0 in range(0, img_zyx.shape[0], batch):
        chunk = np.ascontiguousarray(img_zyx[z0:z0 + batch])
        if use_grit:
            # threshold cut-off from the central 64x64 (training FOV) when slices are bigger
            x = torch.from_numpy(make_channels(chunk, grit_pct[z0:z0 + batch], ref_size=CROP_SIZE)).float().to(device)
        else:
            x = torch.from_numpy(chunk)[:, None].float().to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            p = torch.sigmoid(model(x).float())
            if tta:
                p = p + torch.sigmoid(model(x.flip(-1)).float()).flip(-1)
                p = p + torch.sigmoid(model(x.flip(-2)).float()).flip(-2)
                p = p + torch.sigmoid(model(x.flip(-1, -2)).float()).flip(-1, -2)
                p = p / 4
        out[z0:z0 + batch] = p[:, 0].float().cpu().numpy()
    return out


# ----------------------------------------------------------------------------
# Post-processing
# ----------------------------------------------------------------------------
def postprocess(prob, thr=0.5, min_voxels=8, min_slices=2, keep="all", fill_holes=True, center=None):
    """threshold -> per-slice hole filling -> 3D connected components -> size/extent filter -> keep.
    keep: all | largest | center (component closest to `center` (z,y,x); default = crop centre)."""
    b = prob > thr
    if fill_holes:
        for z in range(b.shape[0]):
            if b[z].any():
                b[z] = ndimage.binary_fill_holes(b[z])
    lab, n = ndimage.label(b, structure=np.ones((3, 3, 3), bool))
    if n == 0:
        return b.astype(np.uint8)
    comps = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        m = lab[sl] == i
        size = int(m.sum())
        z_extent = sl[0].stop - sl[0].start
        if size < min_voxels or z_extent < min_slices:
            continue
        comps.append((i, size, sl))
    if not comps:
        return np.zeros_like(b, dtype=np.uint8)
    if keep == "largest":
        comps = [max(comps, key=lambda c: c[1])]
    elif keep == "center":
        ctr = np.asarray(center if center is not None else (np.array(b.shape) - 1) / 2, dtype=float)

        def dist(c):  # distance from centre to the NEAREST voxel of the component
            i, _, sl = c
            vox = np.argwhere(lab[sl] == i) + np.array([s.start for s in sl])
            return np.min(np.linalg.norm(vox - ctr, axis=1))
        comps = [min(comps, key=dist)]
    return np.isin(lab, [c[0] for c in comps]).astype(np.uint8)


# ----------------------------------------------------------------------------
# Metrics / measurements
# ----------------------------------------------------------------------------
def hd95(a, b):
    if not a.any() or not b.any():
        return float("nan")
    sa = a & ~ndimage.binary_erosion(a)
    sb = b & ~ndimage.binary_erosion(b)
    d_ab = ndimage.distance_transform_edt(~sb)[sa]
    d_ba = ndimage.distance_transform_edt(~sa)[sb]
    return float(np.percentile(np.concatenate([d_ab, d_ba]), 95))


def seg_metrics(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    gt_empty_z = ~gt.any(axis=(1, 2))
    fp_slices = (pred.any(axis=(1, 2)) & gt_empty_z).sum()
    return dict(
        dice=2 * tp / max(2 * tp + fp + fn, 1) if (gt.any() or pred.any()) else 1.0,
        iou=tp / max(tp + fp + fn, 1) if (gt.any() or pred.any()) else 1.0,
        precision=tp / max(tp + fp, 1) if pred.any() else float("nan"),
        recall=tp / max(tp + fn, 1) if gt.any() else float("nan"),
        hd95_mm=hd95(pred, gt),
        pred_vol_mm3=int(pred.sum()), gt_vol_mm3=int(gt.sum()),
        n_gt_empty_slices=int(gt_empty_z.sum()), n_fp_slices_on_empty=int(fp_slices),
    )


def size_measurements(pred_1mm):
    """Measurements of a binary mask on the 1 mm isotropic grid."""
    pred = pred_1mm.astype(bool)
    vol = int(pred.sum())
    out = dict(pred_vol_mm3=vol,
               equiv_sphere_diam_mm=float((6 * vol / np.pi) ** (1 / 3)) if vol else 0.0,
               n_pred_slices=int(pred.any(axis=(1, 2)).sum()),
               max_axial_diam_mm=0.0)
    best = 0.0
    for z in np.where(pred.any(axis=(1, 2)))[0]:
        s = pred[z]
        edge = np.argwhere(s & ~ndimage.binary_erosion(s))
        if len(edge) >= 2:
            best = max(best, float(pdist(edge).max()) + 1.0)  # +1 mm: pixel-centre -> pixel-edge extent
        elif len(edge) == 1:
            best = max(best, 1.0)
    out["max_axial_diam_mm"] = best
    return out


# ----------------------------------------------------------------------------
# PNG visualisation
# ----------------------------------------------------------------------------
GT_COLOR = "#35b8ff"    # ground truth: cyan-blue
PRED_COLOR = "#ff7a1a"  # prediction:   orange   (blue/orange stays distinct for colour-blind readers)


def select_png_slices(pred, gt, mode, center_z=None):
    has_p = pred.any(axis=(1, 2))
    has_g = gt.any(axis=(1, 2)) if gt is not None else np.zeros_like(has_p)
    if mode == "gt":
        keep = has_g if gt is not None else has_p
    elif mode == "all":
        keep = np.ones_like(has_p)
    else:  # "union": every slice with GT or a prediction (shows false positives too)
        keep = has_p | has_g
    zs = np.where(keep)[0]
    if len(zs) == 0 and center_z is not None:  # nothing predicted: still show the target slices
        cz = int(round(center_z))
        zs = np.array([z for z in (cz - 1, cz, cz + 1) if 0 <= z < pred.shape[0]])
    return zs


def save_slice_pngs(img_hu, pred, gt, out_dir, uid, mode="union", dpi=110, center=None):
    """One PNG per axial slice.
    with GT   : CT | prediction | ground truth | both overlaid
    without GT: CT | prediction
    `center` (z,y,x) in crop voxels is marked with a small cross (the given nodule centroid)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out_dir = Path(out_dir) / uid
    out_dir.mkdir(parents=True, exist_ok=True)
    zs = select_png_slices(pred, gt, mode, center_z=None if center is None else center[0])
    for z in zs:
        im, p = img_hu[z], pred[z].astype(bool)
        sp = int(p.sum())
        if gt is not None:
            g = gt[z].astype(bool)
            tp, sg = (p & g).sum(), int(g.sum())
            dice_txt = "both empty" if sp + sg == 0 else f"Dice {2 * tp / (sp + sg):.3f}"
            titles = ["CT", f"Prediction ({sp} px)", f"Ground truth ({sg} px)", f"Overlay  -  {dice_txt}"]
            fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.6))
        else:
            titles = ["CT", f"Prediction ({sp} px)"]
            fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.6))
        for i, ax in enumerate(axes):
            ax.imshow(im, cmap="gray", vmin=-1000, vmax=400, interpolation="nearest")
            ax.set_title(titles[i], fontsize=10)
            ax.axis("off")
        if gt is not None:
            if p.any():
                for ax in (axes[1], axes[3]):
                    ax.contour(p, levels=[0.5], colors=PRED_COLOR, linewidths=1.6)
            if g.any():
                for ax in (axes[2], axes[3]):
                    ax.contour(g, levels=[0.5], colors=GT_COLOR, linewidths=1.6)
            axes[3].legend(handles=[Line2D([], [], color=GT_COLOR, lw=2, label="ground truth"),
                                    Line2D([], [], color=PRED_COLOR, lw=2, label="prediction")],
                           loc="lower right", fontsize=7, framealpha=0.7)
        else:
            if p.any():
                axes[1].contour(p, levels=[0.5], colors=PRED_COLOR, linewidths=1.6)
        if center is not None and abs(z - center[0]) <= 3:
            for ax in axes:
                ax.plot(center[2], center[1], marker="+", color="#ffffff", ms=7, mew=1.0, alpha=0.8)
        fig.suptitle(f"{uid}   slice z={z}", fontsize=10, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(out_dir / f"z{z:03d}.png", dpi=dpi, facecolor="white")
        plt.close(fig)
    return len(zs)


# ----------------------------------------------------------------------------
# Geometry helpers (lets worker processes rebuild grids without passing images)
# ----------------------------------------------------------------------------
def geom_of(img):
    return dict(size=list(img.GetSize()), origin=list(img.GetOrigin()),
                spacing=list(img.GetSpacing()), direction=list(img.GetDirection()))


def header_geom(path):
    rd = sitk.ImageFileReader()
    rd.SetFileName(str(path))
    rd.ReadImageInformation()  # header only, no pixel data
    return dict(size=list(rd.GetSize()), origin=list(rd.GetOrigin()),
                spacing=list(rd.GetSpacing()), direction=list(rd.GetDirection()))


def geom_stub(g):
    """1x1x1 image carrying a grid's origin/spacing/direction (index<->physical mapping only)."""
    s = sitk.Image([1, 1, 1], sitk.sitkUInt8)
    s.SetOrigin(g["origin"])
    s.SetSpacing(g["spacing"])
    s.SetDirection(g["direction"])
    return s


def crop_to_original(pred_crop, ref_geom, start, orig_geom):
    """Resample a 1 mm crop prediction straight onto the original CT grid (outside crop = 0)."""
    s = crop_to_sitk(pred_crop.astype(np.float32), geom_stub(ref_geom), start)
    r = sitk.Resample(s, orig_geom["size"], sitk.Transform(), sitk.sitkLinear,
                      orig_geom["origin"], orig_geom["spacing"], orig_geom["direction"],
                      0.0, sitk.sitkFloat32)
    return sitk.GetArrayFromImage(r) >= 0.5


def array_to_orig_image(arr, orig_geom):
    out = sitk.GetImageFromArray(arr.astype(np.uint8))
    out.SetOrigin(orig_geom["origin"])
    out.SetSpacing(orig_geom["spacing"])
    out.SetDirection(orig_geom["direction"])
    return out


def parse_offset(s):
    if not s:
        return np.zeros(3)
    v = [float(t) for t in s.split(",")]
    assert len(v) == 3, "--offset_mm expects dz,dy,dx"
    return np.array(v)


# ----------------------------------------------------------------------------
# Parallel pipeline
# ----------------------------------------------------------------------------
def init_worker():
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)


def make_pools(n_load, n_save):
    """Create worker pools BEFORE CUDA is initialised (workers never touch the GPU)."""
    ctx = multiprocessing.get_context("fork")
    return ctx.Pool(n_load, initializer=init_worker), ctx.Pool(n_save, initializer=init_worker)


def run_batched_pipeline(jobs, load_fn, finish_fn, cfg, model, device, load_pool, save_pool,
                         batch_size=1024, tta=False):
    """load_fn(job) -> scan dict with 'nodules' (each has 'img': int16 (D,H,W) HU crop),
                       'warnings', or 'error'.
    finish_fn(scan, probs, cfg) -> (list_of_nodule_rows, scan_row_or_None).
    Slices of many nodules/scans are stacked into GPU batches of `batch_size`."""
    rows, scan_rows, pending = [], [], []
    max_pending = 4 * save_pool._processes
    pbar_save = tqdm(total=len(jobs), desc="saved ", position=1)

    def collect(block_until=0):
        while len(pending) > block_until:
            r, s = pending.pop(0).get()
            rows.extend(r)
            if s is not None:
                scan_rows.append(s)
            pbar_save.update(1)

    buffer, n_buf = [], 0

    def flush():
        nonlocal buffer, n_buf
        if not buffer:
            return
        crops = [n["img"] for sc in buffer for n in sc["nodules"]]
        probs = []
        if crops:
            pct = None
            if getattr(model, "grit", None) is not None:  # one threshold level per nodule
                for sc in buffer:
                    for n in sc["nodules"]:
                        c = n.get("center_crop", (np.array(n["img"].shape) - 1) / 2)
                        n["grit_pct"] = grit_level(model, n["img"], c)
                pct = np.concatenate([np.full(n["img"].shape[0], n["grit_pct"])
                                      for sc in buffer for n in sc["nodules"]])
            stacked = normalize_hu(np.concatenate(crops, axis=0))
            prob_all = predict_volume(model, stacked, device, batch=batch_size, tta=tta, grit_pct=pct)
            probs = np.split(prob_all, len(crops), axis=0)
        k = 0
        for sc in buffer:
            nn = len(sc["nodules"])
            pending.append(save_pool.apply_async(finish_fn, (sc, probs[k:k + nn], cfg)))
            k += nn
        collect(block_until=max_pending)  # back-pressure if saving falls behind
        buffer, n_buf = [], 0

    for sc in tqdm(load_pool.imap_unordered(load_fn, jobs), total=len(jobs), desc="loaded", position=0):
        if "error" in sc:
            tqdm.write(f"[skip] {sc['error']}")
            pbar_save.update(1)
            continue
        for w in sc.get("warnings", []):
            tqdm.write(w)
        buffer.append(sc)
        n_buf += sum(n["img"].shape[0] for n in sc["nodules"])
        if n_buf >= batch_size:
            flush()
    flush()
    collect(0)
    pbar_save.close()
    load_pool.close()
    save_pool.close()
    load_pool.join()
    save_pool.join()
    return rows, scan_rows


# ----------------------------------------------------------------------------
# Adaptive field of view (two-pass inference)
# ----------------------------------------------------------------------------
def touches_border(pred, margin=3):
    """True if any predicted voxel lies within `margin` voxels of ANY face of the crop
    (in-plane edges or first/last slices) -> the nodule may be cut off by the crop."""
    if margin <= 0 or not pred.any():
        return False
    m = margin
    return bool(pred[:m].any() or pred[-m:].any() or pred[:, :m].any() or pred[:, -m:].any()
                or pred[:, :, :m].any() or pred[:, :, -m:].any())


def crop_point_to_phys(zyx_in_crop, start, ref_geom):
    """(z,y,x) in crop voxels -> physical LPS point (x,y,z) in mm."""
    z, y, x = (float(v) + float(s) for v, s in zip(zyx_in_crop, start))
    return list(geom_stub(ref_geom).TransformContinuousIndexToPhysicalPoint((x, y, z)))


def phys_to_ref_zyx(phys, ref_geom):
    i, j, k = geom_stub(ref_geom).TransformPhysicalPointToContinuousIndex([float(v) for v in phys])
    return np.array([k, j, i])


def mask_center_phys(pred, start, ref_geom):
    return crop_point_to_phys(ndimage.center_of_mass(pred > 0), start, ref_geom)


def union_from_files(pred_dir, uids, orig_geom):
    """OR of the per-nodule original-space masks <pred_dir>/<uid>_pred.nii.gz."""
    out = None
    for u in uids:
        f = Path(pred_dir) / f"{u}_pred.nii.gz"
        if not f.exists():
            continue
        a = sitk.GetArrayFromImage(sitk.ReadImage(str(f))) > 0
        out = a if out is None else (out | a)
    if out is None:
        out = np.zeros(list(reversed(orig_geom["size"])), bool)
    return out


def gt_union_original_space(gt_paths, orig_geom):
    og = orig_geom
    gg = np.zeros(list(reversed(og["size"])), bool)
    for mp in gt_paths:
        m = sitk.ReadImage(str(mp), sitk.sitkFloat32)
        m = sitk.Resample(sitk.Cast(m > 0, sitk.sitkFloat32), og["size"], sitk.Transform(),
                          sitk.sitkNearestNeighbor, og["origin"], og["spacing"], og["direction"],
                          0.0, sitk.sitkFloat32)
        gg |= sitk.GetArrayFromImage(m) > 0.5
    return gg


def scaled_batch(batch_size, base_size, new_size):
    """Keep GPU memory roughly constant when slices get bigger (e.g. 64^2 -> 128^2: batch / 4)."""
    return max(8, int(batch_size * (base_size[1] * base_size[2]) / (new_size[1] * new_size[2])))
