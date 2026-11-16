#!/usr/bin/env python
"""
Compare two folders of predicted masks (same file names, e.g. before / after a model change):
number of predicted voxels per file, how it moved, and how much the two masks overlap.

  python compare_pred_niftis.py \
      --before /radraid2/mvinet/Lung/Current/code/09_29_26_model_retraining/pred_nlst_subcohort/pred_nifti \
      --after  /radraid2/mvinet/Lung/Current/code/10_13_26_model_retraining/run_digcs/pred_nlst/pred_nifti \
      --out_dir /radraid2/mvinet/Lung/Current/code/10_13_26_model_retraining/compare_before_after

Outputs (in --out_dir):
  voxel_change_dumbbell.png   one column per nodule (sorted by "before" count):
                              coloured dot = before, grey dot = after, line = the move
  voxel_change_scatter.png    before vs after with the identity line
  voxel_comparison.csv        per file: voxels/volume before & after, change, Dice(before, after), flags
  unmatched_files.csv         files present in only one of the two folders
Only per-nodule masks (*_pred.nii.gz) are compared; add --include_scan_union for *_pred_all.nii.gz too.
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import matplotlib
import matplotlib.ticker  # noqa: F401
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import SimpleITK as sitk  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

BEFORE_C = "#2a78d6"   # blue: original position
AFTER_C = "#8c8c8c"    # grey: where it moved to
LINE_C = "#cfcfcf"


def nodule_name(path):
    n = path.name
    return n[:-len(".nii.gz")] if n.endswith(".nii.gz") else path.stem


def compare_one(args):
    name, f_before, f_after = args
    try:
        a = sitk.ReadImage(str(f_before))
        b = sitk.ReadImage(str(f_after))
        ma = sitk.GetArrayFromImage(a) > 0
        mb = sitk.GetArrayFromImage(b) > 0
        va = float(np.prod(a.GetSpacing()))
        vb = float(np.prod(b.GetSpacing()))
        na, nb = int(ma.sum()), int(mb.sum())
        same_grid = ma.shape == mb.shape and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-3) \
            and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-4)
        if same_grid and (na + nb) > 0:
            dice = 2 * int((ma & mb).sum()) / (na + nb)
        elif same_grid:
            dice = 1.0  # both empty
        else:
            dice = np.nan  # different grids: overlap not computed
        return dict(name=name, voxels_before=na, voxels_after=nb,
                    vol_mm3_before=na * va, vol_mm3_after=nb * vb,
                    dice_before_after=dice, same_grid=same_grid, error="")
    except Exception as e:  # noqa: BLE001
        return dict(name=name, error=repr(e))


def dumbbell(df, out_png, unit_label):
    d = df.sort_values(["before", "after"]).reset_index(drop=True)
    x = np.arange(len(d))
    w = min(max(8, len(d) * 0.012), 30)
    fig, ax = plt.subplots(figsize=(w, 5.5))
    ax.vlines(x, d["before"], d["after"], color=LINE_C, lw=0.8, zorder=1)
    ms = 14 if len(d) < 300 else (6 if len(d) < 3000 else 3)
    ax.scatter(x, d["after"], s=ms, color=AFTER_C, zorder=2, lw=0)
    ax.scatter(x, d["before"], s=ms, color=BEFORE_C, zorder=3, lw=0)
    lo = min(d["before"].min(), d["after"].min())
    if lo <= 0:
        ax.set_yscale("symlog", linthresh=10)  # log scale that still shows 0 (empty predictions)
        ax.set_ylim(bottom=0)
    else:
        ax.set_yscale("log")
        ax.set_ylim(bottom=lo / 1.5)
    ax.set_xlim(-1, len(d))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel(f"nodules, sorted by {unit_label} before (n = {len(d)})", color="#444")
    ax.set_ylabel(unit_label, color="#444")
    up, down = int((d.after > d.before).sum()), int((d.after < d.before).sum())
    same = len(d) - up - down
    ax.set_title(f"Predicted {unit_label}: before -> after     "
                 f"increased {up}  |  decreased {down}  |  unchanged {same}",
                 loc="left", fontsize=11, color="#222")
    ax.grid(axis="y", color="#ececec", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bbb")
    ax.tick_params(colors="#555", labelsize=9)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BEFORE_C, label="before"),
                       Line2D([], [], marker="o", ls="", color=AFTER_C, label="after"),
                       Line2D([], [], color=LINE_C, lw=1.5, label="change")],
              frameon=False, loc="lower left", bbox_to_anchor=(1.0, 0.0), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor="white")
    plt.close(fig)


def scatter(df, out_png, unit_label):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["before"], df["after"], s=10, color=BEFORE_C, alpha=0.6, lw=0)
    hi = max(df["before"].max(), df["after"].max(), 10) * 1.2
    ax.plot([0, hi], [0, hi], color=AFTER_C, lw=1, ls="--", label="no change")
    lo = min(df["before"].min(), df["after"].min())
    if lo <= 0:
        ax.set_xscale("symlog", linthresh=10)
        ax.set_yscale("symlog", linthresh=10)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
    else:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo / 1.5, hi)
        ax.set_ylim(lo / 1.5, hi)
    ax.set_xlabel(f"{unit_label} before", color="#444")
    ax.set_ylabel(f"{unit_label} after", color="#444")
    ax.set_title("Before vs after (above the line = grew)", loc="left", fontsize=11, color="#222")
    ax.grid(color="#ececec", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", required=True, help="pred_nifti folder of the original model")
    ap.add_argument("--after", required=True, help="pred_nifti folder of the new model")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--include_scan_union", action="store_true", help="also compare *_pred_all.nii.gz")
    ap.add_argument("--measure", choices=["voxels", "volume"], default="voxels",
                    help="plot voxel counts (default) or volume in mm^3")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def listing(folder):
        files = {nodule_name(p): p for p in Path(folder).glob("*.nii.gz")}
        if not args.include_scan_union:
            files = {k: v for k, v in files.items() if not k.endswith("_pred_all")}
        return files

    fb, fa = listing(args.before), listing(args.after)
    common = sorted(set(fb) & set(fa))
    only_b, only_a = sorted(set(fb) - set(fa)), sorted(set(fa) - set(fb))
    pd.DataFrame({"file": only_b + only_a,
                  "only_in": ["before"] * len(only_b) + ["after"] * len(only_a)}).to_csv(
        out / "unmatched_files.csv", index=False)
    print(f"before: {len(fb)} files | after: {len(fa)} files | matched: {len(common)} | "
          f"only before: {len(only_b)} | only after: {len(only_a)}")
    if not common:
        raise SystemExit("no matching file names between the two folders")

    with Pool(args.workers) as pool:
        recs = pool.map(compare_one, [(n, fb[n], fa[n]) for n in common], chunksize=4)
    df = pd.DataFrame(recs)
    bad = df[df.error != ""]
    if len(bad):
        print(f"[warn] {len(bad)} files could not be read (see 'error' column)")
    df = df[df.error == ""].copy()

    df["change_voxels"] = df.voxels_after - df.voxels_before
    df["change_pct"] = np.where(df.voxels_before > 0,
                                100 * df.change_voxels / df.voxels_before.replace(0, np.nan), np.nan)
    df["became_empty"] = (df.voxels_before > 0) & (df.voxels_after == 0)
    df["became_nonempty"] = (df.voxels_before == 0) & (df.voxels_after > 0)
    df.sort_values("change_voxels").to_csv(out / "voxel_comparison.csv", index=False)
    if len(bad):
        bad[["name", "error"]].to_csv(out / "read_errors.csv", index=False)

    col_b, col_a = ("voxels_before", "voxels_after") if args.measure == "voxels" \
        else ("vol_mm3_before", "vol_mm3_after")
    unit = "voxels" if args.measure == "voxels" else "volume (mm^3)"
    plot_df = df.rename(columns={col_b: "before", col_a: "after"})
    dumbbell(plot_df, out / "voxel_change_dumbbell.png", unit)
    scatter(plot_df, out / "voxel_change_scatter.png", unit)

    # ---- summary
    nz = df[(df.voxels_before > 0) & (df.voxels_after > 0)]
    print("\n=== summary ===")
    print(f"compared: {len(df)}")
    print(f"increased: {(df.change_voxels > 0).sum()}  decreased: {(df.change_voxels < 0).sum()}  "
          f"unchanged: {(df.change_voxels == 0).sum()}")
    print(f"became empty: {df.became_empty.sum()}  became non-empty: {df.became_nonempty.sum()}  "
          f"empty in both: {((df.voxels_before == 0) & (df.voxels_after == 0)).sum()}")
    print(f"voxels before: median {df.voxels_before.median():.0f}   after: median {df.voxels_after.median():.0f}")
    if len(nz):
        print(f"% change (both non-empty): median {nz.change_pct.median():+.1f}%  "
              f"IQR {nz.change_pct.quantile(.25):+.1f}% to {nz.change_pct.quantile(.75):+.1f}%")
    if df.dice_before_after.notna().any():
        print(f"Dice(before, after): median {df.dice_before_after.median():.3f}  "
              f"(< 0.5 for {(df.dice_before_after < 0.5).sum()} files)")
    print(f"outputs -> {out}")


if __name__ == "__main__":
    main()
