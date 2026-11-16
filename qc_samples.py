#!/usr/bin/env python
"""Optional sanity check (run after step 1): save a PNG grid of augmented training
samples (with shift augmentation) so you can confirm masks line up and nodules
are being moved off-centre.

  python qc_samples.py --data_dir /radraid/.../LIDC_2d_prep --n 32 --out qc_samples.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dataset import BalancedDistributedSampler, LIDCSlices2D, load_slice_index  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--max_shift", type=float, default=24)
    ap.add_argument("--out", default="qc_samples.png")
    args = ap.parse_args()

    sl = load_slice_index(args.data_dir)
    ds = LIDCSlices2D(args.data_dir, sl, augment=True, max_shift=args.max_shift)
    idx = list(iter(BalancedDistributedSampler(ds.labels, ds.kinds, seed=np.random.randint(1e6))))[: args.n]
    cols = 8
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, i in zip(np.ravel(axes), idx):
        x, y, lab = ds[i]
        ax.imshow(x[0], cmap="gray", vmin=0, vmax=1)
        if y.any():
            ax.contour(y[0], levels=[0.5], colors="r", linewidths=0.8)
        ax.set_title(f"{'pos' if lab else 'neg'} {ds.kinds[i][:3]}", fontsize=7)
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
