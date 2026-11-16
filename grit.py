"""GRIT: GRadient and Intensity Thresholding input channels.

Three input channels per 64x64 axial slice (all from the normalised [0, 1] image, HU -1000..400):
  1. the image itself
  2. gradient field:  average of |dI/dx| and |dI/dy| (central differences, one-sided at the
                      edges), stored x2 = |dI/dx| + |dI/dy|; a sharp 0 -> 1 edge gives 0.5,
                      values stay in [0, 1]
  3. thresholded image: pixels in the top P % of intensities OF THAT SLICE keep their value,
                      all others are set to 0. P in {12.5, 25, 50, 75} is chosen per nodule:
       descriptor  = mean gradient field in the 10 x 10 patch centred on the nodule centroid
                     (centroid axial slice)
       K-means (K=4) on the descriptors of the TRAINING nodules -> 4 centres, sorted ascending
       and mapped monotonically to the levels (default: lowest gradient -> 12.5 %;
       reverse=True: lowest gradient -> 75 %). New nodules -> nearest centre.
"""
import json

import numpy as np

LEVELS = (12.5, 25.0, 50.0, 75.0)
PATCH = 10


# ----------------------------------------------------------------------------
# Channels
# ----------------------------------------------------------------------------
def gradient_field(img01):
    """img01: (..., H, W) in [0, 1] -> |dx| + |dy| (= 2 x their average), in [0, 1]."""
    a = np.asarray(img01, dtype=np.float32)
    gy = np.gradient(a, axis=-2)
    gx = np.gradient(a, axis=-1)
    return (np.abs(gx) + np.abs(gy)).astype(np.float32)   # = 2 * (|dx|+|dy|)/2


def threshold_image(img01, pct, ref_size=None):
    """Keep intensities of the top `pct` % pixels of each slice, zero the rest.
    img01: (H, W) or (N, H, W); pct: scalar or length-N array.
    ref_size: if a slice is larger than ref_size x ref_size (e.g. a 128 crop at inference), the
    intensity cut-off is taken from its central ref_size x ref_size window - the field of view the
    model was trained on - and applied to the whole slice. Slices of size ref_size are unaffected."""
    a = np.asarray(img01, dtype=np.float32)
    single = a.ndim == 2
    if single:
        a = a[None]
    pct = np.broadcast_to(np.asarray(pct, dtype=np.float64), (a.shape[0],))
    ref = a
    if ref_size is not None and (a.shape[-2] > ref_size or a.shape[-1] > ref_size):
        y0 = max((a.shape[-2] - ref_size) // 2, 0)
        x0 = max((a.shape[-1] - ref_size) // 2, 0)
        ref = a[:, y0:y0 + ref_size, x0:x0 + ref_size]
    flat = ref.reshape(ref.shape[0], -1)
    thr = np.array([np.percentile(f, 100.0 - p) for f, p in zip(flat, pct)], np.float32)
    out = np.where(a >= thr[:, None, None], a, 0.0).astype(np.float32)
    return out[0] if single else out


def make_channels(img01, pct, ref_size=None):
    """(H, W) or (N, H, W) normalised image(s) -> (3, H, W) or (N, 3, H, W)."""
    a = np.asarray(img01, dtype=np.float32)
    ch = np.stack([a, gradient_field(a), threshold_image(a, pct, ref_size)], axis=-3)
    return ch.astype(np.float32)


# ----------------------------------------------------------------------------
# Threshold selection
# ----------------------------------------------------------------------------
def descriptor(img01_slice, center_yx, patch=PATCH):
    """Mean gradient field in a patch x patch window centred on (y, x) of one slice."""
    g = gradient_field(img01_slice)
    cy, cx = (int(round(float(v))) for v in center_yx)
    h = patch // 2
    y0, x0 = max(cy - h, 0), max(cx - h, 0)
    win = g[y0:cy - h + patch, x0:cx - h + patch]
    return float(win.mean()) if win.size else float(g.mean())


def descriptor_from_crop(img01_crop, center_zyx, patch=PATCH):
    """Crop (D, H, W) normalised + nodule centre in crop voxels -> descriptor on the centroid slice."""
    z = int(np.clip(round(float(center_zyx[0])), 0, img01_crop.shape[0] - 1))
    return descriptor(img01_crop[z], center_zyx[1:], patch)


def kmeans_1d(x, k=4, n_init=20, iters=100, seed=0):
    """Plain 1-D K-means (Lloyd, k-means++ init, best of n_init). Returns sorted centres."""
    x = np.asarray(x, dtype=np.float64)
    if len(np.unique(x)) < k:
        raise ValueError(f"need at least {k} distinct descriptor values, got {len(np.unique(x))}")
    rng = np.random.default_rng(seed)
    best, best_inertia = None, np.inf
    for _ in range(n_init):
        c = [x[rng.integers(len(x))]]
        for _ in range(k - 1):                        # k-means++ seeding
            d2 = np.min((x[:, None] - np.array(c)[None]) ** 2, axis=1)
            c.append(x[rng.choice(len(x), p=d2 / d2.sum())])
        c = np.array(c)
        for _ in range(iters):
            lab = np.argmin(np.abs(x[:, None] - c[None]), axis=1)
            new = np.array([x[lab == j].mean() if np.any(lab == j) else c[j] for j in range(k)])
            if np.allclose(new, c):
                break
            c = new
        inertia = float(np.sum((x - c[np.argmin(np.abs(x[:, None] - c[None]), axis=1)]) ** 2))
        if inertia < best_inertia:
            best, best_inertia = c, inertia
    return np.sort(best)


class GritSelector:
    """Maps a nodule descriptor to a threshold level via sorted K-means centres."""

    def __init__(self, centers, levels=LEVELS, reverse=False):
        self.centers = np.sort(np.asarray(centers, dtype=np.float64))
        self.levels = tuple(float(v) for v in levels)
        self.reverse = bool(reverse)
        assert len(self.centers) == len(self.levels)

    @classmethod
    def fit(cls, descriptors, levels=LEVELS, reverse=False, seed=0):
        return cls(kmeans_1d(descriptors, k=len(levels), seed=seed), levels, reverse)

    def level_of(self, desc):
        d = np.atleast_1d(np.asarray(desc, dtype=np.float64))
        j = np.argmin(np.abs(d[:, None] - self.centers[None]), axis=1)
        lv = np.array(self.levels[::-1] if self.reverse else self.levels)
        out = lv[j]
        return float(out[0]) if np.ndim(desc) == 0 else out

    def to_dict(self):
        return dict(centers=self.centers.tolist(), levels=list(self.levels), reverse=self.reverse)

    @classmethod
    def from_dict(cls, d):
        return cls(d["centers"], d["levels"], d["reverse"])

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
