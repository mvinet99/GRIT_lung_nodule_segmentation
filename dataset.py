"""2D slice dataset + balanced, DDP-aware sampler.

Augmentations (training only):
  * nodule shift   - the 64x64 window is moved off the nodule centroid in a random
                     direction (uniform angle) by a random distance in
                     [min_shift, max_shift] px, so the nodule is no longer centred.
                     Uses the real anatomy stored in the context crop; pads with
                     -1000 HU only if the shift exceeds the stored context.
  * random flips (LR / AP) and 90-degree rotations
  * small intensity jitter (scale / offset / gaussian noise)
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from common import CROP_SIZE, HU_MIN, crop_with_pad, normalize_hu
from grit import make_channels


class LIDCSlices2D(Dataset):
    def __init__(self, data_dir, slice_df, patch=CROP_SIZE, augment=True,
                 p_shift=0.8, min_shift=0, max_shift=24, p_flip=0.5, p_rot90=0.5,
                 intensity_jitter=True, fixed_shift=None, eval_shift_seed=None, grit_pct=None):
        self.crop_dir = Path(data_dir) / "crops"
        self.df = slice_df.reset_index(drop=True)
        self.crop_ids = self.df["crop_id"].to_numpy()
        self.zs = self.df["z"].to_numpy().astype(int)
        self.labels = self.df["label"].to_numpy().astype(int)
        self.kinds = self.df["kind"].to_numpy()
        self.patch = patch
        self.augment = augment
        self.p_shift, self.min_shift, self.max_shift = p_shift, min_shift, max_shift
        self.p_flip, self.p_rot90 = p_flip, p_rot90
        self.intensity_jitter = intensity_jitter
        self.fixed_shift = fixed_shift  # (dy, dx) for deterministic evaluation
        # validation: same shift distribution as training, but a FIXED shift per slice
        # (seeded by index) so the val loss is comparable across epochs. No flips/jitter.
        self.eval_shift_seed = eval_shift_seed
        # GRIT: {crop_id: threshold %} -> 3 input channels (image, gradient field, thresholded image)
        self.grit_pct = grit_pct
        self._cache = {}

    def __len__(self):
        return len(self.df)

    def _arr(self, crop_id, kind):
        key = (crop_id, kind)
        a = self._cache.get(key)
        if a is None:  # memory-mapped: only the requested slice is actually read
            a = np.load(self.crop_dir / f"{crop_id}_{kind}.npy", mmap_mode="r")
            if len(self._cache) > 4096:
                self._cache.clear()
            self._cache[key] = a
        return a

    def _sample_shift(self, rng):
        if self.fixed_shift is not None:
            return self.fixed_shift
        shifting = self.augment or self.eval_shift_seed is not None
        if not shifting or rng.random() >= self.p_shift or self.max_shift <= 0:
            return 0, 0
        theta = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(self.min_shift, self.max_shift)
        return int(round(r * math.sin(theta))), int(round(r * math.cos(theta)))

    def __getitem__(self, i):
        if self.eval_shift_seed is not None:
            rng = np.random.default_rng(self.eval_shift_seed + int(i))
        else:
            rng = np.random.default_rng()  # fresh entropy per call; safe across workers
        crop_id, z = self.crop_ids[i], self.zs[i]
        img = np.asarray(self._arr(crop_id, "img")[z], dtype=np.float32)
        msk = np.asarray(self._arr(crop_id, "msk")[z], dtype=np.uint8)
        ctx = img.shape[0]

        dy, dx = self._sample_shift(rng)
        y0 = (ctx - self.patch) // 2 + dy
        x0 = (ctx - self.patch) // 2 + dx
        img = crop_with_pad(img, (y0, x0), (self.patch, self.patch), HU_MIN)
        msk = crop_with_pad(msk, (y0, x0), (self.patch, self.patch), 0)

        img = normalize_hu(img)
        if self.augment:
            if rng.random() < self.p_flip:
                img, msk = img[:, ::-1], msk[:, ::-1]
            if rng.random() < self.p_flip:
                img, msk = img[::-1, :], msk[::-1, :]
            if rng.random() < self.p_rot90:
                k = int(rng.integers(1, 4))
                img, msk = np.rot90(img, k), np.rot90(msk, k)
            if self.intensity_jitter:
                img = img * rng.uniform(0.95, 1.05) + rng.uniform(-0.03, 0.03)
                img = img + rng.normal(0, 0.01, img.shape)
                img = np.clip(img, 0.0, 1.0)

        img = np.ascontiguousarray(img, dtype=np.float32)
        if self.grit_pct is not None:   # channels from the slice the network actually sees
            img = torch.from_numpy(make_channels(img, self.grit_pct[crop_id]))
        else:
            img = torch.from_numpy(img)[None]
        msk = torch.from_numpy(np.ascontiguousarray(msk, dtype=np.float32))[None]
        return img, msk, torch.tensor(int(msk.any()), dtype=torch.long)


class BalancedDistributedSampler(Sampler):
    """Each epoch: ALL positive slices + a fresh random draw of negatives
    (neg_ratio x #positives), of which `bg_frac` come from random background
    lung crops and the rest from nodule crops (slices above/below the nodule).
    The epoch list is shuffled with a seed shared by all ranks, then sharded.
    Works for single-GPU too (num_replicas=1, rank=0)."""

    def __init__(self, labels, kinds, neg_ratio=1.0, bg_frac=0.3, num_replicas=1, rank=0, seed=0):
        labels, kinds = np.asarray(labels), np.asarray(kinds)
        self.pos = np.where(labels == 1)[0]
        self.neg_near = np.where((labels == 0) & (kinds != "background"))[0]
        self.neg_bg = np.where((labels == 0) & (kinds == "background"))[0]
        if neg_ratio < 0:  # use every negative slice
            n_near, n_bg = len(self.neg_near), len(self.neg_bg)
        else:
            n_neg = int(round(neg_ratio * len(self.pos)))
            n_bg = min(int(round(bg_frac * n_neg)), len(self.neg_bg))
            n_near = min(n_neg - n_bg, len(self.neg_near))
            if n_near < n_neg - n_bg:  # not enough near-nodule negatives -> top up from background
                n_bg = min(n_neg - n_near, len(self.neg_bg))
        self.n_bg, self.n_near = n_bg, n_near
        self.num_replicas, self.rank, self.seed = num_replicas, rank, seed
        self.epoch = 0
        total = len(self.pos) + n_near + n_bg
        self.num_samples = math.ceil(total / num_replicas)
        self.total_size = self.num_samples * num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = np.random.default_rng(self.seed + self.epoch)
        idx = np.concatenate([
            self.pos,
            g.choice(self.neg_near, self.n_near, replace=False) if self.n_near else np.array([], int),
            g.choice(self.neg_bg, self.n_bg, replace=False) if self.n_bg else np.array([], int),
        ]).astype(int)
        g.shuffle(idx)
        if len(idx) < self.total_size:
            idx = np.concatenate([idx, idx[: self.total_size - len(idx)]])
        return iter(idx[self.rank: self.total_size: self.num_replicas].tolist())

    def __len__(self):
        return self.num_samples


def split_by_patient(slice_df, val_frac, seed=0):
    """Optional patient-level split (val_frac=0 -> everything is training)."""
    if val_frac <= 0:
        return slice_df, slice_df.iloc[0:0]
    pids = np.array(sorted(slice_df["pid"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)
    n_val = max(1, int(round(val_frac * len(pids))))
    val_p = set(pids[:n_val])
    is_val = slice_df["pid"].isin(val_p)
    return slice_df[~is_val], slice_df[is_val]


def load_slice_index(data_dir):
    return pd.read_csv(Path(data_dir) / "slices_index.csv")
