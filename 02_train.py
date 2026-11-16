#!/usr/bin/env python
"""
Step 2 - Train a 2D V-Net on the preprocessed LIDC slices (multi-GPU DDP).

Run on 3 GPUs:
  torchrun --standalone --nproc_per_node=3 02_train.py \
      --data_dir /radraid/.../LIDC_2d_prep --out_dir /radraid/.../runs/vnet2d_lidc

Single GPU / debugging:
  python 02_train.py --data_dir ... --out_dir ...

GRIT (--grit): 3 input channels per slice - image, gradient field, and the image thresholded to
  its top 12.5/25/50/75 % intensities, the level chosen per nodule by K-means (K=4) on the mean
  gradient in a 10x10 patch at the nodule centroid. K-means is fit on the TRAINING nodules only;
  centres are saved in every checkpoint (and grit_selector.json) and reused at inference.
  --grit_reverse flips the level order. See grit.py.
Early stopping (--early_stop_patience N): stop when val loss has not improved for N epochs.

Notes
  * --batch_size is PER GPU (effective batch = batch_size x #GPUs).
  * Loss = BCE + soft Dice. With --dice_mode all (default) Dice is computed per
    slice INCLUDING empty slices, so any predicted blob on an empty slice is
    penalised strongly (this is what suppresses false positives on nodule-free slices).
  * 80/20 train/validation split at the PATIENT level (--val_frac 0.2), so slices
    of one patient never appear in both sets. The split is saved to
    split.csv, train_paths.csv and val_paths.csv (val_paths.csv can be fed
    directly to 03_predict_postprocess.py).
  * Validation uses the same positive:negative mix as training and the same
    shift distribution, but with a fixed shift per slice and no flips/jitter,
    so the curve is comparable epoch to epoch.
  * Curves: loss_curves.png (train vs val loss, nodule-slice Dice, empty-slice
    FP rate) is rewritten every epoch; the raw numbers are in train_log.csv.
  * Checkpoints: best.pt (lowest val loss), last.pt every epoch,
    epoch_XXX.pt every --save_every epochs, final.pt.
"""
import argparse
import csv
import json
import math
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from common import normalize_hu
from dataset import BalancedDistributedSampler, LIDCSlices2D, load_slice_index, split_by_patient
from grit import GritSelector, descriptor
from vnet2d import VNet2D


# ----------------------------------------------------------------------------
def setup_distributed(timeout_min=10):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)  # bind GPU BEFORE creating the NCCL communicator
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        backend = "nccl" if device.type == "cuda" else "gloo"
        # a stuck collective now raises an error after `timeout_min` instead of hanging forever
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_min))
        rank, world = dist.get_rank(), dist.get_world_size()
        # communication smoke test: fails fast (with a clear message) if NCCL P2P is broken
        t0 = time.time()
        x = torch.ones(1, device=device) * (rank + 1)
        dist.all_reduce(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        expected = world * (world + 1) / 2
        if rank == 0:
            print(f"[dist] {backend} all_reduce OK across {world} GPUs "
                  f"({time.time() - t0:.1f}s, value {x.item():.0f}/{expected:.0f})", flush=True)
    else:
        rank, world = 0, 1
    return rank, world, local_rank, device


def is_main(rank):
    return rank == 0


def all_reduce_sum(t):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


# ----------------------------------------------------------------------------
def seg_loss(logits, target, dice_mode="all", bce_w=1.0, dice_w=1.0, smooth=1.0):
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    if dice_mode == "batch":
        inter = (p * target).sum()
        dice = (2 * inter + smooth) / (p.sum() + target.sum() + smooth)
        dice_loss = 1 - dice
    else:
        dims = (1, 2, 3)
        inter = (p * target).sum(dims)
        dice = (2 * inter + smooth) / (p.sum(dims) + target.sum(dims) + smooth)
        if dice_mode == "pos":
            has = target.sum(dims) > 0
            dice_loss = (1 - dice[has]).mean() if has.any() else logits.sum() * 0
        else:  # "all"
            dice_loss = (1 - dice).mean()
    return bce_w * bce + dice_w * dice_loss


@torch.no_grad()
def batch_stats(logits, target, thr=0.5):
    """Returns sums: [dice_sum_on_pos, n_pos, fp_slices_on_neg, n_neg]."""
    pred = (torch.sigmoid(logits.float()) > thr).float()
    dims = (1, 2, 3)
    has = target.sum(dims) > 0
    inter = (pred * target).sum(dims)
    dice = (2 * inter) / (pred.sum(dims) + target.sum(dims)).clamp_min(1e-6)
    fp = (pred.sum(dims) > 0) & ~has
    return torch.tensor([dice[has].sum().item(), has.sum().item(),
                         fp.sum().item(), (~has).sum().item()], dtype=torch.float64)


def make_scheduler(opt, total_steps, warmup_steps, min_lr_ratio):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


def write_split(data_dir, out_dir, tr_df, va_df):
    """Save the patient-level split + per-split path tables (same columns as LIDC_paths.xlsx)."""
    split = pd.concat([pd.DataFrame({"pid": sorted(tr_df.pid.unique()), "split": "train"}),
                       pd.DataFrame({"pid": sorted(va_df.pid.unique()), "split": "val"})])
    split.to_csv(out_dir / "split.csv", index=False)
    man = pd.read_csv(Path(data_dir) / "crops_manifest.csv")
    man = man[man.kind == "nodule"][["pid", "image_path", "mask_path"]]
    man[man.pid.isin(set(tr_df.pid))].to_csv(out_dir / "train_paths.csv", index=False)
    man[man.pid.isin(set(va_df.pid))].to_csv(out_dir / "val_paths.csv", index=False)


TRAIN_C, VAL_C = "#2a78d6", "#eb6834"   # colour = split, identical in every panel


def plot_curves(log_path, png_path):
    """One PNG: train vs val loss | Dice on nodule slices | FP rate on empty slices."""
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log = pd.read_csv(log_path)
    has_val = log["val_loss"].notna().any()
    panels = [("Loss (BCE + Dice)", "train_loss", "val_loss"),
              ("Dice on nodule slices", "train_pos_dice", "val_pos_dice"),
              ("False-positive rate on empty slices", "train_neg_fp_rate", "val_neg_fp_rate")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (title, tr, va) in zip(axes, panels):
        ax.plot(log["epoch"], log[tr], color=TRAIN_C, lw=2, label="train")
        if has_val:
            ax.plot(log["epoch"], log[va], color=VAL_C, lw=2, label="validation")
        ax.set_title(title, fontsize=11, loc="left", color="#222")
        ax.set_xlabel("epoch", color="#555")
        ax.grid(True, color="#e6e6e6", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#bbb")
        ax.tick_params(colors="#555", labelsize=9)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.legend(frameon=False, fontsize=9)
    if has_val:
        best = log.loc[log["val_loss"].idxmin()]
        axes[0].axvline(best["epoch"], color="#999", lw=1, ls="--")
        axes[0].annotate(f"best val {best['val_loss']:.4f}\n(epoch {int(best['epoch'])})",
                         xy=(best["epoch"], best["val_loss"]), xytext=(6, 12),
                         textcoords="offset points", fontsize=8, color="#444")
    for ax in axes[1:]:
        ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130, facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------
def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def setup_grit(data_dir, out_dir, tr_df, rank, reverse, seed):
    """GRIT threshold level per crop. Descriptor = mean gradient field in the 10x10 patch at the
    nodule centroid (centroid slice of each crop, cached in <data_dir>/grit_descriptors.csv).
    K-means (K=4) is fit on TRAINING nodule crops only, then every crop gets its level."""
    data_dir, cache = Path(data_dir), Path(data_dir) / "grit_descriptors.csv"
    if is_main(rank) and not cache.exists():
        man = pd.read_csv(data_dir / "crops_manifest.csv")
        rows = []
        for r in man.itertuples():
            img = np.load(data_dir / "crops" / f"{r.crop_id}_img.npy", mmap_mode="r")
            cz, cy, cx = r.center_z - r.start_z, r.center_y - r.start_y, r.center_x - r.start_x
            z = int(np.clip(round(cz), 0, img.shape[0] - 1))
            rows.append(dict(crop_id=r.crop_id, pid=r.pid, kind=r.kind,
                             descriptor=descriptor(normalize_hu(np.asarray(img[z])), (cy, cx))))
        pd.DataFrame(rows).to_csv(cache, index=False)
    barrier()
    d = pd.read_csv(cache)
    fit_on = d[(d.kind == "nodule") & d.pid.isin(set(tr_df.pid))]
    sel = GritSelector.fit(fit_on.descriptor.values, reverse=reverse, seed=seed)
    d["grit_pct"] = sel.level_of(d.descriptor.values)
    if is_main(rank):
        sel.save(out_dir / "grit_selector.json")
        d.to_csv(out_dir / "grit_levels.csv", index=False)
        nod = d[d.kind == "nodule"]
        counts = nod.grit_pct.value_counts().sort_index()
        print("[GRIT] K-means on", len(fit_on), "training nodules; centres",
              np.round(sel.centers, 4).tolist(), "->", list(sel.levels[::-1] if reverse else sel.levels), "%")
        print("[GRIT] nodules per threshold level:", {f"{k:g}%": int(v) for k, v in counts.items()})
    return dict(zip(d.crop_id, d.grit_pct)), sel


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True, help="output dir of 01_preprocess.py")
    ap.add_argument("--out_dir", required=True)
    # optimisation
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=32, help="per GPU")
    ap.add_argument("--warmup_epochs", type=float, default=2)
    ap.add_argument("--min_lr_ratio", type=float, default=0.01, help="final lr = lr * this (cosine)")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--num_workers", type=int, default=8, help="dataloader workers per GPU")
    ap.add_argument("--seed", type=int, default=42)
    # loss
    ap.add_argument("--dice_mode", choices=["all", "pos", "batch"], default="all")
    ap.add_argument("--bce_w", type=float, default=1.0)
    ap.add_argument("--dice_w", type=float, default=1.0)
    # model
    ap.add_argument("--norm", choices=["batch", "instance", "group"], default="batch")
    ap.add_argument("--dropout", type=float, default=0.2)
    # sampling / augmentation
    ap.add_argument("--neg_ratio", type=float, default=1.0,
                    help="#negative (empty) slices per epoch = neg_ratio x #positive slices; "
                         "-1 = use every negative slice every epoch")
    ap.add_argument("--bg_frac", type=float, default=0.3,
                    help="fraction of negatives drawn from random background lung crops")
    ap.add_argument("--p_shift", type=float, default=0.8, help="prob. of moving nodule off-centre")
    ap.add_argument("--min_shift", type=float, default=0)
    ap.add_argument("--max_shift", type=float, default=24, help="max shift in px (=mm)")
    ap.add_argument("--p_flip", type=float, default=0.5)
    ap.add_argument("--p_rot90", type=float, default=0.5)
    # bookkeeping
    ap.add_argument("--val_frac", type=float, default=0.2,
                    help="fraction of PATIENTS held out for validation (0.2 = 80/20 split)")
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--grit", action="store_true",
                    help="GRIT: 3 input channels (image, gradient field, per-nodule thresholded image)")
    ap.add_argument("--grit_reverse", action="store_true",
                    help="reverse the level mapping: lowest-gradient cluster -> 75%% (default -> 12.5%%)")
    ap.add_argument("--early_stop_patience", type=int, default=0,
                    help="stop when val loss has not improved for N epochs (0 = off; paper: 300 epochs + ES)")
    ap.add_argument("--log_every", type=int, default=50, help="print progress every N steps")
    ap.add_argument("--resume", default="", help="path to last.pt to resume from")
    ap.add_argument("--max_steps_per_epoch", type=int, default=0, help="debug only")
    args = ap.parse_args()

    rank, world, local_rank, device = setup_distributed()
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.backends.cudnn.benchmark = True
    out_dir = Path(args.out_dir)
    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "config.json", "w") as f:
            json.dump(vars(args) | {"world_size": world}, f, indent=2)

    # ---------------- data
    sl = load_slice_index(args.data_dir)
    tr_df, va_df = split_by_patient(sl, args.val_frac, seed=args.seed)
    grit_pct = None
    args.in_channels = 1
    if args.grit:
        grit_pct, sel = setup_grit(args.data_dir, out_dir, tr_df, rank, args.grit_reverse, args.seed)
        args.grit_selector = sel.to_dict()      # stored in every checkpoint -> used at inference
        args.in_channels = 3
    train_ds = LIDCSlices2D(args.data_dir, tr_df, augment=True, p_shift=args.p_shift,
                            min_shift=args.min_shift, max_shift=args.max_shift,
                            p_flip=args.p_flip, p_rot90=args.p_rot90, grit_pct=grit_pct)
    sampler = BalancedDistributedSampler(train_ds.labels, train_ds.kinds, neg_ratio=args.neg_ratio,
                                         bg_frac=args.bg_frac, num_replicas=world, rank=rank,
                                         seed=args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          num_workers=args.num_workers, pin_memory=device.type == "cuda",
                          drop_last=True, persistent_workers=args.num_workers > 0)
    val_dl = None
    if len(va_df):
        # same pos:neg mix and shift distribution as training, but fixed (identical every epoch)
        val_ds = LIDCSlices2D(args.data_dir, va_df, augment=False, p_shift=args.p_shift,
                              min_shift=args.min_shift, max_shift=args.max_shift,
                              eval_shift_seed=args.seed, grit_pct=grit_pct)
        val_sampler = BalancedDistributedSampler(val_ds.labels, val_ds.kinds, neg_ratio=args.neg_ratio,
                                                 bg_frac=args.bg_frac, num_replicas=world, rank=rank,
                                                 seed=args.seed + 1)  # set_epoch never called -> fixed set
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda",
                            persistent_workers=args.num_workers > 0)

    if is_main(rank):
        write_split(args.data_dir, out_dir, tr_df, va_df)
        print(f"world_size={world}  device={device}  input channels={args.in_channels}"
              + ("  (GRIT)" if args.grit else ""))
        if args.early_stop_patience and val_dl is None:
            print("[warn] --early_stop_patience needs a validation split (--val_frac > 0); ignored")
        print(f"train: {tr_df.pid.nunique()} patients, {len(tr_df)} slices | per epoch: "
              f"{len(sampler.pos)} pos + {sampler.n_near} near-nodule neg + {sampler.n_bg} background neg "
              f"= {sampler.total_size}  ->  {len(train_dl)} steps/epoch/GPU")
        if val_dl is not None:
            print(f"val:   {va_df.pid.nunique()} patients, {len(va_df)} slices | per epoch: "
                  f"{len(val_sampler.pos)} pos + {val_sampler.n_near} near-nodule neg + "
                  f"{val_sampler.n_bg} background neg")

    # ---------------- model / optim
    model = VNet2D(in_channels=args.in_channels, n_classes=1, norm=args.norm, dropout=args.dropout).to(device)
    if world > 1:
        if args.norm == "batch" and device.type == "cuda":  # SyncBN is GPU-only
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
    if is_main(rank):
        print("[setup] model built and wrapped in DDP; starting training "
              "(first epoch also warms up dataloader workers)", flush=True)
    opt =torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_dl) if not args.max_steps_per_epoch else min(len(train_dl), args.max_steps_per_epoch)
    total_steps = steps_per_epoch * args.epochs
    sched = make_scheduler(opt, total_steps, int(args.warmup_epochs * steps_per_epoch), args.min_lr_ratio)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    best_val = float("inf")
    bad_epochs = 0          # epochs since the last val-loss improvement (early stopping)
    last_epoch = start_epoch - 1
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        unwrap(model).load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        best_val = ck.get("best_val", float("inf"))
        bad_epochs = ck.get("bad_epochs", 0)
        if is_main(rank):
            print(f"resumed from {args.resume} at epoch {start_epoch}")

    log_path = out_dir / "train_log.csv"
    if is_main(rank) and not (args.resume and log_path.exists()):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "lr", "train_loss", "train_pos_dice", "train_neg_fp_rate",
                                    "val_loss", "val_pos_dice", "val_neg_fp_rate", "time_s"])

    def save(name, epoch):
        torch.save({"model": unwrap(model).state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_val": best_val, "bad_epochs": bad_epochs,
                    "args": vars(args)}, out_dir / name)

    # ---------------- loop
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        sampler.set_epoch(epoch)
        loss_sum = torch.zeros(2, dtype=torch.float64)
        stats = torch.zeros(4, dtype=torch.float64)
        for step, (x, y, _) in enumerate(train_dl):
            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            loss = seg_loss(logits, y, args.dice_mode, args.bce_w, args.dice_w)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += torch.tensor([loss.item() * x.shape[0], x.shape[0]], dtype=torch.float64)
            stats += batch_stats(logits.detach(), y)
            if is_main(rank) and (step == 0 or (step + 1) % args.log_every == 0):
                print(f"  epoch {epoch + 1} step {step + 1}/{steps_per_epoch}  loss {loss.item():.4f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)

        loss_sum, stats = all_reduce_sum(loss_sum.to(device)).cpu(), all_reduce_sum(stats.to(device)).cpu()
        tr_loss = (loss_sum[0] / loss_sum[1]).item()
        tr_dice = (stats[0] / stats[1].clamp_min(1)).item()
        tr_fp = (stats[2] / stats[3].clamp_min(1)).item()

        va_loss = va_dice = va_fp = float("nan")
        if val_dl is not None:
            model.eval()
            vl = torch.zeros(2, dtype=torch.float64)
            vs = torch.zeros(4, dtype=torch.float64)
            with torch.no_grad():
                for x, y, _ in val_dl:
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                        logits = model(x)
                    vl += torch.tensor([seg_loss(logits, y, args.dice_mode, args.bce_w, args.dice_w).item()
                                        * x.shape[0], x.shape[0]],
                                       dtype=torch.float64)
                    vs += batch_stats(logits, y)
            vl, vs = all_reduce_sum(vl.to(device)).cpu(), all_reduce_sum(vs.to(device)).cpu()
            va_loss = (vl[0] / vl[1]).item()
            va_dice = (vs[0] / vs[1].clamp_min(1)).item()
            va_fp = (vs[2] / vs[3].clamp_min(1)).item()

        # identical on every rank (val metrics are all-reduced) -> all ranks stop together
        improved = val_dl is not None and va_loss < best_val
        if improved:
            best_val, bad_epochs = va_loss, 0
        elif val_dl is not None:
            bad_epochs += 1
        last_epoch = epoch

        if is_main(rank):
            dt = time.time() - t0
            lr = opt.param_groups[0]["lr"]
            print(f"epoch {epoch + 1:3d}/{args.epochs}  lr {lr:.2e}  loss {tr_loss:.4f}  "
                  f"pos-dice {tr_dice:.4f}  neg-FP-rate {tr_fp:.4f}"
                  + (f"  | val loss {va_loss:.4f} dice {va_dice:.4f} FP {va_fp:.4f}" if val_dl else "")
                  + f"  ({dt:.0f}s)", flush=True)
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch + 1, lr, tr_loss, tr_dice, tr_fp, va_loss, va_dice, va_fp, round(dt, 1)])
            if improved:
                save("best.pt", epoch)
                print(f"  new best val loss {va_loss:.4f} -> best.pt", flush=True)
            save("last.pt", epoch)
            try:
                plot_curves(log_path, out_dir / "loss_curves.png")
            except Exception as e:  # never let plotting kill a training run
                print(f"  [warn] could not plot curves: {e}")
            if (epoch + 1) % args.save_every == 0:
                save(f"epoch_{epoch + 1:03d}.pt", epoch)

        if args.early_stop_patience and val_dl is not None and bad_epochs >= args.early_stop_patience:
            if is_main(rank):
                print(f"[early stopping] no val-loss improvement for {bad_epochs} epochs "
                      f"(best {best_val:.4f}); stopping after epoch {epoch + 1}", flush=True)
            break

    if is_main(rank):
        save("final.pt", last_epoch)
        print(f"training finished -> {out_dir / 'final.pt'}"
              + (f"  (best val loss {best_val:.4f} -> best.pt)" if val_dl is not None else ""))
        print(f"curves -> {out_dir / 'loss_curves.png'}")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
