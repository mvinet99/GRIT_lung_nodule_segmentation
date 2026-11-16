#!/usr/bin/env bash
# End-to-end pipeline: edit the paths, then run steps one at a time (or the whole file).
set -euo pipefail
cd "$(dirname "$0")"

XLSX=/path/to/LIDC_paths.xlsx                              # all LIDC rows (split 80/20 by patient in step 2)
PREP=/radraid/luotingzhuang/dataset/LIDC/LIDC_2d_prep       # preprocessed output
RUN=/radraid/luotingzhuang/runs/vnet2d_lidc                 # checkpoints, logs, loss_curves.png

# ---- Step 1: preprocess (CPU, parallel)
python 01_preprocess.py --xlsx "$XLSX" --out_dir "$PREP" --workers 16 --context 128 --bg_crops_per_scan 1

# ---- (optional) eyeball augmented training samples
python qc_samples.py --data_dir "$PREP" --out "$PREP/qc_samples.png"

# ---- Step 2: train on 3 GPUs (100 epochs, lr 1e-3, batch 32/GPU, 80/20 patient-level train/val split)
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 02_train.py \
    --data_dir "$PREP" --out_dir "$RUN" --epochs 100 --lr 1e-3 --batch_size 32 --num_workers 8 --val_frac 0.2
# resume after an interruption:  add  --resume "$RUN/last.pt"
# outputs: best.pt, last.pt, final.pt, loss_curves.png, train_log.csv, split.csv, train_paths.csv, val_paths.csv

# ---- Step 3: inference + post-processing on the validation split (3D metrics, original CT space)
python 03_predict_postprocess.py --xlsx "$RUN/val_paths.csv" --ckpt "$RUN/best.pt" \
    --out_dir "$RUN/pred_val" --save_crops --save_png \
    --batch_size 1024 --load_workers 12 --save_workers 12   # PNGs -> $RUN/pred_val/png/<nodule>/zNNN.png
# robustness to off-centre nodules: add  --random_offset_mm 15   (and/or --roi 96)
# later, on your separate test set:  --xlsx /path/to/test_paths.xlsx --out_dir "$RUN/pred_test"

# ---- Step 4: external / held-out cohort with centroid coordinates (no masks)
#   CSV columns: pid, study_yr, nodule_id, image_path, coordX, coordY, coordZ, (any others are passed through)
TEST_CSV=/path/to/temporal_nodule_volume_analysis_09032026.csv
CUDA_VISIBLE_DEVICES=0 python 04_infer_test_cohort.py --csv "$TEST_CSV" --ckpt "$RUN/best.pt" \
    --out_dir "$RUN/pred_test_cohort" --batch_size 1024 --load_workers 12 --save_workers 12
# outputs: pred_nifti/ (original CT space), crops_64/ (64^3 img/pred/prob), png/, metrics_per_nodule.csv,
#          coord_check.csv (centroids read as LPS by default; the run stops if another convention fits clearly better)
