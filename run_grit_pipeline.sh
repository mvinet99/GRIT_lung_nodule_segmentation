#!/usr/bin/env bash
# Train the GRIT V-Net (and, for comparison, the 1-channel baseline with identical settings),
# then evaluate both on the same validation split. Edit the paths, run step by step.
set -euo pipefail
cd "$(dirname "$0")"

BASE=/radraid2/mvinet/Lung/Current/code/09_29_26_model_retraining
PREP=$BASE/prep                      # existing preprocessed crops (no new preprocessing needed)
GPUS=1,2,3
export NCCL_P2P_DISABLE=1

# ---- 1-channel baseline (same epochs / early stopping as GRIT, for a fair comparison)
CUDA_VISIBLE_DEVICES=$GPUS torchrun --standalone --nproc_per_node=3 02_train.py \
    --data_dir "$PREP" --out_dir "$BASE/run_baseline_es" \
    --epochs 300 --early_stop_patience 30 --lr 1e-3 --batch_size 64 --num_workers 4 --val_frac 0.2

# ---- GRIT: image + gradient field + per-nodule thresholded image
CUDA_VISIBLE_DEVICES=$GPUS torchrun --standalone --nproc_per_node=3 02_train.py \
    --data_dir "$PREP" --out_dir "$BASE/run_grit" --grit \
    --epochs 300 --early_stop_patience 30 --lr 1e-3 --batch_size 64 --num_workers 4 --val_frac 0.2
# reversed level mapping (lowest gradient -> 75 %):  add --grit_reverse and use --out_dir $BASE/run_grit_rev

# ---- evaluate both on the SAME validation patients (same --seed -> same split)
for RUN in run_baseline_es run_grit; do
  CUDA_VISIBLE_DEVICES=1 python 03_predict_postprocess.py --xlsx "$BASE/$RUN/val_paths.csv" \
      --ckpt "$BASE/$RUN/best.pt" --out_dir "$BASE/$RUN/pred_val" --save_png \
      --batch_size 1024 --load_workers 12 --save_workers 12
done
# compare: metrics_per_nodule.csv in each pred_val/ (Dice, precision, recall per nodule),
#          or python compare_pred_niftis.py --before .../run_baseline_es/pred_val/pred_nifti --after .../run_grit/pred_val/pred_nifti
