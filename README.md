# Segmentation of ground glass regions of lung nodules in CT images

```
common.py                  shared I/O, resampling, HU clipping, cropping (used by every step)
01_preprocess.py           STEP 1  xlsx -> 1 mm, HU [-1000,400], 64^3 crops around nodule centroids, slice index
qc_samples.py              (optional) PNG grid of augmented training slices
02_train.py                STEP 2  2D V-Net, DDP on 3 GPUs (torchrun), 100 epochs, lr 1e-3, 80/20 patient split, loss_curves.png
03_predict_postprocess.py  STEP 3  inference -> 3D post-processing -> back to original CT space -> metrics
04_infer_test_cohort.py    STEP 4  external cohort from centroid coordinates -> masks, 64^3 crops, PNGs, volumes
infer_lib.py               shared inference code (model, batching, post-processing, PNGs, geometry)
vnet2d.py / dataset.py     model / dataset, shift augmentation, balanced sampler
run_pipeline.sh            the three steps with example arguments
```

## Run Inference

Download the model weights: 

```bash
pip install -r requirements.txt

# 1) preprocess (CPU)
python 01_preprocess.py --xlsx LIDC_paths.xlsx --out_dir $PREP --workers 16

# 1b) optional sanity check
python qc_samples.py --data_dir $PREP --out $PREP/qc_samples.png

# 2) train on 3 GPUs
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 02_train.py \
    --data_dir $PREP --out_dir $RUN --epochs 100 --lr 1e-3 --batch_size 32

#    -> $RUN/loss_curves.png, best.pt, split.csv, train_paths.csv, val_paths.csv

# 3) predict + post-process the validation split (3D metrics in original CT space)
python 03_predict_postprocess.py --xlsx $RUN/val_paths.csv --ckpt $RUN/best.pt --out_dir $RUN/pred_val
#    later, the same command on your separate held-out test set
```

## Explanation of Project Steps

**Preprocessing**
- Reorients to LPS so numpy axis 0 is always axial, then resamples to 1×1×1 mm. The CT uses linear interpolation. Masks use linear interpolation followed by a ≥0.5 threshold, with nearest-neighbour as a fallback for tiny nodules.
- Clips HU to [-1000, 400]. The network input is (HU + 1000) / 1400, so values are in [0, 1].
- Crops 64 (z) × 64 × 64 around each nodule's centroid, taken from its own mask.
- The label is the **union of all nodules in that scan**. If a neighbouring nodule falls inside the crop, it is labelled foreground instead of being taught as background.
- Each crop is stored with 128×128 in-plane context (`--context`). The central 64³ is the crop you asked for. The extra ring lets the shift augmentation use real anatomy instead of padding. Use `--context 64` to store exactly 64³; shifted windows are then padded with -1000 HU.
- Slice labels:
  - positive = the central 64×64 contains nodule pixels.
  - negative = the whole slice is empty.
  - Slices with nodule pixels only in the outer context ring are skipped.
- Adds `--bg_crops_per_scan` random, nodule-free crops from inside the lungs (default 1). These give extra negatives with vessels, airways and pleura.
- Writes `qc_nifti/` 64³ crops with real geometry so you can open them in ITK-SNAP.

**Training**
- Every epoch uses **all positive slices** plus a fresh random draw of `neg_ratio × #pos` empty slices (default 1:1). By default 30% of those come from the background lung crops (`--bg_frac`).
- **Shift augmentation.** With probability 0.8, the 64×64 window moves off the nodule in a random direction by U(0, 24) px (`--p_shift`, `--max_shift`). Flips, rot90 and a small intensity jitter are also applied.
- Loss is BCE + per-slice soft Dice. Empty slices are included (`--dice_mode all`), so any blob predicted on an empty slice is penalised heavily. This targets the false-positive problem you described.
- Optimiser: AdamW, lr 1e-3 with a 2-epoch warmup and cosine decay. Training uses AMP and SyncBatchNorm across GPUs. `--batch_size` is per GPU.
- **80/20 train/validation split by patient** (`--val_frac 0.2`, seeded by `--seed`). All slices from one patient stay on the same side of the split.
  - `split.csv` lists which patients went where.
  - `train_paths.csv` and `val_paths.csv` use the same columns as your spreadsheet, so `val_paths.csv` can go straight into step 3.
- Validation uses the same positive:negative mix and shift distribution as training. Each validation slice keeps one fixed shift, and there are no flips or jitter, so the validation curve is comparable from epoch to epoch.
- **`loss_curves.png`** is rewritten after every epoch. It shows train vs validation for three things: loss, Dice on nodule slices, and the false-positive rate on empty slices (the fraction of empty slices where anything was predicted). The raw numbers are in `train_log.csv`.
- Checkpoints:
  - `best.pt`: lowest validation loss.
  - `last.pt`: saved every epoch.
  - `epoch_XXX.pt`: every 10 epochs.
  - `final.pt`: end of training.
  - Resume with `--resume $RUN/last.pt`.
- `--neg_ratio -1` uses every empty slice every epoch instead of a 1:1 draw.

**Inference and post-processing**
- Applies identical preprocessing. The crop is centred on the GT-mask centroid, or on `center_x/center_y/center_z` columns (LPS mm) if you have no masks.
- `--random_offset_mm R` or `--offset_mm dz,dy,dx` displaces the crop to test off-centre nodules. `--roi 96` or `--roi 128` feeds a larger field of view; the network is fully convolutional.
- Post-processing:
  1. Threshold.
  2. Fill holes in each slice.
  3. Find 3D connected components.
  4. Drop components smaller than `--min_voxels` mm³ or spanning fewer than `--min_slices` slices.
  5. Keep `all`, the `largest`, or the component nearest the `center` (`--keep`).
- Predictions are resampled back to the **original CT grid** and saved in `pred_nifti/`: one file per nodule plus `*_pred_all.nii.gz` per scan.
- Metrics: `metrics_per_nodule.csv` has Dice, IoU, precision, recall, HD95, volumes and FP slices on GT-empty slices. `metrics_per_scan.csv` has Dice in original space.

**Step 4: external cohort (`04_infer_test_cohort.py`)**
- Input: a CSV or XLSX with `image_path` and `coordX/coordY/coordZ` (nodule centroids in physical mm). Every other column (`pid`, `study_yr`, `nodule_id`, `diag_nodule_1yr`, and so on) is copied through to the output.
- Coordinate convention: centroids are read as **LPS** by default (`--coord_system lps`), the convention of DICOM patient coordinates and SimpleITK. Before the full run, 30 sampled CTs are checked against LPS, RAS and voxel indices. The results go to `coord_check.csv`, and the script stops if another convention clearly fits better. `--coord_system ras`, `voxel` and `auto` are also available.
- Preprocessing, model and post-processing are the same as training. Post-processing uses `--keep center` by default, which keeps only the component at the given centroid so neighbouring nodules don't inflate volumes. Use `--keep all` to keep every component.
- Outputs:
  - `pred_nifti/<uid>_pred.nii.gz`: mask on the original CT grid.
  - `pred_nifti/<scan>_pred_all.nii.gz`: all nodules in that scan combined.
  - `crops_64/<uid>_{img,pred,prob}.nii.gz`: the reconstructed 64×64×64 nodule space at 1 mm, with real-world geometry.
  - `png/<uid>/zNNN.png`: CT | prediction. The given centroid is marked with a small white "+".
  - `metrics_per_nodule.csv`:
    - Volume at 1 mm and in original space, equivalent-sphere diameter, maximum axial diameter.
    - QC flags: empty prediction, centroid inside the prediction, centroid outside the CT, probability at the centroid.
    - Dice, IoU etc. if you pass `--mask_col`.

## Step 4 with pre-cropped `.pt` nodule cubes

`04_infer_test_cohort.py` also accepts `.pt` / `.pth` files, each holding one pre-cropped nodule cube in raw HU. You can pass them in three ways:
- `--pt_dir FOLDER`: every `.pt` / `.pth` file in that folder.
- `--csv` with `.pt` paths in `image_path`. CT rows and cube rows can be mixed; cube rows need no coordinates.
- Both together.

Format and options:
- **Array:** a tensor or numpy array of shape (D, H, W), meaning slices, rows, columns. It can also be a dict holding it under `image`/`img`/`ct`/`volume`/`vol`/`data`/`x`/`cube`. Leading size-1 dimensions are dropped. Use `--pt_axes xyz` if the cubes are stored as (W, H, D).
- **Spacing:** `--pt_spacing x,y,z`, default 1,1,1 mm. Cubes with other spacing are resampled to 1 mm.
- **Nodule position:** the cube centre by default. `--pt_center coords` reads it from coordX/Y/Z instead, as voxel (x, y, z) inside the cube.

Each cube goes through the same 64³ crop, model, post-processing and adaptive crop as a CT. Outside the cube the crop is padded with -1000 HU. The adaptive crop can only add real anatomy if the cube is larger than 64³. Otherwise the row gets `fov_limited_by_cube = True`.

Outputs for cubes:
- `pred_pt/<uid>_pred.pt`: uint8 mask with the same shape and axis order as the input cube.
- `pred_nifti/<uid>_pred.nii.gz`: the same mask on the cube's own grid.
- `crops_64/`, PNGs and CSV rows as for CTs, with `input_type = pt_cube`.

The uid is the file name, unless the CSV row has pid / study_yr / nodule_id.


## Notebook: `nodule_fov_comparison.ipynb`

An interactive comparison for one nodule at a time:
1. Pick a nodule from a table (cohort CSV with coordX/Y/Z in LPS, or `val_paths.csv` with masks).
2. Segment a 128³ crop around it.
3. Segment the 64³ crop plus its 8 neighbouring 64×64 tiles per slice, stitch them, and smooth with an HMM (or Gaussian, or none).
4. Visualize both: summary table (volume, diameters, border contact, Dice vs ground truth), orthogonal views with tile boundaries, a tiles/smoothing panel, and a slice browser.

Edit only the Configuration cell. `ipywidgets` is optional.

## GRIT model (gradient + intensity-threshold input channels)

`02_train.py --grit` trains the same 2D V-Net with 3 input channels per 64×64 slice instead of 1 (code in `grit.py`):

1. **Image:** the normalised image, HU -1000..400 → [0, 1].
2. **Gradient field:** |dI/dx| + |dI/dy|, using central differences.
3. **Thresholded image:** pixels in the top P % of that slice's intensities keep their value; all others are set to 0.

P ∈ {12.5, 25, 50, 75} is chosen per nodule:
- **Descriptor:** the mean gradient field in the 10×10 patch at the nodule centroid, on the centroid slice.
- **K-means (K = 4):** fit on the training nodules only. The centres are sorted and mapped to P, lowest gradient → 12.5 %. `--grit_reverse` reverses that order.
- **Storage:** descriptors are cached in `<prep>/grit_descriptors.csv`. The fitted centres are written to `grit_selector.json` and stored in every checkpoint.

The channels are built on the fly from the exact 64×64 window the network sees, after the shift augmentation.

Model change: in the first V-Net layer, the input is added back as a residual. For 3 channels it is projected to 16 channels with a 1×1 convolution, because 16 is not a multiple of 3. 1-channel models are unchanged.

Other changes:
- **Early stopping:** `--early_stop_patience N` stops when validation loss hasn't improved for N epochs, e.g. `--epochs 300 --early_stop_patience 30`.
- **Inference:** steps 3/4, `.pt` cubes and the notebook detect a GRIT checkpoint automatically. They compute each nodule's descriptor at its centre and assign it to the nearest saved centre. The chosen level is written to the CSV as `grit_threshold_pct`. 1-channel checkpoints keep working unchanged.
