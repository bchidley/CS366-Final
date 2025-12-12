# CS366-Final: Automated Parking Space Detection

Parking-Lot Open-Spot Counter (CS366 Final Project)

This repository contains code to generate masked overlays of parking-lot images, run object-detection to count cars, build a labeled CSV of counts, and train a ResNet-based regression model to predict open parking spots from an image (using precomputed masks to ignore irrelevant areas).

## Contents

- `generate_csv.py` - Runs the SAM-based detector across overlay images and writes `labels/car_counts.csv` (Filename, Car_Count, Full_Path).
- `overlay.py` - Produces black-background overlays by applying per-lot mask alpha channels to raw frames and writes them to `gpu_data/overlays`.
- `label.py` - Small helper(s) for labeling (present in repo root).
- `src/` - Main project code
	- `src/data.py` - `ParkingDataset` (PyTorch Dataset) that matches CSV rows to on-disk images, applies per-lot masks, and returns (image, open_spots) tensors.
	- `src/model.py` - `ResNetCounter` model built on a pretrained ResNet50 backbone.
	- `src/train.py` - Training loop, custom weighted MSE loss, trains/validates the model and saves checkpoints and a predictions scatter plot.
- `labels/` - CSVs and label artifacts (e.g. `car_counts.csv`).

## Design / Contract

- Input: an RGB image (224x224 after resize) with masked-out background.
- Target: single scalar — number of open parking spots in the image.
- Output: trained PyTorch model which predicts open-spots (floating-point regression).

Edge cases to be aware of:
- Missing masks for a lot: `ParkingDataset` will warn and skip masking for that lot.
- CSV rows that don't match on-disk filenames are skipped during dataset build.

## Requirements

- Python 3.8+
- PyTorch (matching your CUDA if using GPU) and torchvision
- Pillow, pandas, matplotlib, seaborn, scikit-learn, tqdm
- `sam3` model code and weights (used by `generate_csv.py` / `label.py`) — these are external and not included in this repo.

Example pip install (adjust versions for your environment):

```bash
python -m pip install torch torchvision pillow pandas matplotlib seaborn scikit-learn tqdm
```

Note: the SAM3 model referenced in `generate_csv.py` expects weights at `/home/bchidley/sam3/sam3_weights/sam3.pt` and custom `sam3` Python modules; ensure those are available or update the script paths.

## Quickstart (high level)

1. Prepare images and masks:
	 - Raw frames are expected in `gpu_data/final/<LotFolder>/...`.
	 - Masks (one per lot) go in `gpu_data/masks/` and are referenced by name in `src/data.py` `FOLDER_TO_MASK` mapping.

2. Create overlays (applies mask alpha to images):

```bash
python overlay.py
```

3. Generate the CSV of car counts using the SAM detector (writes `labels/car_counts.csv`):

```bash
python generate_csv.py
```

4. Train the model (edit paths in `src/train.py` if necessary):

```bash
python src/train.py
```

The training script saves `resnet50_parking_best.pth` and a final scatter plot `scatter_open_spots.png` in the working directory.

## Useful notes / configuration

- Default CSV and image paths are hard-coded in `src/train.py` (variables `CSV_FILE`, `IMAGE_DIR`, `MASK_DIR`). Update them for your environment before training.
- `src/data.py` resizes images to 224x224, normalizes with ImageNet stats, and applies per-lot masks cached at dataset initialization.
- The training loop uses a custom `WeightedMSELoss` (zero-spot targets can be up-weighted) to reduce false negatives on empty lots.

## Authors

- Matt Hake
- Ben Chidley
- Garrett Keyhani
- Josh Smith