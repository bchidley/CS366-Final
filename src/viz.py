"""
viz.py
Matt Hake, Ben Chidley, Garrett Keyhani, Josh Smith
12/11/2025

- Load trained ResNetCounter model from checkpoint
- Build a validation split of ParkingDataset
- Run model to get predicted open spots
- Save side-by-side panels into outputs

Sources:
- Pytorch vision normalization / denormalization pattern: https://pytorch.org/vision/stable/models.html
- PIL ImageDraw text usage: https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html
"""
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader, random_split

from data import ParkingDataset, get_transforms, MEAN, STD
from model import ResNetCounter
from train import CSV_FILE, IMAGE_DIR, MASK_DIR  # Reuse our existing paths

# -CONFIG-
CHECKPOINT_PATH = "resnet50_parking_best.pth"
OUT_DIR = "outputs/samples"
NUM_SAMPLES = 12
VAL_BATCH_SIZE = 1
VAL_SPLIT_RATIO = 0.2
RANDOM_SPLIT_SEED = 42

def tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """
    Undo ImageNet-style normalization and convert (3, H, W) tensor -> PIL RGB image.
    """
    img = img_tensor.detach().cpu().clone()

    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    img = img * std + mean
    img = img.clamp(0.0, 1.0)

    img_np = (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(img_np, mode="RGB")


def add_prediction_text(
    img_pil: Image.Image,
    pred_open_spots: float,
    target_open_spots: float,
) -> Image.Image:
    """
    Draw prediction / target / error text onto a copy of img_pil.
    """
    img = img_pil.convert("RGB").copy()
    draw = ImageDraw.Draw(img)

    pred_rounded = max(0.0, round(pred_open_spots))
    target_rounded = round(float(target_open_spots))
    abs_error = abs(pred_rounded - target_rounded)

    text = (
        f"Predicted open spots: {pred_rounded:.0f}\n"
        f"Actual open spots:    {target_rounded:.0f}\n"
        f"|Error|:               {abs_error:.0f}"
    )

    # Compute text bounding box so we can draw a background rectangle
    margin = 10
    bbox = draw.multiline_textbbox((margin, margin), text)
    x0, y0, x1, y1 = bbox

    # Semi-transparent dark background for readability
    # (Using solid fill in RGB; good enough for our case.)
    draw.rectangle((x0 - 4, y0 - 4, x1 + 4, y1 + 4), fill=(0, 0, 0))

    # Draw text in white
    draw.multiline_text((margin, margin), text, fill=(255, 255, 255))

    return img


def make_panel(
    original_img: Image.Image,
    annotated_img: Image.Image,
) -> Image.Image:
    """
    Create a single row panel: [original | annotated] using PIL only.
    All inputs are PIL Images, assumed same size.
    """
    w, h = original_img.size
    panel = Image.new("RGB", (w * 2, h))
    panel.paste(original_img, (0 * w, 0))
    panel.paste(annotated_img, (1 * w, 0))
    return panel


def build_val_loader() -> DataLoader:
    """
    Rebuild dataset and create a validation DataLoader, similar to train.py.
    The specific val samples may differ from the original training run,
    but this is sufficient for qualitative visualization.
    """
    full_dataset = ParkingDataset(
        csv_file=CSV_FILE,
        img_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        transform=get_transforms(),
    )

    val_size = int(VAL_SPLIT_RATIO * len(full_dataset))
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(RANDOM_SPLIT_SEED)
    _, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )
    return val_loader


def load_model(device: torch.device) -> ResNetCounter:
    # Initialize ResNetCounter and load weights from checkpoint

    model = ResNetCounter().to(device)

    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    # Ensure output directory exists
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for visualization: {device}")

    # Load our model
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found at '{CHECKPOINT_PATH}'. "
            f"Make sure you have trained the model and saved the checkpoint."
        )

    model = load_model(device)
    print(f"Loaded model weights from: {CHECKPOINT_PATH}")

    # Validation loader
    val_loader = build_val_loader()

    saved = 0
    with torch.no_grad():
        for images, targets in val_loader:
            if saved >= NUM_SAMPLES:
                break

            # REFERENCE: images: [1, 3, H, W] //// targets: [1, 1]
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            pred_value = float(outputs[0, 0].cpu().item())
            target_value = float(targets[0, 0].cpu().item())

            # To PIL
            img_tensor = images[0].cpu()
            original_pil = tensor_to_pil(img_tensor)
            annotated_pil = add_prediction_text(original_pil, pred_value, target_value)

            # Make the pane
            panel = make_panel(original_pil, annotated_pil)
            out_path = os.path.join(OUT_DIR, f"sample_{saved:02d}.png")
            panel.save(out_path)
            print("saved", out_path)

            saved += 1

    print("Done - saved", saved, "panels in", OUT_DIR)


if __name__ == "__main__":
    main()
