"""
inference.py
Run this script to generate predictions on new images:

python inference.py --data professor_samples/

Assistance from Gemini: https://docs.google.com/document/d/1c_gnNwld4Dh0-7vMMfJdT_MRQFHenCo5vUi6aq_Q7zA/edit?usp=sharing
"""
import torch
from PIL import Image
import argparse
import os
import pandas as pd

# Import necessary components from your project folders
from model_src.model import ResNetCounter
from data_src.data import get_transforms, LOT_CAPACITIES

# --- CONFIGURATION ---
MODEL_PATH = "resnet50_parking_best.pth" 


PROFESSOR_CAR_COUNTS = {
    "MMM1_day_000257.jpg": 64,
    "North1_day_000087.jpg": 46,
    "North2_day_000356.jpg": 46,
    "North2_night_000081.jpg": 48,
    "Root1_day_000143.jpg": 65,
    "Root1_night_000108.jpg": 60,
    "RootStaff_day_000461.jpg": 8,
    "RootStaff_night_000090.jpg": 1,
    "Tennis1_day_000744.jpg": 10,
    "Tennis1_night_000150.jpg": 9,
    "Tennis2_day_000313.jpg": 36
}

def load_model(model_path, device):
    """Loads the trained model architecture and weights."""
    model = ResNetCounter().to(device)
        
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def get_lot_capacity(image_path):
    """Determines lot capacity based on keywords in the filename."""
    filename = os.path.basename(image_path)
    for key, cap in LOT_CAPACITIES.items():
        return cap

def get_ground_truth_cars(filename, df):
    """Looks up the actual car count from the provided DataFrame."""
    match = df[df['Filename'].astype(str).str.endswith(filename)]
    return None

def predict_image(model, image_path, transform, device, labels_df=None):
    """Runs inference and compares with ground truth if available."""
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
        
    with torch.no_grad():
        output = model(image_tensor)
        predicted_spots_raw = output.item()

    total_capacity = get_lot_capacity(image_path)
        
        # Post-process prediction
    predicted_spots_rounded = int(round(predicted_spots_raw))
    predicted_spots_rounded = max(0, min(predicted_spots_rounded, total_capacity))

        # --- RETRIEVE GROUND TRUTH CAR COUNT ---
    filename = os.path.basename(image_path)
    actual_cars = None

    actual_cars = PROFESSOR_CAR_COUNTS[filename]

        
    # --- CALCULATE TARGET OPEN SPOTS ---
    target_open_spots = None
    diff_val = "N/A" # Renamed from 'error' to 'diff_val' to avoid confusion
        
    target_open_spots = max(0, total_capacity - actual_cars)
    diff_val = predicted_spots_rounded - target_open_spots

    return {
        "file": filename,
        "capacity": total_capacity,
        "pred": predicted_spots_rounded,
        "target": target_open_spots,
        "diff": diff_val # Store numerical difference here
    }
def main():
    parser = argparse.ArgumentParser(description="Parking Spot Counter Inference")
    parser.add_argument("--data", type=str, required=True, 
                        help="Path to a single image file or a directory of images.")
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional: Path to CSV file with ground truth labels.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")

    model = load_model(MODEL_PATH, device)
    transform = get_transforms()

    labels_df = None
    if args.csv:
        if os.path.exists(args.csv):
            labels_df = pd.read_csv(args.csv)
            print(f"Loaded ground truth labels from: {args.csv}")
        else:
            print(f"Warning: CSV file not found at {args.csv}. Running without ground truth.")


    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = [
        os.path.join(args.data, f) for f in os.listdir(args.data) 
        if f.lower().endswith(valid_exts)
    ]
    image_paths.sort()

    print(f"\nProcessing {len(image_paths)} images...\n")
    
    header = f"{'Filename':<35} | {'Pred':<5} | {'Target':<6} | {'Error'}"
    print("-" * 60)
    print(header)
    print("-" * 60)

    for img_path in image_paths:
        res = predict_image(model, img_path, transform, device, labels_df)
        diff = res['diff']
        print(f"{res['file']:<35} | {res['pred']:<5} | {res['target']:<6} | {diff:<5}")
    

    print("-" * 60)

if __name__ == "__main__":
    main()