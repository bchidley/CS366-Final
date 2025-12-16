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
import sys
import pandas as pd

# Import only the model class, we will define capacities locally to be safe
try:
    from model_src.model import ResNetCounter
    from data_src.data import get_transforms
except ImportError:
    from model import ResNetCounter
    from data import get_transforms

# --- CONFIGURATION ---
MODEL_PATH = "resnet50_parking_best.pth" 

# --- ENSURE CAPACITIES ARE CORRECT ---
# We define this HERE to ensure the script uses the correct numbers
LOT_CAPACITIES = {
    'Tennis1': 36, 'Tennis2': 36, 
    'MMM1': 64, 'MMM2': 64,
    'North1': 53, 'North2': 89, 
    'Root1': 77, 'Root2': 77, 
    'RootStaff': 41
}

# --- HARDCODED CAR COUNTS FOR PROFESSOR EXAMPLES ---
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
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at '{model_path}'")
        sys.exit(1)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def get_lot_capacity(image_path):
    """Determines lot capacity based on keywords in the filename."""
    filename = os.path.basename(image_path)
    # Sort keys by length (descending) to ensure 'North2' matches before 'North' if needed
    for key in sorted(LOT_CAPACITIES.keys(), key=len, reverse=True):
        if key in filename:
            return LOT_CAPACITIES[key]
    return 100

def get_ground_truth_cars(filename, df):
    """Looks up the actual car count from the provided DataFrame."""
    if df is None:
        return None
    
    if 'Filename' in df.columns:
        match = df[df['Filename'].astype(str).str.endswith(filename)]
        if not match.empty:
            return int(match.iloc[0]['Car_Count'])

    if 'Full_Path' in df.columns:
        match = df[df['Full_Path'].astype(str).str.contains(filename, regex=False)]
        if not match.empty:
            return int(match.iloc[0]['Car_Count'])     
    return None

def predict_image(model, image_path, transform, device, labels_df=None):
    """Runs inference and compares with ground truth if available."""
    try:
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

        if labels_df is not None:
            actual_cars = get_ground_truth_cars(filename, labels_df)
        
        if actual_cars is None:
            if filename in PROFESSOR_CAR_COUNTS:
                actual_cars = PROFESSOR_CAR_COUNTS[filename]
            else:
                for k, v in PROFESSOR_CAR_COUNTS.items():
                    if filename in k: 
                        actual_cars = v
                        break
        
        # --- CALCULATE TARGET OPEN SPOTS ---
        target_open_spots = None
        diff_val = "N/A"
        
        if actual_cars is not None:
            target_open_spots = max(0, total_capacity - actual_cars)
            diff_val = predicted_spots_rounded - target_open_spots

        return {
            "file": filename,
            "capacity": total_capacity,
            "pred": predicted_spots_rounded,
            "target": target_open_spots,
            "diff": diff_val
        }

    except Exception as e:
        return {"error": str(e), "file": os.path.basename(image_path)}

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
    if args.csv and os.path.exists(args.csv):
        labels_df = pd.read_csv(args.csv)

    if os.path.isfile(args.data):
        image_paths = [args.data]
    elif os.path.isdir(args.data):
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        image_paths = [
            os.path.join(args.data, f) for f in os.listdir(args.data) 
            if f.lower().endswith(valid_exts)
        ]
        image_paths.sort()
    else:
        print(f"Error: Invalid path '{args.data}'")
        return

    # Run Prediction Loop
    print(f"\nProcessing {len(image_paths)} images...\n")
    
    header = f"{'Filename':<35} | {'Pred':<5} | {'Target':<6} | {'Diff':<5} | Status"
    print("-" * 95)
    print(header)
    print("-" * 95)

    for img_path in image_paths:
        res = predict_image(model, img_path, transform, device, labels_df)
        
        if "error" in res:
            print(f"{res['file']:<35} | ERROR: {res['error']}")
            continue

        if res['target'] is not None:
            diff = res['diff']
            status = "Perfect" if diff == 0 else ("Off by " + str(abs(diff)))
            print(f"{res['file']:<35} | {res['pred']:<5} | {res['target']:<6} | {diff:<5} | {status}")
        else:
            print(f"{res['file']:<35} | {res['pred']:<5} | N/A    | N/A   | No label found")

    print("-" * 95)
    print("Done.")

if __name__ == "__main__":
    main()