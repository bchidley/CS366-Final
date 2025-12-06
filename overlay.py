import os
import shutil
from PIL import Image

# --- Configuration ---
ROOT_DIR = "/home/bchidley/gpu_data/final"
MASK_BASE_DIR = "/home/bchidley/gpu_data/masks"
OUTPUT_DIR = "/home/bchidley/gpu_data/overlays"

# Map folders to mask files
FOLDER_TO_MASK = {
    "Tennis1": "Tennis1_mask.png",
    "Tennis2": "Tennis2_mask.png",
    "MMM1":    "MMM_mask.png",
    "MMM2":    "MMM_mask.png",
    "North1":  "North1_mask.png",
    "North2":  "North2_mask.png",
    "Root1":   "Root_mask.png",
    "Root2":   "Root_mask.png",
    "RootStaff": None 
}

# --- Auto-Clear Output Directory ---
if os.path.exists(OUTPUT_DIR):
    print(f"Clearing old files from {OUTPUT_DIR}...")
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

print(f"Processing ALL images... This may take a while.")

for folder_name in os.listdir(ROOT_DIR):
    folder_path = os.path.join(ROOT_DIR, folder_name)
    
    if not os.path.isdir(folder_path):
        continue

    # 1. Get Mask Filename
    if folder_name not in FOLDER_TO_MASK:
        continue

    mask_filename = FOLDER_TO_MASK[folder_name]
    loaded_mask_alpha = None

    # 2. Load the Mask (Extract Alpha Channel Only)
    if mask_filename:
        mask_path = os.path.join(MASK_BASE_DIR, mask_filename)
        try:
            mask_src = Image.open(mask_path).convert("RGBA")
            # Extract the Alpha channel to use as a "cutout" map
            loaded_mask_alpha = mask_src.split()[3] 
            print(f"Loaded mask for {folder_name}")
        except Exception as e:
            print(f"Error loading mask {mask_filename}: {e}")
            continue

    # 3. Process Images (ALL frames)
    # REMOVED [:5] limit here so it does everything!
    frames = sorted(os.listdir(folder_path)) 
    
    count = 0
    for frame_name in frames:
        if not frame_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(folder_path, frame_name)
        save_path = os.path.join(OUTPUT_DIR, f"{folder_name}_{frame_name}")

        try:
            with Image.open(image_path).convert("RGBA") as img:
                
                if mask_filename and loaded_mask_alpha:
                    # Resize the mask alpha to match the image size
                    current_mask_alpha = loaded_mask_alpha.resize(img.size, Image.Resampling.NEAREST)
                    
                    # Create a SOLID BLACK image
                    solid_black = Image.new("RGBA", img.size, (0, 0, 0, 255))
                    
                    # Composite Black onto Image using Mask Alpha
                    img = Image.composite(solid_black, img, current_mask_alpha)

                img.convert("RGB").save(save_path)
                count += 1
                
        except Exception as e:
            print(f"Failed to process {frame_name}: {e}")
            
    print(f"Finished {count} images in {folder_name}")

print("Done! All overlays created.")