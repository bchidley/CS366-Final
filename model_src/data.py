"""
data.py
Matt Hake, Ben Chidley, Garrett Keyhani, Josh Smith
12/11/2025

- Define ParkingDataset with image masking
- Implement image loading with recursive file search
- Pre-load masks for efficiency

Sources:
- PyTorch Dataset class: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
- PIL Image masking: https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.composite
"""
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import os
import pandas as pd
from torchvision import transforms

LOT_CAPACITIES = {
    'Tennis1': 36, 'Tennis2': 36, 'MMM1': 64, 'MMM2': 64,
    'North1': 53, 'North2': 89, 'Root1': 77, 'Root2': 77, 'RootStaff': 41
}

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

# ImageNet Stats
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class ParkingDataset(Dataset):
    def __init__(self, csv_file, img_dir, mask_dir, transform=None):
        raw_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform 
        
        # Pre-Load Masks
        print("Loading and caching masks...")
        self.cached_masks = {}
        
        for lot, mask_filename in FOLDER_TO_MASK.items():
            if mask_filename is None:
                self.cached_masks[lot] = None
                continue
            
            mask_path = os.path.join(self.mask_dir, mask_filename)
            if os.path.exists(mask_path):
                # Open mask, convert to Grayscale (L), resize to target size
                mask_img = Image.open(mask_path).convert("L")
                mask_img = mask_img.resize((224, 224))
                mask_img = ImageOps.invert(mask_img)
                self.cached_masks[lot] = mask_img
            else:
                print(f"WARNING: Mask not found at {mask_path}. Will skip masking for {lot}.")
                self.cached_masks[lot] = None

        # File Finder (Recursive Scan)
        print(f"Scanning {self.img_dir} (and subfolders)...")
        self.image_path_map = {}
        for root, dirs, files in os.walk(self.img_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_path_map[file] = os.path.join(root, file)
        
        # Match CSV to Disk
        valid_indices = []
        self.valid_paths = [] 
        
        for idx, row in raw_df.iterrows():
            csv_filename = os.path.basename(row['Full_Path'])
            
            # Attempt 1: Exact Match
            if csv_filename in self.image_path_map:
                valid_indices.append(idx)
                self.valid_paths.append(self.image_path_map[csv_filename])
                continue

            # Attempt 2: Fix Duplicate Prefix (e.g. MMM1_MMM1_day -> MMM1_day)
            parts = csv_filename.split('_', 1) 
            if len(parts) > 1:
                fixed_name = parts[1]
                if fixed_name in self.image_path_map:
                    valid_indices.append(idx)
                    self.valid_paths.append(self.image_path_map[fixed_name])
                    continue
        
        self.data_frame = raw_df.loc[valid_indices].reset_index(drop=True)
        print(f"Dataset Ready: Matched {len(self.data_frame)} images from CSV.")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.valid_paths[idx]
        filename = os.path.basename(img_path)
        
        # Identify which lot this is
        current_lot = None
        total_capacity = 100
        for key, cap in LOT_CAPACITIES.items():
            if key in filename:
                current_lot = key
                total_capacity = cap
                break

        # Get Labels
        detected_cars = int(self.data_frame.iloc[idx]['Car_Count'])
        open_spots = max(0, total_capacity - detected_cars)
        target = torch.tensor([float(open_spots)], dtype=torch.float32)

        # Load Image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new('RGB', (224, 224))

        # Apply Resize and Mask Manually
        image = image.resize((224, 224))
        
        if current_lot in self.cached_masks and self.cached_masks[current_lot] is not None:
            mask = self.cached_masks[current_lot]
            # Create a black background
            black_bg = Image.new("RGB", image.size, (0, 0, 0))
            # Composite the image onto black using the mask
            image = Image.composite(image, black_bg, mask)

        # Final Transforms
        final_steps = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        image = final_steps(image)

        return image, target

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
#