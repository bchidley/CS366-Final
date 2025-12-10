import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
from torchvision import transforms

LOT_CAPACITIES = {
    'Tennis1': 36, 'Tennis2': 36, 'MMM1': 64, 'MMM2': 64,
    'North1': 53, 'North2': 89, 'Root1': 77, 'Root2': 77, 'RootStaff': 41
}

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class ParkingDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        raw_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # 1. Map every file on the disk to its full path
        print(f"Scanning {self.img_dir} (and subfolders)...")
        self.image_path_map = {}
        for root, dirs, files in os.walk(self.img_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_path_map[file] = os.path.join(root, file)
        
        print(f"Found {len(self.image_path_map)} images on disk.")

        # 2. Match CSV entries to Disk Files (With Auto-Fix)
        valid_indices = []
        self.valid_paths = [] 
        
        for idx, row in raw_df.iterrows():
            # Original filename from CSV: "MMM1_MMM1_day_000055.jpg"
            csv_filename = os.path.basename(row['Full_Path'])
            
            # Attempt 1: Exact Match
            if csv_filename in self.image_path_map:
                valid_indices.append(idx)
                self.valid_paths.append(self.image_path_map[csv_filename])
                continue

            # Attempt 2: Fix Duplicate Prefix
            # Try splitting "MMM1_MMM1_day.jpg" -> "MMM1_day.jpg"
            parts = csv_filename.split('_', 1) # Split only on the first underscore
            if len(parts) > 1:
                fixed_name = parts[1]
                if fixed_name in self.image_path_map:
                    valid_indices.append(idx)
                    self.valid_paths.append(self.image_path_map[fixed_name])
                    continue
        
        self.data_frame = raw_df.loc[valid_indices].reset_index(drop=True)
        print(f"Dataset Ready: Matched {len(self.data_frame)} images from CSV.")
        
        if len(self.data_frame) == 0:
            raise RuntimeError(f"CRITICAL ERROR: Zero matches! The naming pattern is completely different.")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Use the path we found during __init__
        img_path = self.valid_paths[idx]
        
        detected_cars = int(self.data_frame.iloc[idx]['Car_Count'])
        filename = os.path.basename(img_path)

        # Calculate Capacity
        total_capacity = 100 
        for key, cap in LOT_CAPACITIES.items():
            if key in filename:
                total_capacity = cap
                break
        
        open_spots = max(0, total_capacity - detected_cars)
        target = torch.tensor([float(open_spots)], dtype=torch.float32)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, target

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])