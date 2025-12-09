import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
from torchvision import transforms

LOT_CAPACITIES = {
    'Tennis1': 36,
    'Tennis2': 36,
    'MMM1': 64,
    'MMM2': 64,
    'North1': 53,
    'North2': 89,
    'Root1': 77,
    'Root2': 77,
    'RootStaff': 41
}

# ImageNet Statistics (Standard for ResNet)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class ParkingDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. Get the image path and detected car count from CSV
        # Based on your screenshot, 'Full_Path' is the column name
        img_path = self.data_frame.iloc[idx]['Full_Path']
        detected_cars = int(self.data_frame.iloc[idx]['Car_Count'])

        # 2. Determine Total Capacity based on Filename
        # We assume the lot name is part of the filename/path
        total_capacity = 0
        found_lot = False
        
        for key, cap in LOT_CAPACITIES.items():
            if key in img_path:
                total_capacity = cap
                found_lot = True
                break
        
        # Fallback if filename doesn't match (Safety check)
        if not found_lot:
            # You might want to print a warning here or handle it
            total_capacity = 100 # Default or error value

        # 3. Calculate Target: OPEN SPOTS
        open_spots = max(0, total_capacity - detected_cars)
        
        target = torch.tensor([float(open_spots)], dtype=torch.float32)

        # 4. Load and Process Image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank tensor to prevent crash, or handle better
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, target

# --- TRANSFORMS ---
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)), # Resize directly to square
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])