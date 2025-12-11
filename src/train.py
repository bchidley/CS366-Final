import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Import our custom modules
from data import ParkingDataset, get_transforms
from model import ResNetCounter

# --- HYPERPARAMETERS & PATHS ---
# Using the partner's absolute paths as established
CSV_FILE = '/home/bchidley/CS366-Final/labels/car_counts.csv'
IMAGE_DIR = '/home/bchidley/CS366-Final/gpu_data/final'
MASK_DIR = '/home/bchidley/CS366-Final/gpu_data/masks'

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 15
ZERO_SPOT_WEIGHT = 10.0

class WeightedMSELoss(nn.Module):
    def __init__(self, zero_weight=10.0):
        super(WeightedMSELoss, self).__init__()
        self.zero_weight = zero_weight 

    def forward(self, input, target):
        squared_error = (input - target) ** 2
        zero_mask = (target == 0).float()
        weight_map = (zero_mask * self.zero_weight) + (1.0 - zero_mask)
        weighted_squared_error = squared_error * weight_map
        return torch.mean(weighted_squared_error)

def main():
    # 1. Setup Device (GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Prepare Data (Now with Masks!)
    print("Initializing dataset...")
    full_dataset = ParkingDataset(
        csv_file=CSV_FILE, 
        img_dir=IMAGE_DIR, 
        mask_dir=MASK_DIR,
        transform=get_transforms()
    )
    
    # Split: 80% Train, 20% Validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # num_workers=4 is good for Linux
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 3. Initialize Model
    model = ResNetCounter().to(device)
    
    # 4. Loss and Optimizer
    criterion = WeightedMSELoss(zero_weight=ZERO_SPOT_WEIGHT) 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 5. Training Loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 5

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        progress_bar = tqdm(train_loader, desc="Training")
        
        for images, targets in progress_bar:
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = running_loss / len(train_loader)
        
        # Validation Step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Early Stopping Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "resnet50_parking_best.pth")
            print(" -> Best model saved!")
        else:
            epochs_no_improve += 1
            print(f" -> No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"resnet50_parking_epoch{epoch+1}.pth")

    # Save final model
    torch.save(model.state_dict(), "resnet50_parking_final.pth")
    print("Training Complete. Generating Scatter Plot...")
    
    # --- PLOTTING ---
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

    plt.figure(figsize=(10, 8))
    plt.scatter(all_targets, all_preds, alpha=0.5, color='blue', label='Predictions')
    min_val = min(min(all_targets), min(all_preds))
    max_val = max(max(all_targets), max(all_preds))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Fit')
    plt.xlabel('Actual Open Spots')
    plt.ylabel('Predicted Open Spots')
    plt.title('Validation: Predicted vs Actual Open Spots')
    plt.legend()
    plt.grid(True)
    plt.savefig('scatter_open_spots.png')
    print("Scatter plot saved as 'scatter_open_spots.png'")

# --- CRITICAL ENTRY POINT ---
if __name__ == "__main__":
    main()