import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import our custom modules
from data import ParkingDataset, get_transforms
from model import ResNetCounter

# --- HYPERPARAMETERS ---
CSV_FILE = '/home/bchidley/CS366-Final/labels/car_counts.csv' # Check this path!
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 15
ZERO_SPOT_WEIGHT = 10.0  # Weight multiplier for errors on 0 open spots

class WeightedMSELoss(nn.Module):
    def __init__(self, zero_weight=10.0):
        super(WeightedMSELoss, self).__init__()
        # Define the penalty multiplier for errors involving 0 spots
        self.zero_weight = zero_weight 

    def forward(self, input, target):
        # Calculate the standard squared error element-wise
        squared_error = (input - target) ** 2
        
        # Create a mask: True where the actual target is 0 spots
        zero_mask = (target == 0).float()
        
        # Calculate the weight map: Apply the penalty (zero_weight) to errors 
        # where the actual target was 0, and a weight of 1.0 everywhere else.
        weight_map = (zero_mask * self.zero_weight) + (1.0 - zero_mask)
        
        # Apply the weights to the squared error
        weighted_squared_error = squared_error * weight_map
        
        # Return the mean of the weighted squared errors
        return torch.mean(weighted_squared_error)


def main():
    # 1. Setup Device (GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Prepare Data
    full_dataset = ParkingDataset(csv_file=CSV_FILE, transform=get_transforms())
    
    # Split: 80% Train, 20% Validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 3. Initialize Model
    model = ResNetCounter().to(device)
    
    # 4. Loss and Optimizer
    # MSE Loss is standard for regression
    criterion = WeightedMSELoss(zero_weight=ZERO_SPOT_WEIGHT) 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 5. Training Loop
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

        patience = 5
        if epoch == 0:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "resnet50_parking_best.pth")         
        else:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), "resnet50_parking_best.pth")
            else:
                epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break
        
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"resnet50_parking_epoch{epoch+1}.pth")

    # Save final model
    torch.save(model.state_dict(), "resnet50_parking_final.pth")
    print("Training Complete. Model saved.")
    print("Generating Scatter Plot...")
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            # Get model outputs (The model predicts Open Spots)
            outputs = model(images)
            
            # Move data back to CPU and convert to numpy
            preds_list = outputs.cpu().numpy().flatten()
            targets_list = targets.cpu().numpy().flatten()
            
            all_preds.extend(preds_list)
            all_targets.extend(targets_list)

    # Convert to standard lists
    predicted_spots = all_preds
    actual_spots = all_targets

    # Plotting
    plt.figure(figsize=(10, 8))
    
    # 1. The Scatter points
    plt.scatter(actual_spots, predicted_spots, alpha=0.5, color='blue', label='Predictions')
    
    # 2. The "Perfect Prediction" Line (y=x)
    # If a point lands on this red line, the prediction was perfect.
    min_val = min(min(actual_spots), min(predicted_spots))
    max_val = max(max(actual_spots), max(predicted_spots))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Fit')

    plt.xlabel('Actual Open Spots')
    plt.ylabel('Predicted Open Spots')
    plt.title('Validation: Predicted vs Actual Open Spots')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig('scatter_open_spots.png')
    print("Scatter plot saved as 'scatter_open_spots.png'")

if __name__ == "__main__":
    main()
    