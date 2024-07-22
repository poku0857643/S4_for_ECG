from src import clinical_ts
from src.clinical_ts import s4_model, s4_model_lucy
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import time
import psutil
import json

########################################################################
# Logging CPU/GPU usage
def log_device_usage():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders).")
        # Currently, PyTorch does not provide direct APIs for memory usage on MPS.
        # However, we can still log general information.

    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        print(f"Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**3:.1f} GB")
        print(f"Cached: {torch.cuda.memory_reserved(0)/1024**3:.1f} GB")
    else: device = torch.device("cpu")
    print("Using CPU")

    # General CPU usage
    print(f"CPU Usage: {psutil.cpu_percent()}%")

# Example function to log device usage periodically
def log_device_usage_periodically(num_epochs, interval=1):
    for epoch in range(num_epochs):
        log_device_usage()
        time.sleep(interval)

# Call the function with your desired number of epochs and interval
# num_epochs = 30  # Example number of epochs
# log_device_usage_periodically(num_epochs, interval=10)  # Log every 10 seconds
########################################################################
# Main Code Start Here
# pykeops sandbox test
from pykeops.torch import Genred


# load type 2
with open('config.json','r') as file:
    config = json.load(file)

# extract paths
paths = config.get('path', {})

concat_npz_signal_dir = config['path']['signal']
concat_npz_profile_dir = config['path']['profile']

# Check if the files exist
if os.path.exists(concat_npz_signal_dir):
    print("Signal data exists and is accessible.")
else:
    print("Signal data does not exist or is no accessible.")

if os.path.exists(concat_npz_profile_dir):
    print("Profile data exists and is accessible.")
else:
    print("Profile data does not exist or is no accessible.")

# concat_npz_signal_dir = '/Users/eshan/Desktop/2M4 dataset/npz_file_20242M4_10s-001.npz'
# concat_npz_profile_dir = '/Users/eshan/Desktop/2M4 dataset/concatenated_dataset_2M4_20240715115817.csv'


# Load type 2
signal_data = np.load(concat_npz_signal_dir)
signals = signal_data['signal']
filenames = signal_data['filename']
print(filenames[:5])

profile_df = pd.read_csv(concat_npz_profile_dir)
print(profile_df)
filenames_df = profile_df['file_name']
target = profile_df['BS_mg_dl']
print(profile_df)


signal_dict = {fname: signal for fname, signal in zip(filenames, signals)}

X = []
y = []

for idx, row in profile_df.iterrows():
    filename = row['file_name']
    if filename in signal_dict:
        X.append(signal_dict[filename])
        y.append(row['BS_mg_dl'])

X = np.array(X)
y = np.array(y)

# 檢查 NaN 和無窮大值
print(f"NaN in labels: {np.isnan(y).sum()}")
print(f"Infinite values in labels: {np.isinf(y).sum()}")

# 移除 NaN 和無窮大值
valid_mask = np.isfinite(y)
X = X[valid_mask]
y = y[valid_mask]

print(f"Shape of cleaned aug_X: {X.shape}")
print(f"Shape of cleaned aug_y: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
X_test_torch = torch.tensor(X_test, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32)
y_test_torch = torch.tensor(y_test, dtype=torch.float32)



# Create DataLoader for training and testing sets
batch_size = 1
train_dataset = TensorDataset(X_train_torch, y_train_torch)
test_dataset = TensorDataset(X_test_torch, y_test_torch)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model parameters
input_dim =  1 #X_train.shape[1]
hidden_dim = 128
output_dim = 10

# Initialize the model, loss function, and optimizer
model = s4_model_lucy.S4Model(d_input=input_dim, d_output=output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, eps=1e-4)


# TensorBoaed writer
writer = SummaryWriter()

# Checkpoint directory
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=epoch_loss / len(train_loader))
            pbar.update(1)

            # Log loss to TensorBoard
            writer.add_scaler('Train/Loss', loss.item(), epoch * len(train_loader) + pbar.n)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')

    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch+{epoch + 1}.pth')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss / len(train_loader),
    }, checkpoint_path)

    # Log model parameters and gradients
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)
        if param.grad is not None:
            writer.add_histogram(f'{name}.grad', param.grad, epoch)

# Evaluation on test set
model.eval()
test_loss = 0
predictions = []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        test_loss += loss.item()
        outputs_flat = outputs.view(outputs.size(0), -1).cpu().numpy().flatten()
        predictions.append(outputs_flat)

test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.4f}')
writer.add_scaler('Test/Loss', test_loss, num_epochs)

# Convert predictions and true values to numpy arrays for output
predictions = np.concatenate(predictions, axis=0)
y_pred = np.array(predictions).flatten()
y_test_np = np.array(y_test).flatten()

# Output the predictions and true values
print("Predictions:", y_pred[:10])
print("True values:", y_test_np[:10])

print(f'Shape of predicted values: {y_pred.shape}')
print(f'Shape of actual values: {y_test_np.shape}')

# Save the predictions and true values to a CSV file
current_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
output_df = pd.DataFrame({
    'Predict': y_pred,
    'Actual': y_test_np
})

output_df.to_csv(f'predictions_{current_time}.csv', index=False)

print("Predictions and true values saved to predictions.csv")

# Close the TensorBoard writer
writer.close()

# Log device usage periodically during the training
log_device_usage_periodically(num_epochs, interval=10)







