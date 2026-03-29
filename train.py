import torch
import numpy as np
from torch_geometric.data import Data
from model import WeatherGNN

# Load data
X = torch.load('X.pt')  # (363, 65160, 8)
Y = torch.load('Y.pt')  # (363, 65160, 8)

# Load graph edges
import xarray as xr
ds = xr.open_dataset('era5_pressure_levels.nc')
lat = ds.latitude.values[::4]
lon = ds.longitude.values[::4]
n_lat, n_lon = len(lat), len(lon)

edges_src, edges_dst = [], []
for i in range(n_lat):
    for j in range(n_lon):
        node = i * n_lon + j
        if j + 1 < n_lon:
            edges_src.append(node); edges_dst.append(i * n_lon + (j + 1))
        if j - 1 >= 0:
            edges_src.append(node); edges_dst.append(i * n_lon + (j - 1))
        if i + 1 < n_lat:
            edges_src.append(node); edges_dst.append((i + 1) * n_lon + j)
        if i - 1 >= 0:
            edges_src.append(node); edges_dst.append((i - 1) * n_lon + j)

edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

# Model
model = WeatherGNN(in_channels=8, hidden_channels=64, out_channels=8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

# Train
n_epochs = 10
batch_size = 4

print("Starting training...")
for epoch in range(n_epochs):
    model.train()
    total_loss = 0

    for i in range(0, len(X), batch_size):
        x_batch = X[i:i+batch_size]
        y_batch = Y[i:i+batch_size]

        for j in range(len(x_batch)):
            optimizer.zero_grad()
            out = model(x_batch[j], edge_index)
            loss = loss_fn(out, y_batch[j])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    avg_loss = total_loss / len(X)
    print(f"Epoch {epoch+1}/{n_epochs} — Loss: {avg_loss:.4f}")

# Save model
torch.save(model.state_dict(), 'weather_gnn.pt')
print("Model saved as weather_gnn.pt")
# This script trains the Weather GNN model on the prepared ERA5 data.
# It loads the 363 input/output pairs (X.pt and Y.pt) and the graph edge structure.
# The model sees the atmosphere at time T and tries to predict the atmosphere at T+6 hours.
# Loss function is MSE (Mean Squared Error) — measures how wrong the prediction is.
# Optimizer is Adam — adjusts the model weights after each prediction to reduce the loss.
# The loss should decrease each epoch — that means the model is learning.
# After 10 epochs, the trained model is saved as weather_gnn.pt for use in predictions