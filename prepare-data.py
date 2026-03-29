import xarray as xr
import numpy as np
import torch

# Load data
ds = xr.open_dataset('era5_pressure_levels.nc')

# Downsample to match our graph (every 4th point)
z = ds['z'].values[:, :, ::4, ::4]   # geopotential
t = ds['t'].values[:, :, ::4, ::4]   # temperature
u = ds['u'].values[:, :, ::4, ::4]   # wind east-west
v = ds['v'].values[:, :, ::4, ::4]   # wind north-south

print(f"Shape of z: {z.shape}")  # (364, 2, 181, 360)

# Flatten spatial dimensions — (time, levels, lat, lon) -> (time, nodes, levels)
n_time = z.shape[0]
n_nodes = 181 * 360  # 65160

z = z.reshape(n_time, n_nodes, 2)
t = t.reshape(n_time, n_nodes, 2)
u = u.reshape(n_time, n_nodes, 2)
v = v.reshape(n_time, n_nodes, 2)

# Stack all variables — shape becomes (time, nodes, 8)
data = np.concatenate([z, t, u, v], axis=2)
print(f"Full data shape: {data.shape}")  # (364, 65160, 8)

# Normalize — zero mean, unit variance
mean = data.mean(axis=(0, 1), keepdims=True)
std = data.std(axis=(0, 1), keepdims=True)
data = (data - mean) / std

# Save mean and std — we'll need them later to unnormalize predictions
np.save('mean.npy', mean)
np.save('std.npy', std)

# Create input/output pairs
# Input = snapshot at time T
# Output = snapshot at time T+1 (6 hours later)
X = torch.tensor(data[:-1], dtype=torch.float32)  # all except last
Y = torch.tensor(data[1:],  dtype=torch.float32)  # all except first

print(f"X shape: {X.shape}")  # (363, 65160, 8)
print(f"Y shape: {Y.shape}")  # (363, 65160, 8)

# Save
torch.save(X, 'X.pt')
torch.save(Y, 'Y.pt')
print("Data saved! Ready to train.")
# This script prepares the ERA5 data for training the GNN.
# It loads all 4 variables (z, t, u, v) at 2 pressure levels and flattens them into 8 features per node.
# The data is normalized (zero mean, unit variance) so the model trains faster and more stably.
# Mean and std are saved as mean.npy and std.npy — needed later to convert predictions back to real units.
# 363 input/output pairs are created: X is the atmosphere at time T, Y is the atmosphere at T+6 hours.
# Output files: X.pt and Y.pt — these are the direct inputs to the training script.