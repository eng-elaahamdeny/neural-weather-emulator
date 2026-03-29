import xarray as xr
import numpy as np
import torch
from torch_geometric.data import Data

# Load data
ds = xr.open_dataset('era5_pressure_levels.nc')

# We'll work with a coarser grid to keep it manageable on CPU
# Take every 4th point — reduces 721x1440 to ~180x360
lat = ds.latitude.values[::4]
lon = ds.longitude.values[::4]
n_lat = len(lat)
n_lon = len(lon)
n_nodes = n_lat * n_lon

print(f"Grid size: {n_lat} x {n_lon} = {n_nodes} nodes")

# Build edge connections — each node connects to its 4 neighbors
# (up, down, left, right)
edges_src = []
edges_dst = []

for i in range(n_lat):
    for j in range(n_lon):
        node = i * n_lon + j

        # Right neighbor
        if j + 1 < n_lon:
            edges_src.append(node)
            edges_dst.append(i * n_lon + (j + 1))

        # Left neighbor
        if j - 1 >= 0:
            edges_src.append(node)
            edges_dst.append(i * n_lon + (j - 1))

        # Down neighbor
        if i + 1 < n_lat:
            edges_src.append(node)
            edges_dst.append((i + 1) * n_lon + j)

        # Up neighbor
        if i - 1 >= 0:
            edges_src.append(node)
            edges_dst.append((i - 1) * n_lon + j)

edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

print(f"Number of edges: {edge_index.shape[1]}")
print("Graph built successfully!")
print(f"Each node connects to up to 4 neighbors.")
# This script converts the ERA5 grid into a graph structure for the GNN.
# Each grid point becomes a node (65,160 total after downsampling every 4th point).
# Each node is connected to its 4 neighbors (up, down, left, right) = 259,558 edges.
# This graph structure is what makes this project a GNN — weather spreads through
# neighboring locations, just like information spreads through edges in a graph.
# The edge_index tensor is the key output — it tells PyTorch Geometric how nodes are connected.