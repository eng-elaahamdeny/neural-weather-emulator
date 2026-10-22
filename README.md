# Neural Weather Emulator: 6-Hour Atmospheric Forecasting with Graph Neural Networks

A Graph Neural Network that predicts the state of the Earth's atmosphere 6 hours ahead, inspired by DeepMind's GraphCast and trained on ERA5 reanalysis data from the Copernicus Climate Data Store (ECMWF).

---

## What It Does

Every 6 hours, the state of the atmosphere can be described at each point of the globe by variables such as geopotential (pressure), temperature and wind. The model takes one snapshot and predicts the next one, 6 hours later.

Instead of treating the atmosphere as a flat image (CNN approach), the Earth is represented as a graph: each grid location is a node, connected to its geographic neighbors by edges. The GNN passes information along these edges, so each prediction depends on the surrounding region.

## Data

- Source: ERA5 reanalysis, Copernicus Climate Data Store
- Period: January to March 2020, one snapshot every 6 hours (364 time steps)
- Resolution: 0.25 degrees (721 x 1440 grid), 2 pressure levels
- Size: about 12 GB
- Graph: 65,160 nodes and 259,558 edges
- Training set: 363 input/output pairs (snapshot at t, snapshot at t + 6 h)

## Model

- 3-layer Graph Convolutional Network (PyTorch Geometric)
- Trained for 10 epochs
- This is a simplified model inspired by GraphCast. It does not reproduce GraphCast's encoder-processor-decoder architecture.

## Results

Prediction (left) versus ground truth (right) after a 6-hour forecast:

![GNN 6-hour forecast vs reality](the%20result%20of%20my%20GNN's%206-hour%20forecast%20on%20the%20left%2C%20reality%20on%20the%20right..jpg)

Real ERA5 pressure field, 1 January 2020:

![ERA5 pressure field](the%20real%20ERA5%20atmospheric%20pressure%20field%20on%201st%20January%202020..jpg)

Quantitative evaluation (to be completed):

| Model | RMSE (test set) |
|-------|-----------------|
| Persistence baseline (forecast = current state) | [TO FILL] |
| This GNN | [TO FILL] |

## Project Structure

| File | Description |
|------|-------------|
| `download_era5.py` | Downloads the ERA5 data (Jan to Mar 2020) from Copernicus |
| `explore-data.py` | Inspects the dataset structure (time steps, levels, grid) |
| `visualize-data.py` | Plots a 2D world map of the pressure field |
| `globe-viz.py` | Creates an interactive 3D globe of the pressure field |
| `build-graph.py` | Converts the Earth grid into a graph (65,160 nodes, 259,558 edges) |
| `prepare-data.py` | Normalizes the data and builds the 363 training pairs |
| `model.py` | Defines the 3-layer Graph Convolutional Network |
| `train.py` | Trains the GNN for 10 epochs and saves the weights |
| `predict.py` | Runs the trained model and compares prediction and reality on 3D globes |

## How to Run

1. Install the dependencies: `pip install torch torch_geometric xarray plotly cdsapi`
2. Create a free Copernicus account and put your personal access token in `~/.cdsapirc` (in your user folder, not in this repo).
3. Run in this order: `download_era5.py`, `prepare-data.py`, `build-graph.py`, `train.py`, `predict.py`

## Tech Stack

Python 3.11, PyTorch 2.11, PyTorch Geometric 2.7, xarray, Plotly, ERA5 (Copernicus CDS)

## Limitations and Next Steps

- Only 3 months of data and a single 6-hour forecast step
- Time-based train/test split and persistence baseline comparison
- Multi-step rollout (12 h, 24 h)
- Deeper architecture closer to GraphCast

## Inspiration

GraphCast (Lam et al., DeepMind, 2023), a GNN-based global weather forecasting model.

## Author

Elaa Hamdani, Engineering Student at INSAT, Instrumentation & Industrial Maintenance Engineering, specialized in AI & Aerodynamics
