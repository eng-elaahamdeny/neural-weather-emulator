# Neural Weather Emulator — Atmospheric Pressure Prediction with GNNs

I built a Neural Weather Emulator — an AI model that predicts how the Earth's atmosphere evolves over time, inspired by DeepMind's GraphCast and trained on real scientific data from the European Centre for Medium-Range Weather Forecasts.

---

## What It Does

My project is a Neural Weather Emulator — an AI model that predicts atmospheric pressure fields using Graph Neural Networks, inspired by DeepMind's GraphCast, one of the most significant breakthroughs in AI-driven weather forecasting.

The idea is simple but powerful. Every 6 hours, the state of the entire Earth's atmosphere can be described by four variables at every location on the globe: geopotential (pressure), temperature, and wind in two directions. My model looks at one of these snapshots and predicts what the atmosphere will look like 6 hours later.

What makes this project different from a standard deep learning approach is how I represent the atmosphere. Instead of treating it as a flat image and using a CNN, I treat the Earth as a graph — every location is a node, and every node is connected to its geographic neighbors by edges. This is physically meaningful because weather doesn't happen in isolation — a storm system in one location directly influences the surrounding areas. The Graph Neural Network learns to propagate this influence through the edges, exactly like real atmospheric dynamics work.

I trained the model on ERA5 reanalysis data from the Copernicus Climate Data Store — the same dataset DeepMind used for GraphCast. This is 3 months of real atmospheric data from 2020, one snapshot every 6 hours, covering the entire Earth at 0.25 degree resolution. That's over 65,000 locations per snapshot, 363 training pairs, and 12GB of real scientific data.

The results are visualized on an interactive 3D globe where you can rotate the Earth and compare the model's predicted pressure field against the ground truth.

---

## Project Structure

| File | Description |
|------|-------------|
| `test_env.py` | Verifies that PyTorch, PyTorch Geometric and xarray are installed correctly |
| `download_era5.py` | Downloads 12GB of real ERA5 atmospheric data (Jan–Mar 2020) from Copernicus |
| `explore_data.py` | Inspects the dataset structure — 364 time steps, 2 pressure levels, 721×1440 grid |
| `visualize_data.py` | Plots a 2D world map of the atmospheric pressure field |
| `globe_viz.py` | Creates an interactive 3D globe of the pressure field, rotatable in the browser |
| `build_graph.py` | Converts the Earth grid into a graph — 65,160 nodes and 259,558 edges |
| `model.py` | Defines the 3-layer Graph Convolutional Network architecture |
| `prepare_data.py` | Normalizes the data and creates 363 input/output training pairs |
| `train.py` | Trains the GNN for 10 epochs and saves the model as `weather_gnn.pt` |
| `predict.py` | Runs the trained model and visualizes prediction vs reality on dual 3D globes |
## Tech Stack

- Python 3.11
- PyTorch 2.11
- PyTorch Geometric 2.7
- xarray
- ERA5 data from Copernicus Climate Data Store
- Plotly

---

## Inspiration

Inspired by DeepMind's GraphCast (2023) — the first AI model to outperform traditional numerical weather prediction at global scale.

---

## About

This project demonstrates how Graph Neural Networks can be applied to real-world physical systems, and represents my first step into the intersection of aerospace engineering and artificial intelligence.

## Author
**Elaa Hamdani**  
Engineering Student at INSAT – Instrumentation & Industrial Maintenance Engineering  
Specialized in AI & Aerodynamics