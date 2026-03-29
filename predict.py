import torch
import numpy as np
import xarray as xr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from model import WeatherGNN

# Load data
X = torch.load('X.pt')
Y = torch.load('Y.pt')
mean = np.load('mean.npy')
std = np.load('std.npy')

# Load graph edges
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

# Load trained model
model = WeatherGNN(in_channels=8, hidden_channels=64, out_channels=8)
model.load_state_dict(torch.load('weather_gnn.pt'))
model.eval()

# Pick one snapshot
idx = 0
x_input = X[idx]
y_true = Y[idx]

with torch.no_grad():
    y_pred = model(x_input, edge_index)

# Unnormalize
y_pred_np = y_pred.numpy() * std[0, 0] + mean[0, 0]
y_true_np = y_true.numpy() * std[0, 0] + mean[0, 0]

pred_z = y_pred_np[:, 0].reshape(n_lat, n_lon)
true_z = y_true_np[:, 0].reshape(n_lat, n_lon)

lon_grid, lat_grid = np.meshgrid(lon, lat)

# Stars background
np.random.seed(42)
n_stars = 1000
star_lats = np.random.uniform(-90, 90, n_stars)
star_lons = np.random.uniform(0, 360, n_stars)
star_sizes = np.random.uniform(0.5, 2.5, n_stars)

fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "scattergeo"}, {"type": "scattergeo"}]],
    subplot_titles=("⚡ GNN Prediction", "🌍 Ground Truth"),
    horizontal_spacing=0.05
)

for col, (data, name) in enumerate([(pred_z, "Prediction"), (true_z, "Ground Truth")], 1):

    # Stars
    fig.add_trace(go.Scattergeo(
        lat=star_lats,
        lon=star_lons,
        mode='markers',
        marker=dict(
            size=star_sizes,
            color='white',
            opacity=0.6,
        ),
        showlegend=False,
        hoverinfo='skip',
    ), row=1, col=col)

    # Atmosphere
    fig.add_trace(go.Scattergeo(
        lat=lat_grid.flatten(),
        lon=lon_grid.flatten(),
        mode='markers',
        marker=dict(
            size=2.5,
            color=data.flatten(),
            colorscale=[
                [0.0,  '#0a0a2e'],
                [0.15, '#1a1a6e'],
                [0.3,  '#2255cc'],
                [0.45, '#44aaff'],
                [0.5,  '#ffffff'],
                [0.55, '#ffdd44'],
                [0.7,  '#ff6600'],
                [0.85, '#cc1100'],
                [1.0,  '#660000'],
            ],
            colorbar=dict(
                title=dict(text='Geopotential<br>(m²/s²)', font=dict(color='white', size=11)),
                tickfont=dict(color='white'),
                x=1.02 if col == 2 else -0.08,
                thickness=12,
            ),
            opacity=0.9,
        ),
        showlegend=False,
        hoverinfo='skip',
    ), row=1, col=col)

geo_settings = dict(
    showland=True,
    landcolor='rgb(15, 25, 15)',
    showocean=True,
    oceancolor='rgb(5, 10, 30)',
    showcoastlines=True,
    coastlinecolor='rgba(100, 200, 255, 0.6)',
    coastlinewidth=0.8,
    showlakes=True,
    lakecolor='rgb(5, 10, 30)',
    showrivers=False,
    showcountries=True,
    countrycolor='rgba(100, 200, 255, 0.2)',
    projection_type='orthographic',
    showframe=False,
    bgcolor='rgba(0,0,0,0)',
    projection_rotation=dict(lon=20, lat=20, roll=0),
)

fig.update_geos(geo_settings)

fig.update_layout(
    title=dict(
        text='🌌 Neural Weather Emulator — 6h Atmospheric Forecast',
        font=dict(size=22, color='white', family='Arial Black'),
        x=0.5,
        xanchor='center',
        y=0.97,
    ),
    paper_bgcolor='#000008',
    plot_bgcolor='#000008',
    font=dict(color='white'),
    margin=dict(t=80, b=20, l=20, r=20),
    annotations=[
        dict(
            text="GNN Prediction",
            x=0.22, y=1.04,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=15, color='#44aaff', family='Arial Black'),
        ),
        dict(
            text="Ground Truth",
            x=0.78, y=1.04,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=15, color='#44ff88', family='Arial Black'),
        ),
        dict(
            text="Inspired by DeepMind GraphCast · ERA5 Data · PyTorch Geometric · Author: Elaa HAMDANI",
            x=0.5, y=-0.02,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=10, color='rgba(255,255,255,0.4)'),
        ),
    ]
)

fig.write_html('prediction_vs_truth.html')
print("Saved! Open prediction_vs_truth.html in your browser.")