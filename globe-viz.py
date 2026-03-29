import xarray as xr
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# Load data
ds = xr.open_dataset('era5_pressure_levels.nc')
z = ds['z'].isel(valid_time=0, pressure_level=1).values
lats = ds.latitude.values[::4]
lons = ds.longitude.values[::4]
z_small = z[::4, ::4]

lon_grid, lat_grid = np.meshgrid(lons, lats)

# Stars
np.random.seed(42)
n_stars = 1500
star_lats = np.random.uniform(-90, 90, n_stars)
star_lons = np.random.uniform(0, 360, n_stars)
star_sizes = np.random.uniform(0.5, 2.5, n_stars)

fig = go.Figure()

# Stars
fig.add_trace(go.Scattergeo(
    lat=star_lats,
    lon=star_lons,
    mode='markers',
    marker=dict(size=star_sizes, color='white', opacity=0.5),
    showlegend=False,
    hoverinfo='skip',
))

# Atmosphere
fig.add_trace(go.Scattergeo(
    lat=lat_grid.flatten(),
    lon=lon_grid.flatten(),
    mode='markers',
    marker=dict(
        size=2.5,
        color=z_small.flatten(),
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
            thickness=12,
        ),
        opacity=0.9,
    ),
    showlegend=False,
    hoverinfo='skip',
))

fig.update_layout(
    title=dict(
        text='🌌 Atmospheric Pressure Field — 1 Jan 2020, 500 hPa',
        font=dict(size=20, color='white', family='Arial Black'),
        x=0.5, xanchor='center', y=0.97,
    ),
    geo=dict(
        showland=True,
        landcolor='rgb(15, 25, 15)',
        showocean=True,
        oceancolor='rgb(5, 10, 30)',
        showcoastlines=True,
        coastlinecolor='rgba(100, 200, 255, 0.6)',
        coastlinewidth=0.8,
        showcountries=True,
        countrycolor='rgba(100, 200, 255, 0.2)',
        projection_type='orthographic',
        showframe=False,
        bgcolor='rgba(0,0,0,0)',
        projection_rotation=dict(lon=20, lat=20, roll=0),
    ),
    paper_bgcolor='#000008',
    font=dict(color='white'),
    margin=dict(t=80, b=20, l=20, r=20),
    annotations=[
        dict(
            text="Inspired by DeepMind GraphCast · ERA5 Data · PyTorch Geometric · Author: Elaa HAMDANI",
            x=0.5, y=-0.02,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=10, color='rgba(255,255,255,0.4)'),
        ),
    ]
)

fig.write_html('globe.html')
print("Saved! Open globe.html in your browser.")

# This script visualizes the ERA5 atmospheric pressure field (geopotential) on a 3D interactive globe.
# The data is from 1st January 2020 at 500 hPa (roughly 5.5km altitude).
# Blue = high pressure (calm, stable air, clear skies) — Red = low pressure (cold, unstable air, storms).
# The globe is interactive — you can rotate it by clicking and dragging in the browser.
# The data is downsampled (every 4th point) to run faster — the full resolution is 721x1440 points.
# This is the target variable our Graph Neural Network will learn to predict:
# given the pressure field at time T, predict what it looks like at T+6 hours.