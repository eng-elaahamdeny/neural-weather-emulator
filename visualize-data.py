import xarray as xr
import matplotlib.pyplot as plt

# Load the data
ds = xr.open_dataset('era5_pressure_levels.nc')

# Take one snapshot — first time step, pressure level 500 hPa
z = ds['z'].isel(valid_time=0, pressure_level=1)

# Plot it
plt.figure(figsize=(14, 7))
plt.contourf(ds.longitude, ds.latitude, z, levels=50, cmap='RdBu_r')
plt.colorbar(label='Geopotential (m²/s²)')
plt.title('Atmospheric Pressure Field — 1 Jan 2020, 500 hPa')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('pressure_map.png')
plt.show()
print("Map saved as pressure_map.png")
# This script plots a flat 2D world map of the atmospheric pressure field (geopotential).
# It takes one snapshot: the first time step (1 Jan 2020) at 500 hPa pressure level.
# The colormap RdBu_r is used: red = high pressure (calm), blue = low pressure (storms).
# The output is saved as pressure_map.png in the project folder.
# This was our first look at the data before building the interactive 3D globe version.