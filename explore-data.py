import xarray as xr

ds = xr.open_dataset('era5_pressure_levels.nc')
print(ds)
# This script loads the ERA5 dataset and prints its structure using xarray.
# It shows the dimensions: 364 time steps, 2 pressure levels, 721 latitudes, 1440 longitudes.
# The 4 data variables are: z (geopotential), t (temperature), u (east-west wind), v (north-south wind).
# Total dataset size is 12GB — real scientific data from the European Centre for Medium-Range Weather Forecasts.
# Run this script any time you want to inspect the shape or contents of the dataset.