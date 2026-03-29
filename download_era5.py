import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'variable': ['geopotential', 'temperature', 
                     'u_component_of_wind', 'v_component_of_wind'],
        'pressure_level': ['500', '850'],
        'year': ['2020'],
        'month': ['01', '02', '03'],
        'day': [f'{d:02d}' for d in range(1, 32)],
        'time': ['00:00', '06:00', '12:00', '18:00'],
        'format': 'netcdf',
    },
    'era5_pressure_levels.nc'
)

print("Download complete!")
# This script downloads 3 months of ERA5 reanalysis data from the Copernicus Climate Data Store.
# It requests 4 atmospheric variables: geopotential (z), temperature (t), and wind components (u, v).
# Data is downloaded at 2 pressure levels: 500 hPa (5.5km altitude) and 850 hPa (1.5km altitude).
# Time range: January to March 2020, one snapshot every 6 hours = 364 time steps.
# The output file era5_pressure_levels.nc is 12GB and is the training dataset for our GNN model.
# NetCDF (.nc) is the standard file format for scientific climate and weather data.s