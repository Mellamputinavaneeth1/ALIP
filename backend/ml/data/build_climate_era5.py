"""
ALIP - ERA5-Land Climate Pipeline Stub

PIPELINE (requires downloaded data):
  ERA5-Land NetCDF (0.1 degree grid)
      -> Clip to India bbox [6N,68E,38N,98E]
      -> For each district polygon (india_districts.geojson):
           -> Spatial intersection with grid cells
           -> Area-weighted mean temperature
           -> Sum precipitation
      -> Aggregate to Kharif (Jun-Oct) / Rabi (Nov-Mar) seasonal stats
      -> Apply 10-year rolling window ending at Y-1 (leakage-safe)

DATA DOWNLOAD:
  1. Register: https://cds.climate.copernicus.eu/
  2. pip install cdsapi
  3. Create ~/.cdsapirc with UID and key
  4. python ml/data/downloaders/download_era5.py
  5. Place .nc files in ml/data/raw/era5/
  6. Then run: python ml/data/build_climate_era5.py

OUTPUT SCHEMA (same as climate_features.parquet):
  district_id, target_year, season,
  temp_mean, rainfall_total, rainfall_variability,
  preceding_monsoon_rainfall
  + climate_source = 'era5_land'
"""

ERA5_DATA_DIR = 'ml/data/raw/era5'
ERA5_OUT = 'ml/data/processed/climate_features_era5.parquet'

def build_climate_era5(verbose=True):
    import os
    if not os.path.isdir(ERA5_DATA_DIR) or not os.listdir(ERA5_DATA_DIR):
        raise FileNotFoundError(
            'ERA5-Land data not found. See module docstring for download instructions. '
            'Expected directory: ' + ERA5_DATA_DIR
        )
    raise NotImplementedError(
        'ERA5-Land pipeline not yet implemented. '
        'See ml/data/build_climate_era5.py for the pipeline design and download instructions.'
    )

if __name__ == '__main__':
    build_climate_era5()
