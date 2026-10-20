"""
ALIP - IMD Gridded Rainfall Climate Pipeline Stub

PIPELINE (requires downloaded data):
  IMD source: https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html
  IMD provides rainfall only (0.25 degree grid).
  For temperature, combine with ERA5 or Open-Meteo.
  Output: climate_features_imd.parquet, climate_source='imd_gridded'
"""

IMD_DATA_DIR = 'ml/data/raw/imd'
IMD_OUT = 'ml/data/processed/climate_features_imd.parquet'

def build_climate_imd(verbose=True):
    import os
    if not os.path.isdir(IMD_DATA_DIR) or not os.listdir(IMD_DATA_DIR):
        raise FileNotFoundError(
            'IMD data not found. See module docstring for download instructions. '
            'Expected directory: ' + IMD_DATA_DIR
        )
    raise NotImplementedError(
        'IMD pipeline not yet implemented. '
        'See ml/data/build_climate_imd.py for the pipeline design and download instructions.'
    )

if __name__ == '__main__':
    build_climate_imd()
