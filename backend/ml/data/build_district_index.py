"""
Builds a geopandas spatial index from Indian district boundaries.
Run once after downloading india_districts.geojson.
Output: ml/data/raw/district_spatial_index.pkl

Usage: python ml/data/build_district_index.py
"""

import os, sys, pickle
sys.path.insert(0, '.')

GEOJSON_PATH = os.path.join('ml', 'data', 'raw', 'india_districts.geojson')
INDEX_PATH = os.path.join('ml', 'data', 'raw', 'district_spatial_index.pkl')

def build_spatial_index(verbose=True):
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError('Run: pip install geopandas shapely')
    
    if not os.path.exists(GEOJSON_PATH):
        raise FileNotFoundError('Download india_districts.geojson first. See build_district_index.py')
    
    gdf = gpd.read_file(GEOJSON_PATH)
    if len(gdf) == 0:
        raise ValueError('GeoJSON is empty or is a stub. Download the real file.')
    
    # Normalize column names - handle different GeoJSON schemas
    # Common column names in Indian district shapefiles:
    name_col  = next((c for c in gdf.columns if c.upper() in ['DISTRICT','DISTRICT_N','DTNAME','NAME']), gdf.columns[0])
    state_col = next((c for c in gdf.columns if c.upper() in ['ST_NM','STATE','STATENAME','ST_NAME']), None)
    
    gdf['district_norm'] = gdf[name_col].str.upper().str.strip()
    if state_col:
        gdf['state_norm'] = gdf[state_col].str.upper().str.strip()
    else:
        gdf['state_norm'] = ''
    gdf['district_col'] = name_col
    gdf['state_col'] = state_col or ''
    
    with open(INDEX_PATH, 'wb') as f:
        pickle.dump(gdf, f)
    
    if verbose:
        print('Spatial index saved: %d districts' % len(gdf))
        print('Index path: %s' % INDEX_PATH)
    return gdf

if __name__ == '__main__':
    build_spatial_index()
