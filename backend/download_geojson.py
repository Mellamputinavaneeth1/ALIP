import urllib.request
import os

urls = [
    "https://raw.githubusercontent.com/datameet/maps/master/Districts/districts.geojson",
    "https://raw.githubusercontent.com/HindustanTimesLabs/shapefiles/master/india/district/district.geojson"
]
out_path = 'ml/data/raw/india_districts.geojson'
os.makedirs(os.path.dirname(out_path), exist_ok=True)

success = False
for url in urls:
    try:
        print(f"Trying {url}...")
        urllib.request.urlretrieve(url, out_path)
        print("Success!")
        success = True
        break
    except Exception as e:
        print(f"Failed: {e}")

if not success:
    print("Warning: Both URLs failed. Creating stub file. Download india_districts.geojson manually.")
    with open(out_path, 'w') as f:
        f.write('{"type":"FeatureCollection","features":[]}')
