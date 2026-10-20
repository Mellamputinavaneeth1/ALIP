import json

path = 'ml/data/raw/climate_cache.json'
with open(path, 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()

# Find last occurrence of '}, "'
pattern = '}, "'
idx = text.rfind(pattern)
if idx != -1:
    repaired = text[:idx+1] + '}'
    try:
        data = json.loads(repaired)
        print('Repaired successfully! Total districts:', len(data))
        with open(path, 'w', encoding='utf-8') as f_out:
            json.dump(data, f_out)
        print('Saved clean cache!')
    except Exception as e:
        print('Failed to parse repaired json:', e)
else:
    print('Pattern not found')
