import cdsapi, os
c = cdsapi.Client()
os.makedirs('ml/data/raw/era5', exist_ok=True)
for year in range(2000, 2023):
    out = 'ml/data/raw/era5/era5_land_%d.nc' % year
    if os.path.exists(out):
        print('Already downloaded: %s' % out)
        continue
    print('Downloading year %d...' % year)
    c.retrieve(
        'reanalysis-era5-land-monthly-means',
        {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': ['2m_temperature', 'total_precipitation'],
            'year': str(year),
            'month': ['%02d' % m for m in range(1, 13)],
            'time': '00:00',
            'area': [38, 68, 6, 98],
            'format': 'netcdf',
        },
        out
    )
    print('Saved: %s' % out)
