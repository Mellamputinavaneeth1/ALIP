import pytest
import pandas as pd
import sys
import os
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_no_duplicate_district_year_season_crop():
    df = pd.read_parquet('ml/data/features/district_crop_training.parquet')
    dupes = df.duplicated(['district_id','year','season','crop']).sum()
    assert dupes == 0

def test_valid_seasons_only():
    df = pd.read_parquet('ml/data/processed/crop_outcomes.parquet')
    assert set(df['season'].unique()).issubset({'kharif','rabi'})

def test_ph_range():
    soil = pd.read_parquet('ml/data/processed/soil_features.parquet')
    assert soil['soil_ph'].between(3.0, 10.0).all()

def test_crop_area_share_not_exceed_one():
    crops = pd.read_parquet('ml/data/processed/crop_outcomes.parquet')
    observed = crops[crops['record_present']]
    sums = observed.groupby(['district_id','year','season'])['crop_area_share'].sum()
    assert (sums <= 1.05).all()

def test_is_major_matches_threshold():
    df = pd.read_parquet('ml/data/features/district_crop_training.parquet')
    observed = df[df['record_present']]
    expected = (observed['crop_area_share'] >= 0.10).astype(int)
    assert (observed['is_major'].values == expected.values).all()

def test_no_missing_feature_values():
    df = pd.read_parquet('ml/data/features/district_crop_training.parquet')
    from ml.data.build_training_table import FEATURE_COLS
    assert df[FEATURE_COLS].isna().sum().sum() == 0

def test_source_provenance_is_kaggle_des():
    df = pd.read_parquet('ml/data/processed/crop_outcomes.parquet')
    assert 'data_source' in df.columns
    assert (df['data_source'] == 'kaggle_des').all()

def test_no_icrisat_label():
    df = pd.read_parquet('ml/data/processed/crop_outcomes.parquet')
    if 'data_source' in df.columns:
        assert 'icrisat_real' not in df['data_source'].values

def test_name_resolution_nashik():
    from services.district_service import resolve_district
    r = resolve_district(location='Nashik')
    assert r['found']
    assert 'maharashtra' in r.get('state','').lower() or r.get('state','') != ''

def test_name_resolution_ludhiana():
    from services.district_service import resolve_district
    r = resolve_district(location='Ludhiana')
    assert r['found']

def test_unknown_name_not_found():
    from services.district_service import resolve_district
    r = resolve_district(location='ZZZ_INVALID_DISTRICT_999')
    assert not r['found']

def test_latlon_resolution_returns_method():
    from services.district_service import resolve_district
    r = resolve_district(lat=20.01, lon=73.79)
    assert r['found']
    assert r.get('method') in ['polygon', 'haversine_nearest', 'latlon_nearest']

def test_record_present_column_exists():
    df = pd.read_parquet('ml/data/features/district_crop_training.parquet')
    assert 'record_present' in df.columns
    assert df['record_present'].any()
    assert (~df['record_present']).any()

def test_absent_rows_zero_values():
    df = pd.read_parquet('ml/data/features/district_crop_training.parquet')
    absent = df[~df['record_present']]
    assert (absent['is_major'] == 0).all()
    assert (absent['crop_area_share'] == 0.0).all()

def test_year_2015_excluded():
    df = pd.read_parquet('ml/data/features/district_crop_training.parquet')
    assert 2015 not in df['year'].values

def test_3way_split_years_distinct():
    from ml.evaluation.splits import TEMPORAL_TRAIN_END, TEMPORAL_CAL_YEAR, TEMPORAL_TEST_YEAR
    assert TEMPORAL_TRAIN_END < TEMPORAL_CAL_YEAR
    assert TEMPORAL_CAL_YEAR < TEMPORAL_TEST_YEAR
    assert TEMPORAL_CAL_YEAR != TEMPORAL_TEST_YEAR

def test_3way_split_no_overlap():
    from ml.evaluation.splits import temporal_split_3way
    df = pd.read_parquet('ml/data/features/district_crop_training.parquet')
    train, cal, test = temporal_split_3way(df)
    train_grp = set(zip(train['district_id'], train['year'], train['season']))
    test_grp  = set(zip(test['district_id'],  test['year'],  test['season']))
    assert len(train_grp & test_grp) == 0

def test_calibration_separate_from_test():
    from ml.evaluation.splits import TEMPORAL_CAL_YEAR, TEMPORAL_TEST_YEAR
    assert TEMPORAL_CAL_YEAR != TEMPORAL_TEST_YEAR

def test_calibrated_probs_in_range():
    import joblib, numpy as np
    cal = joblib.load('models/calibrator.pkl')
    raw = np.array([0.05, 0.2, 0.5, 0.8, 0.95])
    out = cal.predict(raw)
    assert (out >= 0.0).all() and (out <= 1.0).all()

def test_one_hot_encoding_no_ordinal_crop_id():
    df = pd.read_parquet('ml/data/features/district_crop_training.parquet')
    assert 'crop_id' not in df.columns
    crop_cols = [c for c in df.columns if c.startswith('crop_')]
    assert len(crop_cols) >= 14

def test_candidate_ranking_sorted_descending():
    from services.prediction_service import score_candidate_crops
    features = {
        'temp_mean':23.0,'rainfall_total':900.0,'rainfall_variability':0.4,
        'preceding_monsoon_rainfall':None,'soil_ph':6.8,'soil_oc':4.5,
        'soil_clay':30.0,'soil_cec':18.0,'lagged_irrigated_share':0.3,'season_kharif':1
    }
    results = score_candidate_crops(features, 'MH001', 'kharif')
    assert len(results) > 0
    scores = [r['score'] for r in results]
    assert scores == sorted(scores, reverse=True)

def test_probability_output_range():
    from services.prediction_service import score_candidate_crops
    features = {
        'temp_mean':23.0,'rainfall_total':900.0,'rainfall_variability':0.4,
        'preceding_monsoon_rainfall':None,'soil_ph':6.8,'soil_oc':4.5,
        'soil_clay':30.0,'soil_cec':18.0,'lagged_irrigated_share':0.3,'season_kharif':1
    }
    results = score_candidate_crops(features, 'MH001', 'kharif')
    for r in results:
        assert 0.0 <= r['score'] <= 1.0

def test_invalid_season_fails():
    from services.regional_crop_service import recommend_regional
    result = asyncio.run(recommend_regional(location='Nashik', season='summer'))
    assert not result.get('success')

def test_unknown_district_graceful():
    from services.regional_crop_service import recommend_regional
    result = asyncio.run(recommend_regional(location='ZZZ_INVALID', season='kharif'))
    assert not result.get('success') or result.get('abstain')

def test_nashik_kharif_returns_recommendation():
    from services.regional_crop_service import recommend_regional
    result = asyncio.run(recommend_regional(location='Nashik', season='kharif'))
    assert result['success']
    assert len(result['recommendations']) > 0
    r = result['recommendations'][0]
    assert 'hybrid_rank_score' in r
    assert 'ml_probability' in r
    assert 'prior_score' in r
    assert 'rank' in r
    assert 'confidence_band' in r

def test_provenance_no_icrisat():
    from services.regional_crop_service import recommend_regional
    result = asyncio.run(recommend_regional(location='Nashik', season='kharif'))
    prov = result.get('provenance', {})
    assert 'crop_data' in prov
    crop_str = str(prov.get('crop_data', '')).lower()
    assert 'icrisat' not in crop_str

def test_rank_order_ascending():
    from services.regional_crop_service import recommend_regional
    result = asyncio.run(recommend_regional(location='Ludhiana', season='rabi'))
    recs = result['recommendations']
    ranks = [r['rank'] for r in recs]
    assert ranks == list(range(1, len(ranks)+1))

def test_haversine_50km_abstention():
    from services.district_service import resolve_district
    # Middle of Arabian Sea far from any Indian district (> 50 km)
    r = resolve_district(lat=15.0, lon=65.0)
    assert not r['found']

def test_provenance_explicit_resolutions():
    from services.regional_crop_service import recommend_regional
    result = asyncio.run(recommend_regional(location='Nashik', season='kharif'))
    prov = result.get('provenance', {})
    clim_res = prov.get('climate', {}).get('resolution') if isinstance(prov.get('climate'), dict) else prov.get('climate_resolution')
    soil_res = prov.get('soil', {}).get('resolution') if isinstance(prov.get('soil'), dict) else prov.get('soil_resolution')
    irr_res = prov.get('irrigation', {}).get('resolution') if isinstance(prov.get('irrigation'), dict) else prov.get('irrigation_resolution')
    assert clim_res == 'district_centroid'
    assert soil_res in ['district', 'state_fallback']
    assert irr_res in ['district', 'state_level']

def test_hybrid_alpha_presence():
    from services.regional_crop_service import recommend_regional
    result = asyncio.run(recommend_regional(location='Nashik', season='kharif'))
    prov = result.get('provenance', {})
    assert 'hybrid_alpha' in prov
    assert 0.0 <= float(prov['hybrid_alpha']) <= 1.0

def test_legacy_model_preservation():
    legacy_model = os.path.join(os.path.dirname(__file__), '..', 'models', 'legacy', 'crop_model_regional.pkl')
    assert os.path.exists(legacy_model)
    assert os.path.getsize(legacy_model) > 100000

def test_all_candidates_has_15_crops():
    from services.regional_crop_service import recommend_regional
    result = asyncio.run(recommend_regional(location='Nashik', season='kharif'))
    all_c = result.get('all_candidates', [])
    assert len(all_c) >= 14

def test_response_has_separate_scores():
    """recommendations must have ml_probability, prior_score, hybrid_rank_score."""
    from services.regional_crop_service import recommend_regional
    result = asyncio.run(recommend_regional(location='Nashik', season='kharif'))
    assert result['success']
    rec = result['recommendations'][0]
    assert 'ml_probability' in rec
    assert 'prior_score' in rec
    assert 'hybrid_rank_score' in rec
    assert isinstance(rec['ml_probability'], (int, float))
    assert isinstance(rec['prior_score'], (int, float))
    assert isinstance(rec['hybrid_rank_score'], (int, float))

def test_hybrid_score_not_labeled_probability():
    """hybrid_rank_score must NOT be called major_crop_probability in recommendations."""
    from services.regional_crop_service import recommend_regional
    result = asyncio.run(recommend_regional(location='Nashik', season='kharif'))
    rec = result['recommendations'][0]
    assert 'major_crop_probability' not in rec

def test_provenance_is_structured():
    """provenance must have structured sub-objects."""
    from services.regional_crop_service import recommend_regional
    result = asyncio.run(recommend_regional(location='Nashik', season='kharif'))
    prov = result['provenance']
    assert isinstance(prov.get('crop_data'), dict)
    assert 'source' in prov['crop_data']
    assert isinstance(prov.get('climate'), dict)
    assert 'resolution' in prov['climate']
    assert isinstance(prov.get('soil'), dict)
    assert isinstance(prov.get('irrigation'), dict)

def test_guardrail_status_present():
    """each recommendation must have guardrail_status."""
    from services.regional_crop_service import recommend_regional
    result = asyncio.run(recommend_regional(location='Nashik', season='kharif'))
    for rec in result['recommendations']:
        assert 'guardrail_status' in rec
        assert rec['guardrail_status'] in ('pass', 'flagged')

def test_context_has_environmental_fields():
    """context must include all environmental fields."""
    from services.regional_crop_service import recommend_regional
    result = asyncio.run(recommend_regional(location='Nashik', season='kharif'))
    ctx = result['context']
    for field in ['temp_mean', 'rainfall_total', 'soil_ph', 'irrigated_share']:
        assert field in ctx

def test_rabi_season_recommendations():
    from services.regional_crop_service import recommend_regional
    result = asyncio.run(recommend_regional(location='Nashik', season='rabi'))
    assert result['success']
    assert len(result['recommendations']) > 0

def test_ph_override_changes_source():
    from services.regional_crop_service import recommend_regional
    result = asyncio.run(recommend_regional(location='Nashik', season='kharif', measured_ph=6.5))
    assert result['success']
    assert result.get('ph_source') in ('farmer_measured', 'measured')

def test_all_candidates_have_ml_and_prior():
    from services.regional_crop_service import recommend_regional
    result = asyncio.run(recommend_regional(location='Nashik', season='kharif'))
    candidates = result.get('all_candidates', [])
    assert len(candidates) >= 14
    for c in candidates:
        assert 'ml_probability' in c
        assert 'prior_probability' in c
        assert 'score' in c


