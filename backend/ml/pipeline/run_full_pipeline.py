"""
ALIP - Full Pipeline Runner (v2.1 Regional Hybrid Engine)
Runs the complete data -> model -> calibration -> hybrid ranking -> serving lookup pipeline.

Features:
- Leakage-safe 10-year rolling climate window
- Spatial provenance tracking (soil/irrigation/climate resolution)
- Temporal 3-way split: Train (<=2012), Calibrate (2013), Test (2014)
- Leakage-safe Isotonic Calibration fitted on 2013 calibration set
- Validation-tuned Hybrid Ranker: P_hybrid = alpha * P_ML + (1 - alpha) * P_prior
- Multi-year rolling-origin temporal evaluation (2013, 2014)
- Safe artifact persistence in backend/models/regional_v2.1/
"""

import os
import sys
import json
import argparse
import time
import shutil
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FEATURES_DIR = os.path.join(BASE_DIR, 'ml', 'data', 'features')
MODELS_DIR   = os.path.join(BASE_DIR, 'ml', 'models')
LOOKUP_DIR   = os.path.join(BASE_DIR, 'data', 'lookup')
V21_DIR      = os.path.join(BASE_DIR, 'models', 'regional_v2.1')
LEGACY_DIR   = os.path.join(BASE_DIR, 'models', 'legacy')
BASE_MODELS  = os.path.join(BASE_DIR, 'models')

os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,   exist_ok=True)
os.makedirs(LOOKUP_DIR,   exist_ok=True)
os.makedirs(V21_DIR,      exist_ok=True)
os.makedirs(LEGACY_DIR,   exist_ok=True)


def run_data_pipeline(skip_fetch: bool = False):
    print("\n" + "="*60)
    print("PHASE 1: DATA PIPELINE (v2.1)")
    print("="*60)

    # 1a. Crop outcomes
    print("\n[1/5] Building crop outcomes dataset...")
    from ml.data.build_crop_dataset import build_crop_outcomes
    build_crop_outcomes(verbose=True)

    # 1b. Irrigation
    print("\n[2/5] Building irrigation features...")
    from ml.data.build_irrigation_features import build_irrigation_features
    build_irrigation_features(verbose=True)

    if not skip_fetch:
        # 1c. Soil
        print("\n[3/5] Fetching soil features...")
        from ml.data.build_soil_features import build_soil_features
        build_soil_features(verbose=True)

        # 1d. Climate
        print("\n[4/5] Building climate features (10-year trailing window)...")
        from ml.data.build_climate_features import build_climate_features
        build_climate_features(verbose=True)
    else:
        print("\n[3/5] Skipping API fetches (--skip-fetch). Using cached soil.")
        print("[4/5] Skipping API fetches (--skip-fetch). Using cached climate.")
        from ml.data.build_climate_features import build_climate_features
        build_climate_features(verbose=True)

    # 1e. Join training table
    print("\n[5/5] Building training table...")
    from ml.data.build_training_table import build_training_table
    df = build_training_table(verbose=True)
    print(f"  Training table ready: {df.shape}")
    return df


def run_ml_pipeline(df=None):
    print("\n" + "="*60)
    print("PHASE 2: MODEL TRAINING, CALIBRATION & HYBRID RANKING")
    print("="*60)

    if df is None:
        path = os.path.join(FEATURES_DIR, 'district_crop_training.parquet')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Run data pipeline first: {path}")
        df = pd.read_parquet(path)

    from ml.evaluation.splits import temporal_split_3way, rolling_origin_temporal_splits
    from ml.evaluation.ranking_metrics import (
        compute_ranking_metrics, compute_baselines,
        compute_district_season_prior, compute_hybrid_evaluation, optimize_hybrid_alpha
    )
    from ml.evaluation.calibration import calibrate_model
    from ml.data.build_training_table import FEATURE_COLS, LABEL_COL

    train_df, cal_df, test_df = temporal_split_3way(df)

    # 1. Baselines on Test Set (2014)
    print("\n[1/6] Running historical baselines on test set (2014)...")
    baseline_results = compute_baselines(train_df, test_df)

    # Compute empirical prior table from train_df
    prior_df = compute_district_season_prior(train_df)

    # 2. Train ML models on train_df (<= 2012)
    print("\n[2/6] Training ML algorithms (<= 2012)...")
    from ml.models.train_rf import train_random_forest
    rf_model, rf_meta = train_random_forest(verbose=False)

    from ml.models.train_xgb import train_xgb
    xgb_model, xgb_meta = train_xgb(verbose=False)

    from ml.models.train_lgbm import train_lgbm
    lgbm_model, lgbm_meta = train_lgbm(verbose=False)

    def get_predictions(model, data_df):
        if model is None: return np.zeros(len(data_df))
        X = data_df[FEATURE_COLS].fillna(0)
        return model.predict_proba(X)[:, 1]

    # Evaluate ML models raw
    raw_models = {}
    for name, model, meta in [("RandomForest", rf_model, rf_meta),
                              ("XGBoost", xgb_model, xgb_meta),
                              ("LightGBM", lgbm_model, lgbm_meta)]:
        if model is not None:
            df_eval = test_df.copy()
            df_eval['prob'] = get_predictions(model, test_df)
            metrics = compute_ranking_metrics(df_eval)
            raw_models[name] = (model, meta, metrics)

    best_ml_name = max(raw_models, key=lambda k: raw_models[k][2]['ndcg_at_3'])
    best_ml_model, best_ml_meta, best_ml_metrics = raw_models[best_ml_name]
    print(f"  Best raw ML model: {best_ml_name} (NDCG@3 = {best_ml_metrics['ndcg_at_3']:.4f})")

    # 3. Fit Calibrator on 2013 Calibration Data (Leakage-safe)
    print("\n[3/6] Fitting Isotonic Calibrator on 2013 validation partition...")
    val_X = cal_df[FEATURE_COLS].fillna(0)
    val_y = cal_df[LABEL_COL].values
    calibrator = calibrate_model(best_ml_model, val_X, val_y)

    val_prob_raw = best_ml_model.predict_proba(val_X)[:, 1]
    val_prob_cal = np.clip(calibrator.predict(val_prob_raw), 0.0, 1.0)

    # 4. Tune Hybrid Alpha strictly on 2013 Validation Data
    print("\n[4/6] Optimizing Hybrid Alpha (alpha * ML + (1-alpha) * Prior) on 2013 validation set...")
    cal_prior_merge = cal_df.merge(prior_df, on=['district_id', 'season', 'crop'], how='left')
    val_prior_probs = cal_prior_merge['prior_prob'].fillna(0.0).values

    best_alpha, best_val_hybrid_metrics, all_alpha_results = optimize_hybrid_alpha(
        cal_df, val_prob_cal, val_prior_probs, cal_df[LABEL_COL]
    )
    print(f"  Optimal alpha tuned on validation set: alpha = {best_alpha:.2f}")
    print(f"  Validation NDCG@3 at alpha={best_alpha:.2f}: {best_val_hybrid_metrics['ndcg_at_3']:.4f}")

    # 5. Final Evaluation on 2014 Holdout Test Set
    print("\n[5/6] Final Holdout Evaluation (2014 Test Set)...")
    test_prob_raw = best_ml_model.predict_proba(test_df[FEATURE_COLS].fillna(0))[:, 1]
    test_prob_cal = np.clip(calibrator.predict(test_prob_raw), 0.0, 1.0)

    test_prior_merge = test_df.merge(prior_df, on=['district_id', 'season', 'crop'], how='left')
    test_prior_probs = test_prior_merge['prior_prob'].fillna(0.0).values

    # Calibrated ML metrics
    df_cal_eval = test_df.copy()
    df_cal_eval['prob'] = test_prob_cal
    cal_ml_metrics = compute_ranking_metrics(df_cal_eval)

    # Hybrid Model metrics using optimal alpha
    hybrid_test_metrics = compute_hybrid_evaluation(
        test_prob_cal, test_prior_probs, test_df[LABEL_COL], test_df, alpha=best_alpha
    )

    print("\n" + "-"*75)
    print(f"{'Model / Approach':<30} {'NDCG@3':<10} {'Recall@1':<10} {'Recall@3':<10} {'Brier':<10}")
    print("-"*75)
    for name, res in baseline_results.items():
        print(f"{name:<30} {res['ndcg_at_3']:<10.4f} {res['recall_at_1']:<10.4f} {res['recall_at_3']:<10.4f} {res['brier_score']:<10.4f}")
    for name, (m, meta, res) in raw_models.items():
        print(f"Raw {name:<26} {res['ndcg_at_3']:<10.4f} {res['recall_at_1']:<10.4f} {res['recall_at_3']:<10.4f} {res['brier_score']:<10.4f}")
    print(f"Calibrated {best_ml_name:<19} {cal_ml_metrics['ndcg_at_3']:<10.4f} {cal_ml_metrics['recall_at_1']:<10.4f} {cal_ml_metrics['recall_at_3']:<10.4f} {cal_ml_metrics['brier_score']:<10.4f}")
    print(f"Hybrid ({best_ml_name}+Prior, a={best_alpha:.2f}) {hybrid_test_metrics['ndcg_at_3']:<10.4f} {hybrid_test_metrics['recall_at_1']:<10.4f} {hybrid_test_metrics['recall_at_3']:<10.4f} {hybrid_test_metrics['brier_score']:<10.4f}")
    print("-"*75)

    # 6. Multi-year rolling-origin evaluation
    print("\n[6/6] Running Multi-Year Rolling-Origin Evaluation (2013, 2014)...")
    rolling_results = []
    for train_f, cal_f, test_f, t_year in rolling_origin_temporal_splits(df):
        f_prior = compute_district_season_prior(train_f)
        f_prior_test = test_f.merge(f_prior, on=['district_id', 'season', 'crop'], how='left')
        f_prior_test['prob'] = f_prior_test['prior_prob'].fillna(0.0)
        f_prior_test_probs = f_prior_test['prob'].values
        prior_res = compute_ranking_metrics(f_prior_test)

        # Train model for fold
        f_X_train = train_f[FEATURE_COLS].fillna(0)
        f_y_train = train_f[LABEL_COL].values
        from lightgbm import LGBMClassifier
        fold_model = LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, verbose=-1)
        fold_model.fit(f_X_train, f_y_train)

        # Calibrate on cal_f
        f_cal_X = cal_f[FEATURE_COLS].fillna(0)
        f_cal_y = cal_f[LABEL_COL].values
        fold_calibrator = calibrate_model(fold_model, f_cal_X, f_cal_y)

        # Optimize alpha on cal_f
        f_cal_prior = cal_f.merge(f_prior, on=['district_id', 'season', 'crop'], how='left')
        f_cal_prior_probs = f_cal_prior['prior_prob'].fillna(0.0).values
        f_cal_raw = fold_model.predict_proba(f_cal_X)[:, 1]
        f_cal_probs = np.clip(fold_calibrator.predict(f_cal_raw), 0.0, 1.0)
        fold_alpha, _, _ = optimize_hybrid_alpha(cal_f, f_cal_probs, f_cal_prior_probs, f_cal_y)

        # Test on test_f
        f_test_X = test_f[FEATURE_COLS].fillna(0)
        f_test_raw = fold_model.predict_proba(f_test_X)[:, 1]
        f_test_probs = np.clip(fold_calibrator.predict(f_test_raw), 0.0, 1.0)
        ml_res = compute_ranking_metrics(test_f.assign(prob=f_test_probs))
        hyb_res = compute_hybrid_evaluation(f_test_probs, f_prior_test_probs, test_f[LABEL_COL], test_f, alpha=fold_alpha)

        rolling_results.append({
            'test_year': t_year,
            'fold_alpha': fold_alpha,
            'prior_ndcg3': round(float(prior_res['ndcg_at_3']), 4),
            'ml_ndcg3': round(float(ml_res['ndcg_at_3']), 4),
            'hybrid_ndcg3': round(float(hyb_res['ndcg_at_3']), 4),
            'hybrid_rec1': round(float(hyb_res['recall_at_1']), 4),
            'hybrid_rec3': round(float(hyb_res['recall_at_3']), 4),
        })
        print(f"  Year {t_year}: Prior NDCG@3={prior_res['ndcg_at_3']:.4f}, ML={ml_res['ndcg_at_3']:.4f}, Hybrid(a={fold_alpha:.2f})={hyb_res['ndcg_at_3']:.4f}")

    # Ablation study
    print("\n--- ABLATION STUDY ---")
    from ml.evaluation.ablation import run_ablation
    ablation_results = run_ablation(df, verbose=False)

    # Spatial validation (leave-state-out)
    print("\n--- SPATIAL VALIDATION (Leave-State-Out) ---")
    spatial_results = _run_spatial_validation(df, FEATURE_COLS, LABEL_COL)

    from ml.data.district_centroids import ALL_CROPS
    best_ml_meta.update({
        "model_version": "v2.1-regional-hybrid",
        "dataset_version": "v2.1-kaggle-des",
        "feature_schema_version": "25-dim-v2.1",
        "crop_schema_version": "15-annual-field-crops",
        "boundary_version": "kaggle_des_district_names",
        "climate_version": "open_meteo_centroid_10yr",
        "soil_version": "nbss_state_fallback",
        "irrigation_version": "des_state_level_lagged",
        "training_period": "<= 2012",
        "calibration_period": "2013",
        "test_period": "2014",
        "crop_list": sorted(ALL_CROPS),
        "feature_list": FEATURE_COLS,
        "target_definition": "is_major = 1 if crop_area_share >= 0.10",
        "major_threshold": 0.10,
        "hybrid_alpha": float(best_alpha),
        "alpha_selection_method": "grid_search_on_2013_validation_ndcg3",
        "split_info": "Train <=2012, Cal=2013, Test=2014",
        "provenance": "Kaggle Crop Production in India (DES)",
        "hybrid_test_metrics": hybrid_test_metrics,
        "baseline_results": baseline_results,
        "rolling_results": rolling_results,
        "ablation_results": ablation_results,
        "spatial_validation": spatial_results,
    })

    return best_ml_model, best_ml_meta, calibrator, best_alpha, hybrid_test_metrics, prior_df, df


def _run_spatial_validation(df, feature_cols, label_col):
    """Leave-state-out validation."""
    from ml.evaluation.ranking_metrics import compute_ranking_metrics
    from lightgbm import LGBMClassifier

    state_col = None
    for col_name in ['state', 'state_name', 'State_Name']:
        if col_name in df.columns:
            state_col = col_name
            break

    if state_col is None:
        print("  WARNING: No state column found, skipping spatial validation")
        return []

    states = df[state_col].dropna().unique()
    results = []

    for hold_state in sorted(states):
        test_mask = df[state_col] == hold_state
        train_part = df[~test_mask]
        test_part = df[test_mask]

        if len(test_part) < 30 or test_part[label_col].sum() < 3:
            continue

        model = LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, verbose=-1)
        X_tr = train_part[feature_cols].fillna(0)
        y_tr = train_part[label_col].values
        model.fit(X_tr, y_tr)

        X_te = test_part[feature_cols].fillna(0)
        test_eval = test_part.copy()
        test_eval['prob'] = model.predict_proba(X_te)[:, 1]
        metrics = compute_ranking_metrics(test_eval)

        results.append({
            'state': str(hold_state),
            'n_rows': int(len(test_part)),
            'ndcg_at_3': round(float(metrics['ndcg_at_3']), 4),
            'recall_at_1': round(float(metrics['recall_at_1']), 4),
            'recall_at_3': round(float(metrics['recall_at_3']), 4),
            'brier_score': round(float(metrics['brier_score']), 4),
        })

    if results:
        ndcg_vals = [r['ndcg_at_3'] for r in results]
        print("  Leave-state-out: %d states evaluated" % len(results))
        print("  NDCG@3 mean=%.4f, std=%.4f, min=%.4f, max=%.4f" % (np.mean(ndcg_vals), np.std(ndcg_vals), np.min(ndcg_vals), np.max(ndcg_vals)))
    else:
        print("  No states had sufficient data for spatial validation")

    return results


def run_export_and_serving(best_model, best_meta, calibrator, best_alpha, hybrid_test_metrics, prior_df, df):
    print("\n" + "="*60)
    print("PHASE 3: PERSIST ARTIFACTS (v2.1) & BUILD SERVING LOOKUP")
    print("="*60)

    # 1. Save artifacts to models/regional_v2.1/
    v21_model_path = os.path.join(V21_DIR, 'crop_model_regional.pkl')
    v21_cal_path   = os.path.join(V21_DIR, 'calibrator.pkl')
    v21_meta_path  = os.path.join(V21_DIR, 'model_metadata.json')
    v21_prior_path = os.path.join(V21_DIR, 'district_season_prior.parquet')
    v21_cfg_path   = os.path.join(V21_DIR, 'hybrid_config.json')

    joblib.dump(best_model, v21_model_path)
    joblib.dump(calibrator, v21_cal_path)

    prior_df.to_parquet(v21_prior_path, index=False)

    hybrid_cfg = {
        "alpha": float(best_alpha),
        "model_version": "v2.1-regional-hybrid",
        "best_base_model": best_meta.get("model_name", "LightGBM"),
        "test_metrics": hybrid_test_metrics,
        "rolling_results": best_meta.get("rolling_results", []),
    }
    with open(v21_cfg_path, 'w') as f:
        json.dump(hybrid_cfg, f, indent=2, default=str)

    with open(v21_meta_path, 'w') as f:
        json.dump(best_meta, f, indent=2, default=str)

    # Save feature schema
    from ml.data.build_training_table import FEATURE_COLS, ENVIRONMENTAL_COLS, CROP_COLS, LABEL_COL
    from ml.data.district_centroids import ALL_CROPS

    feature_schema = {
        "schema_version": "25-dim-v2.1",
        "total_features": len(FEATURE_COLS),
        "environmental_features": ENVIRONMENTAL_COLS,
        "crop_features": CROP_COLS,
        "all_features_ordered": FEATURE_COLS,
        "target": LABEL_COL,
        "target_definition": "is_major = 1 if crop_area_share >= 0.10 else 0",
        "major_threshold": 0.10,
        "crop_list": sorted(ALL_CROPS),
        "crop_count": len(ALL_CROPS),
        "supported_seasons": ["kharif", "rabi"],
    }
    feature_schema_path = os.path.join(V21_DIR, 'feature_schema.json')
    with open(feature_schema_path, 'w') as f:
        json.dump(feature_schema, f, indent=2)

    print(f"  Saved v2.1 model:       {v21_model_path}")
    print(f"  Saved v2.1 calibrator:  {v21_cal_path}")
    print(f"  Saved v2.1 prior:       {v21_prior_path}")
    print(f"  Saved v2.1 hybrid cfg:  {v21_cfg_path}")
    print(f"  Saved v2.1 metadata:    {v21_meta_path}")
    print(f"  Saved feature schema:   {feature_schema_path}")

    # Also copy to root backend/models/ for backward compatibility
    shutil.copy(v21_model_path, os.path.join(BASE_MODELS, 'crop_model_regional.pkl'))
    shutil.copy(v21_cal_path,   os.path.join(BASE_MODELS, 'calibrator.pkl'))
    shutil.copy(v21_prior_path, os.path.join(BASE_MODELS, 'district_season_prior.parquet'))
    shutil.copy(v21_cfg_path,   os.path.join(BASE_MODELS, 'hybrid_config.json'))
    shutil.copy(v21_meta_path,  os.path.join(BASE_MODELS, 'model_metadata.json'))
    shutil.copy(feature_schema_path, os.path.join(BASE_MODELS, 'feature_schema.json'))
    print("  Copied artifacts to root backend/models/ for backward compatibility.")

    # 2. Build serving lookup
    print("\nBuilding serving lookup table...")
    _build_serving_lookup(df, best_meta)


def _build_serving_lookup(df, meta):
    """
    Builds data/lookup/district_season_features.parquet.
    Uses most recent year's features as the current feature vector.
    """
    from ml.data.build_training_table import FEATURE_COLS

    latest_year = df['year'].max()
    latest = df[df['year'] == latest_year].copy()

    agg_cols = [c for c in FEATURE_COLS if c not in ('crop_id', 'season_kharif') and not c.startswith('crop_')]
    group_cols = ['district_id', 'district_name', 'state', 'season']

    # Numeric mean aggregation
    serving = latest.groupby(group_cols)[agg_cols].mean().reset_index()

    # Metadata columns aggregation
    meta_cols = [
        'soil_source', 'soil_resolution', 'soil_is_fallback',
        'irrigation_source', 'irrigation_resolution', 'irrigation_is_fallback',
        'climate_source'
    ]
    for col in meta_cols:
        if col in latest.columns:
            meta_map = latest.groupby(group_cols)[col].first().reset_index()
            serving = serving.merge(meta_map, on=group_cols, how='left')

    serving['season_kharif'] = (serving['season'] == 'kharif').astype(int)
    serving['data_quality_score'] = 'good'
    serving['model_version'] = meta.get('model_version', 'v2.1-regional-hybrid')
    serving['last_updated'] = latest_year
    serving['climate_resolution'] = 'district_centroid'

    out_path = os.path.join(LOOKUP_DIR, 'district_season_features.parquet')
    serving.to_parquet(out_path, index=False)
    print(f"  Serving lookup saved: {out_path}")
    print(f"  Districts in lookup: {serving['district_id'].nunique()}")


def main():
    parser = argparse.ArgumentParser(description='ALIP Crop Recommendation Pipeline')
    parser.add_argument('--skip-fetch', action='store_true', help='Skip API calls, use cached data')
    parser.add_argument('--train-only', action='store_true', help='Skip data pipeline, retrain only')
    args = parser.parse_args()

    t0 = time.time()

    if not args.train_only:
        df = run_data_pipeline(skip_fetch=args.skip_fetch)
    else:
        df = None

    (best_model, best_meta, calibrator, best_alpha,
     hybrid_test_metrics, prior_df, df) = run_ml_pipeline(df)

    run_export_and_serving(best_model, best_meta, calibrator, best_alpha,
                           hybrid_test_metrics, prior_df, df)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
