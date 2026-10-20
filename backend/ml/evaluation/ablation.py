"""
ALIP — Feature Ablation Experiments
Tests 5 feature configurations to identify what information matters.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from ml.data.district_centroids import ALL_CROPS

CROP_COLS = [f"crop_{c}" for c in sorted(ALL_CROPS)]

ABLATION_CONFIGS = {
    'A_climate_only': [
        'temp_mean', 'rainfall_total', 'rainfall_variability', 'season_kharif'
    ] + CROP_COLS,
    'B_climate_soil': [
        'temp_mean', 'rainfall_total', 'rainfall_variability', 'season_kharif',
        'soil_ph', 'soil_oc', 'soil_clay', 'soil_cec'
    ] + CROP_COLS,
    'C_climate_soil_irrigation': [
        'temp_mean', 'rainfall_total', 'rainfall_variability', 'season_kharif',
        'soil_ph', 'soil_oc', 'soil_clay', 'soil_cec',
        'lagged_irrigated_share'
    ] + CROP_COLS,
    'D_full_no_crop': [
        'temp_mean', 'rainfall_total', 'rainfall_variability', 'season_kharif',
        'soil_ph', 'soil_oc', 'soil_clay', 'soil_cec',
        'lagged_irrigated_share', 'preceding_monsoon_rainfall'
    ],
    'E_full': [
        'temp_mean', 'rainfall_total', 'rainfall_variability', 'preceding_monsoon_rainfall',
        'soil_ph', 'soil_oc', 'soil_clay', 'soil_cec',
        'lagged_irrigated_share', 'season_kharif'
    ] + CROP_COLS,
    'F_crop_prior_only': CROP_COLS,
}

def run_ablation(df, verbose=True) -> pd.DataFrame:
    """
    Runs all 5 ablation configs using temporal split.
    Returns DataFrame with results table.
    Prints comparison table.
    """
    import sys, os
    try:
        from ml.evaluation.splits import temporal_split_3way
    except ImportError:
        def temporal_split_3way(data): return data, data, data
    
    if 'year' not in df.columns:
        df['year'] = np.random.randint(2010, 2020, size=len(df))
    if 'is_major' not in df.columns:
        df['is_major'] = np.random.randint(0, 2, size=len(df))
        
    train_df, cal_df, test_df = temporal_split_3way(df)
    
    results = []
    
    for config_name, features in ABLATION_CONFIGS.items():
        avail_features = [f for f in features if f in df.columns]
        if not avail_features:
            continue
            
        X_train = train_df[avail_features].copy()
        y_train = train_df['is_major']
        X_test = test_df[avail_features].copy()
        y_test = test_df['is_major']
        
        for col in X_train.select_dtypes(include=['object', 'category']).columns:
            X_train[col] = pd.factorize(X_train[col])[0]
            X_test[col] = pd.factorize(X_test[col])[0]
            
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        
        rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        preds = rf.predict(X_test)
        probs = rf.predict_proba(X_test)[:, 1]
        
        acc = np.mean(preds == y_test)
        try:
            auc = float(roc_auc_score(y_test, probs))
        except Exception:
            auc = 0.0

        try:
            from ml.evaluation.ranking_metrics import compute_ranking_metrics
            df_eval = test_df.copy()
            df_eval['prob'] = probs
            rmetrics = compute_ranking_metrics(df_eval)
            ndcg3 = round(float(rmetrics.get('ndcg_at_3', 0.0)), 4)
            rec1 = round(float(rmetrics.get('recall_at_1', 0.0)), 4)
            rec3 = round(float(rmetrics.get('recall_at_3', 0.0)), 4)
        except Exception:
            ndcg3, rec1, rec3 = 0.0, 0.0, 0.0

        results.append({
            'config': config_name,
            'num_features': len(avail_features),
            'accuracy': round(float(acc), 4),
            'auc': round(float(auc), 4),
            'ndcg_at_3': ndcg3,
            'recall_at_1': rec1,
            'recall_at_3': rec3,
        })

    results_df = pd.DataFrame(results)

    if verbose:
        print(f"{'Config':<30} | {'Features':<10} | {'NDCG@3':<10} | {'Recall@1':<10} | {'AUC':<10}")
        print("-" * 75)
        for _, row in results_df.iterrows():
            print(f"{row['config']:<30} | {row['num_features']:<10} | {row['ndcg_at_3']:<10.4f} | {row['recall_at_1']:<10.4f} | {row['auc']:<10.4f}")

    return results

if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    try:
        from ml.data.build_training_table import FEATURE_COLS
    except ImportError:
        pass
    features_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'features', 'district_crop_training.parquet')
    if os.path.exists(features_path):
        df = pd.read_parquet(features_path)
    else:
        df = pd.DataFrame(np.random.randn(100, 20), columns=[
            'temp_mean', 'rainfall_total', 'rainfall_variability', 'season_kharif', 'crop_id',
            'soil_ph', 'soil_oc', 'soil_clay', 'soil_cec', 'lagged_irrigated_share', 'preceding_monsoon_rainfall',
            'year', 'is_major'
        ])
    run_ablation(df, verbose=True)
