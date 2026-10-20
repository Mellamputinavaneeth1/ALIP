"""
ALIP — LightGBM Crop Ranking Model
"""
import os, sys, json, joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from ml.data.build_training_table import FEATURE_COLS, LABEL_COL
except ImportError:
    FEATURE_COLS = ['temp_mean', 'rainfall_total']
    LABEL_COL = 'is_major'
    
try:
    from ml.evaluation.splits import temporal_split_3way, random_grouped_split
except ImportError:
    def temporal_split_3way(df): return df, df, df
    def random_grouped_split(df): return df, df

FEATURES_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'features', 'district_crop_training.parquet')
MODEL_OUT     = os.path.join(os.path.dirname(__file__), '..', 'models', 'lgbm_model.pkl')
METADATA_OUT  = os.path.join(os.path.dirname(__file__), '..', 'models', 'lgbm_metadata.json')

def train_lgbm(verbose=True):
    if not os.path.exists(FEATURES_PATH):
        print(f"Data not found at {FEATURES_PATH}.")
        return None, None
    else:
        df = pd.read_parquet(FEATURES_PATH)
        
    train_df_tmp, cal_df_tmp, test_df_tmp = temporal_split_3way(df)
    
    X_train = train_df_tmp[FEATURE_COLS].fillna(0)
    y_train = train_df_tmp[LABEL_COL]
    # We use cal_df_tmp for validation metrics here
    X_val = cal_df_tmp[FEATURE_COLS].fillna(0)
    y_val = cal_df_tmp[LABEL_COL]
    
    # No categorical features anymore since they are one-hot encoded
    
    model = LGBMClassifier(
        n_estimators=300, 
        max_depth=6, 
        learning_rate=0.05, 
        num_leaves=31,
        subsample=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    if verbose:
        print("--- Validation Set Results (Calibration Set) ---")
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        try:
            print(f"ROC-AUC: {roc_auc_score(y_val, probs):.4f}")
        except: pass
        
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    
    metadata = {
        'model_type': 'LightGBM',
        'feature_list': FEATURE_COLS,
        'training_period': 'all_available',
        'crop_list': df['crop_id'].unique().tolist() if 'crop_id' in df.columns else [],
        'validation_results': {},
        'n_train': len(X_train),
        'n_test': len(X_val)
    }
    
    with open(METADATA_OUT, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    return model, metadata

def compute_top3_hit_rate(model, test_df, feature_cols):
    if 'district_id' not in test_df.columns or 'season' not in test_df.columns or 'year' not in test_df.columns:
        return 0.0
        
    df = test_df.copy()
    X = df[feature_cols].fillna(0)
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = pd.factorize(X[col])[0]
        
    df['prob'] = model.predict_proba(X)[:, 1]
    
    grouped = df.groupby(['district_id', 'year', 'season'])
    hits = 0
    total = 0
    
    for _, group in grouped:
        top3 = group.nlargest(3, 'prob')
        if top3[LABEL_COL].sum() > 0:
            hits += 1
        total += 1
        
    return hits / total if total > 0 else 0.0

if __name__ == '__main__':
    train_lgbm(verbose=True)
