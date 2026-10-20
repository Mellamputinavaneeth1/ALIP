"""
ALIP — Baseline Models for Crop Recommendation
Three baselines that ML models must beat.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

class HistoricalFrequencyBaseline:
    """
    For each district-season, predict based on historical crop frequency.
    crop with highest avg crop_area_share historically → is_major=1.
    """
    def __init__(self):
        self.freq_map = {}

    def fit(self, train_df):
        if 'crop_area_share' in train_df.columns:
            grouped = train_df.groupby(['district_id', 'season', 'crop_id'])['crop_area_share'].mean().reset_index()
            idx = grouped.groupby(['district_id', 'season'])['crop_area_share'].idxmax()
            best_crops = grouped.loc[idx]
            for _, row in best_crops.iterrows():
                self.freq_map[(row['district_id'], row['season'])] = row['crop_id']
        else:
            grouped = train_df[train_df['is_major'] == 1].groupby(['district_id', 'season', 'crop_id']).size().reset_index(name='count')
            if not grouped.empty:
                idx = grouped.groupby(['district_id', 'season'])['count'].idxmax()
                best_crops = grouped.loc[idx]
                for _, row in best_crops.iterrows():
                    self.freq_map[(row['district_id'], row['season'])] = row['crop_id']

    def predict(self, test_df) -> pd.Series:
        preds = []
        for _, row in test_df.iterrows():
            best_crop = self.freq_map.get((row['district_id'], row.get('season', 'kharif')), None)
            preds.append(1 if best_crop == row['crop_id'] else 0)
        return pd.Series(preds, index=test_df.index)

    def evaluate(self, test_df) -> dict:
        y_pred = self.predict(test_df)
        y_true = test_df['is_major']
        return evaluate_metric(y_true, y_pred)

class StateSeasonPriorBaseline:
    """
    Use state × season crop frequencies as prior.
    Most common crop in that state-season → is_major=1.
    """
    def __init__(self):
        self.freq_map = {}

    def fit(self, train_df):
        if 'state' in train_df.columns:
            grouped = train_df[train_df['is_major'] == 1].groupby(['state', 'season', 'crop_id']).size().reset_index(name='count')
            if not grouped.empty:
                idx = grouped.groupby(['state', 'season'])['count'].idxmax()
                best_crops = grouped.loc[idx]
                for _, row in best_crops.iterrows():
                    self.freq_map[(row['state'], row['season'])] = row['crop_id']

    def predict(self, test_df) -> pd.Series:
        preds = []
        for _, row in test_df.iterrows():
            best_crop = self.freq_map.get((row.get('state', ''), row.get('season', 'kharif')), None)
            preds.append(1 if best_crop == row['crop_id'] else 0)
        return pd.Series(preds, index=test_df.index)

    def evaluate(self, test_df) -> dict:
        y_pred = self.predict(test_df)
        y_true = test_df['is_major']
        return evaluate_metric(y_true, y_pred)

class AgronomicRuleBaseline:
    """
    Hard-coded agronomic suitability rules.
    Returns 1 if crop is known to be grown in that season.
    """
    SEASON_CROPS = {
        'kharif': ['rice','maize','soybean','cotton','groundnut','pigeonpeas',
                   'mungbean','jowar','bajra','ragi'],
        'rabi':   ['wheat','chickpea','mustard','lentil','barley','maize',
                   'jowar','ragi']
    }
    def predict(self, test_df) -> pd.Series:
        preds = []
        for _, row in test_df.iterrows():
            season = str(row.get('season', 'kharif')).lower()
            crop = str(row['crop_id']).lower()
            valid_crops = self.SEASON_CROPS.get(season, [])
            preds.append(1 if crop in valid_crops else 0)
        return pd.Series(preds, index=test_df.index)

    def evaluate(self, test_df) -> dict:
        y_pred = self.predict(test_df)
        y_true = test_df['is_major']
        return evaluate_metric(y_true, y_pred)

def evaluate_metric(y_true, y_pred, y_scores=None) -> dict:
    """
    Returns: top1_hit_rate, precision, recall, f1
    Also calculates top3_hit_rate if scores provided.
    """
    metrics = {
        'top1_hit_rate': np.mean(y_true[y_pred == 1] == 1) if sum(y_pred) > 0 else 0.0,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    return metrics

def run_all_baselines(train_df, test_df, verbose=True) -> dict:
    """Runs all 3 baselines and returns comparison dict."""
    baselines = {
        'Historical Frequency': HistoricalFrequencyBaseline(),
        'State-Season Prior': StateSeasonPriorBaseline(),
        'Agronomic Rule': AgronomicRuleBaseline()
    }
    
    results = {}
    for name, model in baselines.items():
        if hasattr(model, 'fit'):
            model.fit(train_df)
        res = model.evaluate(test_df)
        results[name] = res
        
    if verbose:
        print(f"{'Baseline Model':<25} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9}")
        print("-" * 60)
        for name, res in results.items():
            print(f"{name:<25} | {res['precision']:.4f}    | {res['recall']:.4f}    | {res['f1']:.4f}")
            
    return results
