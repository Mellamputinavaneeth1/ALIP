import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

def compute_ranking_metrics(df_with_scores, k=3):
    """
    Computes ranking metrics for crop recommendations.
    df_with_scores must contain: ['district_id', 'season', 'year', 'crop', 'is_major', 'prob']
    """
    # Calculate brier score globally
    brier = brier_score_loss(df_with_scores['is_major'], df_with_scores['prob'])
    
    # Group by instance
    groups = df_with_scores.groupby(['district_id', 'season', 'year'])
    
    rec_at_1_list = []
    rec_at_3_list = []
    ndcg_list = []
    map_list = []
    
    for _, group in groups:
        # Sort by predicted probability
        sorted_group = group.sort_values(by='prob', ascending=False)
        actual_labels = sorted_group['is_major'].values
        
        # Recall @ 1
        r1 = actual_labels[0] if len(actual_labels) > 0 else 0
        rec_at_1_list.append(r1)
        
        # Recall @ 3
        r3 = 1 if sum(actual_labels[:k]) > 0 else 0
        rec_at_3_list.append(r3)
        
        # NDCG @ 3
        dcg = 0
        idcg = 0
        ideal_labels = sorted(actual_labels, reverse=True)
        for i in range(min(k, len(actual_labels))):
            dcg += (2**actual_labels[i] - 1) / np.log2(i + 2)
            idcg += (2**ideal_labels[i] - 1) / np.log2(i + 2)
        ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)
        
        # MAP @ 3
        ap = 0
        hits = 0
        for i in range(min(k, len(actual_labels))):
            if actual_labels[i] == 1:
                hits += 1
                ap += hits / (i + 1)
        num_relevant = min(k, sum(actual_labels))
        map_list.append(ap / num_relevant if num_relevant > 0 else 0.0)
        
    return {
        'recall_at_1': np.mean(rec_at_1_list),
        'recall_at_3': np.mean(rec_at_3_list),
        'ndcg_at_3': np.mean(ndcg_list),
        'map_at_3': np.mean(map_list),
        'brier_score': brier
    }

def compute_baselines(train_df, test_df):
    """
    Computes baselines based on historical frequencies.
    """
    results = {}
    
    # Calculate historical frequencies
    district_season_freq = train_df.groupby(['district_id', 'season', 'crop'])['is_major'].mean().reset_index()
    district_season_freq = district_season_freq.rename(columns={'is_major': 'prob'})
    
    state_season_freq = train_df.groupby(['state', 'season', 'crop'])['is_major'].mean().reset_index()
    state_season_freq = state_season_freq.rename(columns={'is_major': 'prob'})
    
    global_freq = train_df.groupby('crop')['is_major'].mean().reset_index()
    global_freq = global_freq.rename(columns={'is_major': 'prob'})
    
    # 1. DistrictSeasonPrior
    df_ds = test_df.copy()
    df_ds = df_ds.merge(district_season_freq, on=['district_id', 'season', 'crop'], how='left')
    df_ds['prob'] = df_ds['prob'].fillna(0.0)
    results['DistrictSeasonPrior'] = compute_ranking_metrics(df_ds)
    
    # 2. StateSeasonPrior
    df_ss = test_df.copy()
    df_ss = df_ss.merge(state_season_freq, on=['state', 'season', 'crop'], how='left')
    df_ss['prob'] = df_ss['prob'].fillna(0.0)
    results['StateSeasonPrior'] = compute_ranking_metrics(df_ss)
    
    # 3. MajorityCrop
    df_mc = test_df.copy()
    df_mc = df_mc.merge(global_freq, on='crop', how='left')
    df_mc['prob'] = df_mc['prob'].fillna(0.0)
    results['MajorityCrop'] = compute_ranking_metrics(df_mc)
    
    return results


def compute_district_season_prior(train_df):
    """
    Computes empirical major-crop probability for each district x season x crop.
    """
    ds_freq = train_df.groupby(['district_id', 'season', 'crop'])['is_major'].mean().reset_index()
    ds_freq = ds_freq.rename(columns={'is_major': 'prior_prob'})
    return ds_freq


def compute_hybrid_evaluation(ml_probs, prior_probs, labels, groups_df, alpha=0.5, k=3):
    """
    Computes ranking metrics for hybrid score = alpha * ml_prob + (1 - alpha) * prior_prob.
    """
    df_eval = groups_df[['district_id', 'season', 'year', 'crop']].copy()
    df_eval['is_major'] = labels.values if hasattr(labels, 'values') else labels
    df_eval['prob'] = alpha * np.array(ml_probs) + (1.0 - alpha) * np.array(prior_probs)
    return compute_ranking_metrics(df_eval, k=k)


def optimize_hybrid_alpha(val_df, ml_probs, prior_probs, labels):
    """
    Searches alpha in [0.0, 1.0] by steps of 0.05 to maximize NDCG@3 on validation data.
    Returns (best_alpha, best_metrics, all_results).
    """
    best_alpha = 0.5
    best_ndcg = -1.0
    best_metrics = None
    all_results = {}
    
    for a in np.linspace(0.0, 1.0, 21):
        alpha_val = round(float(a), 2)
        metrics = compute_hybrid_evaluation(ml_probs, prior_probs, labels, val_df, alpha=alpha_val)
        all_results[alpha_val] = metrics
        if metrics['ndcg_at_3'] > best_ndcg:
            best_ndcg = metrics['ndcg_at_3']
            best_alpha = alpha_val
            best_metrics = metrics
            
    return best_alpha, best_metrics, all_results

