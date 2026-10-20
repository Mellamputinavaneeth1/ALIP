"""
ALIP — Validation Split Strategies
Four leakage-safe splits for robust model evaluation.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.cluster import KMeans

# ── Constants ────────────────────────────────────────────────────────────────
TEMPORAL_CUTOFF_YEAR = 2012   # Legacy
TEMPORAL_TRAIN_END = 2012
TEMPORAL_CAL_YEAR = 2013
TEMPORAL_TEST_YEAR = 2014

def random_grouped_split(df, group_col='district_id', test_size=0.2, seed=42):
    """
    Groups all rows of the same district together before splitting.
    Prevents same-district rows from leaking between train/test.
    Returns: (train_df, test_df)
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=df[group_col]))
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
    print(f"Random Grouped Split -> Train: {len(train_df)}, Test: {len(test_df)}")
    return train_df, test_df

def spatial_block_split(df, n_blocks=5, test_blocks=1, seed=42):
    """
    Clusters districts geographically using state + district patterns.
    One geographic cluster is held out as test set.
    Returns: (train_df, test_df)
    """
    if 'state' in df.columns:
        state_encoded = pd.factorize(df['state'])[0]
    else:
        state_encoded = np.zeros(len(df))
    
    kmeans = KMeans(n_clusters=n_blocks, random_state=seed, n_init=10)
    blocks = kmeans.fit_predict(state_encoded.reshape(-1, 1))
    
    test_block_ids = np.random.RandomState(seed).choice(range(n_blocks), size=test_blocks, replace=False)
    
    test_mask = np.isin(blocks, test_block_ids)
    train_df = df[~test_mask]
    test_df = df[test_mask]
    print(f"Spatial Block Split -> Train: {len(train_df)}, Test: {len(test_df)}")
    return train_df, test_df

def leave_state_out_split(df, holdout_state: str):
    """
    Entire state held out as test set.
    Returns: (train_df, test_df)
    """
    train_df = df[df['state'] != holdout_state]
    test_df = df[df['state'] == holdout_state]
    print(f"Leave State Out Split ('{holdout_state}') -> Train: {len(train_df)}, Test: {len(test_df)}")
    return train_df, test_df

def temporal_split(df, cutoff_year: int = TEMPORAL_CUTOFF_YEAR):
    """
    DEPRECATED. Use temporal_split_3way instead.
    Earlier years -> train, later years -> test.
    Returns: (train_df, test_df)
    """
    import warnings
    warnings.warn("temporal_split is deprecated. Use temporal_split_3way.", DeprecationWarning)
    train_df = df[df['year'] <= cutoff_year]
    test_df = df[df['year'] > cutoff_year]
    print(f"Temporal Split (cutoff {cutoff_year}) -> Train: {len(train_df)}, Test: {len(test_df)}")
    return train_df, test_df

def temporal_split_3way(df):
    """
    Train: <= 2012
    Calibration: 2013
    Test: 2014
    (2015 is excluded entirely due to incomplete data)
    """
    train_df = df[df['year'] <= 2012]
    cal_df = df[df['year'] == 2013]
    test_df = df[df['year'] == 2014]
    print(f"Temporal 3-way Split -> Train: {len(train_df)}, Cal: {len(cal_df)}, Test: {len(test_df)}")
    return train_df, cal_df, test_df


def rolling_origin_temporal_splits(df):
    """
    Rolling-origin temporal evaluation folds:
    Fold 1: Train <= 2011, Cal 2012, Test 2013
    Fold 2: Train <= 2012, Cal 2013, Test 2014
    Returns list of (train_df, cal_df, test_df, test_year)
    """
    folds = []
    for test_year in [2013, 2014]:
        cal_year = test_year - 1
        train_df = df[df['year'] < cal_year]
        cal_df   = df[df['year'] == cal_year]
        test_df  = df[df['year'] == test_year]
        folds.append((train_df, cal_df, test_df, test_year))
    return folds
