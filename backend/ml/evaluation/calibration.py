"""
ALIP — Model Probability Calibration
Calibrates is_major probability output using isotonic regression.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

def calibrate_model_iso(val_prob_raw, val_y):
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(val_prob_raw, val_y)
    return ir

def calibrate_model(model, val_X, val_y):
    """
    Fits isotonic regression calibrator on validation predictions.
    Returns calibrator object.
    """
    raw_probs = model.predict_proba(val_X)[:, 1]
    return calibrate_model_iso(raw_probs, val_y)

def apply_calibration(calibrator, raw_probs: np.ndarray) -> np.ndarray:
    """Applies calibrator to raw probabilities."""
    return calibrator.predict(raw_probs)

def compute_calibration_metrics(y_true, y_prob_raw, y_prob_cal) -> dict:
    """
    Returns: brier_score_raw, brier_score_cal, ece_raw, ece_cal
    ECE = Expected Calibration Error
    """
    def expected_calibration_error(y, proba, bins=10):
        prob_true, prob_pred = calibration_curve(y, proba, n_bins=bins)
        return np.mean(np.abs(prob_true - prob_pred))
        
    metrics = {
        'brier_score_raw': brier_score_loss(y_true, y_prob_raw),
        'brier_score_cal': brier_score_loss(y_true, y_prob_cal),
        'ece_raw': expected_calibration_error(y_true, y_prob_raw),
        'ece_cal': expected_calibration_error(y_true, y_prob_cal)
    }
    return metrics

def save_calibrator(calibrator, path: str):
    joblib.dump(calibrator, path)

def load_calibrator(path: str):
    return joblib.load(path)
