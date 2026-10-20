"""
ALIP — SHAP Explainability
Generates per-prediction SHAP explanations translated to farmer language.
"""
import numpy as np
import shap

# Human-readable templates for each feature
SHAP_TEMPLATES = {
    'temp_mean': {
        'positive': 'Seasonal temperature is associated with higher scores for this crop in the model',
        'negative': 'Temperature is associated with lower scores for this crop in the model'
    },
    'rainfall_total': {
        'positive': 'Seasonal rainfall is associated with this crop recommendation',
        'negative': 'Rainfall is associated with lower scores for this crop in the model'
    },
    'rainfall_variability': {
        'positive': 'Rainfall pattern is associated with higher scores for this crop in the model',
        'negative': 'Erratic rainfall is associated with lower scores for this crop in the model'
    },
    'preceding_monsoon_rainfall': {
        'positive': 'Strong monsoon is associated with higher scores for this crop in the model',
        'negative': 'Low preceding monsoon is associated with lower scores for this crop in the model'
    },
    'soil_ph': {
        'positive': 'Soil pH is associated with higher scores for this crop in the model',
        'negative': 'Soil pH is associated with lower scores for this crop in the model'
    },
    'soil_oc': {
        'positive': 'Soil organic matter is associated with higher scores for this crop in the model',
        'negative': 'Soil organic matter is associated with lower scores for this crop in the model'
    },
    'soil_clay': {
        'positive': 'Soil texture is associated with higher scores for this crop in the model',
        'negative': 'Soil texture is associated with lower scores for this crop in the model'
    },
    'soil_cec': {
        'positive': 'Soil nutrient retention capacity is associated with higher scores for this crop in the model',
        'negative': 'Soil nutrient retention is associated with lower scores for this crop in the model'
    },
    'lagged_irrigated_share': {
        'positive': 'Irrigation availability is associated with this crop recommendation',
        'negative': 'Irrigation levels are associated with lower scores for this crop in the model'
    },
    'season_kharif': {
        'positive': 'This season is associated with higher scores for this crop in the model',
        'negative': 'This season is associated with lower scores for this crop in the model'
    },
}

class CropExplainer:
    def __init__(self, model, feature_cols):
        self.model = model
        self.feature_cols = feature_cols
        self.explainer = shap.TreeExplainer(model)
    
    def explain_prediction(self, feature_vector: np.ndarray, top_n: int = 3) -> list:
        """
        Returns top_n reasons as human-readable strings.
        feature_vector: 1D numpy array matching feature_cols order.
        """
        shap_values = self.explainer.shap_values(feature_vector.reshape(1, -1))
        
        if isinstance(shap_values, list):
            sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            sv = shap_values[0]
            if len(sv.shape) > 1: # if multiclass format in array
                sv = sv[:, 1] if sv.shape[1] > 1 else sv[:, 0]
            
        top_indices = np.argsort(np.abs(sv))[::-1][:top_n]
        
        reasons = []
        for idx in top_indices:
            feat_name = self.feature_cols[idx]
            val = sv[idx]
            
            if feat_name in SHAP_TEMPLATES:
                direction = 'positive' if val > 0 else 'negative'
                reasons.append(SHAP_TEMPLATES[feat_name][direction])
            else:
                reasons.append(f"{feat_name} was an important factor in the model score for this crop")
                
        return reasons
    
    def explain_batch(self, X: np.ndarray) -> list:
        """Explains all rows in X. Returns list of reason lists."""
        reasons_list = []
        for i in range(X.shape[0]):
            reasons_list.append(self.explain_prediction(X[i]))
        return reasons_list

if __name__ == '__main__':
    print('SHAP explainer module loaded OK')
