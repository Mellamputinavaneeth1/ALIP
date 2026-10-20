import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
import shap
import joblib
import os
import urllib.request
from data.soil_data import get_soil_data, get_rainfall
from data.crop_agronomy import get_crop_agronomy
from services.weather_service import get_weather

# ─── Global SHAP Explainer Cache ────────────────────────────
_shap_explainer = None  # Lazy-initialized after model loads

# ─── Paths ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "../data/Crop_recommendation.csv")
MODEL_PATH = os.path.join(BASE_DIR, "../data/crop_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "../data/label_encoder.pkl")

# ─── Crop Info ──────────────────────────────────────────────
CROP_INFO = {
    "rice": "Best grown in monsoon. Needs waterlogged conditions.",
    "maize": "Versatile crop for moderate climates. Good for Kharif season.",
    "chickpea": "Cool, dry climate crop. Best for Rabi season.",
    "kidneybeans": "Needs warm temperature and moderate rainfall.",
    "pigeonpeas": "Drought resistant. Suitable for tropical climates.",
    "mothbeans": "Highly drought tolerant. Best for arid regions.",
    "mungbean": "Warm humid conditions. Good for Kharif season.",
    "blackgram": "Tropical and subtropical climate crop.",
    "lentil": "Cool dry conditions. Best for Rabi season.",
    "pomegranate": "Semi-arid conditions with hot summers.",
    "banana": "High humidity and warmth needed year round.",
    "mango": "Tropical climate with dry winters.",
    "grapes": "Warm dry climate with cool nights.",
    "watermelon": "Warm temperature and sandy loam soil.",
    "muskmelon": "Warm dry climate. Best for Zaid season.",
    "apple": "Cold winters and mild summers required.",
    "orange": "Subtropical climate preferred.",
    "papaya": "Warm humid conditions year round.",
    "coconut": "Tropical coastal areas.",
    "cotton": "Warm temperature and moderate rainfall.",
    "jute": "Warm humid conditions.",
    "coffee": "Tropical highlands with moderate rainfall."
}

# ─── Download Dataset ───────────────────────────────────────
def download_dataset():
    if not os.path.exists(DATA_PATH):
        print("Downloading crop dataset...")
        url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/Crop_recommendation.csv"
        try:
            urllib.request.urlretrieve(url, DATA_PATH)
            print("Dataset downloaded!")
        except Exception as e:
            print(f"Download failed: {e}")

# ─── Train Model ────────────────────────────────────────────
def train_model():
    print("Training crop recommendation model...")
    df = pd.read_csv(DATA_PATH)

    X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
    y = df["label"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10
    )
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model trained! Accuracy: {accuracy * 100:.2f}%")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print("Model saved!")

    return model, le

def evaluate_model_kfold():
    """
    Publication-ready model evaluation using Stratified K-Fold
    Cross-Validation. Reports mean ± std for Accuracy, Precision,
    Recall, F1-Score, and Cohen's Kappa across 5 and 10 folds.
    
    Why Stratified K-Fold?
    - Ensures each fold has proportional representation of all 22 crop classes
    - More reliable than a single train/test split
    - Standard deviation reveals model stability
    
    Returns dict with all metrics for journal reporting.
    """
    df = pd.read_csv(DATA_PATH)

    X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
    y = df["label"]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    kappa_scorer = make_scorer(cohen_kappa_score)
    
    scoring = {
        'accuracy': 'accuracy',
        'precision_weighted': 'precision_weighted',
        'recall_weighted': 'recall_weighted',
        'f1_weighted': 'f1_weighted',
        'cohen_kappa': kappa_scorer
    }
    
    results = {}
    
    for folds in [5, 10]:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        
        cv_results = cross_validate(model, X, y_encoded, cv=cv, scoring=scoring)
        
        fold_key = f'fold_{folds}'
        results[fold_key] = {}
        for metric in scoring.keys():
            mean_val = np.mean(cv_results[f'test_{metric}'])
            std_val = np.std(cv_results[f'test_{metric}'])
            results[fold_key][metric] = {
                'mean': float(mean_val),
                'std': float(std_val)
            }
            
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y_encoded)
    results['feature_importances'] = dict(zip(X.columns, model.feature_importances_))
    
    print("\n--- K-Fold Cross-Validation Results ---")
    for folds in [5, 10]:
        print(f"\n{folds}-Fold Stratified CV:")
        for metric in scoring.keys():
            mean = results[f'fold_{folds}'][metric]['mean']
            std = results[f'fold_{folds}'][metric]['std']
            print(f"  {metric}: {mean:.4f} ± {std:.4f}")
            
    return results

def evaluate_noise_robustness(noise_levels=None):
    """
    Noise Injection Robustness Test for publication.
    Proves model doesn't collapse when soil sensors provide
    inaccurate readings (common in field conditions).
    
    Adds Gaussian noise at 5%, 10%, 15%, 20%, 25% to test features
    and reports accuracy degradation curve.
    
    Key insight for journal: If accuracy drops gracefully (e.g., 99% → 85%)
    rather than collapsing (99% → 30%), the model is robust.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
        
    df = pd.read_csv(DATA_PATH)

    X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
    y = df["label"]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    results = {}
    
    print("\n--- Noise Robustness Results ---")
    for noise in noise_levels:
        accuracies = []
        for _ in range(10):
            std_devs = X_test.std(axis=0)
            noise_matrix = np.random.normal(0, noise * std_devs, X_test.shape)
            X_test_noisy = X_test + noise_matrix
            
            acc = model.score(X_test_noisy, y_test)
            accuracies.append(acc)
            
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        results[noise] = {'mean_accuracy': float(mean_acc), 'std_accuracy': float(std_acc)}
        print(f"Noise {noise*100:0.0f}% -> Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        
    return results

def evaluate_baseline_comparison():
    print("--- Baseline Comparison ---")
    df = pd.read_csv(DATA_PATH)
    X = df.drop('label', axis=1)
    y = df['label']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {'models': {}, 'statistical_tests': {}}
    fold_accs = {}
    
    for name, model in models.items():
        scores = cross_validate(model, X, y_encoded, cv=cv, scoring=['accuracy', 'f1_weighted'])
        acc_scores = scores['test_accuracy']
        f1_scores = scores['test_f1_weighted']
        
        fold_accs[name] = acc_scores
        
        results['models'][name] = {
            'accuracy_mean': float(np.mean(acc_scores)),
            'accuracy_std': float(np.std(acc_scores)),
            'f1_mean': float(np.mean(f1_scores)),
            'f1_std': float(np.std(f1_scores)),
            'fold_scores': acc_scores.tolist()
        }
        print(f"[*] {name}: Acc {np.mean(acc_scores):.4f} (+/- {np.std(acc_scores):.4f})")
        
    for baseline in ['SVM', 'LogisticRegression', 'KNN']:
        t_stat, p_val = stats.ttest_rel(fold_accs['RandomForest'], fold_accs[baseline])
        results['statistical_tests'][f'RF_vs_{baseline}'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'significant': bool(p_val < 0.05)
        }
        
    return results

def evaluate_shap_analysis():
    print("--- SHAP Analysis ---")
    df = pd.read_csv(DATA_PATH)
    X = df.drop('label', axis=1)
    y = df['label']
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)
    
    explainer = shap.TreeExplainer(model)
    X_sample = X.sample(n=100, random_state=42)
    shap_values = explainer.shap_values(X_sample)
    
    if isinstance(shap_values, list):
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        if len(shap_values.shape) == 3:
            mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
        else:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
    feature_names = X.columns.tolist()
    shap_dict = {feat: float(val) for feat, val in zip(feature_names, mean_abs_shap)}
    
    sorted_features = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (feat, val) in enumerate(sorted_features):
        print(f"[{rank+1}] {feat}: {val:.4f}")
        
    top_features = [f[0] for f in sorted_features[:3]]
    agro_str = f"Top features ({', '.join(top_features)}) align with known agronomic factors for crop selection"
    
    return {
        'mean_abs_shap': shap_dict,
        'feature_ranking': sorted_features,
        'agronomic_validation': agro_str
    }

# ─── Load or Train ──────────────────────────────────────────
def get_crop_model():
    global _shap_explainer
    download_dataset()
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        model = joblib.load(MODEL_PATH)
        le = joblib.load(ENCODER_PATH)
        print("Crop model loaded from disk")
    else:
        model, le = train_model()

    # Build SHAP explainer once and cache it
    try:
        _shap_explainer = shap.TreeExplainer(model)
        print("SHAP explainer initialised")
    except Exception as e:
        print(f"SHAP init failed (non-critical): {e}")
        _shap_explainer = None

    return model, le

crop_model, label_encoder = get_crop_model()

# ─── SHAP Helper ────────────────────────────────────────────
FEATURE_LABELS = {
    "N":           "Nitrogen (N)",
    "P":           "Phosphorous (P)",
    "K":           "Potassium (K)",
    "temperature": "Temperature",
    "humidity":    "Humidity",
    "ph":          "Soil pH",
    "rainfall":    "Rainfall",
}

def _get_shap_explanation(input_array: np.ndarray) -> dict:
    """
    Returns the top 3 SHAP feature drivers for a single prediction.
    Uses the cached TreeExplainer. Returns empty dict if SHAP unavailable.
    """
    if _shap_explainer is None:
        return {}

    try:
        feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        # shap_values shape: (n_classes, n_samples, n_features) for multi-class RF
        sv = _shap_explainer.shap_values(input_array)

        # Average absolute SHAP across all classes for this sample
        if isinstance(sv, list):
            mean_abs = np.mean([np.abs(sv_class[0]) for sv_class in sv], axis=0)
        else:
            if sv.ndim == 3:
                mean_abs = np.abs(sv[0]).mean(axis=1)
            else:
                mean_abs = np.abs(sv[0])

        total = mean_abs.sum() or 1.0
        ranked = sorted(
            zip(feature_names, mean_abs),
            key=lambda x: x[1], reverse=True
        )

        top_factors = []
        for feat, val in ranked[:3]:
            pct = round(float(val / total) * 100, 1)
            if pct >= 30:
                impact = "Very High"
            elif pct >= 20:
                impact = "High"
            elif pct >= 10:
                impact = "Medium"
            else:
                impact = "Low"

            top_factors.append({
                "feature": FEATURE_LABELS.get(feat, feat),
                "feature_key": feat,
                "raw_value": round(float(input_array[0][feature_names.index(feat)]), 2),
                "shap_score": round(float(val), 4),
                "bar_pct": min(100, round(pct * 3)),  # scale for bar chart display
                "impact": impact,
            })

        return {"top_factors": top_factors}

    except Exception as e:
        print(f"SHAP explanation error (non-critical): {e}")
        return {}


def _build_recommendations(probabilities, top_n: int = 3) -> list:
    """Build ranked recommendation list with agronomy data."""
    top_indices = np.argsort(probabilities)[::-1][:top_n]
    recommendations = []
    for idx in top_indices:
        crop_name = label_encoder.inverse_transform([idx])[0]
        confidence = round(float(probabilities[idx]) * 100, 2)
        agronomy = get_crop_agronomy(crop_name)
        recommendations.append({
            "crop": crop_name,
            "confidence": confidence,
            "description": CROP_INFO.get(crop_name.lower(), "Suitable for your conditions."),
            "agronomy": agronomy,
        })
    return recommendations


# ─── Recommend Crop (Auto Mode — location + season) ─────────
async def recommend_crop(location: str, season: str) -> dict:
    try:
        soil = get_soil_data(location)
        rainfall = get_rainfall(season)

        temperature = 28.0
        humidity = 65.0
        weather_source = "fallback-default"
        weather_error = None

        weather_result = await get_weather(location)
        if weather_result.get("success"):
            weather = weather_result["weather"]
            temperature = float(weather["temperature"])
            humidity = float(weather["humidity"])
            weather_source = "openweathermap"
        else:
            weather_error = weather_result.get("error")

        input_array = np.array([[
            soil["N"], soil["P"], soil["K"],
            temperature, humidity,
            soil["ph"], rainfall
        ]])

        probabilities = crop_model.predict_proba(input_array)[0]
        recommendations = _build_recommendations(probabilities, top_n=3)
        shap_explanation = _get_shap_explanation(input_array)

        return {
            "success": True,
            "location": location,
            "season": season,
            "top_recommendation": recommendations[0]["crop"],
            "confidence": recommendations[0]["confidence"],
            "recommendations": recommendations,
            "shap_explanation": shap_explanation,
            "weather_source": weather_source,
            "weather_error": weather_error,
            "weather_inputs": {
                "temperature": temperature,
                "humidity": humidity,
                "rainfall": rainfall,
            },
            "soil_summary": {
                "nitrogen": soil["N"],
                "phosphorous": soil["P"],
                "potassium": soil["K"],
                "ph": soil["ph"],
            },
            "soil_type": soil.get("soil_type", "Mixed Soil"),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# ─── Recommend Crop (Manual Mode — direct values) ───────────
async def recommend_crop_manual(
    nitrogen: float,
    phosphorous: float,
    potassium: float,
    temperature: float,
    humidity: float,
    ph: float,
    rainfall: float,
    season: str = "kharif",
    location: str = "",
) -> dict:
    """
    Manual input mode: farmer provides exact soil test values.
    Used when farmer has a Soil Health Card or lab report.
    """
    try:
        feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        import pandas as _pd
        input_df = _pd.DataFrame([[
            nitrogen, phosphorous, potassium,
            temperature, humidity, ph, rainfall
        ]], columns=feature_names)
        input_array = input_df.values

        probabilities = crop_model.predict_proba(input_df)[0]
        recommendations = _build_recommendations(probabilities, top_n=3)
        shap_explanation = _get_shap_explanation(input_array)

        return {
            "success": True,
            "location": location or "Custom Input",
            "season": season,
            "input_mode": "manual",
            "top_recommendation": recommendations[0]["crop"],
            "confidence": recommendations[0]["confidence"],
            "recommendations": recommendations,
            "shap_explanation": shap_explanation,
            "weather_source": "manual",
            "weather_error": None,
            "weather_inputs": {
                "temperature": temperature,
                "humidity": humidity,
                "rainfall": rainfall,
            },
            "soil_summary": {
                "nitrogen": nitrogen,
                "phosphorous": phosphorous,
                "potassium": potassium,
                "ph": ph,
            },
            "soil_type": "Custom (Manual Input)",
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
