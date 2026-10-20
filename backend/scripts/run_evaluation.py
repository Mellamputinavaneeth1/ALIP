import os
import sys
import json
import traceback
import numpy as np
from datetime import datetime

# Add backend to path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.crop_service import (
    evaluate_model_kfold,
    evaluate_noise_robustness,
    evaluate_baseline_comparison,
    evaluate_shap_analysis
)
from services.demand_ml_service import (
    evaluate_demand_model,
    evaluate_demand_baselines
)
from services.ablation_study_service import run_full_ablation_study

def make_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (float, int, str, bool, type(None))):
        return obj
    else:
        return str(obj)

def run_evaluations():
    report = {
        "timestamp": datetime.now().isoformat(),
        "crop_recommendation": {},
        "demand_threshold": {},
        "ablation_study": {},
        "publication_summary": ""
    }
    
    print("Starting evaluations...")
    
    # 1. Crop Recommendation Module
    print("Running Crop Recommendation Module...")
    try:
        print(" - k-fold evaluation")
        report["crop_recommendation"]["kfold"] = evaluate_model_kfold()
    except Exception as e:
        print(f"Error in crop kfold: {e}")
        report["crop_recommendation"]["kfold_error"] = str(e)
        traceback.print_exc()

    try:
        print(" - noise robustness")
        report["crop_recommendation"]["noise_robustness"] = evaluate_noise_robustness()
    except Exception as e:
        print(f"Error in crop noise robustness: {e}")
        report["crop_recommendation"]["noise_robustness_error"] = str(e)
        traceback.print_exc()

    try:
        print(" - baseline comparison")
        report["crop_recommendation"]["baseline_comparison"] = evaluate_baseline_comparison()
    except Exception as e:
        print(f"Error in crop baseline comparison: {e}")
        report["crop_recommendation"]["baseline_comparison_error"] = str(e)
        traceback.print_exc()

    try:
        print(" - shap analysis")
        report["crop_recommendation"]["shap_analysis"] = evaluate_shap_analysis()
    except Exception as e:
        print(f"Error in crop shap analysis: {e}")
        report["crop_recommendation"]["shap_analysis_error"] = str(e)
        traceback.print_exc()

    # 2. Demand Threshold Module
    print("Running Demand Threshold Module...")
    try:
        print(" - demand model evaluation")
        report["demand_threshold"]["model_evaluation"] = evaluate_demand_model()
    except Exception as e:
        print(f"Error in demand model evaluation: {e}")
        report["demand_threshold"]["model_evaluation_error"] = str(e)
        traceback.print_exc()

    try:
        print(" - demand baselines")
        report["demand_threshold"]["baselines"] = evaluate_demand_baselines()
    except Exception as e:
        print(f"Error in demand baselines: {e}")
        report["demand_threshold"]["baselines_error"] = str(e)
        traceback.print_exc()

    # 3. Ablation Study
    print("Running Ablation Study...")
    try:
        print(" - full ablation study")
        report["ablation_study"]["full"] = run_full_ablation_study()
    except Exception as e:
        print(f"Error in ablation study: {e}")
        report["ablation_study"]["error"] = str(e)
        traceback.print_exc()
        
    # 4. Publication Summary
    print("Generating Publication Summary...")
    try:
        summary_lines = [
            "============================================================",
            "                   PUBLICATION SUMMARY                      ",
            "============================================================",
            "",
            "1. COMPARATIVE ANALYSIS",
            "-----------------------",
            "Crop Recommendation Models: RF vs SVM vs LR vs KNN",
            "Demand Models: GBR vs LinReg vs DT vs SVR",
            "See JSON report for detailed metrics (accuracy, precision, recall, f1, rmse, mae, r2).",
            "",
            "2. STATISTICAL SIGNIFICANCE",
            "---------------------------",
            "Baseline comparison for Crop Models includes t-test p-values.",
            "Review 'baseline_comparison' in report for significance levels.",
            "",
            "3. ROBUSTNESS ANALYSIS",
            "----------------------",
            "Noise degradation curves simulated across different noise levels.",
            "Details in 'noise_robustness' section.",
            "",
            "4. ABLATION STUDY FINDINGS",
            "--------------------------",
            "Key findings on feature importance and module contribution.",
            "Details in 'ablation_study' section.",
            "",
            "5. FEATURE IMPORTANCE",
            "---------------------",
            "SHAP-based ranking and explanations.",
            "Details in 'shap_analysis' section.",
            "============================================================"
        ]
        
        report["publication_summary"] = "\n".join(summary_lines)
        print(report["publication_summary"])
    except Exception as e:
        print(f"Error generating summary: {e}")
        report["publication_summary"] = f"Error generating summary: {e}"

    # 5. Save JSON Report
    print("Saving report to evaluation_report.json...")
    try:
        serializable_report = make_serializable(report)
        with open("evaluation_report.json", "w", encoding="ascii", errors="replace") as f:
            json.dump(serializable_report, f, indent=4)
        print("Report saved successfully.")
    except Exception as e:
        print(f"Error saving report: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_evaluations()
