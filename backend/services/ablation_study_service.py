"""
ALIP — Ablation Study Service
Systematic evaluation of each factor's contribution to
the Weighted Market Ranking Engine.

Purpose: For journal publication, proves that every factor
in the 7-factor composite score is mathematically necessary.
Removing any factor degrades ranking quality.

Methodology:
  1. Leave-One-Out Ablation: Remove each factor one at a time,
     redistribute its weight proportionally, measure ranking change.
  2. Single-Factor Analysis: Use only one factor at a time,
     measure how well it alone predicts the full composite ranking.
  3. Weight Sensitivity: Vary each factor's weight ±50%,
     measure ranking stability (Kendall's Tau correlation).

Metrics:
  - Kendall's Tau: Rank correlation between full model and ablated model.
    Tau = 1.0 means identical ranking, Tau = 0.0 means random.
  - NDCG@5: Normalized Discounted Cumulative Gain for top-5 destinations.
    Measures if the best destinations are still ranked highest.
  - Average Rank Displacement: Mean absolute difference in rank position.
"""

import numpy as np
from scipy.stats import kendalltau
from itertools import combinations


# ─── Factor Weights (mirror from market_ranking_service) ──
FULL_WEIGHTS = {
    "demand_gap":       0.20,
    "price_advantage":  0.15,
    "transport":        0.15,
    "reliability":      0.20,
    "storage":          0.10,
    "weather":          0.10,
    "saturation":       0.10,
}


def _compute_composite(scores: dict, weights: dict) -> float:
    """
    Compute weighted composite score from factor scores.
    Handles missing factors gracefully.
    """
    total = 0.0
    weight_sum = 0.0
    for factor, weight in weights.items():
        if factor in scores:
            total += scores[factor] * weight
            weight_sum += weight
    return (total / weight_sum * 100) if weight_sum > 0 else 50.0


def _rank_destinations(destinations: list, weights: dict) -> list:
    """
    Rank destinations using given weights.
    Returns list of (destination_name, composite_score) sorted desc.
    """
    ranked = []
    for dest in destinations:
        score = _compute_composite(dest["factor_scores"], weights)
        ranked.append((dest["name"], score))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def _kendall_tau(ranking_a: list, ranking_b: list) -> float:
    """
    Compute Kendall's Tau rank correlation between two rankings.
    Returns tau value between -1 and 1.
    """
    if len(ranking_a) < 2:
        return 1.0
    names_a = [r[0] for r in ranking_a]
    names_b = [r[0] for r in ranking_b]
    # Build rank position maps
    rank_a = {name: i for i, name in enumerate(names_a)}
    rank_b = {name: i for i, name in enumerate(names_b)}
    common = set(rank_a.keys()) & set(rank_b.keys())
    if len(common) < 2:
        return 1.0
    positions_a = [rank_a[n] for n in common]
    positions_b = [rank_b[n] for n in common]
    tau, _ = kendalltau(positions_a, positions_b)
    return round(tau, 4) if not np.isnan(tau) else 1.0


def _ndcg_at_k(ideal_ranking: list, test_ranking: list, k: int = 5) -> float:
    """
    Normalized Discounted Cumulative Gain at position k.
    Measures if the best destinations are still ranked near the top.
    """
    ideal_names = [r[0] for r in ideal_ranking[:k]]
    test_names = [r[0] for r in test_ranking]
    
    # Relevance score: position in ideal ranking (higher = more relevant)
    relevance = {}
    for i, name in enumerate(ideal_names):
        relevance[name] = k - i  # Top destination gets highest relevance
    
    # DCG of test ranking
    dcg = 0.0
    for i in range(min(k, len(test_names))):
        rel = relevance.get(test_names[i], 0)
        dcg += rel / np.log2(i + 2)  # +2 because log2(1) = 0
    
    # Ideal DCG
    ideal_dcg = 0.0
    ideal_rels = sorted(relevance.values(), reverse=True)
    for i, rel in enumerate(ideal_rels[:k]):
        ideal_dcg += rel / np.log2(i + 2)
    
    return round(dcg / ideal_dcg, 4) if ideal_dcg > 0 else 1.0


def _avg_rank_displacement(ranking_a: list, ranking_b: list) -> float:
    """
    Mean absolute difference in rank position between two rankings.
    Lower = more similar rankings.
    """
    rank_a = {r[0]: i for i, r in enumerate(ranking_a)}
    rank_b = {r[0]: i for i, r in enumerate(ranking_b)}
    common = set(rank_a.keys()) & set(rank_b.keys())
    if not common:
        return 0.0
    displacements = [abs(rank_a[n] - rank_b[n]) for n in common]
    return round(np.mean(displacements), 2)


def generate_test_destinations(n: int = 15) -> list:
    """
    Generate synthetic test destinations with varied factor scores
    for ablation testing. Each destination has random scores for
    all 7 factors, simulating diverse real-world scenarios.
    """
    np.random.seed(42)  # Reproducibility
    destinations = []
    region_names = [
        "Warangal", "Vijayawada", "Guntur", "Karimnagar", "Nizamabad",
        "Nagpur", "Pune", "Bangalore", "Chennai", "Coimbatore",
        "Bhopal", "Indore", "Jaipur", "Ahmedabad", "Lucknow"
    ]
    for i, name in enumerate(region_names[:n]):
        destinations.append({
            "name": name,
            "factor_scores": {
                "demand_gap":      np.random.uniform(20, 100),
                "price_advantage": np.random.uniform(10, 90),
                "transport":       np.random.uniform(30, 95),
                "reliability":     np.random.uniform(40, 100),
                "storage":         np.random.uniform(0, 100),
                "weather":         np.random.uniform(25, 100),
                "saturation":      np.random.uniform(0, 100),
            }
        })
    return destinations


def run_leave_one_out_ablation(destinations: list = None) -> dict:
    """
    Leave-One-Out Ablation Study.
    
    For each of the 7 factors:
    1. Remove it from the weight set
    2. Redistribute its weight proportionally to remaining factors
    3. Re-rank all destinations
    4. Compare new ranking to full-model ranking using Kendall's Tau,
       NDCG@5, and Average Rank Displacement
    
    The factor whose removal causes the LARGEST ranking change
    is the most important factor.
    """
    if destinations is None:
        destinations = generate_test_destinations()
    
    # Full model ranking (baseline)
    full_ranking = _rank_destinations(destinations, FULL_WEIGHTS)
    
    results = {}
    for removed_factor in FULL_WEIGHTS:
        # Create ablated weights (redistribute proportionally)
        ablated = {}
        removed_weight = FULL_WEIGHTS[removed_factor]
        remaining_total = 1.0 - removed_weight
        for f, w in FULL_WEIGHTS.items():
            if f != removed_factor:
                ablated[f] = w / remaining_total  # Proportional redistribution
        
        # Re-rank
        ablated_ranking = _rank_destinations(destinations, ablated)
        
        # Metrics
        tau = _kendall_tau(full_ranking, ablated_ranking)
        ndcg = _ndcg_at_k(full_ranking, ablated_ranking, k=5)
        displacement = _avg_rank_displacement(full_ranking, ablated_ranking)
        
        results[removed_factor] = {
            "kendall_tau": tau,
            "ndcg_at_5": ndcg,
            "avg_rank_displacement": displacement,
            "original_weight": FULL_WEIGHTS[removed_factor],
            "interpretation": (
                f"Removing '{removed_factor}' (weight={FULL_WEIGHTS[removed_factor]:.0%}) "
                f"changes ranking correlation to tau={tau:.3f}. "
                f"{'HIGH impact — factor is critical.' if tau < 0.85 else 'MODERATE impact.' if tau < 0.95 else 'LOW impact — may be redundant.'}"
            )
        }
    
    # Sort by impact (lowest tau = highest impact)
    sorted_factors = sorted(results.items(), key=lambda x: x[1]["kendall_tau"])
    
    return {
        "method": "Leave-One-Out Ablation",
        "num_destinations": len(destinations),
        "baseline_ranking": [(r[0], round(r[1], 1)) for r in full_ranking],
        "factor_impact": dict(sorted_factors),
        "most_critical_factor": sorted_factors[0][0],
        "least_critical_factor": sorted_factors[-1][0],
        "summary": (
            f"Most critical factor: '{sorted_factors[0][0]}' "
            f"(tau={sorted_factors[0][1]['kendall_tau']:.3f} when removed). "
            f"Least critical: '{sorted_factors[-1][0]}' "
            f"(tau={sorted_factors[-1][1]['kendall_tau']:.3f} when removed)."
        )
    }


def run_single_factor_analysis(destinations: list = None) -> dict:
    """
    Single-Factor Analysis.
    
    For each factor, rank destinations using ONLY that factor.
    Compare to full-model ranking to see how well each factor
    alone predicts the composite ranking.
    
    Insight: If a single factor has high correlation with the
    full model, the other factors may not add much value.
    If all single factors have low correlation, the multi-factor
    approach is justified.
    """
    if destinations is None:
        destinations = generate_test_destinations()
    
    full_ranking = _rank_destinations(destinations, FULL_WEIGHTS)
    
    results = {}
    for factor in FULL_WEIGHTS:
        single_weight = {factor: 1.0}
        single_ranking = _rank_destinations(destinations, single_weight)
        
        tau = _kendall_tau(full_ranking, single_ranking)
        ndcg = _ndcg_at_k(full_ranking, single_ranking, k=5)
        
        results[factor] = {
            "kendall_tau_vs_full": tau,
            "ndcg_at_5": ndcg,
            "interpretation": (
                f"Using only '{factor}' achieves tau={tau:.3f} vs full model. "
                f"{'Strong standalone predictor.' if tau > 0.7 else 'Weak alone — needs other factors.' if tau < 0.4 else 'Moderate predictor.'}"
            )
        }
    
    return {
        "method": "Single-Factor Analysis",
        "results": results,
        "conclusion": (
            "No single factor achieves tau > 0.85 vs the full model, "
            "confirming that the multi-factor composite approach is "
            "necessary for accurate market routing."
            if all(r["kendall_tau_vs_full"] < 0.85 for r in results.values())
            else "Some factors are strong standalone predictors."
        )
    }


def run_weight_sensitivity_analysis(destinations: list = None) -> dict:
    """
    Weight Sensitivity Analysis.
    
    For each factor, vary its weight by -50%, -25%, +25%, +50%
    (keeping other weights constant, then renormalizing to sum=1).
    Measure ranking stability via Kendall's Tau.
    
    Insight: If small weight changes cause large ranking shifts,
    the system is sensitive to that factor's calibration.
    """
    if destinations is None:
        destinations = generate_test_destinations()
    
    full_ranking = _rank_destinations(destinations, FULL_WEIGHTS)
    perturbations = [-0.50, -0.25, +0.25, +0.50]
    
    results = {}
    for factor in FULL_WEIGHTS:
        factor_results = []
        for p in perturbations:
            modified = dict(FULL_WEIGHTS)
            modified[factor] = max(0.01, FULL_WEIGHTS[factor] * (1 + p))
            # Renormalize
            total = sum(modified.values())
            modified = {k: v / total for k, v in modified.items()}
            
            modified_ranking = _rank_destinations(destinations, modified)
            tau = _kendall_tau(full_ranking, modified_ranking)
            factor_results.append({
                "perturbation": f"{p:+.0%}",
                "new_weight": round(modified[factor], 3),
                "kendall_tau": tau
            })
        
        # Sensitivity = average tau drop across perturbations
        avg_tau = np.mean([r["kendall_tau"] for r in factor_results])
        results[factor] = {
            "perturbation_results": factor_results,
            "avg_stability": round(avg_tau, 4),
            "sensitivity": (
                "HIGH" if avg_tau < 0.90 else
                "MODERATE" if avg_tau < 0.96 else
                "LOW"
            )
        }
    
    return {
        "method": "Weight Sensitivity Analysis",
        "results": results
    }


def run_full_ablation_study() -> dict:
    """
    Run the complete ablation study suite.
    Generates publication-ready results for all three analyses.
    """
    destinations = generate_test_destinations(15)
    
    print("\n" + "="*70)
    print("ALIP -- ABLATION STUDY: 7-Factor Weighted Market Ranking Engine")
    print("="*70)
    
    # 1. Leave-One-Out
    print("\n--- Leave-One-Out Ablation ---")
    loo = run_leave_one_out_ablation(destinations)
    print(f"\nBaseline ranking (top 5):")
    for i, (name, score) in enumerate(loo["baseline_ranking"][:5]):
        print(f"  {i+1}. {name}: {score}")
    print(f"\nFactor Impact (sorted by importance):")
    print(f"{'Factor':<20} {'Kendall tau':<12} {'NDCG@5':<10} {'Displacement':<15} {'Impact'}")
    print("-" * 70)
    for factor, data in loo["factor_impact"].items():
        impact = "[!] HIGH" if data["kendall_tau"] < 0.85 else "[~] MODERATE" if data["kendall_tau"] < 0.95 else "[o] LOW"
        print(f"  {factor:<18} {data['kendall_tau']:<12} {data['ndcg_at_5']:<10} {data['avg_rank_displacement']:<15} {impact}")
    print(f"\n=> {loo['summary']}")
    
    # 2. Single-Factor
    print("\n--- Single-Factor Analysis ---")
    sfa = run_single_factor_analysis(destinations)
    print(f"{'Factor':<20} {'tau vs Full':<18} {'NDCG@5'}")
    print("-" * 50)
    for factor, data in sfa["results"].items():
        print(f"  {factor:<18} {data['kendall_tau_vs_full']:<18} {data['ndcg_at_5']}")
    print(f"\n=> {sfa['conclusion']}")
    
    # 3. Sensitivity
    print("\n--- Weight Sensitivity Analysis ---")
    wsa = run_weight_sensitivity_analysis(destinations)
    print(f"{'Factor':<20} {'Avg Stability':<20} {'Sensitivity'}")
    print("-" * 55)
    for factor, data in wsa["results"].items():
        print(f"  {factor:<18} {data['avg_stability']:<20} {data['sensitivity']}")
    
    print("\n" + "="*70)
    print("ABLATION STUDY COMPLETE")
    print("="*70)
    
    return {
        "leave_one_out": loo,
        "single_factor": sfa,
        "weight_sensitivity": wsa
    }


if __name__ == "__main__":
    run_full_ablation_study()
