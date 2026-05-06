import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr, kendalltau

def calculate_entropy(scores):
    scores = np.array(scores)
    scores = scores[scores > 0]
    if len(scores) == 0:
        return 0.0
    scores = scores / scores.sum()
    return -np.sum(scores * np.log2(scores))

def calculate_gini(scores):
    scores = np.array(scores)
    if len(scores) == 0:
        return 0.0
    scores = np.sort(scores)
    n = len(scores)
    cum_scores = np.cumsum(scores)
    if cum_scores[-1] == 0:
        return 0.0
    return (n + 1 - 2 * np.sum(cum_scores) / cum_scores[-1]) / n

def calculate_top_k_mass(scores, k):
    scores = np.array(scores)
    if len(scores) == 0:
        return 0.0
    scores = np.sort(scores)[::-1]
    return float(np.sum(scores[:k]))

def calculate_effective_num_steps(scores):
    # Effective number of steps = 2^entropy
    ent = calculate_entropy(scores)
    return float(2 ** ent)

def compute_correlations(df, score_col, target_col):
    df = df.dropna(subset=[score_col, target_col])
    if len(df) < 2:
        return {"pearson": 0.0, "spearman": 0.0, "kendall": 0.0}
    
    x = df[score_col].values
    y = df[target_col].values
    
    # Avoid constant arrays which return NaN
    if np.std(x) == 0 or np.std(y) == 0:
        return {"pearson": 0.0, "spearman": 0.0, "kendall": 0.0}
    
    p, _ = pearsonr(x, y)
    s, _ = spearmanr(x, y)
    k, _ = kendalltau(x, y)
    
    return {
        "pearson": float(p) if not np.isnan(p) else 0.0,
        "spearman": float(s) if not np.isnan(s) else 0.0,
        "kendall": float(k) if not np.isnan(k) else 0.0
    }

def cosine_sim(a, b):
    num = np.dot(a, b)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den == 0:
        return 0.0
    return float(num / den)

def compare_tool_distributions(tool_df, col1, col2):
    """
    Expects tool_df with unique tools and two columns with their normalized credits.
    """
    if len(tool_df) == 0:
        return {"cosine_sim": 0.0, "js_divergence": 0.0, "rank_correlation": 0.0}
    
    v1 = tool_df[col1].fillna(0.0).values
    v2 = tool_df[col2].fillna(0.0).values
    
    # Normalize to probability distributions for JS Divergence
    s1 = v1.sum()
    s2 = v2.sum()
    
    p1 = v1 / s1 if s1 > 0 else np.zeros_like(v1)
    p2 = v2 / s2 if s2 > 0 else np.zeros_like(v2)
    
    # Jensen-Shannon Divergence
    js_div = jensenshannon(p1, p2)
    if np.isnan(js_div):
        js_div = 0.0
        
    # Cosine Similarity
    c_sim = cosine_sim(v1, v2)
    
    # Rank Correlation (Spearman)
    if len(tool_df) < 2:
        rank_corr = 0.0
    else:
        rank_corr, _ = spearmanr(v1, v2)
        if np.isnan(rank_corr):
            rank_corr = 0.0
            
    return {
        "cosine_sim": float(c_sim),
        "js_divergence": float(js_div),
        "rank_correlation": float(rank_corr)
    }

def evaluate_redundancy_and_concentration(df, groupby_col="filename", score_cols=["lexical_credit_norm", "semantic_credit_norm"]):
    results = {col: {"entropy": [], "gini": [], "top1": [], "top3": [], "eff_steps": []} for col in score_cols}
    
    for _, group in df.groupby(groupby_col):
        for col in score_cols:
            scores = group[col].values
            results[col]["entropy"].append(calculate_entropy(scores))
            results[col]["gini"].append(calculate_gini(scores))
            results[col]["top1"].append(calculate_top_k_mass(scores, 1))
            results[col]["top3"].append(calculate_top_k_mass(scores, 3))
            results[col]["eff_steps"].append(calculate_effective_num_steps(scores))
            
    # Aggregate means
    means = {}
    for col in score_cols:
        means[col] = {
            "mean_entropy": float(np.mean(results[col]["entropy"])) if results[col]["entropy"] else 0.0,
            "mean_gini": float(np.mean(results[col]["gini"])) if results[col]["gini"] else 0.0,
            "mean_top1_mass": float(np.mean(results[col]["top1"])) if results[col]["top1"] else 0.0,
            "mean_top3_mass": float(np.mean(results[col]["top3"])) if results[col]["top3"] else 0.0,
            "mean_effective_num_steps": float(np.mean(results[col]["eff_steps"])) if results[col]["eff_steps"] else 0.0,
        }
    return means
