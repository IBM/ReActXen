import os
import json
import numpy as np
import pandas as pd
import sys

# We need to import semantic_similarity and find_final_output from credit_assignment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from credit_assignment import semantic_similarity, find_final_output

def load_traj(filepath):
    if not os.path.exists(filepath):
        return None, []
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    final_output = find_final_output(data)
    if not final_output:
        return None, []
        
    final_passages = [p["text"] for p in final_output.get("supporting_passages", []) if "text" in p]
    final_justification = final_output.get("justification", "")
    final_texts = final_passages + ([final_justification] if final_justification else [])
    
    steps = data.get("trajectroy_log", [])
    obs_list = []
    for step in steps:
        obs = step.get("observation", "")
        if obs:
            obs_list.append(obs)
            
    return final_texts, obs_list

def calculate_max_semantic_sim(obs_list, final_texts):
    if not obs_list or not final_texts:
        return 0.0
    combined_obs = " ".join(obs_list)
    return max([semantic_similarity(combined_obs, t) for t in final_texts])

def run_ablation(df_steps, traj_dir, top_k_list=[1, 3, 5]):
    results = []
    
    for filename, group in df_steps.groupby("filename"):
        traj_path = os.path.join(traj_dir, filename)
        final_texts, all_obs = load_traj(traj_path)
        
        if not final_texts or not all_obs:
            continue
            
        # We assume the steps in df match the non-empty observations in order
        if len(all_obs) != len(group):
            print(f"Warning: Obs count mismatch for {filename}: {len(all_obs)} vs {len(group)}")
            continue
            
        base_sim = calculate_max_semantic_sim(all_obs, final_texts)
        
        for col in ["lexical_credit_norm", "semantic_credit_norm"]:
            sorted_indices = np.argsort(group[col].values)[::-1]
            
            for k in top_k_list:
                if k >= len(all_obs):
                    rem_sim = 0.0
                else:
                    remove_idx = sorted_indices[:k]
                    remaining_obs = [obs for i, obs in enumerate(all_obs) if i not in remove_idx]
                    rem_sim = calculate_max_semantic_sim(remaining_obs, final_texts)
                    
                deg = base_sim - rem_sim
                results.append({
                    "filename": filename,
                    "method": "lexical" if "lexical" in col else "semantic",
                    "k": k,
                    "degradation": float(deg)
                })
                
    df_res = pd.DataFrame(results)
    
    auc_results = {}
    if not df_res.empty:
        for method in ["lexical", "semantic"]:
            method_df = df_res[df_res["method"] == method]
            if method_df.empty:
                auc_results[method] = 0.0
                continue
            means = method_df.groupby("k")["degradation"].mean()
            # Trapezoidal AUC
            k_vals = list(means.index)
            deg_vals = list(means.values)
            auc = 0.0
            if len(k_vals) > 1:
                for i in range(len(k_vals) - 1):
                    w = k_vals[i+1] - k_vals[i]
                    h = (deg_vals[i] + deg_vals[i+1]) / 2.0
                    auc += w * h
            auc_results[method] = float(auc)
    else:
        auc_results = {"lexical": 0.0, "semantic": 0.0}
        
    return df_res, auc_results
