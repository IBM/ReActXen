import argparse
import os
import json
import pandas as pd
import numpy as np
from evaluation import credit_metrics, ablation, plotting

def load_quality_labels(df_steps, input_dir):
    """Attempt to load final quality labels from the JSON files."""
    quality_labels = {}
    for filename in df_steps["filename"].unique():
        json_path = os.path.join(input_dir, filename)
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    # Try common keys
                    score = data.get("factuality_score") or data.get("grounded_score") or data.get("final_score")
                    if score is not None:
                        quality_labels[filename] = float(score)
            except Exception:
                pass
    
    if not quality_labels:
        return df_steps, False
        
    df_steps["final_quality"] = df_steps["filename"].map(quality_labels)
    return df_steps, True

def determine_winner(lex_val, sem_val, metric_name):
    """Return the winning method name. For ablation, higher degradation is better."""
    if np.isnan(lex_val) or np.isnan(sem_val):
        return "N/A"
    
    # Most metrics we define (degradation, auc) are "higher is better"
    # For entropy, top1 mass, etc. there is no clear "winner" without context
    if sem_val > lex_val:
        return "semantic"
    elif lex_val > sem_val:
        return "lexical"
    return "tie"

def main():
    parser = argparse.ArgumentParser(description="Benchmark Credit Assignment Methods")
    parser.add_argument("--traj_dir", type=str, required=True, help="Path to trajectory JSON files")
    parser.add_argument("--csv_dir", type=str, default=".", help="Path to credit assignment CSV outputs")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save benchmark outputs")
    parser.add_argument("--top_k", type=int, nargs='+', default=[1, 3, 5], help="Top K steps for ablation")
    parser.add_argument("--human_eval_sample_size", type=int, default=100, help="Number of steps to sample for human evaluation")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    steps_csv = os.path.join(args.csv_dir, "credit_assigned_steps.csv")
    tools_csv = os.path.join(args.csv_dir, "tool_credit_summary_with_filenames.csv")
    
    if not os.path.exists(steps_csv):
        print(f"Error: Could not find {steps_csv}. Please run credit_assignment.py first.")
        return
        
    df_steps = pd.read_csv(steps_csv)
    df_tools = pd.read_csv(tools_csv) if os.path.exists(tools_csv) else pd.DataFrame()
    
    print(f"Loaded {len(df_steps)} steps from {len(df_steps['filename'].unique())} files.")
    
    # Attempt to load final quality
    df_steps, has_quality = load_quality_labels(df_steps, args.traj_dir)
    
    # --- Metrics ---
    # Redundancy & Concentration
    means = credit_metrics.evaluate_redundancy_and_concentration(df_steps)
    
    # Tool Distribution Comparison
    tool_comp = credit_metrics.compare_tool_distributions(
        df_tools, "tool_credit_lexical", "tool_credit_semantic"
    )
    
    # Ablation
    print("Running step ablation proxy evaluation...")
    df_ablation, auc_results = ablation.run_ablation(df_steps, args.traj_dir, args.top_k)
    
    # Organize Summary
    summary = {
        "num_files": len(df_steps["filename"].unique()),
        "num_steps": len(df_steps),
        "final_quality_labels_available": has_quality,
        "lexical": {
            "mean_entropy": means["lexical_credit_norm"]["mean_entropy"],
            "mean_gini": means["lexical_credit_norm"]["mean_gini"],
            "mean_top1_mass": means["lexical_credit_norm"]["mean_top1_mass"],
            "mean_top3_mass": means["lexical_credit_norm"]["mean_top3_mass"],
            "mean_effective_num_steps": means["lexical_credit_norm"]["mean_effective_num_steps"],
            "ablation_auc": auc_results.get("lexical", 0.0)
        },
        "semantic": {
            "mean_entropy": means["semantic_credit_norm"]["mean_entropy"],
            "mean_gini": means["semantic_credit_norm"]["mean_gini"],
            "mean_top1_mass": means["semantic_credit_norm"]["mean_top1_mass"],
            "mean_top3_mass": means["semantic_credit_norm"]["mean_top3_mass"],
            "mean_effective_num_steps": means["semantic_credit_norm"]["mean_effective_num_steps"],
            "ablation_auc": auc_results.get("semantic", 0.0)
        },
        "tool_distribution_comparison": tool_comp,
        "winner_by_metric": {
            "ablation_auc": determine_winner(auc_results.get("lexical", 0.0), auc_results.get("semantic", 0.0), "ablation_auc")
        }
    }
    
    # Populate ablation @ k
    for method in ["lexical", "semantic"]:
        if not df_ablation.empty:
            method_df = df_ablation[df_ablation["method"] == method]
            for k in args.top_k:
                k_mean = method_df[method_df["k"] == k]["degradation"].mean()
                if not np.isnan(k_mean):
                    summary[method][f"ablation_degradation_at_{k}"] = float(k_mean)
                else:
                    summary[method][f"ablation_degradation_at_{k}"] = 0.0
                    
    for k in args.top_k:
        lex_val = summary["lexical"].get(f"ablation_degradation_at_{k}", 0.0)
        sem_val = summary["semantic"].get(f"ablation_degradation_at_{k}", 0.0)
        summary["winner_by_metric"][f"ablation_at_{k}"] = determine_winner(lex_val, sem_val, f"ablation_at_{k}")
    
    # Optional correlation if quality labels exist
    if has_quality:
        file_credits = df_steps.groupby("filename").agg({
            "lexical_credit_norm": "mean", # This is simplistic, but works for file-level
            "semantic_credit_norm": "mean",
            "final_quality": "first"
        }).reset_index()
        
        lex_corr = credit_metrics.compute_correlations(file_credits, "lexical_credit_norm", "final_quality")
        sem_corr = credit_metrics.compute_correlations(file_credits, "semantic_credit_norm", "final_quality")
        
        summary["lexical"]["correlation_to_final_quality"] = lex_corr
        summary["semantic"]["correlation_to_final_quality"] = sem_corr
    
    # --- Saves and Outputs ---
    
    # 1. Benchmark Summary
    summary_path = os.path.join(args.output_dir, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    # 2. Raw Comparisons
    df_steps.to_csv(os.path.join(args.output_dir, "per_step_credit_comparison.csv"), index=False)
    if not df_tools.empty:
        df_tools.to_csv(os.path.join(args.output_dir, "per_tool_credit_comparison.csv"), index=False)
    if not df_ablation.empty:
        df_ablation.to_csv(os.path.join(args.output_dir, "ablation_results.csv"), index=False)
        
    # 3. Human Eval Sample
    if args.human_eval_sample_size > 0:
        sample_size = min(args.human_eval_sample_size, len(df_steps))
        cols = [
            "filename", "step", "tool", "observation", 
            "rouge_l", "jaccard", "semantic_sim", 
            "lexical_credit_score", "semantic_credit_score",
            "lexical_credit_norm", "semantic_credit_norm"
        ]
        # Only keep existing columns
        cols = [c for c in cols if c in df_steps.columns]
        df_sample = df_steps[cols].sample(sample_size, random_state=42)
        df_sample.to_csv(os.path.join(args.output_dir, "human_eval_sample.csv"), index=False)
        
    # 4. Plots
    plotting.plot_credit_distributions(df_steps, os.path.join(args.output_dir, "credit_distribution_plots.png"))
    if not df_tools.empty:
        plotting.plot_tool_distributions(df_tools, os.path.join(args.output_dir, "tool_credit_distribution_plots.png"))
    if not df_ablation.empty:
        plotting.plot_ablation_curves(df_ablation, os.path.join(args.output_dir, "ablation_curves.png"))

    print(f"Benchmark completed successfully! Outputs saved to {args.output_dir}")

if __name__ == "__main__":
    main()
