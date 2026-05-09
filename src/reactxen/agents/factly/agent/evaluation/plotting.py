import matplotlib.pyplot as plt
import os
import seaborn as sns

def plot_credit_distributions(df_steps, out_path):
    if df_steps.empty:
        return
    plt.figure(figsize=(10, 5))
    sns.histplot(df_steps["lexical_credit_norm"], color="blue", label="Lexical", kde=True, alpha=0.5)
    sns.histplot(df_steps["semantic_credit_norm"], color="green", label="Semantic", kde=True, alpha=0.5)
    plt.title("Step Credit Distribution Comparison")
    plt.xlabel("Normalized Credit Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_tool_distributions(df_tools, out_path):
    if df_tools.empty:
        return
        
    tools = df_tools["tool"].values
    lex = df_tools["tool_credit_lexical"].values
    sem = df_tools["tool_credit_semantic"].values
    
    x = range(len(tools))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([i - width/2 for i in x], lex, width, label='Lexical', color="blue", alpha=0.7)
    ax.bar([i + width/2 for i in x], sem, width, label='Semantic', color="green", alpha=0.7)
    
    ax.set_ylabel('Total Normalized Credit')
    ax.set_title('Tool-Level Credit Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(tools, rotation=45, ha="right")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_ablation_curves(df_ablation, out_path):
    if df_ablation.empty:
        return
        
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df_ablation, x="k", y="degradation", hue="method", marker="o", errorbar=None)
    plt.title("Step Ablation Degradation (Higher is Better)")
    plt.xlabel("Top-K Steps Removed")
    plt.ylabel("Semantic Similarity Degradation")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
