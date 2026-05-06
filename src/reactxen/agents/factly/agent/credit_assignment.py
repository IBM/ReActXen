#add step wise importance
import sys
import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
from sklearn.metrics import jaccard_score
from sklearn.feature_extraction.text import CountVectorizer


from sentence_transformers import SentenceTransformer, util
_embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# --- Similarity Functions ---
def rouge_l_score(a, b):
    return SequenceMatcher(None, a, b).ratio()

def jaccard_sim(a, b):
    vect = CountVectorizer(binary=True).fit([a, b])
    X = vect.transform([a, b]).toarray()
    return jaccard_score(X[0], X[1], average='macro')

def semantic_similarity(a, b):
    """
    Computes cosine similarity between two texts using sentence embeddings.
    Returns a float in [0, 1].
    """
    embeddings = _embed_model.encode([a, b], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1])
    return float(score)

# --- Robust final output extractor with debug ---
def find_final_output(data):
    for step in data.get("trajectroy_log", []):
        if step.get("action", "").strip() == "Finish":
            for field in ["output", "action_input"]:
                raw = step.get(field, "")
                if isinstance(raw, str):
                    try:
                        raw = json.loads(raw)
                    except Exception:
                        continue

                # Nested action_input check
                if isinstance(raw, dict) and "action_input" in raw:
                    inner = raw["action_input"]
                    if isinstance(inner, str):
                        try:
                            inner = json.loads(inner)
                        except Exception:
                            continue
                    if isinstance(inner, dict) and ("justification" in inner or "supporting_passages" in inner):
                        return inner

                # Flat dict fallback
                if isinstance(raw, dict) and ("justification" in raw or "supporting_passages" in raw):
                    return raw

    # Fallback to top-level keys
    for key in ["Final Answer", "final_output", "final_answer", "output"]:
        if key in data:
            val = data[key]
            if isinstance(val, dict) and "output" in val:
                val = val["output"]
            try:
                parsed = json.loads(val) if isinstance(val, str) else val
                return parsed
            except Exception:
                continue

    return None


# --- Analyze single file with debug ---
def analyze_file_auto_verbose(filepath, debug_logs):
    print(f"\n=== Analyzing {filepath} ===")
    with open(filepath) as f:
        data = json.load(f)

    steps = data.get("trajectroy_log", [])
    final_output = find_final_output(data)

    if not final_output:
        debug_logs.append((os.path.basename(filepath), "Missing final output"))
        return []

    final_passages = [p["text"] for p in final_output.get("supporting_passages", []) if "text" in p]
    final_justification = final_output.get("justification", "")
    final_texts = final_passages + ([final_justification] if final_justification else [])

    print(f"> Extracted {len(final_passages)} supporting passages")
    print(f"> Justification length: {len(final_justification)}")

    if not final_texts:
        debug_logs.append((os.path.basename(filepath), "Finish output empty or missing justification/passages"))
        return []

    results = []
    for idx, step in enumerate(steps):
        obs = step.get("observation", "")
        if not obs:
            continue
        tool = step.get("action", "").split("[")[0].strip()
        rouge_sim = max([rouge_l_score(obs, t) for t in final_texts])
        jacc_sim = max([jaccard_sim(obs, t) for t in final_texts])
        sem_sim = max([semantic_similarity(obs, t) for t in final_texts])

        results.append({
            "filename": os.path.basename(filepath),
            "step": idx + 1,
            "tool": tool,
            "rouge_l": round(rouge_sim, 3),
            "jaccard": round(jacc_sim, 3),
            "semantic_sim": round(sem_sim, 3),
            "observation": obs[:200]
        })

    if not results:
        debug_logs.append((os.path.basename(filepath), "No valid observation steps"))
    else:
        debug_logs.append((os.path.basename(filepath), f"Processed {len(results)} steps"))

    return results

def normalize_scores(scores, eps=1e-12):
    scores = scores.fillna(0.0)
    total = scores.sum()
    if total <= eps:
        return scores * 0.0
    return scores / total

# --- Main pipeline with per-file plots ---
def run_credit_assignment_filewise_only(input_folder, method="semantic", output_folder="credit_assignment_results"):
    os.makedirs(output_folder, exist_ok=True)
    json_files = glob.glob(os.path.join(input_folder, "*_traj_output.json"))
    debug_logs = []
    combined_results = []
    all_filewise_credits = []

    for file in json_files:
        print(f"\n--- Processing: {file} ---")
        try:
            file_results = analyze_file_auto_verbose(file, debug_logs)
            if not file_results:
                continue

            df_file = pd.DataFrame(file_results)
            
            # --- Score Calculations ---
            # lexical_credit_score is the original baseline: 0.7 * ROUGE-L + 0.3 * Jaccard
            df_file["lexical_credit_score"] = 0.7 * df_file["rouge_l"] + 0.3 * df_file["jaccard"]
            
            # semantic_credit_score is the new embedding-based method
            df_file["semantic_credit_score"] = df_file["semantic_sim"]

            # Compute normalized scores side by side
            df_file["lexical_credit_norm"] = normalize_scores(df_file["lexical_credit_score"])
            df_file["semantic_credit_norm"] = normalize_scores(df_file["semantic_credit_score"])

            # Default fallback column based on active method
            if method == "lexical":
                df_file["credit_score"] = df_file["lexical_credit_score"]
                df_file["normalized_credit"] = df_file["lexical_credit_norm"]
            else:
                df_file["credit_score"] = df_file["semantic_credit_score"]
                df_file["normalized_credit"] = df_file["semantic_credit_norm"]

            combined_results.extend(df_file.to_dict(orient="records"))

            # --- Tool-Level Aggregation ---
            file_credit = df_file.groupby("tool").agg(
                tool_credit_lexical=("lexical_credit_norm", "sum"),
                tool_credit_semantic=("semantic_credit_norm", "sum"),
                normalized_credit=("normalized_credit", "sum")
            ).reset_index()
            
            file_credit["source_file"] = os.path.basename(file)
            file_credit["tool_credit"] = file_credit["normalized_credit"] # Backward compatibility mapping
            
            all_filewise_credits.append(file_credit)

            # Save per-file plot
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(file_credit["tool"], file_credit["normalized_credit"])
            ax.set_title(f"Tool Credit: {os.path.basename(file)} ({method.capitalize()})")
            ax.set_ylabel(f"Normalized Credit ({method})")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plot_filename = f"{os.path.splitext(os.path.basename(file))[0]}_{method}_credit_plot.png"
            plot_path = os.path.join(output_folder, plot_filename)
            plt.savefig(plot_path)
            plt.close()
            print(f"> Saved plot: {plot_path}")

        except Exception as e:
            debug_logs.append((os.path.basename(file), f"Exception: {str(e)}"))

    if not combined_results:
        return None, None, pd.DataFrame(debug_logs, columns=["filename", "status"])

    df_all = pd.DataFrame(combined_results)
    filewise_credit_df = pd.concat(all_filewise_credits, ignore_index=True)
    debug_df = pd.DataFrame(debug_logs, columns=["filename", "status"])
    return df_all, filewise_credit_df, debug_df


if __name__ == "__main__":
    #input_dir = "/Users/nishugarg/Documents/research/30June/factly-main/factly/traj_store"  
    input_dir = "../traj_store"  # made the hardcoded path general to work for everyone

    method = "semantic"
    if len(sys.argv) > 1:
        method = sys.argv[1].lower()

    print(f"Running credit assignment with primary method: {method}")
    output_folder = "credit_assignment_results"
    df_all_steps, df_filewise_credit, df_debug_logs = run_credit_assignment_filewise_only(input_dir, method=method, output_folder=output_folder)

    if df_all_steps is not None:
        df_all_steps.to_csv("credit_assigned_steps.csv", index=False)
        df_filewise_credit.to_csv("tool_credit_summary_with_filenames.csv", index=False)
        df_debug_logs.to_csv("credit_assignment_debug_log.csv", index=False)

        print("\n Output Saved:")
        print("credit_assigned_steps.csv")
        print("tool_credit_summary_with_filenames.csv")
        print("credit_assignment_debug_log.csv")
        print(f"And plots saved in: {output_folder}/")
    else:
        print("\n No valid data to process.")