#add step wise importance
import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
from sklearn.metrics import jaccard_score
from sklearn.feature_extraction.text import CountVectorizer

# --- Similarity Functions ---
def rouge_l_score(a, b):
    return SequenceMatcher(None, a, b).ratio()

def jaccard_sim(a, b):
    vect = CountVectorizer(binary=True).fit([a, b])
    X = vect.transform([a, b]).toarray()
    return jaccard_score(X[0], X[1], average='macro')

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

        results.append({
            "filename": os.path.basename(filepath),
            "step": idx + 1,
            "tool": tool,
            "rouge_l": round(rouge_sim, 3),
            "jaccard": round(jacc_sim, 3),
            "observation": obs[:200]
        })

    if not results:
        debug_logs.append((os.path.basename(filepath), "No valid observation steps"))
    else:
        debug_logs.append((os.path.basename(filepath), f"Processed {len(results)} steps"))

    return results

# --- Main pipeline with per-file plots ---
def run_credit_assignment_filewise_only(input_folder):
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
            df_file["credit_score"] = 0.7 * df_file["rouge_l"] + 0.3 * df_file["jaccard"]
            total_score = df_file["credit_score"].sum()
            df_file["normalized_credit"] = df_file["credit_score"] / total_score if total_score > 0 else 0

            combined_results.extend(df_file.to_dict(orient="records"))

            file_credit = df_file.groupby("tool")["normalized_credit"].sum().reset_index()
            file_credit["source_file"] = os.path.basename(file)
            all_filewise_credits.append(file_credit)

            # Save per-file plot
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(file_credit["tool"], file_credit["normalized_credit"])
            ax.set_title(f"Tool Credit: {os.path.basename(file)}")
            ax.set_ylabel("Normalized Credit")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plot_name = f"{os.path.splitext(os.path.basename(file))[0]}_credit_plot.png"
            plt.savefig(plot_name)
            plt.close()
            print(f"> Saved plot: {plot_name}")

        except Exception as e:
            debug_logs.append((os.path.basename(file), f"Exception: {str(e)}"))

    if not combined_results:
        return None, None, pd.DataFrame(debug_logs, columns=["filename", "status"])

    df_all = pd.DataFrame(combined_results)
    filewise_credit_df = pd.concat(all_filewise_credits, ignore_index=True)
    debug_df = pd.DataFrame(debug_logs, columns=["filename", "status"])
    return df_all, filewise_credit_df, debug_df

# --- Run It ---
if __name__ == "__main__":
    input_dir = "/Users/nishugarg/Documents/research/30June/factly-main/factly/traj_store"  

    df_all_steps, df_filewise_credit, df_debug_logs = run_credit_assignment_filewise_only(input_dir)

    if df_all_steps is not None:
        df_all_steps.to_csv("credit_assigned_steps.csv", index=False)
        df_filewise_credit.to_csv("tool_credit_summary_with_filenames.csv", index=False)
        df_debug_logs.to_csv("credit_assignment_debug_log.csv", index=False)

        print("\n Output Saved:")
        print("credit_assigned_steps.csv")
        print("tool_credit_summary_with_filenames.csv")
        print("credit_assignment_debug_log.csv")
        print("And *_credit_plot.png per file")
    else:
        print("\n No valid data to process.")
