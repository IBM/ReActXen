import re
import json_repair
import numpy as np

filename = "../result/llm_generated_asset_summary_result.json"  # or .txt if applicable

try:
    decoded_object = json_repair.from_file(filename)
except:
    pass

scores = []

total_s = 0
for key, value in decoded_object.items():
    try:
        if "groundedness_score" in value:
            pattern = re.compile(r'"groundedness_score"\s*:\s*([0-9.]+)')
            matches = pattern.findall(value)
            unique_scores = list(set(float(m) for m in matches)) 
            if len(unique_scores) > 0:
                scores.append(unique_scores[0])            
    except Exception as e:
        print(f"Failed to parse for key {key}: {e}")


# Assuming `scores` is your list of groundedness_score values (as floats)
scores_array = np.array(scores)

# Basic statistics
mean_score = np.mean(scores_array)
median_score = np.median(scores_array)
std_dev = np.std(scores_array)
min_score = np.min(scores_array)
max_score = np.max(scores_array)

# Histogram distribution (e.g., how many in [0.0–0.2], [0.2–0.4], ...)
hist, bin_edges = np.histogram(scores_array, bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
hist_distribution = dict(zip([f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(hist))], hist))

# Print stats
print(f"Total Evaluations: {len(scores_array)}")
print(f"Mean Score: {mean_score:.3f}")
print(f"Median Score: {median_score:.3f}")
print(f"Standard Deviation: {std_dev:.3f}")
print(f"Min Score: {min_score}")
print(f"Max Score: {max_score}")
print("Distribution Histogram (bin range → count):")
for k, v in hist_distribution.items():
    print(f"  {k}: {v}")
