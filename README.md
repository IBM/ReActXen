# ReAct eXtENded Agent Design (ReActXen)

[![EMNLP 2025](https://img.shields.io/badge/EMNLP-2025-blue)](https://openreview.net/forum?id=luETrQw0j6)
[![PDF](https://img.shields.io/badge/PDF-Available-green)](https://openreview.net/pdf?id=luETrQw0j6)

This repository contains the implementation of the **ReActXen** framework, focused on agent-based design and interaction.  
It has been referred to as **ReAct++**, **Enhanced ReAct**, etc.

The current version is 0.0.8. This uses python = ">=3.12,<3.14" and langchain = "^1.1.0".

The previous version is tagged `0.0.7`


📘 [Tutorial PDF](./docs/tutorial/ReActXen_IoT_Agent_EMNLP_2025.pdf)

---

## 📄 Publication

Our work has been accepted at **[EMNLP 2025 Industry Track](https://openreview.net/forum?id=luETrQw0j6)** 🎉  

> **ReAct Meets Industrial IoT: Language Agents for Data Access**  
> *James T. Rayfield, Shuxin Lin, Nianjun Zhou, Dhaval C. Patel*  

📑 [Read the Paper (OpenReview PDF)](https://openreview.net/pdf?id=luETrQw0j6)

### Citation
If you use this work, please cite:
```bibtex
@inproceedings{patel2025react,
  title     = {ReAct Meets Industrial IoT: Language Agents for Data Access},
  author    = {James T. Rayfield and Shuxin Lin and Nianjun Zhou and Dhaval C. Patel},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track},
  year      = {2025},
  url       = {https://openreview.net/forum?id=luETrQw0j6}
}
```

### Overview

The **ReActXen** framework is built on the **Agent-Family** that has multiple helper agents as outlined in the following diagram. The ReAct agent supports both **text generation** and **code generation** based on the specified action. 

<div style="text-align: center;">
  <img src="./src/reactxen/resources/ReActXen.png" width="300" height="250" />
</div>

## Table of Contents

- [Project Setup Instructions](#project-setup-instructions)
- [Setting Up the LLM](#getting-ready-to-use-reactxen)
- [Getting Ready to use ReActXen](#getting-ready-to-use-reactxen)
- [Hello Word ReActXen](#hello-world-reactxen)
- [API Functions](#api-functions)
- [Publications](#publications)

## Project Setup Instructions

To get started with this project, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone git@github.ibm.com:GenAIApps/ReActXen.git
    cd reactxen
    ```

2. **Set up a Python virtual environment (Python 3.11 or greater is required)**:
    ```bash
    python3.11 -m venv reactxen
    source reactxen/bin/activate  # On macOS/Linux
    # .\\reactxen\\Scripts\\activate  # On Windows
    ```

3. **Install the required dependencies**:
    Simply run the following command to install the package and its dependencies:
    ```bash
    pip install .
    ```

4. If you need to clean or remove the Python virtual environment (`reactxen`), follow these steps:

    ```bash
    deactivate
    rm -rf reactxen
    ```

## Getting Ready to use ReActXen

1. Setting Up Environment Variables. Copy the `.env_template` file to `.env`:
   ```bash
   cp env/.env_template .env
   ```

2. Edit the .env file and replace the placeholders with your actual values:

    ```bash
    WATSONX_APIKEY=your_watsonx_apikey
    WATSONX_URL=https://your-watsonx-url.com
    WATSONX_PROJECT_ID=your_project_id
    ```

## Hello World ReActXen

In the **hello_world_math.py** demo, the ReActXen framework is used to solve the following mathematical equation:

**Question:** Find the \(r\) that satisfies the equation:

  \[
  \log_{16} (r+16) = \frac{5}{4}
  \]

The question asks the agent to find the value of \(r\) that satisfies the logarithmic equation.

1. Execute the demo script (id 15 = granite 3.2, ibm/granite-3-2-8b-instruct)

    ```bash
    cd src/reactxen/demo
    python hello_world_math.py --mode code --model_id 15 # for code model
    python hello_world_math.py --mode text --model_id 15 # for text model
    ```

## API Functions

- **`create_reactxen_agent`**  
  *Description*: Initializes and configures a ReActXen agent.  
  *Example Usage*:  
  ```python
  agent = create_reactxen_agent(question="Find the r...", key="", ...)
  ```

- **`run_reactxen_agent`**
  *Description*: Runs the agent and returns the result.
  *Example Usage*:
  ```python
  agent.run()
  ```
  You can view the review agent output here: [Review](./src/reactxen/resources/sample_review_math_problem.json)

- **`export_benchmark_metric`**
  *Description*: Exports benchmark metrics from the agent's run.
  *Example Usage*:
  ```python
  agent.export_benchmark_metric()
  ```

  You can view the benchmark metric here: [Benchmark](./src/reactxen/resources/sample_metric_math_problem.json)

- **`export_trajectory`**
  *Description*: Exports the trajectory of the agent's decisions.
  *Example Usage*:
  ```python
  agent.export_trajectory()
  ```

  You can view the sample exported trajectory here: [Trajectory](./src/reactxen/resources/sample_traj_math_problem.json)

  ---

  # Running Experiments

> **Before running anything**, navigate to the following directory:
> ```
> src/reactxen/agents/factly/agent
> ```

---

## Running Research Question 1

For more information, see `docs/credit_assignment_algorithms.md` and `docs/prompt_optimizer_benchmark.md` at the root of the project.

### Running `credit_assignment.py`

```bash
python credit_assignment.py semantic
python credit_assignment.py lexical
```

Results will be saved in the `credit_assignment_results` folder.

---

### Running `benchmark_credit_assignment.py`

```bash
python benchmark_credit_assignment.py --traj_dir ../traj_store --csv_dir . --output_dir benchmark_outputs
```

---

## Running Research Question 2

For more information, see `docs/prompt_optimizer_benchmark.md` at the root of the project.

```bash
python benchmark_prompt_optimizers.py --n 5 --methods baseline,v1
```

---

## Running Research Question 3

For more information, see `docs/prompt_optimizer_benchmark.md` at the root of the project.

```bash
python benchmark_prompt_optimizers.py --n 5 --methods v1,v2_lexical,v2_semantic
```

---

# Project Source Code

Please find the codes that we created and modified in this directory:

```
src/reactxen/agents/factly/agent/
```

Below is a summary of every file we created or modified for this project.

## Core Agent Code

| File | Status | Description |
|---|---|---|
| `truthful_mcqa_checker.py` | Modified | Main Factly agent — runs TruthfulQA fact-checking with ReActXen. Contains `PROMPT_OG` (baseline prompt) and the full agent execution pipeline. |
| `truthful_mcqa_checker_demo.py` | Created | Interactive Streamlit demo for side-by-side prompt comparison and credit assignment benchmark visualization. |

## Credit Assignment (RQ1)

| File | Status | Description |
|---|---|---|
| `credit_assignment.py` | Created | Assigns per-step credit to agent trajectories using either lexical (ROUGE-L / Jaccard) or semantic (embedding cosine similarity) scoring. |
| `benchmark_credit_assignment.py` | Created | Benchmarks lexical vs semantic credit assignment — computes concentration metrics, ablation degradation, and generates comparison outputs. |
| `evaluation/__init__.py` | Created | Package init for the evaluation sub-module. |
| `evaluation/credit_metrics.py` | Created | Mathematical metric functions: entropy, Gini coefficient, Top-K mass, effective number of steps, correlation, cosine similarity, and Jensen-Shannon divergence. |
| `evaluation/ablation.py` | Created | Ablation evaluation — removes Top-k credited steps and measures degradation in similarity to the final answer; computes AUC via the trapezoidal rule. |
| `evaluation/plotting.py` | Created | Visualization utilities for credit distributions, tool credit bar charts, and ablation curves. |

## Prompt Optimization (RQ2 & RQ3)

| File | Status | Description |
|---|---|---|
| `prompt_optimizer.py` | Created | V1 (Behavioral) prompt optimizer — iteratively refines the system prompt by analyzing agent trajectories and extracting behavioral rules. |
| `prompt_optimizer_v2.py` | Created | V2 (Lexical) prompt optimizer — uses ROUGE-L / Jaccard credit assignment to guide prompt refinement. |
| `prompt_optimizer_v2_semantic.py` | Created | V2 (Semantic) prompt optimizer — uses embedding-based credit assignment to guide prompt refinement. |
| `benchmark_prompt_optimizers.py` | Created | Benchmarks optimized prompts on unseen TruthfulQA questions — computes success rate, groundedness, fault rate, and composite score. |

---

# Pre-Computed Results

All experiment outputs are committed to the repository so results can be inspected without re-running the scripts.

## Credit Assignment Results (RQ1)

| File / Folder | Description |
|---|---|
| `src/reactxen/agents/factly/agent/credit_assigned_steps.csv` | Per-step credit scores (lexical & semantic) for every trajectory |
| `src/reactxen/agents/factly/agent/tool_credit_summary_with_filenames.csv` | Aggregated credit per tool type |
| `src/reactxen/agents/factly/agent/credit_assignment_results/` | Per-trajectory credit distribution plots |
| `src/reactxen/agents/factly/traj_store/` | Raw trajectory JSON files used as input |

## Credit Assignment Benchmark Results (RQ1)

All outputs from `benchmark_credit_assignment.py` are stored in:

```
src/reactxen/agents/factly/agent/benchmark_outputs/
```

| File | Description |
|---|---|
| `benchmark_summary.json` | Lexical vs Semantic summary metrics (entropy, Gini, Top-K mass, ablation AUC) |
| `per_step_credit_comparison.csv` | Side-by-side lexical and semantic credit for every step |
| `per_tool_credit_comparison.csv` | Per-tool credit distribution comparison |
| `ablation_results.csv` | Degradation values at each Top-k removal level |
| `ablation_curves.png` | Ablation degradation curves (lexical vs semantic) |
| `credit_distribution_plots.png` | Histogram of credit score distributions |
| `tool_credit_distribution_plots.png` | Bar chart of credit allocated to each tool |
| `human_eval_sample.csv` | Sampled steps for optional human evaluation |

## Prompt Optimizer Benchmark Results (RQ2 & RQ3)

| File | Description |
|---|---|
| `benchmark_outputs/prompt_benchmark_summary.json` | Rankings for V1, V2_Lexical, V2_Semantic (n=5) |
| `benchmark_outputs/prompt_benchmark_summary_n=5_baseline_v1.json` | Baseline vs V1 comparison (n=5) |
| `benchmark_outputs/prompt benchmark results (n5)/prompt_benchmark_summary.json` | Earlier benchmark run with 3 optimizers |

## Optimized Prompts

The final system prompts produced by each optimizer are stored under `genai-proj-results/`:

| Folder | Optimizer | Contents |
|---|---|---|
| `genai-proj-results/prompt_optimizer_v1/` | V1 (Behavioral) | `system_prompt.txt`, per-loop trajectories |
| `genai-proj-results/prompt_optimizer_v2_default/` | V2 (Lexical) | `system_prompt.txt`, per-loop trajectories |
| `genai-proj-results/prompt_optimizer_v2_semantic/` | V2 (Semantic) | `system_prompt.txt`, per-loop trajectories |

Each folder contains loop subdirectories (`loop_1/`, `loop_2/`, …) with the trajectory outputs and meta-analysis prompts from that optimization iteration, as well as intermediate `system_prompt_loop_*.txt` snapshots.

## Documentation

| File | Description |
|---|---|
| `docs/credit_assignment_algorithms.md` | Detailed explanation of the credit assignment algorithms |
| `docs/prompt_optimizer_benchmark.md` | Methodology and metrics for the prompt optimizer benchmark |
| `docs/workflow_explanation.md` | End-to-end workflow and evaluation methodology |


