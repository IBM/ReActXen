# Prompt Optimizer Benchmarking Plan

This document explains the three prompt optimizer implementations and proposes a benchmarking framework to rank their performance.

## 1. Analysis of the Prompt Optimizers

The codebase contains three distinct prompt optimization scripts. All three operate via a "meta-prompt" loop where the WatsonX LLM is asked to improve the agent's system prompt based on past performance, but they differ in the data they provide to the LLM.

### Method 1: `prompt_optimizer.py` (V1 - Behavioral Only)
- **Concept:** A single-phase optimizer focused strictly on agent *behavior*.
- **Mechanism:** It runs the agent, collects trajectories, and calculates behavioral statistics (e.g., how often the agent repeats the same tool, finishes too early, or forgets to verify). It feeds these stats to the optimizer LLM to generate rules like *"Do not repeat identical tool calls."*

### Method 2: `prompt_optimizer_v2.py` (V2 Lexical)
- **Concept:** A two-phase optimizer adding *Lexical Credit Assignment*.
- **Mechanism:** Phase 1 fixes behaviors. Phase 2 tries to optimize *Tool Selection*. It measures how much each tool contributed to the final answer using **ROUGE-L and Jaccard similarity (word overlap)**. It feeds a table like `[Wikipedia: 0.8, Wikidata: 0.2]` to the optimizer LLM so it writes rules like *"Prefer Wikipedia."*

### Method 3: `prompt_optimizer_v2_semantic.py` (V2 Semantic)
- **Concept:** A two-phase optimizer adding *Semantic Credit Assignment*.
- **Mechanism:** Identical to V2, but it replaces the flawed word-overlap metric with **Cosine Similarity on Sentence Embeddings (`all-MiniLM-L6-v2`)**. This allows the optimizer LLM to see which tools *actually* provided the meaning/facts, even if the LLM paraphrased the final answer.

---

## 2. The Benchmarking Workflow (Training vs. Validation)

To properly evaluate which optimizer is "best", we must separate the data into two phases to avoid overfitting.

### Phase 1: The Optimization (Training) Phase
First, we run the three optimizer scripts (`prompt_optimizer.py`, `prompt_optimizer_v2.py`, etc.) separately.
Each script analyzes a set of $N$ questions (e.g., 20 questions) from the dataset. It observes the agent's mistakes on these specific questions and generates a new `system_prompt.txt` containing rules designed to fix those exact mistakes.

### Phase 2: The Benchmark (Validation) Phase
If we want to know which of the 3 optimizers is *actually* the best, we cannot test them on those original 20 questions. That would be "cheating", because the prompts were specifically engineered to beat those exact questions.

Instead, the `benchmark_prompt_optimizers.py` script takes the 3 final prompts and tests them on a set of **completely new, unseen questions**. By measuring the agent's performance on unseen data, the benchmark proves which optimizer wrote the most robust, generalized rules, rather than which one just memorized the training set.

---

## 3. Evaluation Metrics

During the Validation Phase, the `benchmark_prompt_optimizers.py` script tracks the following metrics for each prompt:

1. **Success Rate (40% Weight):** The percentage of questions the agent successfully answered.
2. **Average Groundedness Score (40% Weight):** The agent's self-reported confidence/accuracy score based on the evidence it found.
3. **Behavioral Fault Rate (-20% Penalty):** The percentage of runs where the agent exhibited poor behavior (e.g., repeating the same tool 3+ times, or concluding without attempting to verify the facts).
4. **Efficiency (Avg Steps):** The average number of tool calls required to reach the answer (tracked for analysis, but not directly scored).

### The Composite Score
The script calculates a final Composite Score for each method using the formula:
`Composite Score = (Success Rate * 0.4) + (Avg Groundedness * 0.4) - (Fault Rate * 0.2)`

The scores are normalized to a 0-100 scale. The optimizer that produces the prompt with the highest composite score on the unseen validation dataset is declared the winner in `prompt_benchmark_summary.json`.
