# Credit Assignment Workflow and Evaluation Methodology

This document details the complete algorithmic workflow for the ReActXen Factly credit assignment module, including the transition from lexical to semantic scoring and the mathematical framework used to evaluate their performance.

---

## 1. The Core Problem: What is Credit Assignment?
When an AI agent (like ReActXen) attempts to solve a problem, it generates a "trajectory" of multiple steps—searching the web, using a calculator, retrieving documents, etc. When the agent finally arrives at a correct answer and writes a justification, **Credit Assignment** is the process of looking back at the trajectory and asking: *"Which specific tool calls (observations) were actually responsible for providing the information used in the final answer?"*

---

## 2. The Baseline: Lexical Credit Assignment
Originally, the project relied exclusively on a lexical (word-level) approach. It computes credit by directly comparing the raw text of what a tool returned (the observation) against the raw text of the final justification.

### How it works:
*   **ROUGE-L Algorithm:** Measures the "Longest Common Subsequence" of words between the observation and the final answer. It checks if sequences of words appear in the same order.
*   **Jaccard Similarity Algorithm:** Treats sentences as a "bag of words" and calculates the Intersection over Union (IoU) of the word tokens.
*   **The Formula:** `Score = 0.7 * ROUGE-L + 0.3 * Jaccard`

### The Flaw
This method requires *exact word overlap*. If a search tool returns "The CEO of Apple," and the agent's final answer says "Tim Cook leads the tech giant," the Lexical score will be extremely low. It unfairly penalizes highly useful steps simply because the LLM paraphrased or synthesized the information when writing the final answer.

---

## 3. The Improvement: Semantic Credit Assignment
To address the paraphrasing flaw, we introduced the Semantic approach, which compares the underlying *meaning* of the texts rather than their exact characters.

### How it works:
*   **Sentence-Transformers (HuggingFace):** We use a deep learning embedding model (specifically `all-MiniLM-L6-v2`) to encode both the tool's observation and the final answer into dense, high-dimensional mathematical vectors.
*   **Cosine Similarity Algorithm:** We calculate the mathematical angle between these two vectors. If the vectors point in the exact same direction (Cosine Sim = 1.0), it means the two texts contain the exact same semantic information, even if they use completely different vocabulary.

---

## 4. The Evaluation Metric: Proxy Ablation Study
Now we have two different methods assigning scores to the same trajectory steps. Because we don't have "ground truth" human labels telling us which steps were actually the most important, we engineered a **Proxy Ablation Evaluation** framework to mathematically declare a winner.

Here is the exact step-by-step algorithm used in `benchmark_credit_assignment.py`:

### Step 1: Rank the Steps
For a given trajectory, we sort all the agent's steps from "Most Important" to "Least Important" based on the credit they were assigned. We do this twice in parallel: once using the Lexical rankings, and once using the Semantic rankings.

### Step 2: The Ablation (Removal) Test
We simulate what happens if we delete the supposedly "most important" steps from the agent's memory.
*   Remove the Top $K$ steps (e.g., $K=1, 3, 5$).
*   Take whatever observations are left over (the ones the method deemed unimportant) and combine them into a single "Remaining Context" string.

### Step 3: Calculate Degradation
We then measure the semantic similarity between this "Remaining Context" and the final answer. 
*   If the removed steps were truly useless, the Remaining Context will still contain all the evidence needed, and similarity to the final answer will stay high.
*   If the removed steps were highly critical, the Remaining Context will be missing key facts, and similarity to the final answer will drop significantly.
*   The difference between the original baseline similarity and this new remaining similarity is our **Degradation Score**.

### Step 4: Declare the Winner using AUC
We calculate the **Area Under the Curve (AUC)** for the degradation across different values of $K$. 

> [!IMPORTANT]
> **The method with the HIGHER AUC wins.** 
> A higher AUC means that when we deleted its top-ranked steps, the final answer was completely unsupported. This proves that the method successfully identified the most critical, load-bearing steps in the trajectory!

---

## 5. Secondary Evaluation Metrics
The benchmark script also calculates a few statistical properties of the credit assignment to give us deeper insights into how the algorithms behave:

*   **Entropy & Gini Coefficient:** These measure "Concentration." Does the method give 99% of the credit to a single step (low entropy/high Gini), or does it spread credit evenly across all steps (high entropy/low Gini)?
*   **Jensen-Shannon (JS) Divergence:** A statistical measurement of how differently the two methods distribute credit across tools. If JS Divergence is high, it proves that the Lexical and Semantic methods fundamentally disagree on which tools (e.g., Wikipedia vs DuckDuckGo) are actually useful.
