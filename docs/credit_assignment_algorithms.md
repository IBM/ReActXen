# Credit Assignment Algorithms

## Part 1: The Semantic Analysis Algorithm

This is the process that occurs inside `credit_assignment.py` for every single step the agent takes.

### 1. Input Collection

Let `$O$` be the text of the tool's Observation, such as the search result text.

Let `$J$` be the text of the agent's Final Justification, meaning the final answer it provided to the user.

---

### 2. Vector Embedding with SentenceTransformer

Both `$O$` and `$J$` are passed through the `all-MiniLM-L6-v2` transformer model.

The model maps the semantic meaning of the sentences into a 384-dimensional mathematical space.

This produces two dense vectors:

- `$\vec{V}_O$`: the observation vector
- `$\vec{V}_J$`: the justification vector

---

### 3. Cosine Similarity Calculation

We calculate the angle between these two vectors using the Cosine Similarity formula:

$$
\text{Similarity} = \frac{\vec{V}_O \cdot \vec{V}_J}{|\vec{V}_O| |\vec{V}_J|}
$$

This outputs a score between `-1.0` and `1.0`.

A score closer to `1.0` means the vectors point in the exact same direction, proving that the tool output and the final answer contain the exact same semantic facts, even if they use different vocabulary.

---

### 4. Credit Assignment

The step's `semantic_credit_score` is set to this Similarity value.

Once all steps are scored, the scores are normalized so that the total credit across all steps in the trajectory sums to exactly `1.0`.

---

## Part 2: The Benchmark Evaluation Algorithm

This is the process that occurs inside `benchmark_credit_assignment.py` to determine if Semantic is better than Lexical.

---

### 1. Establish the Baseline

For a given problem trajectory, collect all tool observations:

$$
O_1, O_2, \dots, O_n
$$

Concatenate them all into one massive string called the Full Context:

$$
C_{full}
$$

Calculate the semantic similarity between `$C_{full}$` and the Final Answer `$(J)$`.

We call this the Base Similarity:

$$
S_{base}
$$

This represents the maximum possible support the agent had.

---

### 2. Ranking by Method

Sort the list of observations from Highest Credit to Lowest Credit according to the Lexical scores.

Sort the list again according to the Semantic scores.

---

### 3. The Ablation Loop

This process is repeated for both methods.

For a threshold `$k$`, such as removing the top `1`, `3`, and `5` steps:

1. Remove the top `$k$` steps from the ranked list.
2. Concatenate the remaining, unimportant observations into a Remaining Context string:

$$
C_{rem}
$$

3. Calculate the semantic similarity between `$C_{rem}$` and the Final Answer `$(J)$` to get the Remaining Similarity:

$$
S_{rem}
$$

4. Calculate the Degradation:

$$
\text{Degradation}_k = S_{base} - S_{rem}
$$

If the removed steps were highly critical, `$S_{rem}$` will be very low, making the Degradation very high.

---

### 4. The AUC Calculation

We now have a degradation score for:

$$
k = 1, 3, 5
$$

We plot these points on a graph:

- X-axis: `$k$` steps removed
- Y-axis: Degradation

Using numerical integration, specifically the Trapezoidal Rule via `numpy.trapz`, we calculate the Area Under the Curve, or AUC, for both the Lexical line and the Semantic line.

---

### 5. Declaring the Winner

The benchmark compares the two AUC values.

The method with the higher AUC wins.

A higher AUC mathematically proves that when we trusted that method's rankings and deleted its top steps, the evidence supporting the final answer collapsed the fastest.

Therefore, that method is the most accurate at identifying the most important steps.