# Prompt Optimization Algorithm — Factly ReActXen Agent

This directory contains three iteratively improved implementations of an automated prompt optimization loop for the Factly ReActXen fact-checking agent. Each file evolves the core algorithm to add more sophisticated signal about how the agent uses its tools.

---

## Table of Contents

- [Overview](#overview)
- [File Comparison](#file-comparison)
- [Core Pipeline (all versions)](#core-pipeline-all-versions)
- [Phase Structure (v2 and v2\_semantic)](#phase-structure-v2-and-v2_semantic)
- [Credit Assignment Methods](#credit-assignment-methods)
- [Behavioral Metrics](#behavioral-metrics)
- [Directory Layout](#directory-layout)
- [CLI Reference](#cli-reference)
- [Environment Variables](#environment-variables)
- [Dependencies](#dependencies)

---

## Overview

The optimizer treats prompt engineering as an automated feedback loop. On each iteration:

1. The Factly agent runs on a sample of TruthfulQA questions using the **current** system prompt.
2. Each agent trajectory is parsed to extract behavioral metrics.
3. Aggregate statistics are compiled into a **meta-analysis prompt** describing what went wrong and why.
4. A WatsonX LLM (`meta-llama/llama-3-3-70b-instruct`) rewrites the system prompt to fix those failures.
5. The new prompt is saved and used as the starting point for the next loop.

This closes the loop: each iteration produces a strictly-better prompt grounded in observed agent behavior.

---

## File Comparison

| Feature | `prompt_optimizer.py` | `prompt_optimizer_v2.py` | `prompt_optimizer_v2_semantic.py` |
|---|---|---|---|
| Optimization phases | 1 | 2 | 2 |
| Behavioral metrics | Yes | Yes | Yes |
| Credit assignment | No | Yes — lexical | Yes — semantic |
| Credit signal | — | ROUGE-L + Jaccard | Cosine similarity (embeddings) |
| Extra dependency | — | None | `sentence-transformers` |
| Default loops | 3 | 3 phase-1 + 2 phase-2 | 3 phase-1 + 2 phase-2 |

**Recommended version:** `prompt_optimizer_v2_semantic.py` — it captures meaning even when the agent and the final answer use different phrasing for the same concept.

---

## Core Pipeline (all versions)

Every version runs the same five-step cycle per loop:

```
COLLECT → SUMMARIZE → ANALYZE → OPTIMIZE → PERSIST
```

### Step 1 — Collect

`collect_trajectories()` samples N rows from `TruthfulQA.csv` and runs the `ReactAgent` on each one using the current system prompt. Rows are dispatched in batches across worker threads (default: 8 threads, 10 rows/batch, 30-minute timeout).

Each agent run:
- Builds a fact-checking prompt around the TruthfulQA question, correct answer, and a shuffled incorrect answer as a distractor.
- Has access to five tools: `wikipedia`, `arxiv`, `ddg-search`, `wikidata`, `semanticscholar`.
- Is limited to 12 ReAct steps.
- Saves its full trajectory JSON to `trajectories/loop_N/<key>_traj_output.json`.

### Step 2 — Summarize

`summarize_trajectory()` parses each raw trajectory into a structured summary with these fields:

| Field | What it detects |
|---|---|
| `tool_sequence` | Ordered list of tools called (excludes Finish) |
| `unique_tools_used` | Deduplicated tool list (order-preserving) |
| `n_steps` | Total ReAct steps taken |
| `groundedness_score` | Float 0.0–1.0 extracted from the Finish action JSON |
| `success` | Whether the run completed without an exception |
| `reached_final_answer` | Whether a non-empty Finish action was produced |
| `repeated_tool_calls` | `True` if any 3 consecutive steps used the same tool |
| `missing_verification` | `True` if no thought contained verify/check/validate/confirm/cross/contradict |
| `premature_finish` | `True` if fewer than 3 tools were called but the run was marked successful |

### Step 3 — Analyze

`build_meta_analysis()` (or `build_meta_analysis_phase1/2()` in v2) assembles a structured prompt containing:

- The **current system prompt** (so the LLM can see what rules already exist).
- **Aggregate statistics** across all N trajectories: success rate, average groundedness, per-failure counts, tool usage frequency.
- **Per-trajectory summaries** — one block per run with all fields from Step 2.
- **Explicit optimization instructions** telling the LLM what failure patterns to fix and how to produce a complete, non-contradictory rewrite.

The meta-analysis prompt is saved to `trajectories/loop_N/meta_analysis_prompt.txt` for inspection.

### Step 4 — Optimize

`call_optimizer_llm()` sends the meta-analysis prompt to WatsonX (`llama-3-3-70b-instruct`) with greedy decoding and asks for a full system prompt rewrite. The LLM output is post-processed to strip:

- WatsonX stop tokens (`<|eom_id|>`, `<|eot_id|>`)
- Accidental JSON or markdown code-fence wrappers
- Trailing meta-commentary or self-narration the LLM bleeds in after the prompt
- Duplicate prompt copies (LLM sometimes reproduces the input before rewriting)
- Stray `{"action": ...}` artifacts from ReAct-format generation

### Step 5 — Persist

The cleaned rewrite is written to `prompts/system_prompt.txt`, replacing the previous prompt. A versioned copy is saved to `trajectories/loop_N/optimized_prompt.txt`. The next loop iteration automatically picks up the new file.

---

## Phase Structure (v2 and v2_semantic)

The two-phase versions split the optimization signal into complementary objectives:

### Phase 1 — Behavioral Failure Correction

Runs for the first `--phase1-loops` iterations (default: 3). The meta-analysis focuses exclusively on **what the agent did wrong**:

- Repeated identical tool calls
- Finishing with fewer than 3 tool calls
- Never cross-verifying (no verification intent in any thought)
- Never using `wikidata` or `semanticscholar`
- Low groundedness score despite multiple steps

The optimizer LLM is asked to write **concrete, actionable rules** — not vague directives. Example of the distinction enforced in the prompt:

> **Vague:** "Use more tools."
> **Concrete:** "After a Wikipedia result, always issue one follow-up query to wikidata or semanticscholar before assigning a groundedness score above 0.7."

### Phase 2 — Tool Contribution Refinement

Runs for the next `--phase2-loops` iterations (default: 2). The same behavioral metrics are included, but the meta-analysis is **augmented with a credit assignment table** showing which tools actually contributed evidence to correct final answers. This lets the LLM write tool-selection rules grounded in data rather than intuition.

The credit table is saved to `trajectories/loop_N/credit_summary.json`.

---

## Credit Assignment Methods

Credit assignment answers: *which tool's observations actually ended up in the final answer?*

For each trajectory where a Finish action was produced, the algorithm:

1. Extracts the final output's supporting passages and justification text.
2. Scores each tool observation against every final text.
3. Normalizes scores within the trajectory so they sum to 1.
4. Averages normalized credit per tool across all trajectories.

### v2 — Lexical (ROUGE-L + Jaccard)

```
credit(observation, final_text) = max over all final_texts of:
    0.7 × ROUGE-L(observation, final_text)
  + 0.3 × Jaccard(observation, final_text)
```

- **ROUGE-L** measures longest common subsequence ratio (via `difflib.SequenceMatcher`).
- **Jaccard** measures token-level overlap: `|A ∩ B| / |A ∪ B|`.

Works well when the agent copies verbatim passages. Misses paraphrases.

### v2_semantic — Semantic (Sentence Embeddings)

```
credit(observation, final_text) = max cosine_similarity(
    embed(observation), embed(final_text)
) over all final_texts
```

Uses `all-MiniLM-L6-v2` from `sentence-transformers`, loaded lazily on first Phase 2 call. Embeddings capture meaning regardless of surface wording — if the agent retrieved a relevant Wikipedia paragraph that was then paraphrased in the justification, semantic credit correctly assigns credit to `wikipedia`.

### Interpreting the Credit Table

The meta-analysis prompt includes this interpretation guide for the optimizer LLM:

| Credit | Usage | Meaning | Action |
|---|---|---|---|
| High | High | Tool is working well | Reinforce it |
| High | Low | Tool is underused | Add "prefer X" rules |
| Low | High | Tool wastes steps | Add "skip X unless Y" rules |
| Low | Low | Tool may be irrelevant | Add domain-specific triggers or ignore |

---

## Behavioral Metrics

All three detectors operate on the `trajectroy_log` (note: this key name preserves a typo from the original trajectory export format; both `trajectroy_log` and `steps` are checked as a fallback).

**Repeated tool calls** — detected if any three consecutive entries in `tool_sequence` are identical:

```python
any(seq[i] == seq[i+1] == seq[i+2] for i in range(len(seq) - 2))
```

**Missing verification** — `True` if no step's `thought` field contains any of: `verify`, `check`, `cross`, `validate`, `confirm`, `contradict`.

**Premature finish** — `True` if fewer than 3 tools were called AND the run was marked successful (i.e., the agent gave up too early, not because it crashed).

**Groundedness score** — extracted by JSON-parsing the Finish action's `action_input` dict and reading the `groundedness_score` field (float 0.0–1.0). Averaged across all runs that produced a parseable score.

---

## Directory Layout

```
factly/
├── agent/
│   ├── prompt_optimizer.py              ← v1: single-phase loop
│   ├── prompt_optimizer_v2.py           ← v2: two-phase, lexical credit
│   ├── prompt_optimizer_v2_semantic.py  ← v2_semantic: two-phase, semantic credit
│   ├── prompts/
│   │   ├── system_prompt.txt            ← live prompt (overwritten each loop)
│   │   └── system_prompt_loop_N.txt     ← snapshot of prompt before loop N ran
│   └── trajectories/
│       └── loop_N/
│           ├── <id>_<category>_traj_output.json
│           ├── meta_analysis_prompt.txt
│           ├── optimized_prompt.txt
│           └── credit_summary.json      ← phase 2 only
└── data/
    └── TruthfulQA.csv
```

`system_prompt.txt` is the single source of truth. All optimizer files read from and write to this path. Snapshots (`system_prompt_loop_N.txt`) preserve a copy of the prompt *before* loop N modifies it, so you can audit the evolution.

---

## CLI Reference

### prompt_optimizer.py (v1)

```bash
# Defaults: 3 loops, 20 trajectories each, 8 threads
python prompt_optimizer.py

# Custom configuration
python prompt_optimizer.py --loops 5 --n 10 --threads 4 --batch-size 5

# Start from a specific prompt file
python prompt_optimizer.py --base-prompt path/to/my_prompt.txt

# Skip agent runs for loop 1 by loading pre-existing trajectories
python prompt_optimizer.py --seed-trajectory-dir trajectories/loop_0
```

| Argument | Default | Description |
|---|---|---|
| `--loops` | 3 | Number of optimization iterations |
| `--n` | 20 | Trajectories to collect per loop |
| `--threads` | 8 | Parallel worker threads |
| `--batch-size` | 10 | Rows per thread batch |
| `--base-prompt` | None | Path to an initial prompt file |
| `--seed-trajectory-dir` | None | Load existing trajectories for loop 1 |

### prompt_optimizer_v2.py and prompt_optimizer_v2_semantic.py (v2)

```bash
# Defaults: 3 phase-1 loops + 2 phase-2 loops, 20 trajectories each
python prompt_optimizer_v2.py
python prompt_optimizer_v2_semantic.py

# Custom phase lengths
python prompt_optimizer_v2.py --phase1-loops 4 --phase2-loops 3 --n 15

# Bootstrap loop 1 from an existing trajectory store
python prompt_optimizer_v2.py --seed-trajectory-dir ../traj_store
```

| Argument | Default | Description |
|---|---|---|
| `--phase1-loops` | 3 | Behavioral correction loops |
| `--phase2-loops` | 2 | Credit-assignment refinement loops |
| `--n` | 20 | Trajectories per loop |
| `--threads` | 8 | Parallel worker threads |
| `--batch-size` | 10 | Rows per thread batch |
| `--base-prompt` | None | Path to an initial prompt file |
| `--seed-trajectory-dir` | None | Load existing trajectories for loop 1 |

---

## Environment Variables

All three files use the WatsonX LLM for the optimizer step. Set these before running:

| Variable | Required | Description |
|---|---|---|
| `WATSONX_APIKEY` | Yes | IBM WatsonX API key |
| `WATSONX_PROJECT_ID` | Yes | WatsonX project ID |
| `WATSONX_URL` | No | WatsonX endpoint (default: `https://us-south.ml.cloud.ibm.com`) |

A `.env` file in the working directory is automatically loaded via `python-dotenv`.

---

## Dependencies

**Core (all versions):**

```
pandas
python-dotenv
ibm-watsonx-ai
langchain-community
reactxen  (this repo)
```

**Semantic version only (`prompt_optimizer_v2_semantic.py`):**

```
sentence-transformers
```

The sentence transformer model (`all-MiniLM-L6-v2`) is downloaded from HuggingFace on first use and cached locally. It is loaded lazily — Phase 1 runs do not incur the download cost.
