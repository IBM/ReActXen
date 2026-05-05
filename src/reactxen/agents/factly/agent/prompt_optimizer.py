"""
prompt_optimizer.py
--------------------
Self-contained, fully automated Prompt Optimization Loop for the Factly
ReActXen Agent.

Just run:
    python prompt_optimizer.py

What happens on each iteration
--------------------------------
1. COLLECT  – Run the Factly agent (via truthful_mcqa_checker) on N rows
               sampled from TruthfulQA.csv using the *current* prompt.
               Trajectories are saved under trajectories/loop_<N>/.

2. SUMMARIZE – Each trajectory is parsed to extract: tool call sequence,
               step count, repeated-tool loops, missing-verification flags,
               groundedness score, and success status.

3. ANALYZE  – Aggregate statistics across all N trajectories are compiled
               into a structured meta-prompt.

4. OPTIMIZE – The meta-prompt is sent to a WatsonX LLM, which returns a
               fully rewritten, augmented system prompt.

5. PERSIST  – The augmented prompt is written to prompts/system_prompt.txt,
               replacing the current prompt so the next loop iteration
               automatically picks it up.

6. REPEAT   – Steps 1-5 repeat for --loops iterations.

Directory layout (all paths relative to this file):
    factly/
    ├── agent/
    │   ├── prompt_optimizer.py          ← this file
    │   ├── truthful_mcqa_checker.py
    │   ├── prompts/
    │   │   ├── system_prompt.txt        ← live prompt (updated each loop)
    │   │   └── system_prompt_loop_N.txt ← snapshot before loop N runs
    │   └── trajectories/
    │       ├── loop_1/
    │       │   ├── <key>_traj_output.json  ...
    │       │   └── meta_analysis_prompt.txt
    │       └── loop_2/ ...
    └── data/
        └── TruthfulQA.csv

Usage
-----
    # Default: 3 optimization loops, 20 trajectories each
    python prompt_optimizer.py

    # Custom: 5 loops, 10 trajectories each, starting from a specific prompt
    python prompt_optimizer.py --loops 5 --n 10 --base-prompt prompts/my_prompt.txt

    # Load pre-existing trajectories for the first loop instead of running
    python prompt_optimizer.py --seed-trajectory-dir trajectories/loop_0
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path constants – everything anchored to this file's location
# ---------------------------------------------------------------------------
_AGENT_DIR  = Path(__file__).resolve().parent        # factly/agent/
_FACTLY_DIR = _AGENT_DIR.parent                      # factly/
_DATA_DIR   = _FACTLY_DIR / "data"                   # factly/data/
_PROMPTS_DIR      = _AGENT_DIR / "prompts"           # factly/agent/prompts/
_TRAJECTORIES_DIR = _AGENT_DIR / "trajectories"      # factly/agent/trajectories/

# The single "live" prompt file – updated in-place after each loop
_LIVE_PROMPT_PATH = _PROMPTS_DIR / "system_prompt.txt"

# Repo root so package imports resolve
sys.path.insert(0, str(_FACTLY_DIR.parents[3]))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_N_TRAJECTORIES = 20
DEFAULT_N_LOOPS        = 3
DEFAULT_N_THREADS      = 8
DEFAULT_BATCH_SIZE     = 10


# ===========================================================================
# Prompt helpers
# ===========================================================================

def load_prompt(path: Path) -> str:
    """Read the current live prompt, or return an empty string if absent."""
    if path.exists():
        text = path.read_text(encoding="utf-8")
        logger.info("Loaded prompt from %s (%d chars)", path, len(text))
        return text
    logger.warning("Prompt file not found at %s – starting with empty prompt.", path)
    return ""


def save_prompt(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.info("Prompt written to %s (%d chars)", path, len(text))


def snapshot_prompt(current_prompt: str, loop_index: int) -> None:
    """Save a timestamped copy of the prompt before the loop modifies it."""
    snapshot_path = _PROMPTS_DIR / f"system_prompt_loop_{loop_index}.txt"
    save_prompt(current_prompt, snapshot_path)
    logger.info("Snapshot saved → %s", snapshot_path)


# ===========================================================================
# Step 1 – Collect trajectories via truthful_mcqa_checker machinery
# ===========================================================================

def collect_trajectories(
    current_prompt: str,
    n: int,
    loop_dir: Path,
    n_threads: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    """
    Run the ReactAgent on *n* rows sampled from TruthfulQA.csv using
    *current_prompt* as the system prompt.  Trajectories are saved under
    *loop_dir* and also returned as a list for immediate summarisation.
    """
    # --- Lazy imports (heavy deps only loaded when actually running) ---------
    try:
        from dotenv import load_dotenv
        load_dotenv()

        import pandas as pd
        from reactxen.agents.react.agents import ReactAgent
        from langchain_community.agent_toolkits.load_tools import load_tools
        from langchain_community.tools.wikidata.tool import (
            WikidataAPIWrapper, WikidataQueryRun,
        )
        from langchain_community.tools.semanticscholar.tool import (
            SemanticScholarQueryRun,
        )
        from reactxen.utils.model_inference import watsonx_llm
    except ImportError as exc:
        raise ImportError(
            "Missing dependency for trajectory collection. "
            f"Original error: {exc}"
        ) from exc

    loop_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data -----------------------------------------------------------
    csv_path = _DATA_DIR / "TruthfulQA.csv"
    data = pd.read_csv(csv_path)
    data["question_id"] = list(range(len(data)))

    # Sample n rows (reproducible within a loop via the loop index embedded
    # in loop_dir name, but fresh randomness across loops)
    sample = data.sample(n=min(n, len(data)), random_state=None).reset_index(drop=True)
    logger.info("Collected %d rows from TruthfulQA for this loop.", len(sample))

    # Shared results dict written by worker threads
    collected: dict[str, dict[str, Any]] = {}

    # -------------------------------------------------------------------------
    # Inner helpers – mirrors the logic in truthful_mcqa_checker.py but writes
    # trajectories into loop_dir instead of a global traj_store, and returns
    # structured dicts instead of raw JSON strings.
    # -------------------------------------------------------------------------

    react_example = textwrap.dedent("""
        Question: I want to verify if 'Lubrication System Failure' in gearboxes
        can be monitored or detected using the 'Power Input' sensor.
        Thought 1: let me use wikipedia to search the content.
        Action 1: wikipedia
        Action Input 1: gearbox lubrication failure using power input sensor
        Observation 1: suppressed due to length
        Thought 2: Based on the evidence, I can now conclude the analysis.
        Action 2: Finish
        Action Input 2: Final Answer: { "asset_type": "gearbox", "claim": "...",
        "supporting_passages": [], "contradicting_passages": [],
        "groundedness_score": 0.8, "justification": "..." }
    """).strip()

    def _build_prompt(asset_name: str, claim_text: str, answer_text: str) -> str:
        # Re-use the exact prompt template from truthful_mcqa_checker.py,
        # but prepend the current optimized system prompt so the agent
        # benefits from any improvements made in prior loops.
        system_section = (
            f"[SYSTEM INSTRUCTIONS]\n{current_prompt}\n\n"
            if current_prompt.strip()
            else ""
        )
        task_prompt = textwrap.dedent(f"""
            Task: Fact-Check an LLM-Generated Answer to a Multiple-Choice Question.

            Objective: Determine whether the LLM-generated answer is factually
            grounded by identifying supporting or contradicting evidence from
            trusted sources.

            Inputs:
            - Question Category: "{asset_name}"
            - Multiple-Choice Question: "{claim_text}"
            - LLM-Generated Answer: "{answer_text}"

            Steps:
            1. Search Wikipedia, arXiv, DuckDuckGo, Wikidata, and Semantic Scholar.
            2. Extract the most relevant passages. Cite each as supporting or
               contradicting.
            3. Assign a groundedness score 0.0–1.0.
            4. End with a Finish action containing a valid JSON object.

            IMPORTANT: Respond only with a single valid JSON object with exactly
            two fields: "action" and "action_input".  The "action" must be one of:
            arxiv, wikipedia, ddg-search, wikidata, semanticscholar, Finish.

            Final Output Format (for Finish only):
            {{
              "action": "Finish",
              "action_input": {{
                "asset_type": "{asset_name}",
                "question": "{claim_text}",
                "answer": "{answer_text}",
                "supporting_passages": [{{"text":"...","source":"...","highlight":"..."}}],
                "contradicting_passages": [{{"text":"...","source":"...","highlight":"..."}}],
                "groundedness_score": 0.0,
                "justification": "..."
              }}
            }}
        """).strip()
        return system_section + task_prompt

    def _process_row(row: "pd.Series") -> None:  # type: ignore[name-defined]
        asset_name  = row.get("Category", "unknown")
        entry_id    = row.get("question_id", "unknown")
        unique_key  = f"{entry_id}_{asset_name}"
        question    = row.get("Question", "")
        answer      = row.get("Best Answer", "")
        incorrect   = row.get("Best Incorrect Answer", "")

        choices     = [answer, incorrect]
        random.shuffle(choices)
        options_text = ", ".join(
            [f"{chr(65+i)}. {c}" for i, c in enumerate(choices)]
        )
        claim_text  = f"Q: {question}\nOptions: {options_text}"
        prompt      = _build_prompt(asset_name, claim_text, answer)

        try:
            wikidata        = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())
            semantic_scholar = SemanticScholarQueryRun()
            tools = load_tools(["arxiv", "wikipedia", "ddg-search"])
            tools.extend([wikidata, semantic_scholar])

            selected_llm = partial(watsonx_llm, max_tokens=5000)

            ragent = ReactAgent(
                question=prompt,
                key="",
                cbm_tools=tools,
                tool_desc=None,
                max_steps=12,
                react_llm_model_id=16,
                react_example=react_example,
                handle_context_length_overflow=False,
                debug=True,
                apply_loop_detection_check=True,
                log_structured_messages=True,
                use_tool_cache=True,
                llm=selected_llm,
                enable_tool_partial_match=True,
                skip_token_counting=True,
            )
            ragent.run(name="optimizer_run")
            raw_answer = ragent.answer or ""

            # Export raw trajectory JSON
            try:
                traj_data = ragent.export_trajectory()
            except Exception:
                traj_data = {}

            # Parse steps from trajectory export
            steps = traj_data.get("steps", [])

            # Try to extract groundedness score from the final answer JSON
            groundedness_score = None
            try:
                cleaned = raw_answer.replace("Final Answer:", "").strip()
                parsed  = json.loads(cleaned)
                ai      = parsed.get("action_input", parsed)
                groundedness_score = float(ai.get("groundedness_score", 0.0))
            except Exception:
                pass

            result = {
                "question":           question,
                "claim_text":         claim_text,
                "answer":             answer,
                "asset_name":         asset_name,
                "steps":              steps,
                "final_answer":       raw_answer,
                "groundedness_score": groundedness_score,
                "success":            True,
                "error":              None,
            }

            # Save individual trajectory file
            traj_path = loop_dir / f"{unique_key}_traj_output.json"
            with traj_path.open("w", encoding="utf-8") as fh:
                json.dump({**result, "raw_trajectory": traj_data}, fh,
                          indent=2, ensure_ascii=False)

            collected[unique_key] = result
            logger.info("✓ Trajectory saved: %s", unique_key)

        except Exception as exc:
            logger.error("✗ Failed trajectory %s: %s", unique_key, exc)
            collected[unique_key] = {
                "question":           question,
                "claim_text":         claim_text,
                "answer":             answer,
                "asset_name":         asset_name,
                "steps":              [],
                "final_answer":       None,
                "groundedness_score": None,
                "success":            False,
                "error":              str(exc),
            }

    # --- Threaded execution (mirrors generate_solution in checker) -----------
    batches = [
        sample.iloc[i : i + batch_size]
        for i in range(0, len(sample), batch_size)
    ]
    logger.info("Dispatching %d batches across %d threads…", len(batches), n_threads)

    def _process_batch(batch: "pd.DataFrame") -> None:  # type: ignore[name-defined]
        for _, row in batch.iterrows():
            _process_row(row)

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(_process_batch, b) for b in batches]
        done, not_done = wait(futures, timeout=1800)
        for f in not_done:
            f.cancel()
            logger.warning("A batch timed out and was cancelled.")

    logger.info("Collected %d/%d trajectories.", len(collected), n)
    return list(collected.values())


# ===========================================================================
# Step 2 – Summarize individual trajectories
# ===========================================================================

def summarize_trajectory(traj: dict[str, Any]) -> dict[str, Any]:
    steps = traj.get("steps", [])

    tool_sequence = [
        step.get("action", "unknown")
        for step in steps
        if step.get("action") not in (None, "", "Final Answer", "Finish")
    ]

    # Detect 3+ consecutive identical tool calls
    repeated_tools = any(
        tool_sequence[i] == tool_sequence[i+1] == tool_sequence[i+2]
        for i in range(len(tool_sequence) - 2)
    ) if len(tool_sequence) >= 3 else False

    # Flag if no verification intent appeared in any thought
    verify_kws = {"verify", "check", "cross", "validate", "confirm", "contradict"}
    missing_verification = not any(
        any(kw in (step.get("thought") or "").lower() for kw in verify_kws)
        for step in steps
    )

    # Detect premature finish (< 3 tool calls before Final Answer)
    premature_finish = len(tool_sequence) < 3 and traj.get("success", False)

    # Tool diversity (unique tools used)
    unique_tools = list(dict.fromkeys(tool_sequence))  # order-preserving dedup

    return {
        "question":            traj.get("question", ""),
        "asset_name":          traj.get("asset_name", ""),
        "tool_sequence":       tool_sequence,
        "unique_tools_used":   unique_tools,
        "n_tools_used":        len(unique_tools),
        "n_steps":             len(steps),
        "groundedness_score":  traj.get("groundedness_score"),
        "reached_final_answer": bool(traj.get("final_answer")),
        "success":             traj.get("success", False),
        "error":               traj.get("error"),
        "repeated_tool_calls": repeated_tools,
        "missing_verification": missing_verification,
        "premature_finish":    premature_finish,
    }


# ===========================================================================
# Step 3 – Build the meta-analysis prompt
# ===========================================================================

def build_meta_analysis(
    summaries: list[dict[str, Any]],
    current_prompt: str,
    loop_index: int,
) -> str:
    n  = len(summaries)
    ns = sum(1 for s in summaries if s["success"])
    na = sum(1 for s in summaries if s["reached_final_answer"])
    nr = sum(1 for s in summaries if s["repeated_tool_calls"])
    nv = sum(1 for s in summaries if s["missing_verification"])
    np_ = sum(1 for s in summaries if s["premature_finish"])

    scores = [s["groundedness_score"] for s in summaries if s["groundedness_score"] is not None]
    avg_score = f"{sum(scores)/len(scores):.3f}" if scores else "N/A"

    # Tool usage frequency across all trajectories
    tool_counts: dict[str, int] = {}
    for s in summaries:
        for t in s["tool_sequence"]:
            tool_counts[t] = tool_counts.get(t, 0) + 1
    tool_freq = ", ".join(
        f"{t}={c}" for t, c in sorted(tool_counts.items(), key=lambda x: -x[1])
    ) or "none"

    trajectory_blocks = "\n\n".join(
        textwrap.dedent(f"""
            --- Trajectory {i+1} ---
            Category        : {s['asset_name']}
            Question        : {s['question'][:120]}
            Tool sequence   : {' → '.join(s['tool_sequence']) or '(none)'}
            Unique tools    : {', '.join(s['unique_tools_used']) or 'none'}
            Steps taken     : {s['n_steps']}
            Groundedness    : {s['groundedness_score'] if s['groundedness_score'] is not None else 'N/A'}
            Success         : {s['success']}
            Repeated tools  : {s['repeated_tool_calls']}
            Missing verify  : {s['missing_verification']}
            Premature finish: {s['premature_finish']}
            Error           : {s['error'] or 'none'}
        """).strip()
        for i, s in enumerate(summaries)
    )

    return textwrap.dedent(f"""
        You are an expert prompt engineer for ReAct-style fact-checking agents.

        This is optimization loop {loop_index}. Below are statistics and summaries
        from {n} agent trajectories run using the CURRENT system prompt. Your job
        is to produce an IMPROVED system prompt that fixes the failure patterns
        you observe.

        ================================================================
        CURRENT SYSTEM PROMPT (loop {loop_index} input)
        ================================================================
        {current_prompt if current_prompt.strip() else "(no system prompt yet)"}

        ================================================================
        AGGREGATE STATISTICS ({n} trajectories)
        ================================================================
        Successful runs             : {ns} / {n}
        Runs reaching Final Answer  : {na} / {n}
        Average groundedness score  : {avg_score}
        Runs with repeated tool calls : {nr} / {n}
        Runs missing verification   : {nv} / {n}
        Runs with premature finish  : {np_} / {n}
        Tool usage frequency        : {tool_freq}

        ================================================================
        INDIVIDUAL TRAJECTORY SUMMARIES
        ================================================================
        {trajectory_blocks}

        ================================================================
        YOUR TASK
        ================================================================
        1. Identify the TOP 3-5 recurring failure patterns from the data above.
           Focus on:
             a) Wrong or suboptimal tool selection (e.g., never using wikidata
                or semanticscholar when they would help)
             b) Missing cross-verification steps before concluding
             c) Looping / repeating the same tool call
             d) Finishing too early with insufficient evidence
             e) Low groundedness scores despite multiple tool calls

        2. For each failure pattern, write a CONCRETE, ACTIONABLE guideline
           (not vague advice) that directly prevents it. Examples of concrete
           vs vague:
             Vague:   "Use more tools."
             Concrete: "After retrieving a Wikipedia passage, always issue at
                        least one follow-up query to either wikidata or
                        semanticscholar to cross-reference the claim before
                        assigning a groundedness score above 0.7."

        3. Output the COMPLETE AUGMENTED SYSTEM PROMPT.
           - Keep ALL existing instructions intact.
           - Append a new section at the end:

               ## Optimization Loop {loop_index} Additions

           - List only the new guidelines in that section.
           - Do NOT wrap in markdown fences or add any preamble.
           - Output ONLY the prompt text — nothing else.
    """).strip()


# ===========================================================================
# Step 4 – Call the WatsonX optimizer LLM
# ===========================================================================

def call_optimizer_llm(meta_prompt: str) -> str:
    try:
        from ibm_watsonx_ai.foundation_models import ModelInference  # type: ignore
        from ibm_watsonx_ai import Credentials                        # type: ignore
    except ImportError as exc:
        raise ImportError(
            "ibm_watsonx_ai is required. Install with: pip install ibm-watsonx-ai"
        ) from exc

    api_key    = os.environ.get("WATSONX_APIKEY")
    url        = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    project_id = os.environ.get("WATSONX_PROJECT_ID")

    if not api_key or not project_id:
        raise EnvironmentError(
            "Set WATSONX_APIKEY and WATSONX_PROJECT_ID environment variables."
        )

    model = ModelInference(
        model_id="meta-llama/llama-3-3-70b-instruct",
        credentials=Credentials(api_key=api_key, url=url),
        project_id=project_id,
        params={
            "decoding_method": "greedy",
            "max_new_tokens":  4096,
            "temperature":     0.2,
            "repetition_penalty": 1.05,
        },
    )

    logger.info("Calling optimizer LLM…")
    return model.generate_text(prompt=meta_prompt)


# ===========================================================================
# Single optimization loop
# ===========================================================================

def run_one_loop(
    loop_index: int,
    n: int,
    n_threads: int,
    batch_size: int,
    seed_trajectory_dir: Path | None = None,
) -> str:
    """
    Execute one full optimize cycle:
      collect → summarize → analyze → optimize → persist

    Returns the new prompt text.
    """
    logger.info("=" * 60)
    logger.info("LOOP %d  START", loop_index)
    logger.info("=" * 60)

    loop_dir = _TRAJECTORIES_DIR / f"loop_{loop_index}"

    # --- Load current prompt -------------------------------------------------
    current_prompt = load_prompt(_LIVE_PROMPT_PATH)

    # --- Snapshot current prompt before we overwrite it ----------------------
    snapshot_prompt(current_prompt, loop_index)

    # --- Step 1: Collect trajectories ----------------------------------------
    if seed_trajectory_dir is not None and loop_index == 1:
        logger.info("Loading seed trajectories from %s", seed_trajectory_dir)
        trajectories = _load_trajectories_from_dir(seed_trajectory_dir)
        # Copy them into this loop's directory for consistency
        loop_dir.mkdir(parents=True, exist_ok=True)
        for src in seed_trajectory_dir.glob("*.json"):
            shutil.copy(src, loop_dir / src.name)
    else:
        trajectories = collect_trajectories(
            current_prompt=current_prompt,
            n=n,
            loop_dir=loop_dir,
            n_threads=n_threads,
            batch_size=batch_size,
        )

    if not trajectories:
        logger.error("No trajectories collected for loop %d – skipping.", loop_index)
        return current_prompt

    # --- Step 2: Summarize ---------------------------------------------------
    logger.info("Summarizing %d trajectories…", len(trajectories))
    summaries = [summarize_trajectory(t) for t in trajectories]

    # --- Step 3: Build meta-analysis prompt ----------------------------------
    logger.info("Building meta-analysis prompt…")
    meta_prompt = build_meta_analysis(summaries, current_prompt, loop_index)

    meta_path = loop_dir / "meta_analysis_prompt.txt"
    meta_path.write_text(meta_prompt, encoding="utf-8")
    logger.info("Meta-analysis prompt → %s", meta_path)

    # --- Step 4: Call optimizer LLM ------------------------------------------
    new_prompt = call_optimizer_llm(meta_prompt)

    # --- Step 5: Persist new prompt ------------------------------------------
    save_prompt(new_prompt, _LIVE_PROMPT_PATH)

    # Also save a versioned copy alongside the loop's trajectories
    loop_prompt_path = loop_dir / "optimized_prompt.txt"
    save_prompt(new_prompt, loop_prompt_path)

    logger.info("LOOP %d  COMPLETE.  New prompt: %d chars.", loop_index, len(new_prompt))
    return new_prompt


def _load_trajectories_from_dir(directory: Path) -> list[dict[str, Any]]:
    files = sorted(directory.glob("*_traj_output.json"))
    if not files:
        raise FileNotFoundError(f"No *_traj_output.json files found in {directory}")
    results = []
    for f in files:
        with f.open(encoding="utf-8") as fh:
            results.append(json.load(fh))
    return results


# ===========================================================================
# Main optimization loop
# ===========================================================================

def run_optimization_loop(
    n_loops: int,
    n_trajectories: int,
    n_threads: int,
    batch_size: int,
    seed_trajectory_dir: Path | None,
) -> None:
    """
    Runs the full prompt optimization loop *n_loops* times.

    Loop N uses the prompt produced by Loop N-1. Loop 1 uses whatever
    is currently in prompts/system_prompt.txt (or an empty string if
    the file does not exist yet).
    """
    _PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    _TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()
    logger.info(
        "Starting prompt optimization: %d loops × %d trajectories each",
        n_loops, n_trajectories,
    )

    for loop_idx in range(1, n_loops + 1):
        run_one_loop(
            loop_index=loop_idx,
            n=n_trajectories,
            n_threads=n_threads,
            batch_size=batch_size,
            seed_trajectory_dir=seed_trajectory_dir if loop_idx == 1 else None,
        )

    elapsed = datetime.now() - start_time
    logger.info(
        "All %d loops complete in %s. Final prompt at: %s",
        n_loops, elapsed, _LIVE_PROMPT_PATH,
    )


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automated Prompt Optimization Loop for the Factly ReActXen Agent.\n"
            "Runs collect → summarize → optimize in a loop, updating the live\n"
            "system prompt after each iteration.\n\n"
            "Just run:  python prompt_optimizer.py"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--loops", type=int, default=DEFAULT_N_LOOPS,
        help=f"Number of optimize iterations (default: {DEFAULT_N_LOOPS}).",
    )
    parser.add_argument(
        "--n", type=int, default=DEFAULT_N_TRAJECTORIES,
        help=f"Trajectories to collect per loop (default: {DEFAULT_N_TRAJECTORIES}).",
    )
    parser.add_argument(
        "--threads", type=int, default=DEFAULT_N_THREADS,
        help=f"Worker threads for parallel agent runs (default: {DEFAULT_N_THREADS}).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Rows per batch handed to each thread (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--base-prompt", type=Path, default=None,
        help=(
            "Path to an initial system prompt file.  If supplied, it is copied "
            "to prompts/system_prompt.txt before loop 1 runs, overwriting any "
            "existing file there."
        ),
    )
    parser.add_argument(
        "--seed-trajectory-dir", type=Path, default=None,
        help=(
            "If set, load pre-existing trajectory JSON files from this directory "
            "for loop 1 instead of running the agent.  Useful for bootstrapping "
            "the first optimization from an existing run."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # If caller supplied a starting prompt, install it as the live prompt
    if args.base_prompt:
        if not args.base_prompt.exists():
            raise FileNotFoundError(f"--base-prompt file not found: {args.base_prompt}")
        shutil.copy(args.base_prompt, _LIVE_PROMPT_PATH)
        logger.info("Installed base prompt from %s", args.base_prompt)

    run_optimization_loop(
        n_loops=args.loops,
        n_trajectories=args.n,
        n_threads=args.threads,
        batch_size=args.batch_size,
        seed_trajectory_dir=args.seed_trajectory_dir,
    )


if __name__ == "__main__":
    main()
