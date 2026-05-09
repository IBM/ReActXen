"""
prompt_optimizer_v2.py
-----------------------
Two-phase, fully automated Prompt Optimization Loop for the Factly
ReActXen Agent.

PHASE 1 (loops 1..phase1_loops):
    Behavioral failure correction.
    Collect trajectories → summarize behavioral metrics → optimize prompt.
    Metrics: repeated tools, premature finish, missing verification,
             groundedness score, success rate.

PHASE 2 (loops phase1_loops+1..total_loops):
    Tool contribution refinement via credit assignment.
    Same collect → summarize pipeline, PLUS Script-2-style semantic/lexical
    credit assignment is run on each loop's trajectories and appended to the
    meta-analysis prompt so the optimizer LLM can write tool-selection
    guidelines grounded in actual evidence contribution.

Just run:
    python prompt_optimizer_v2.py

    # Custom phases
    python prompt_optimizer_v2.py --phase1-loops 3 --phase2-loops 2 --n 20

    # Bootstrap phase 1 from existing traj_store trajectories
    python prompt_optimizer_v2.py --seed-trajectory-dir ../traj_store

Directory layout (all paths relative to this file):
    factly/
    ├── agent/
    │   ├── prompt_optimizer_v2.py       ← this file
    │   ├── prompts/
    │   │   ├── system_prompt.txt        ← live prompt (updated each loop)
    │   │   └── system_prompt_loop_N.txt ← snapshot before loop N
    │   └── trajectories/
    │       ├── loop_1/  loop_2/  ...
    │       │   ├── <key>_traj_output.json
    │       │   ├── meta_analysis_prompt.txt
    │       │   ├── optimized_prompt.txt
    │       │   └── credit_summary.json   (phase 2 only)
    └── data/
        └── TruthfulQA.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import shutil
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime
from difflib import SequenceMatcher
from functools import partial
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
_AGENT_DIR        = Path(__file__).resolve().parent
_FACTLY_DIR       = _AGENT_DIR.parent
_DATA_DIR         = _FACTLY_DIR / "data"
_PROMPTS_DIR      = _AGENT_DIR / "prompts"
_TRAJECTORIES_DIR = _AGENT_DIR / "trajectories"
_LIVE_PROMPT_PATH = _PROMPTS_DIR / "system_prompt.txt"

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
DEFAULT_PHASE1_LOOPS   = 3
DEFAULT_PHASE2_LOOPS   = 2
DEFAULT_N_TRAJECTORIES = 20
DEFAULT_N_THREADS      = 8
DEFAULT_BATCH_SIZE     = 10


# ===========================================================================
# Prompt helpers
# ===========================================================================

def load_prompt(path: Path) -> str:
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
    snapshot_path = _PROMPTS_DIR / f"system_prompt_loop_{loop_index}.txt"
    save_prompt(current_prompt, snapshot_path)
    logger.info("Snapshot saved → %s", snapshot_path)


# ===========================================================================
# Step 1 – Collect trajectories
# ===========================================================================

def collect_trajectories(
    current_prompt: str,
    n: int,
    loop_dir: Path,
    n_threads: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    """
    Run the ReactAgent on n rows sampled from TruthfulQA.csv.
    Trajectories are saved under loop_dir and returned as structured dicts.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
        import pandas as pd
        from reactxen.agents.react.agents import ReactAgent
        from langchain_community.agent_toolkits.load_tools import load_tools
        from langchain_community.tools.wikidata.tool import (
            WikidataAPIWrapper, WikidataQueryRun,
        )
        from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
        from reactxen.utils.model_inference import watsonx_llm
    except ImportError as exc:
        raise ImportError(f"Missing dependency: {exc}") from exc

    loop_dir.mkdir(parents=True, exist_ok=True)

    csv_path = _DATA_DIR / "TruthfulQA.csv"
    data = pd.read_csv(csv_path)
    data["question_id"] = list(range(len(data)))

    sample = data.sample(n=min(n, len(data)), random_state=None).reset_index(drop=True)
    logger.info("Sampled %d rows from TruthfulQA.", len(sample))

    collected: dict[str, dict[str, Any]] = {}

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
        system_section = (
            f"[SYSTEM INSTRUCTIONS]\n{current_prompt}\n\n"
            if current_prompt.strip() else ""
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
            two fields: "action" and "action_input". The "action" must be one of:
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

    def _process_row(row: Any) -> None:
        asset_name = row.get("Category", "unknown")
        entry_id   = row.get("question_id", "unknown")
        unique_key = f"{entry_id}_{asset_name}"
        question   = row.get("Question", "")
        answer     = row.get("Best Answer", "")
        incorrect  = row.get("Best Incorrect Answer", "")

        choices = [answer, incorrect]
        random.shuffle(choices)
        options_text = ", ".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        claim_text   = f"Q: {question}\nOptions: {options_text}"
        prompt       = _build_prompt(asset_name, claim_text, answer)

        try:
            wikidata         = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())
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

            try:
                traj_data = ragent.export_trajectory()
            except Exception:
                traj_data = {}

            # The trajectory log is stored under "trajectroy_log" (typo preserved
            # from the original export format)
            steps = traj_data.get("trajectroy_log", traj_data.get("steps", []))

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
                "trajectroy_log":     steps,   # preserve original key spelling
                "final_answer":       raw_answer,
                "groundedness_score": groundedness_score,
                "success":            True,
                "error":              None,
            }

            traj_path = loop_dir / f"{unique_key}_traj_output.json"
            with traj_path.open("w", encoding="utf-8") as fh:
                json.dump({**result, "raw_trajectory": traj_data}, fh,
                          indent=2, ensure_ascii=False)

            collected[unique_key] = result
            logger.info("✓ Saved: %s", unique_key)

        except Exception as exc:
            logger.error("✗ Failed %s: %s", unique_key, exc)
            collected[unique_key] = {
                "question":           question,
                "claim_text":         claim_text,
                "answer":             answer,
                "asset_name":         asset_name,
                "trajectroy_log":     [],
                "final_answer":       None,
                "groundedness_score": None,
                "success":            False,
                "error":              str(exc),
            }

    def _process_batch(batch: Any) -> None:
        for _, row in batch.iterrows():
            _process_row(row)

    batches = [sample.iloc[i: i + batch_size] for i in range(0, len(sample), batch_size)]
    logger.info("Dispatching %d batches across %d threads…", len(batches), n_threads)

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(_process_batch, b) for b in batches]
        _, not_done = wait(futures, timeout=1800)
        for f in not_done:
            f.cancel()
            logger.warning("A batch timed out and was cancelled.")

    logger.info("Collected %d/%d trajectories.", len(collected), n)
    return list(collected.values())


# ===========================================================================
# Step 2 – Summarize individual trajectories (behavioral metrics)
# ===========================================================================

def summarize_trajectory(traj: dict[str, Any]) -> dict[str, Any]:
    steps = traj.get("trajectroy_log", traj.get("steps", []))

    tool_sequence = [
        step.get("action", "unknown")
        for step in steps
        if step.get("action") not in (None, "", "Final Answer", "Finish")
    ]

    repeated_tools = any(
        tool_sequence[i] == tool_sequence[i+1] == tool_sequence[i+2]
        for i in range(len(tool_sequence) - 2)
    ) if len(tool_sequence) >= 3 else False

    verify_kws = {"verify", "check", "cross", "validate", "confirm", "contradict"}
    missing_verification = not any(
        any(kw in (step.get("thought") or "").lower() for kw in verify_kws)
        for step in steps
    )

    premature_finish = len(tool_sequence) < 3 and traj.get("success", False)
    unique_tools     = list(dict.fromkeys(tool_sequence))

    return {
        "question":             traj.get("question", ""),
        "asset_name":           traj.get("asset_name", ""),
        "tool_sequence":        tool_sequence,
        "unique_tools_used":    unique_tools,
        "n_tools_used":         len(unique_tools),
        "n_steps":              len(steps),
        "groundedness_score":   traj.get("groundedness_score"),
        "reached_final_answer": bool(traj.get("final_answer")),
        "success":              traj.get("success", False),
        "error":                traj.get("error"),
        "repeated_tool_calls":  repeated_tools,
        "missing_verification": missing_verification,
        "premature_finish":     premature_finish,
    }


# ===========================================================================
# Phase 2 addition – Credit assignment (Script 2 logic, refactored)
# ===========================================================================

def _rouge_l(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity."""
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _find_final_output(traj: dict[str, Any]) -> dict[str, Any] | None:
    """
    Extract the Finish action's structured output from a trajectory dict.
    Mirrors the robust extractor in Script 2 (credit_assignment.py).
    """
    for step in traj.get("trajectroy_log", traj.get("steps", [])):
        if step.get("action", "").strip() == "Finish":
            for field in ["output", "action_input"]:
                raw = step.get(field, "")
                if isinstance(raw, str):
                    try:
                        raw = json.loads(raw)
                    except Exception:
                        continue
                if isinstance(raw, dict) and "action_input" in raw:
                    inner = raw["action_input"]
                    if isinstance(inner, str):
                        try:
                            inner = json.loads(inner)
                        except Exception:
                            continue
                    if isinstance(inner, dict) and (
                        "justification" in inner or "supporting_passages" in inner
                    ):
                        return inner
                if isinstance(raw, dict) and (
                    "justification" in raw or "supporting_passages" in raw
                ):
                    return raw

    # Fallback: top-level keys
    for key in ["Final Answer", "final_output", "final_answer", "output"]:
        if key in traj:
            val = traj[key]
            if isinstance(val, dict) and "output" in val:
                val = val["output"]
            try:
                parsed = json.loads(val) if isinstance(val, str) else val
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
    return None


_TOOL_NAME_ALIASES: dict[str, str] = {
    "Wikidata":          "wikidata",
    "duckduckgo_search": "ddg-search",
}


def _normalize_tool_name(name: str) -> str:
    return _TOOL_NAME_ALIASES.get(name, name)


def compute_credit_assignment(
    trajectories: list[dict[str, Any]],
) -> dict[str, float]:
    """
    For each trajectory, score each tool observation against the final output
    (supporting passages + justification) using ROUGE-L and Jaccard similarity.
    Aggregate normalized credit per tool across all trajectories.

    Returns a dict mapping tool name → mean normalized credit (0–1).
    """
    tool_credit_totals: dict[str, float] = {}
    tool_credit_counts: dict[str, int]   = {}

    for traj in trajectories:
        final_output = _find_final_output(traj)
        if not final_output:
            continue

        final_passages = [
            p["text"] for p in final_output.get("supporting_passages", [])
            if "text" in p
        ]
        justification = final_output.get("justification", "")
        final_texts   = final_passages + ([justification] if justification else [])
        if not final_texts:
            continue

        steps = traj.get("trajectroy_log", traj.get("steps", []))
        step_scores = []

        for step in steps:
            obs  = step.get("observation", "")
            tool = _normalize_tool_name(step.get("action", "unknown"))
            if not obs or tool in ("Finish", "Final Answer", "", None):
                continue

            rouge = max(_rouge_l(obs, t) for t in final_texts)
            jacc  = max(_jaccard(obs, t) for t in final_texts)
            credit = 0.7 * rouge + 0.3 * jacc
            step_scores.append((tool, credit))

        if not step_scores:
            continue

        total = sum(s for _, s in step_scores)
        for tool, raw_score in step_scores:
            norm = raw_score / total if total > 0 else 0.0
            tool_credit_totals[tool] = tool_credit_totals.get(tool, 0.0) + norm
            tool_credit_counts[tool] = tool_credit_counts.get(tool, 0) + 1

    # Average normalized credit per tool across all trajectories
    result = {
        tool: tool_credit_totals[tool] / tool_credit_counts[tool]
        for tool in tool_credit_totals
    }
    return dict(sorted(result.items(), key=lambda x: -x[1]))


# ===========================================================================
# Step 3a – Build meta-analysis prompt (Phase 1: behavioral only)
# ===========================================================================

def build_meta_analysis_phase1(
    summaries: list[dict[str, Any]],
    current_prompt: str,
    loop_index: int,
) -> str:
    n   = len(summaries)
    ns  = sum(1 for s in summaries if s["success"])
    na  = sum(1 for s in summaries if s["reached_final_answer"])
    nr  = sum(1 for s in summaries if s["repeated_tool_calls"])
    nv  = sum(1 for s in summaries if s["missing_verification"])
    np_ = sum(1 for s in summaries if s["premature_finish"])

    scores    = [s["groundedness_score"] for s in summaries if s["groundedness_score"] is not None]
    avg_score = f"{sum(scores)/len(scores):.3f}" if scores else "N/A"

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

        This is PHASE 1 optimization loop {loop_index}. Your goal is to fix
        BEHAVIORAL failures: repeated tool calls, premature finishing, missing
        verification steps, and low success rates.

        ================================================================
        CURRENT SYSTEM PROMPT (loop {loop_index} input)
        ================================================================
        {current_prompt if current_prompt.strip() else "(no system prompt yet)"}

        ================================================================
        AGGREGATE STATISTICS ({n} trajectories)
        ================================================================
        Successful runs               : {ns} / {n}
        Runs reaching Final Answer    : {na} / {n}
        Average groundedness score    : {avg_score}
        Runs with repeated tool calls : {nr} / {n}
        Runs missing verification     : {nv} / {n}
        Runs with premature finish    : {np_} / {n}
        Tool usage frequency          : {tool_freq}

        ================================================================
        INDIVIDUAL TRAJECTORY SUMMARIES
        ================================================================
        {trajectory_blocks}

        ================================================================
        YOUR TASK
        ================================================================
        1. Identify the TOP 3-5 recurring BEHAVIORAL failure patterns.
           Focus on:
             a) Repeated identical tool calls
             b) Missing cross-verification before concluding
             c) Finishing with fewer than 3 tool calls
             d) Never using wikidata or semanticscholar
             e) Low groundedness despite multiple steps

        2. For each failure, write a CONCRETE, ACTIONABLE guideline.
           Vague:    "Use more tools."
           Concrete: "After a Wikipedia result, always issue one follow-up
                      query to wikidata or semanticscholar before scoring
                      above 0.7."

        3. Produce a COMPLETE REWRITE of the system prompt that:
           - Preserves every rule from the current prompt that is working well.
           - Fixes or replaces rules that are causing the failure patterns above.
           - Adds new CONCRETE, ACTIONABLE guidelines to address those patterns.
             Examples of concrete vs vague:
               Vague:   "Use more tools."
               Concrete: "After retrieving a Wikipedia passage, always issue at
                          least one follow-up query to either wikidata or
                          semanticscholar to cross-reference the claim before
                          assigning a groundedness score above 0.7."
           - Resolves any contradictions between existing rules. For example,
             if one rule says "stop after one source" and another says "always
             verify with two sources", pick the more appropriate one given the
             observed failure patterns and remove or qualify the other.
           - Removes or merges rules that are redundant, overlapping, or
             contradictory with each other.
           - Absorbs any "## Phase N Optimization" sections into the main body
             — do NOT carry those headers forward.

        4. Output ONLY the full rewritten system prompt.
           - No preamble, no commentary, no "## Phase N Optimization" headers,
             no markdown fences, no JSON wrapper.
           - Start directly with the first line of the new system prompt.
           - Stop immediately after the last guideline. Output nothing else.
    """).strip()


# ===========================================================================
# Step 3b – Build meta-analysis prompt (Phase 2: behavioral + credit)
# ===========================================================================

def build_meta_analysis_phase2(
    summaries: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
    current_prompt: str,
    loop_index: int,
) -> tuple[str, dict[str, float]]:
    """
    Returns (meta_prompt_string, credit_dict).
    Runs credit assignment on the trajectories and appends the results
    to the meta-analysis prompt so the optimizer LLM can write
    tool-selection guidelines grounded in actual evidence contribution.
    """
    n   = len(summaries)
    ns  = sum(1 for s in summaries if s["success"])
    na  = sum(1 for s in summaries if s["reached_final_answer"])
    nr  = sum(1 for s in summaries if s["repeated_tool_calls"])
    nv  = sum(1 for s in summaries if s["missing_verification"])
    np_ = sum(1 for s in summaries if s["premature_finish"])

    scores    = [s["groundedness_score"] for s in summaries if s["groundedness_score"] is not None]
    avg_score = f"{sum(scores)/len(scores):.3f}" if scores else "N/A"

    tool_counts: dict[str, int] = {}
    for s in summaries:
        for t in s["tool_sequence"]:
            tool_counts[t] = tool_counts.get(t, 0) + 1
    tool_freq = ", ".join(
        f"{t}={c}" for t, c in sorted(tool_counts.items(), key=lambda x: -x[1])
    ) or "none"

    # --- Credit assignment ---------------------------------------------------
    logger.info("Running credit assignment for phase 2 meta-analysis…")
    credit_scores = compute_credit_assignment(trajectories)
    credit_lines  = "\n".join(
        f"  {tool:<20} {score:.3f}" for tool, score in credit_scores.items()
    ) or "  (no data)"

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

    meta_prompt = textwrap.dedent(f"""
        You are an expert prompt engineer for ReAct-style fact-checking agents.

        This is PHASE 2 optimization loop {loop_index}. Behavioral failures have
        been addressed in Phase 1. Your goal now is to refine TOOL SELECTION
        based on which tools actually contributed evidence to correct final answers.

        ================================================================
        CURRENT SYSTEM PROMPT (loop {loop_index} input)
        ================================================================
        {current_prompt if current_prompt.strip() else "(no system prompt yet)"}

        ================================================================
        AGGREGATE STATISTICS ({n} trajectories)
        ================================================================
        Successful runs               : {ns} / {n}
        Runs reaching Final Answer    : {na} / {n}
        Average groundedness score    : {avg_score}
        Runs with repeated tool calls : {nr} / {n}
        Runs missing verification     : {nv} / {n}
        Runs with premature finish    : {np_} / {n}
        Tool usage frequency          : {tool_freq}

        ================================================================
        TOOL CONTRIBUTION – CREDIT ASSIGNMENT
        (normalized mean credit toward final supporting passages +
         justification, averaged across all trajectories)
        Tool                 Mean Credit
        ----------------------------------------------------------------
{credit_lines}
        ----------------------------------------------------------------
        Interpretation guide:
        - High credit + high usage  → tool is working well; reinforce it.
        - High credit + low usage   → tool is underused; encourage more use.
        - Low credit  + high usage  → tool wastes steps; deprioritize or
                                      restrict to specific claim types.
        - Low credit  + low usage   → tool may be irrelevant for this domain.

        ================================================================
        INDIVIDUAL TRAJECTORY SUMMARIES
        ================================================================
        {trajectory_blocks}

        ================================================================
        YOUR TASK
        ================================================================
        1. Interpret the credit assignment table above.
           Identify tools that are:
             a) High-contributing but underused → add explicit "prefer X" rules
             b) Low-contributing but overused   → add "skip X unless Y" rules
             c) Never used despite relevance    → add domain-specific triggers

        2. Write CONCRETE, ACTIONABLE tool-selection guidelines.
           Examples:
             "For scientific/medical claims, call semanticscholar FIRST.
              Skip wikipedia unless semanticscholar returns no results."
             "wikidata has low credit on general-knowledge questions; only
              use it to verify structured facts (dates, IDs, relationships)."

        3. Also address any remaining behavioral failures still visible
           in the statistics above.

        4. Produce a COMPLETE REWRITE of the system prompt that:
           - Preserves every rule from the current prompt that is working well.
           - Incorporates the tool-selection guidelines above.
           - Fixes or replaces rules that are causing remaining failures.
           - Resolves any contradictions between existing rules. For example,
             if one rule says "stop after one source" and another says "always
             verify with two sources", pick the more appropriate one given the
             observed failure patterns and remove or qualify the other.
           - Removes or merges rules that are redundant, overlapping, or
             contradictory with each other.
           - Absorbs any "## Phase N Optimization" sections into the main body
             — do NOT carry those headers forward.

        5. Output ONLY the full rewritten system prompt.
           - No preamble, no commentary, no "## Phase N Optimization" headers,
             no markdown fences, no JSON wrapper.
           - Start directly with the first line of the new system prompt.
           - Stop immediately after the last guideline. Output nothing else.
    """).strip()

    return meta_prompt, credit_scores


# ===========================================================================
# Step 4 – Call the WatsonX optimizer LLM
# ===========================================================================

def call_optimizer_llm(meta_prompt: str) -> str:
    try:
        from ibm_watsonx_ai.foundation_models import ModelInference  # type: ignore
        from ibm_watsonx_ai import Credentials                        # type: ignore
    except ImportError as exc:
        raise ImportError(
            "ibm_watsonx_ai is required. pip install ibm-watsonx-ai"
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
            "decoding_method":    "greedy",
            "max_new_tokens":     2048,
            "temperature":        0.2,
            "repetition_penalty": 1.05,
            "stop_sequences":     ["<|eom_id|>", "<|eot_id|>"],
        },
    )
    logger.info("Calling optimizer LLM…")
    return model.generate_text(prompt=meta_prompt)


# ===========================================================================
# Single loop runner
# ===========================================================================

def run_one_loop(
    loop_index: int,
    phase: int,           # 1 or 2
    n: int,
    n_threads: int,
    batch_size: int,
    seed_trajectory_dir: Path | None = None,
) -> str:
    logger.info("=" * 60)
    logger.info("PHASE %d  LOOP %d  START", phase, loop_index)
    logger.info("=" * 60)

    loop_dir = _TRAJECTORIES_DIR / f"loop_{loop_index}"

    current_prompt = load_prompt(_LIVE_PROMPT_PATH)
    snapshot_prompt(current_prompt, loop_index)

    # --- Collect -------------------------------------------------------------
    if seed_trajectory_dir is not None and loop_index == 1:
        logger.info("Loading seed trajectories from %s", seed_trajectory_dir)
        trajectories = _load_trajectories_from_dir(seed_trajectory_dir)
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
        logger.error("No trajectories for loop %d – skipping.", loop_index)
        return current_prompt

    # --- Summarize -----------------------------------------------------------
    logger.info("Summarizing %d trajectories…", len(trajectories))
    summaries = [summarize_trajectory(t) for t in trajectories]

    # --- Build meta-analysis -------------------------------------------------
    credit_scores: dict[str, float] = {}

    if phase == 1:
        logger.info("Phase 1: building behavioral meta-analysis…")
        meta_prompt = build_meta_analysis_phase1(summaries, current_prompt, loop_index)
    else:
        logger.info("Phase 2: building behavioral + credit meta-analysis…")
        meta_prompt, credit_scores = build_meta_analysis_phase2(
            summaries, trajectories, current_prompt, loop_index
        )
        # Save credit summary alongside trajectories
        credit_path = loop_dir / "credit_summary.json"
        with credit_path.open("w", encoding="utf-8") as fh:
            json.dump(credit_scores, fh, indent=2)
        logger.info("Credit summary saved → %s", credit_path)

    meta_path = loop_dir / "meta_analysis_prompt.txt"
    meta_path.write_text(meta_prompt, encoding="utf-8")
    logger.info("Meta-analysis prompt → %s", meta_path)

    # --- Optimize ------------------------------------------------------------
    new_prompt = call_optimizer_llm(meta_prompt)

    # Strip stop tokens that WatsonX includes in the returned text
    new_prompt = new_prompt.replace("<|eom_id|>", "").replace("<|eot_id|>", "").strip()

    # Strip leading markdown code fence (``` or ```json at position 0)
    new_prompt = re.sub(r'^```(?:json)?\s*\n', '', new_prompt).strip()

    # Strip JSON wrapper if the LLM responded in ReAct format
    try:
        parsed = json.loads(new_prompt)
        action_input = parsed.get("action_input", "")
        if isinstance(action_input, dict):
            action_input = action_input.get("text", "")
        if isinstance(action_input, str) and action_input.strip():
            new_prompt = action_input.strip()
    except (json.JSONDecodeError, TypeError):
        pass

    # Strip trailing markdown code fence containing JSON (```json {...} ```)
    new_prompt = re.sub(r'\n```(?:json)?\n[\s\S]*?```\s*$', '', new_prompt).strip()

    # Strip trailing bare code fence markers (``` alone, possibly repeated)
    new_prompt = re.sub(r'(\n```)+\s*$', '', new_prompt).strip()

    # Strip trailing bare JSON block starting on its own line
    idx = new_prompt.rfind('\n{')
    if idx != -1:
        try:
            json.loads(new_prompt[idx:].strip())
            new_prompt = new_prompt[:idx].strip()
        except (json.JSONDecodeError, ValueError):
            pass

    # Strip trailing "The final answer is: ..." lines (LLM math-style artifact)
    new_prompt = re.sub(r'(\n+The final answer is:.*)+\s*$', '', new_prompt, flags=re.IGNORECASE).strip()

    # Strip trailing JSON blob artifact (e.g. {"action": ...} appended by model)
    tail_marker = '{"action":'
    tail_idx = new_prompt.find(tail_marker)
    if tail_idx > 0:
        new_prompt = new_prompt[:tail_idx].strip()

    # Strip duplicate prompt — LLM sometimes reproduces the current prompt then
    # adds a second copy; truncate at the second IMPORTANT: if present.
    first_idx = new_prompt.find("IMPORTANT:")
    second_idx = new_prompt.find("IMPORTANT:", first_idx + 1)
    if second_idx > 0:
        new_prompt = new_prompt[:second_idx].strip()

    # Warn if the LLM returned only an additions section instead of a full rewrite
    phase_marker = f"## Phase {phase} Optimization – Loop {loop_index} Additions"
    if phase_marker in new_prompt:
        logger.warning(
            "LLM returned an additions section instead of a full rewrite for "
            "phase %d loop %d – stripping the header and using the content.",
            phase, loop_index,
        )
        new_prompt = new_prompt[new_prompt.rfind(phase_marker):].strip()
        new_prompt = new_prompt[len(phase_marker):].strip()

    # Strip trailing leaked meta-commentary ("Note:", "## Step N:", stray "}$", etc.)
    # Also catches LLM self-narration that bleeds past the actual system prompt.
    leak_match = re.search(
        r'\n+(?:Note:|}\$|## Step \d|The following|The above'
        r'|Based on the analysis|Given these insights'
        r'|In conclusion[,\s]|By following these|Ultimately[,\s]'
        r'|The system should|The refined system|The emphasis on'
        r'|The decision to move|The minimum tool|The action field must)',
        new_prompt,
        flags=re.IGNORECASE,
    )
    if leak_match:
        logger.warning(
            "Stripping leaked LLM meta-commentary from phase %d loop %d output.",
            phase, loop_index,
        )
        new_prompt = new_prompt[:leak_match.start()].strip()

    # Normalize tool names — LLM sometimes uses duckduckgo_search instead of ddg-search
    new_prompt = new_prompt.replace("duckduckgo_search", "ddg-search")

    # --- Persist: replace the prompt with the full rewrite -------------------
    if new_prompt:
        final_prompt = new_prompt
    else:
        logger.warning("No rewrite produced for loop %d – keeping existing prompt.", loop_index)
        final_prompt = current_prompt

    save_prompt(final_prompt, _LIVE_PROMPT_PATH)
    loop_prompt_path = loop_dir / "optimized_prompt.txt"
    save_prompt(final_prompt, loop_prompt_path)

    logger.info("PHASE %d  LOOP %d  COMPLETE. New prompt: %d chars.",
                phase, loop_index, len(final_prompt))
    return final_prompt


def _load_trajectories_from_dir(directory: Path) -> list[dict[str, Any]]:
    files = sorted(directory.glob("*_traj_output.json"))
    if not files:
        raise FileNotFoundError(f"No *_traj_output.json files in {directory}")
    results = []
    for f in files:
        with f.open(encoding="utf-8") as fh:
            results.append(json.load(fh))
    logger.info("Loaded %d trajectories from %s", len(results), directory)
    return results


# ===========================================================================
# Main two-phase optimization loop
# ===========================================================================

def run_optimization_loop(
    phase1_loops: int,
    phase2_loops: int,
    n_trajectories: int,
    n_threads: int,
    batch_size: int,
    seed_trajectory_dir: Path | None,
) -> None:
    """
    Phase 1: loops 1..phase1_loops  — behavioral failure correction.
    Phase 2: loops phase1_loops+1.. — tool contribution refinement
                                       via credit assignment.
    """
    _PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    _TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)

    total_loops = phase1_loops + phase2_loops
    start_time  = datetime.now()

    logger.info(
        "Starting two-phase optimization: Phase 1 = %d loops, Phase 2 = %d loops, "
        "%d trajectories each.",
        phase1_loops, phase2_loops, n_trajectories,
    )

    for loop_idx in range(1, total_loops + 1):
        phase = 1 if loop_idx <= phase1_loops else 2

        run_one_loop(
            loop_index=loop_idx,
            phase=phase,
            n=n_trajectories,
            n_threads=n_threads,
            batch_size=batch_size,
            seed_trajectory_dir=seed_trajectory_dir if loop_idx == 1 else None,
        )

    elapsed = datetime.now() - start_time
    logger.info(
        "All %d loops (%d phase-1 + %d phase-2) complete in %s.\n"
        "Final prompt → %s",
        total_loops, phase1_loops, phase2_loops, elapsed, _LIVE_PROMPT_PATH,
    )


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""
            Two-phase automated Prompt Optimization Loop for the Factly ReActXen Agent.

            Phase 1: fix behavioral failures (repeated tools, premature finish,
                     missing verification).
            Phase 2: refine tool selection using credit assignment — which tools
                     actually contributed evidence to correct final answers.

            Just run:  python prompt_optimizer_v2.py
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase1-loops", type=int, default=DEFAULT_PHASE1_LOOPS,
        help=f"Phase 1 (behavioral) loops (default: {DEFAULT_PHASE1_LOOPS}).",
    )
    parser.add_argument(
        "--phase2-loops", type=int, default=DEFAULT_PHASE2_LOOPS,
        help=f"Phase 2 (credit assignment) loops (default: {DEFAULT_PHASE2_LOOPS}).",
    )
    parser.add_argument(
        "--n", type=int, default=DEFAULT_N_TRAJECTORIES,
        help=f"Trajectories per loop (default: {DEFAULT_N_TRAJECTORIES}).",
    )
    parser.add_argument(
        "--threads", type=int, default=DEFAULT_N_THREADS,
        help=f"Worker threads (default: {DEFAULT_N_THREADS}).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Rows per batch (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--base-prompt", type=Path, default=None,
        help=(
            "Path to an initial system prompt file. Copied to "
            "prompts/system_prompt.txt before loop 1 runs."
        ),
    )
    parser.add_argument(
        "--seed-trajectory-dir", type=Path, default=None,
        help=(
            "Load pre-existing *_traj_output.json files for loop 1 instead of "
            "running the agent. Useful for bootstrapping from an existing traj_store."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.base_prompt:
        if not args.base_prompt.exists():
            raise FileNotFoundError(f"--base-prompt not found: {args.base_prompt}")
        shutil.copy(args.base_prompt, _LIVE_PROMPT_PATH)
        logger.info("Installed base prompt from %s", args.base_prompt)

    run_optimization_loop(
        phase1_loops=args.phase1_loops,
        phase2_loops=args.phase2_loops,
        n_trajectories=args.n,
        n_threads=args.threads,
        batch_size=args.batch_size,
        seed_trajectory_dir=args.seed_trajectory_dir,
    )


if __name__ == "__main__":
    main()
