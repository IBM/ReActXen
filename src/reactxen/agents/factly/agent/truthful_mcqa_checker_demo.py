"""
Factly Prompt Comparison Demo
-----------------------------
An interactive Streamlit demo to compare the execution traces and final answers
of the Best and Worst optimized prompt variants side-by-side.
"""

import sys
import io
import re
import contextlib
import logging
import json
import random
import textwrap
from functools import partial
from pathlib import Path

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Ensure the correct path to the ReactXen modules
_AGENT_DIR = Path(__file__).resolve().parent
_FACTLY_DIR = _AGENT_DIR.parent
# Insert the 'src' directory (parents[2] of _FACTLY_DIR) into sys.path
sys.path.insert(0, str(_FACTLY_DIR.parents[2]))

from reactxen.agents.react.agents import ReactAgent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from reactxen.utils.model_inference import watsonx_llm


# =====================================================================
# Manual Prompt Variables
# =====================================================================

_ROOT_DIR = _FACTLY_DIR.parents[3]

v2_semantic_path = _ROOT_DIR / "genai-proj-results" / "prompt_optimizer_v2_semantic" / "system_prompt.txt"
if v2_semantic_path.exists():
    BEST_PROMPT_TEMPLATE = v2_semantic_path.read_text(encoding="utf-8")
else:
    BEST_PROMPT_TEMPLATE = ""

v2_default_path = _ROOT_DIR / "genai-proj-results" / "prompt_optimizer_v2_default" / "system_prompt.txt"
if v2_default_path.exists():
    WORST_PROMPT_TEMPLATE = v2_default_path.read_text(encoding="utf-8")
else:
    WORST_PROMPT_TEMPLATE = ""

# =====================================================================
# Benchmark Helper Functions
# =====================================================================

BENCHMARK_SUMMARY_PATH = str(_AGENT_DIR / "benchmark_outputs" / "prompt_benchmark_summary.json")

_BENCH_DIR = _AGENT_DIR / "benchmark_outputs"
CREDIT_SUMMARY_PATH   = str(_BENCH_DIR / "benchmark_summary.json")
PER_STEP_CSV_PATH     = str(_BENCH_DIR / "per_step_credit_comparison.csv")
PER_TOOL_CSV_PATH     = str(_BENCH_DIR / "per_tool_credit_comparison.csv")
ABLATION_CSV_PATH     = str(_BENCH_DIR / "ablation_results.csv")
ABLATION_CURVES_IMG   = str(_BENCH_DIR / "ablation_curves.png")
CREDIT_DIST_IMG       = str(_BENCH_DIR / "credit_distribution_plots.png")
TOOL_CREDIT_DIST_IMG  = str(_BENCH_DIR / "tool_credit_distribution_plots.png")

@st.cache_data
def load_benchmark_summary(path: str) -> dict:
    try:
        p = Path(path)
        if not p.exists():
            alt_path = Path(path.replace("summary.json", "summery.json"))
            if alt_path.exists():
                p = alt_path
            else:
                st.warning(f"Benchmark summary file not found at {path}. Live execution will still work.")
                return {}
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        st.warning(f"Error loading benchmark JSON: {e}")
        return {}

def find_best_worst_prompt_records(summary: dict) -> tuple[dict | None, dict | None]:
    if not summary:
        return None, None
        
    best = None
    worst = None
    
    # Format 1: Explicit best/worst
    if "best_prompt" in summary and "worst_prompt" in summary:
        best = summary["best_prompt"]
        worst = summary["worst_prompt"]
        return best, worst
        
    # Format 2: Rankings list
    if "rankings" in summary and isinstance(summary["rankings"], list) and len(summary["rankings"]) > 0:
        best = summary["rankings"][0]
        worst = summary["rankings"][-1]
        return best, worst
        
    # Format 3: Prompts dict (infer from max/min score)
    if "prompts" in summary and isinstance(summary["prompts"], dict) and len(summary["prompts"]) > 0:
        prompts = summary["prompts"]
        score_keys = ["composite_score", "score", "overall_score", "success_rate"]
        best_score = -float('inf')
        worst_score = float('inf')
        
        for p_name, p_data in prompts.items():
            if not isinstance(p_data, dict):
                continue
            
            p_score = 0
            for sk in score_keys:
                if sk in p_data and isinstance(p_data[sk], (int, float)):
                    p_score = p_data[sk]
                    break
                    
            if p_score > best_score:
                best_score = p_score
                best = p_data
                best["_inferred_name"] = p_name
            if p_score < worst_score:
                worst_score = p_score
                worst = p_data
                worst["_inferred_name"] = p_name
                
    return best, worst

def get_numeric_comparison(best: dict | None, worst: dict | None) -> pd.DataFrame:
    if not best or not worst:
        return pd.DataFrame()
        
    rows = []
    for k, v_best in best.items():
        if isinstance(v_best, (int, float)):
            v_worst = worst.get(k)
            if isinstance(v_worst, (int, float)):
                diff = v_best - v_worst
                rows.append({
                    "Metric": k,
                    "Semantic": v_best,
                    "Lexical": v_worst,
                    "Difference": round(diff, 4)
                })
    return pd.DataFrame(rows)

def render_prompt_benchmark_card(title: str, record: dict | None) -> None:
    st.subheader(title)
    if not record:
        st.info("No benchmark data found.")
        return
        
    for k, v in record.items():
        if k == "method_name" or k == "_inferred_name":
            st.markdown(f"**Name:** `{v}`")
        elif isinstance(v, (int, float)):
            st.markdown(f"**{k}:** `{v}`")
        elif isinstance(v, str):
            if len(v) > 100:
                with st.expander(f"{k} (Click to expand)"):
                    st.text(v)
            else:
                st.markdown(f"**{k}:** {v}")
        else:
            st.markdown(f"**{k}:** {v}")

def render_benchmark_comparison(best_record: dict | None, worst_record: dict | None) -> None:
    df = get_numeric_comparison(best_record, worst_record)
    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No numeric comparison data available.")

# =====================================================================
# Helper Functions
# =====================================================================

@st.cache_data
def load_truthfulqa(csv_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(csv_path)
        data['question_id'] = list(range(len(data)))
        return data
    except Exception as e:
        st.error(f"Failed to load TruthfulQA.csv from {csv_path}: {e}")
        return pd.DataFrame()


def build_final_prompt(prompt_template: str, asset_name: str, claim_text: str, answer_text: str) -> str:
    # If the user included the injection variables, use the template directly
    if "{asset_name}" in prompt_template and "{claim_text}" in prompt_template and "{answer_text}" in prompt_template:
        prompt = prompt_template
        prompt = prompt.replace("{asset_name}", asset_name)
        prompt = prompt.replace("{claim_text}", claim_text)
        prompt = prompt.replace("{answer_text}", answer_text)
        return prompt.strip()
    
    # Otherwise, fallback to prepending the template to a default task prompt
    system_section = f"[SYSTEM INSTRUCTIONS]\n{prompt_template.strip()}\n\n" if prompt_template.strip() else ""
    
    task_prompt = textwrap.dedent(f"""
        Task: Fact-Check an LLM-Generated Answer to a Multiple-Choice Question about General Factual Knowledge.

        Objective: Determine whether the LLM-generated answer is factually grounded by identifying relevant supporting or contradicting evidence from trusted technical or scientific sources.

        Inputs:
        - Question Category: "{asset_name}"
        - Multiple-Choice Question: "{claim_text}"
        - LLM-Generated Answer: "{answer_text}"

        Steps:
        1. Search trusted sources such as Wikipedia, arXiv, DuckDuckGo, Wikidata, and Semantic Scholar to find passages that support or contradict the LLM-generated answer.
        2. Extract the most relevant passages from these sources. Provide clear citations, and label them as either supporting or contradicting.
        3. Evaluate all retrieved evidence and determine whether the LLM’s answer is factually grounded.
        4. Assign a groundedness score from 0.0 (no support or contradicted) to 1.0 (fully supported).
        5. Provide a concise justification explaining how the evidence supports or contradicts the answer.
        6. End with a Finish action that includes a valid JSON object containing all extracted information.

        IMPORTANT:
        - You must respond only with a single valid JSON object with exactly two fields: "action" and "action_input".
        - The "action" field must be one of the following tools (case-sensitive):
        - "arxiv"
        - "wikipedia"
        - "ddg-search"
        - "wikidata"
        - "semanticscholar"
        - "Finish"
        - The "action_input" must be a valid query string for the chosen tool.
        - Do NOT include any extra text, reasoning, or comments outside the JSON object.

        Search Strategy Guidance:
        - Use multiple tools (e.g., wikipedia, arXiv, ddg-search, wikidata, semanticscholar) during the search process.
        - If the initial results do not yield strong supporting or contradicting evidence, reformulate your queries and try alternative tools.
        - Most of the time, how you query the tools is critical—effective queries can significantly influence what evidence is retrieved.
        - Use information from one tool (e.g., Wikipedia context or keywords from Semantic Scholar) to refine queries for another tool.
        - Iterate the search process until you are confident that sufficient effort has been made to locate the most relevant evidence.
        - Your goal is not to stop at the first relevant passage, but to ensure comprehensive coverage to justify the groundedness score accurately.

        Final Output Format (for Finish action only):

        {{
        "action": "Finish",
        "action_input": {{
            "asset_type": "{asset_name}",
            "question": "{claim_text}",
            "answer": "{answer_text}",
            "supporting_passages": [
            {{
                "text": "...",
                "source": "...",
                "highlight": "..."
            }}
            ],
            "contradicting_passages": [
            {{
                "text": "...",
                "source": "...",
                "highlight": "..."
            }}
            ],
            "groundedness_score": 0.0,
            "justification": "..."
        }}
        }}
    """).strip()
    return system_section + task_prompt


def extract_final_answer(raw_answer: str) -> dict:
    if not raw_answer:
        return {}
    cleaned = raw_answer.replace("Final Answer:", "").strip()
    start = cleaned.find("{")
    if start == -1:
        return {"raw": raw_answer}
    try:
        # raw_decode stops after the first complete JSON object,
        # ignoring any duplicate or trailing content in the string
        parsed, _ = json.JSONDecoder().raw_decode(cleaned, start)
        if "action_input" in parsed:
            inner = parsed["action_input"]
            # unwrap a second nesting if the agent double-wrapped
            if isinstance(inner, dict) and "action_input" in inner:
                return inner["action_input"]
            return inner
        return parsed
    except Exception:
        return {"raw": raw_answer}


def run_checker_with_prompt(question_row: pd.Series, prompt_template: str, label: str):
    load_dotenv()
    
    # Extract question data
    asset_name = question_row.get("Category", "unknown_asset")
    question = question_row.get("Question", "")
    answer = question_row.get("Best Answer", "")
    incorrect = question_row.get("Best Incorrect Answer", "")
    
    choices = [answer, incorrect]
    random.seed(42) # Deterministic shuffle for demo consistency
    random.shuffle(choices)
    
    options_id = ['A','B']
    options_text = ", ".join([f"{options_id[i]}. {choices[i]}" for i in range(len(choices))])
    claim_text = f"Q: {question}\nOptions: {options_text}"
    
    prompt = build_final_prompt(prompt_template, asset_name, claim_text, answer)
    
    react_example = textwrap.dedent("""
        Question: I want to verify if 'Lubrication System Failure' in gearboxes can be monitored or detected using the 'Power Input' sensor.
        Thought 1: let me use wikipedia to search the content.
        Action 1: wikipedia
        Action Input 1: gearbox lubrication failure using power input sensor
        Observation 1: suppressed due to length
        Thought 2: Based on the evidence, I can now conclude the analysis.  
        Action 2: Finish  
        Action Input 2: Final Answer: { "asset_type": "gearbox", "claim": "...", "supporting_passages": [], "contradicting_passages": [], "groundedness_score": 0.8, "justification": "..." }
    """).strip()

    # Capture logs and stdout
    log_stream = io.StringIO()
    
    try:
        wikidata = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())
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
        
        with contextlib.redirect_stdout(log_stream), contextlib.redirect_stderr(log_stream):
            print(f"--- Starting {label} Run ---")
            print(f"Question: {question}")
            print(f"Evaluating LLM Answer: {answer}")
            print("="*40)
            ragent.run(name="demo_run")
            
        raw_answer = ragent.answer or ""
        parsed_answer = extract_final_answer(raw_answer)
        
        return {
            "success": True,
            "trace": log_stream.getvalue(),
            "parsed_answer": parsed_answer,
            "raw_answer": raw_answer
        }
        
    except Exception as e:
        import traceback
        with contextlib.redirect_stdout(log_stream), contextlib.redirect_stderr(log_stream):
            traceback.print_exc()
        return {
            "success": False,
            "trace": log_stream.getvalue(),
            "error": str(e),
            "parsed_answer": {},
            "raw_answer": ""
        }

# =====================================================================
# Trace & Answer Rendering
# =====================================================================

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\[[0-9;]+m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def render_trace(trace: str) -> None:
    """Parse a ReAct agent trace into collapsible per-step cards."""
    if not trace:
        st.info("No trace captured.")
        return

    thought_re       = re.compile(r"^Thought\s+(\d+)\s*:\s*(.*)",      re.IGNORECASE)
    action_input_re  = re.compile(r"^Action\s+Input\s+(\d+)\s*:\s*(.*)", re.IGNORECASE)
    action_re        = re.compile(r"^Action\s+(\d+)\s*:\s*(.*)",        re.IGNORECASE)
    observation_re   = re.compile(r"^Observation\s+(\d+)\s*:\s*(.*)",   re.IGNORECASE)
    noise_re         = re.compile(r"^(Debug Info|Input Question:|Scratch Pad|I am .+with ReAct|-{5,}|={5,}|\*{5,})", re.IGNORECASE)

    header_lines: list[str] = []
    steps: dict[int, dict] = {}
    current_step_num: int | None = None
    current_field: str | None = None
    current_field_lines: list[str] = []
    in_header = True
    skip_until_step = False

    def is_step_line(t: str) -> bool:
        return bool(thought_re.match(t) or action_input_re.match(t) or action_re.match(t) or observation_re.match(t))

    def flush() -> None:
        nonlocal current_field, current_field_lines
        if current_step_num is not None and current_field is not None:
            if current_field not in steps.setdefault(current_step_num, {}):
                steps[current_step_num][current_field] = "\n".join(current_field_lines).strip()
        current_field = None
        current_field_lines = []

    for raw_line in trace.split("\n"):
        line = _strip_ansi(raw_line).strip()

        if not line:
            if current_field is not None and not skip_until_step:
                current_field_lines.append("")
            continue

        # Blocks whose content should be skipped until the next ReAct step line
        if re.match(r"^(Debug Info|Input Question:)", line, re.IGNORECASE):
            flush()
            skip_until_step = True
            continue

        # Decorative / noise lines — silently drop
        if noise_re.match(line):
            continue

        if skip_until_step:
            if is_step_line(line):
                skip_until_step = False
            else:
                continue

        am_thought      = thought_re.match(line)
        am_action_input = action_input_re.match(line)
        am_action       = action_re.match(line)
        am_obs          = observation_re.match(line)

        if am_thought:
            in_header = False
            flush()
            current_step_num = int(am_thought.group(1))
            current_field = "thought"
            current_field_lines = [am_thought.group(2)]
        elif am_action_input:
            in_header = False
            flush()
            current_step_num = int(am_action_input.group(1))
            current_field = "action_input"
            current_field_lines = [am_action_input.group(2)]
        elif am_action:
            in_header = False
            flush()
            current_step_num = int(am_action.group(1))
            current_field = "action"
            current_field_lines = [am_action.group(2)]
        elif am_obs:
            in_header = False
            flush()
            current_step_num = int(am_obs.group(1))
            current_field = "observation"
            current_field_lines = [am_obs.group(2)]
        else:
            if in_header:
                header_lines.append(line)
            elif current_field is not None:
                current_field_lines.append(line)

    flush()

    header_text = "\n".join(l for l in header_lines if l).strip()
    if header_text:
        st.caption(header_text)

    if not steps:
        with st.expander("Raw Trace", expanded=True):
            st.code(_strip_ansi(trace), language="text")
        return

    tool_icons = {
        "wikipedia":          "📖",
        "arxiv":              "📄",
        "ddg-search":         "🔍",
        "duckduckgo_search":  "🔍",
        "wikidata":           "🗃️",
        "semanticscholar":    "🎓",
        "finish":             "✅",
    }
    tool_labels = {
        "wikipedia":          "Wikipedia",
        "arxiv":              "arXiv",
        "ddg-search":         "DuckDuckGo",
        "duckduckgo_search":  "DuckDuckGo",
        "wikidata":           "Wikidata",
        "semanticscholar":    "Semantic Scholar",
        "finish":             "Finish",
    }

    for step_num in sorted(steps.keys()):
        step        = steps[step_num]
        thought     = step.get("thought", "")
        action      = step.get("action", "").strip()
        action_input = step.get("action_input", "")
        observation = step.get("observation", "")
        action_lower = action.lower()

        icon       = tool_icons.get(action_lower,  "🔧")
        name       = tool_labels.get(action_lower, action if action else f"Step {step_num}")
        is_finish  = action_lower == "finish"
        is_error   = "invalid action" in observation.lower()
        status_tag = " ⚠️" if is_error else ""

        with st.expander(f"Step {step_num} — {icon} {name}{status_tag}", expanded=False):
            if thought:
                st.markdown("**💭 Thought**")
                st.info(thought)

            if not is_finish:
                if action or action_input:
                    left, right = st.columns([1, 3])
                    if action:
                        left.markdown("**Tool**")
                        left.code(action)
                    if action_input:
                        right.markdown("**Query**")
                        right.code(action_input)
                if observation:
                    st.markdown("**📄 Observation**")
                    if is_error:
                        st.warning(observation)
                    else:
                        preview = observation[:600]
                        if len(observation) > 600:
                            preview += f"\n\n*…({len(observation) - 600} more characters)*"
                        st.text(preview)
            else:
                st.markdown("**📋 Final Output**")
                if action_input:
                    try:
                        cleaned = action_input.replace("Final Answer:", "").strip()
                        start   = cleaned.find("{")
                        parsed, _ = json.JSONDecoder().raw_decode(cleaned, max(start, 0))
                        if "action_input" in parsed:
                            inner = parsed["action_input"]
                            parsed = inner["action_input"] if isinstance(inner, dict) and "action_input" in inner else inner
                        render_final_answer(parsed)
                    except Exception:
                        st.text(action_input)


def render_final_answer(parsed: dict) -> None:
    """Render the parsed final answer as structured UI components."""
    if not parsed:
        st.info("No answer was returned.")
        return

    if "raw" in parsed and len(parsed) == 1:
        st.warning("Could not parse a structured answer from the agent output.")
        st.text(parsed["raw"])
        return

    score = parsed.get("groundedness_score")
    justification = parsed.get("justification", "")
    supporting = parsed.get("supporting_passages", [])
    contradicting = parsed.get("contradicting_passages", [])
    question = parsed.get("question", "")
    answer = parsed.get("answer", "")

    if question or answer:
        fact_check_md = ""
        if question:
            fact_check_md += f"**Statement being fact-checked:**\n\n{question}"
        if answer:
            fact_check_md += f"\n\n**LLM Answer evaluated:** {answer}"
        st.info(fact_check_md)

    if score is not None:
        try:
            score_float = float(score)
            st.metric("Groundedness Score", f"{score_float:.2f} / 1.00")
        except (ValueError, TypeError):
            st.markdown(f"**Groundedness Score:** {score}")

    if justification:
        st.markdown("**Justification**")
        st.info(justification)

    if isinstance(supporting, list) and supporting:
        st.markdown(f"**Supporting Evidence** ({len(supporting)} passage{'s' if len(supporting) != 1 else ''})")
        for i, p in enumerate(supporting):
            source = p.get("source", f"Source {i + 1}") if isinstance(p, dict) else f"Passage {i + 1}"
            with st.expander(f"Supporting {i + 1}: {source}"):
                if isinstance(p, dict):
                    if p.get("highlight"):
                        st.markdown(f"*{p['highlight']}*")
                    if p.get("text"):
                        st.markdown(p["text"])
                else:
                    st.text(str(p))

    if isinstance(contradicting, list) and contradicting:
        st.markdown(f"**Contradicting Evidence** ({len(contradicting)} passage{'s' if len(contradicting) != 1 else ''})")
        for i, p in enumerate(contradicting):
            source = p.get("source", f"Source {i + 1}") if isinstance(p, dict) else f"Passage {i + 1}"
            with st.expander(f"Contradicting {i + 1}: {source}"):
                if isinstance(p, dict):
                    if p.get("highlight"):
                        st.markdown(f"*{p['highlight']}*")
                    if p.get("text"):
                        st.markdown(p["text"])
                else:
                    st.text(str(p))

    skip_keys = {"groundedness_score", "justification", "supporting_passages",
                 "contradicting_passages", "asset_type", "question", "answer"}
    extra = {k: v for k, v in parsed.items() if k not in skip_keys}
    if extra:
        with st.expander("Additional Fields"):
            for k, v in extra.items():
                st.markdown(f"**{k}:** {v}")


# =====================================================================
# Credit Assignment Benchmark Helpers
# =====================================================================

@st.cache_data
def load_credit_summary(path: str) -> dict:
    try:
        p = Path(path)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        return {}

@st.cache_data
def load_csv_safe(path: str) -> pd.DataFrame:
    try:
        p = Path(path)
        return pd.read_csv(p) if p.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def render_credit_assignment_tab() -> None:
    st.header("Credit Assignment Benchmark")
    st.markdown(
        "Compares **lexical** (ROUGE-L / Jaccard) and **semantic** (embedding similarity) "
        "credit assignment methods across retrieval steps."
    )

    summary = load_credit_summary(CREDIT_SUMMARY_PATH)

    # ── Summary metric cards ─────────────────────────────────────────
    if summary:
        lex = summary.get("lexical", {})
        sem = summary.get("semantic", {})
        winners = summary.get("winner_by_metric", {})
        tool_dist = summary.get("tool_distribution_comparison", {})

        metric_labels = {
            "mean_entropy":           "Mean Entropy",
            "mean_gini":              "Mean Gini",
            "mean_top1_mass":         "Top-1 Credit Mass",
            "mean_top3_mass":         "Top-3 Credit Mass",
            "mean_effective_num_steps": "Effective # Steps",
            "ablation_auc":           "Ablation AUC",
        }

        st.subheader("Lexical vs Semantic — Summary Metrics")
        header_cols = st.columns([2, 1, 1, 1])
        header_cols[0].markdown("**Metric**")
        header_cols[1].markdown("**Lexical**")
        header_cols[2].markdown("**Semantic**")
        header_cols[3].markdown("**Better**")

        for key, label in metric_labels.items():
            v_lex = lex.get(key)
            v_sem = sem.get(key)
            if v_lex is None and v_sem is None:
                continue
            cols = st.columns([2, 1, 1, 1])
            cols[0].markdown(label)
            cols[1].markdown(f"`{v_lex:.4f}`" if isinstance(v_lex, float) else str(v_lex))
            cols[2].markdown(f"`{v_sem:.4f}`" if isinstance(v_sem, float) else str(v_sem))
            # higher AUC / top mass / effective steps = better; lower entropy / gini = better
            lower_is_better = key in ("mean_entropy", "mean_gini")
            if isinstance(v_lex, float) and isinstance(v_sem, float):
                best = "Lexical" if (v_lex < v_sem if lower_is_better else v_lex > v_sem) else "Semantic"
                cols[3].markdown(f"{'Lexical' if best == 'Lexical' else 'Semantic'}")

        st.divider()

        # ── Winner summary ───────────────────────────────────────────
        if winners:
            st.subheader("Winner by Ablation Metric")
            w_cols = st.columns(len(winners))
            for col, (metric, winner) in zip(w_cols, winners.items()):
                col.metric(metric.replace("_", " ").title(), winner.title())

        st.divider()

        # ── Tool distribution comparison ──────────────────────────────
        if tool_dist:
            st.subheader("Tool Distribution Similarity (Lexical vs Semantic)")
            td_cols = st.columns(len(tool_dist))
            for col, (k, v) in zip(td_cols, tool_dist.items()):
                col.metric(k.replace("_", " ").title(), f"{v:.4f}")

        st.divider()

    # ── Per-step table ───────────────────────────────────────────────
    st.subheader("Per-Step Credit Scores")
    per_step_df = load_csv_safe(PER_STEP_CSV_PATH)
    if not per_step_df.empty:
        display_cols = [c for c in ["step", "tool", "lexical_credit_norm", "semantic_credit_norm",
                                     "rouge_l", "jaccard", "semantic_sim"] if c in per_step_df.columns]
        st.dataframe(per_step_df[display_cols].rename(columns={
            "lexical_credit_norm":  "Lexical Credit (norm)",
            "semantic_credit_norm": "Semantic Credit (norm)",
            "rouge_l":  "ROUGE-L",
            "jaccard":  "Jaccard",
            "semantic_sim": "Semantic Sim",
        }), use_container_width=True)

        if "lexical_credit_norm" in per_step_df.columns and "semantic_credit_norm" in per_step_df.columns:
            chart_df = per_step_df[["step", "tool", "lexical_credit_norm", "semantic_credit_norm"]].copy()
            chart_df = chart_df.rename(columns={
                "lexical_credit_norm": "Lexical",
                "semantic_credit_norm": "Semantic",
            })
            chart_df["label"] = chart_df["step"].astype(str) + " – " + chart_df["tool"].astype(str)
            st.bar_chart(chart_df.set_index("label")[["Lexical", "Semantic"]])
    else:
        st.info("per_step_credit_comparison.csv not found.")

    st.divider()

    # ── Per-tool table ───────────────────────────────────────────────
    st.subheader("Per-Tool Credit Distribution")
    per_tool_df = load_csv_safe(PER_TOOL_CSV_PATH)
    if not per_tool_df.empty:
        display_cols = [c for c in ["tool", "tool_credit_lexical", "tool_credit_semantic"] if c in per_tool_df.columns]
        st.dataframe(per_tool_df[display_cols].rename(columns={
            "tool_credit_lexical":  "Lexical Credit",
            "tool_credit_semantic": "Semantic Credit",
        }), use_container_width=True)

        if "tool_credit_lexical" in per_tool_df.columns and "tool_credit_semantic" in per_tool_df.columns:
            st.bar_chart(
                per_tool_df.set_index("tool")[["tool_credit_lexical", "tool_credit_semantic"]].rename(columns={
                    "tool_credit_lexical":  "Lexical",
                    "tool_credit_semantic": "Semantic",
                })
            )
    else:
        st.info("per_tool_credit_comparison.csv not found.")

    st.divider()

    # ── Ablation results ─────────────────────────────────────────────
    st.subheader("Ablation Results — Degradation at Top-k Steps")
    ablation_df = load_csv_safe(ABLATION_CSV_PATH)
    if not ablation_df.empty:
        st.dataframe(ablation_df, use_container_width=True)
        if {"method", "k", "degradation"}.issubset(ablation_df.columns):
            pivot = ablation_df.pivot_table(index="k", columns="method", values="degradation")
            st.line_chart(pivot)
    else:
        st.info("ablation_results.csv not found.")

    st.divider()

    # ── Visualisation images ─────────────────────────────────────────
    st.subheader("Visualisations")
    img_cols = st.columns(3)
    for col, (img_path, caption) in zip(img_cols, [
        (CREDIT_DIST_IMG,    "Credit Distribution"),
        (TOOL_CREDIT_DIST_IMG, "Tool Credit Distribution"),
        (ABLATION_CURVES_IMG,  "Ablation Curves"),
    ]):
        p = Path(img_path)
        if p.exists():
            col.image(str(p), caption=caption, use_container_width=True)
        else:
            col.info(f"{caption} image not found.")


# =====================================================================
# Streamlit App Layout
# =====================================================================

def main():
    st.set_page_config(page_title="Factly Prompt Comparison Demo", layout="wide")

    # Hide Streamlit chrome
    st.markdown(
        "<style>header {visibility: hidden !important;} footer {visibility: hidden !important;}</style>",
        unsafe_allow_html=True,
    )

    st.title("Factly Demo")

    tab1, tab2 = st.tabs(["Prompt Comparison", "Credit Assignment Benchmark"])

    # ═══════════════════════════════════════════════════════════════════
    # TAB 1 — Prompt Comparison
    # ═══════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("""
            Compare the execution traces and final answers of the **best** and **"worst"**
            performing optimized prompt variants side-by-side on the same TruthfulQA question.
        """)

        summary_data = load_benchmark_summary(BENCHMARK_SUMMARY_PATH)
        best_record, worst_record = find_best_worst_prompt_records(summary_data)

        st.subheader("Semantic vs Lexical Benchmark Comparison")
        render_benchmark_comparison(best_record, worst_record)

        st.divider()

        csv_path = str(_FACTLY_DIR / "data" / "TruthfulQA.csv")
        df = load_truthfulqa(csv_path)

        if df.empty:
            st.stop()

        question_options = [
            f"[{row['question_id']}] {row['Category']}: {row['Question'][:60]}..."
            for _, row in df.iterrows()
        ]

        selected_option = st.selectbox("Select a Question from TruthfulQA", question_options)
        selected_idx = question_options.index(selected_option)
        selected_row = df.iloc[selected_idx]

        st.markdown("### Selected Question Details")
        st.info(
            f"**Question:** {selected_row['Question']}\n\n"
            f"**Correct Answer:** {selected_row['Best Answer']}\n\n"
            f"**Incorrect Answer:** {selected_row['Best Incorrect Answer']}"
        )

        if st.button("Run Comparison", type="primary"):
            with st.spinner("Executing agent with both prompts sequentially… Please wait."):

                best_results  = run_checker_with_prompt(selected_row, BEST_PROMPT_TEMPLATE,  "Semantic Prompt")
                worst_results = run_checker_with_prompt(selected_row, WORST_PROMPT_TEMPLATE, "Lexical Prompt")

            st.success("Execution Complete!")

            col1, col2 = st.columns(2)

            with col1:
                st.header("🏆 Semantic Prompt")
                if best_record:
                    b_name  = best_record.get("method_name", best_record.get("_inferred_name", "Unknown"))
                    b_score = best_record.get("composite_score", best_record.get("score", "N/A"))
                    st.caption(f"**Composite Score:** {b_score}")
                else:
                    st.caption("Benchmark metadata not found for this prompt.")

                with st.expander("Agent Reasoning Trace", expanded=False):
                    render_trace(best_results["trace"])
                st.markdown("#### Final Answer")
                if best_results["success"]:
                    render_final_answer(best_results["parsed_answer"])
                else:
                    st.error(f"Execution Failed: {best_results.get('error', 'Unknown Error')}")
                    if best_results.get("trace"):
                        with st.expander("Error Trace"):
                            st.code(best_results["trace"], language="text")

            with col2:
                st.header("📉 Lexical Prompt")
                if worst_record:
                    w_name  = worst_record.get("method_name", worst_record.get("_inferred_name", "Unknown"))
                    w_score = worst_record.get("composite_score", worst_record.get("score", "N/A"))
                    st.caption(f"**Method:** {w_name} | **Score:** {w_score}")
                else:
                    st.caption("Benchmark metadata not found for this prompt.")

                with st.expander("Agent Reasoning Trace", expanded=False):
                    render_trace(worst_results["trace"])
                st.markdown("#### Final Answer")
                if worst_results["success"]:
                    render_final_answer(worst_results["parsed_answer"])
                else:
                    st.error(f"Execution Failed: {worst_results.get('error', 'Unknown Error')}")
                    if worst_results.get("trace"):
                        with st.expander("Error Trace"):
                            st.code(worst_results["trace"], language="text")


    # ═══════════════════════════════════════════════════════════════════
    # TAB 2 — Credit Assignment Benchmark
    # ═══════════════════════════════════════════════════════════════════
    with tab2:
        render_credit_assignment_tab()


if __name__ == "__main__":
    main()
