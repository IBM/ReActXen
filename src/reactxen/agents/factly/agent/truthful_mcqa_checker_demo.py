"""
Factly Prompt Comparison Demo
-----------------------------
An interactive Streamlit demo to compare the execution traces and final answers
of the Best and Worst optimized prompt variants side-by-side.
"""

import sys
import io
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

v2_semantic_path = _ROOT_DIR / "genai-proj-results" / "n=5" / "prompt_optimizer_v2_semantic" / "system_prompt.txt"
if v2_semantic_path.exists():
    BEST_PROMPT_TEMPLATE = v2_semantic_path.read_text(encoding="utf-8")
else:
    BEST_PROMPT_TEMPLATE = ""

# Keep Worst as empty so it defaults to the unoptimized baseline task_prompt
WORST_PROMPT_TEMPLATE = ""

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
    try:
        cleaned = raw_answer.replace("Final Answer:", "").strip()
        parsed = json.loads(cleaned)
        if "action_input" in parsed:
            return parsed["action_input"]
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
# Streamlit App Layout
# =====================================================================

def main():
    st.set_page_config(page_title="Factly Prompt Comparison Demo", layout="wide")
    
    st.title("Factly Prompt Comparison Demo")
    st.markdown("""
        This interactive demo allows you to run two different system prompts side-by-side 
        on the exact same question. The goal is to visually compare how the agent reasons 
        through the problem, which tools it selects, and how the final groundedness scores differ.
    """)
    
    csv_path = str(_FACTLY_DIR / "data" / "TruthfulQA.csv")
    df = load_truthfulqa(csv_path)
    
    if df.empty:
        st.stop()
        
    # Hide Streamlit header (rerun, settings, deploy) and footer
    hide_streamlit_style = """
        <style>
        header {visibility: hidden !important;}
        footer {visibility: hidden !important;}
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    # Create dropdown strings: "ID - Category: Question snippet"
    question_options = [
        f"[{row['question_id']}] {row['Category']}: {row['Question'][:60]}..." 
        for _, row in df.iterrows()
    ]
    
    selected_option = st.selectbox("Select a Question from TruthfulQA", question_options)
    selected_idx = question_options.index(selected_option)
    selected_row = df.iloc[selected_idx]
    
    st.markdown("### Selected Question Details")
    st.info(f"**Question:** {selected_row['Question']}\n\n**Correct Answer:** {selected_row['Best Answer']}\n\n**Incorrect Answer:** {selected_row['Best Incorrect Answer']}")
    
    if st.button("Run Comparison", type="primary"):
        with st.spinner("Executing agent with both prompts sequentially... Please wait."):
            
            # Run Best
            best_results = run_checker_with_prompt(selected_row, BEST_PROMPT_TEMPLATE, "Best Prompt")
            
            # Run Worst
            worst_results = run_checker_with_prompt(selected_row, WORST_PROMPT_TEMPLATE, "Worst Prompt")
            
            st.success("Execution Complete!")
            
            # --- Rendering Side-by-Side Panels ---
            col1, col2 = st.columns(2)
            
            with col1:
                st.header("🏆 Best Prompt")
                st.markdown("#### Terminal Trace")
                st.code(best_results["trace"], language="text")
                st.markdown("#### Final Answer Output")
                if best_results["success"]:
                    st.json(best_results["parsed_answer"])
                else:
                    st.error(f"Execution Failed: {best_results.get('error', 'Unknown Error')}")
                    
            with col2:
                st.header("📉 Worst Prompt")
                st.markdown("#### Terminal Trace")
                st.code(worst_results["trace"], language="text")
                st.markdown("#### Final Answer Output")
                if worst_results["success"]:
                    st.json(worst_results["parsed_answer"])
                else:
                    st.error(f"Execution Failed: {worst_results.get('error', 'Unknown Error')}")
                    
            # --- Comparison Summary ---
            st.divider()
            st.subheader("Comparison Summary")
            
            best_score = best_results["parsed_answer"].get("groundedness_score", "N/A") if best_results["success"] else "Error"
            worst_score = worst_results["parsed_answer"].get("groundedness_score", "N/A") if worst_results["success"] else "Error"
            
            st.markdown(f"- **Ground Truth Correct Answer**: {selected_row['Best Answer']}")
            st.markdown(f"- **Best Prompt Assigned Score**: `{best_score}`")
            st.markdown(f"- **Worst Prompt Assigned Score**: `{worst_score}`")
            
            best_justification = best_results["parsed_answer"].get("justification", "None provided") if best_results["success"] else "N/A"
            worst_justification = worst_results["parsed_answer"].get("justification", "None provided") if worst_results["success"] else "N/A"
            
            c1, c2 = st.columns(2)
            with c1:
                st.info(f"**Best Prompt Justification:**\n\n{best_justification}")
            with c2:
                st.warning(f"**Worst Prompt Justification:**\n\n{worst_justification}")

if __name__ == "__main__":
    main()
