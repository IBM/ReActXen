import json
import logging
import os
from functools import partial
from dotenv import load_dotenv

from reactxen.agents.react.agents import ReactAgent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from reactxen.utils.model_inference import watsonx_llm

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)

# --- Tool Setup (including Wikidata and SemanticScholar) ---
wikidata = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())
semantic_scholar = SemanticScholarQueryRun()

# Load standard tools (note correct "ddg-search" with hyphen)
tools = load_tools(["arxiv", "wikipedia", "ddg-search"])
tools.append(wikidata)
tools.append(semantic_scholar)

# Define LLM - here we can also pass some temparature
selected_llm_family = partial(watsonx_llm, max_tokens=5000)

# Load Q&A data
input_path = "../data/fmsr_full.json"
with open(input_path, "r") as file:
    data = json.load(file)
    print(f"Loaded {len(data)} Q&A entries")

# Load previous results if available
all_results = {}
results_file = "results.json"
if os.path.exists(results_file):
    try:
        with open(results_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            all_results = json.loads(content) if content else {}
    except (json.JSONDecodeError, IOError) as e:
        logging.warning(f"Could not load results from {results_file}, starting fresh. Reason: {e}")
        all_results = {}

react_example = ""

# Generate prompt with .format() and escaped braces for JSON examples
def generate_groundedness_check_prompt(asset_name, claim_text, answer_text):
    prompt = '''Task: Fact-Check an LLM-Generated Answer to a Multiple-Choice Question about an Industrial Asset.

Objective: Determine whether the LLM-generated answer is factually grounded by identifying relevant supporting or contradicting evidence from trusted technical or scientific sources.

Inputs:
- Asset Type: "{asset_name}"
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
'''
    return prompt.format(asset_name=asset_name, claim_text=claim_text, answer_text=answer_text)

tmp_react_example = """Question: I want to verify if 'Lubrication System Failure' in gearboxes can be monitored or detected using the 'Power Input' sensor.
Thought 1: let me use wikipedia to search the content.
Action 1: wikipedia
Action Input 1: gearbox lubrication failure using power input sensor
Observation 1: suppressed due to length
Thought 2: Based on the evidence, I can now conclude the analysis.  
Action 2: Finish  
Action Input 2: Final Answer: { "asset_type": "gearbox", "claim": "Lubrication failure in Gearbox 3 can be detected using power input sensor", "supporting_passages": [{ "text": "Power consumption increases when lubrication is insufficient in rotating machinery like gearboxes.", "source": "Wikipedia: Gearbox", "highlight": "power consumption increases when lubrication is insufficient" }], "contradicting_passages": [], "groundedness_score": 0.8, "justification": "Evidence indicates that lubrication failure leads to increased power draw, making power input a valid monitoring signal." }"""

# Main processing loop
for entry in data:
    asset_name = entry.get("asset_name", "unknown_asset")
    entry_id = entry.get("id", "unknown_id")
    unique_key = f"{entry_id}_{asset_name}"

    if unique_key in all_results:
        continue  # already processed

    question = entry.get("question", "")
    choices = entry.get("choices", {})
    answer_label = entry.get("answer", "")
    answer_text = choices.get(answer_label, "Unknown answer")

    # Format the choices into a readable list: A. Option1, B. Option2, ...
    options_text = ", ".join([f"{label}. {text}" for label, text in choices.items()])

    # Final claim text includes question, options, and selected answer
    claim_text = f"Q: {question} \nOptions: {options_text}"
    asset_claim = {asset_name: claim_text}
    print (claim_text)

    prompt = generate_groundedness_check_prompt(asset_name, claim_text, answer_text)

    try:
        ragent = ReactAgent(
            question=prompt,
            key="",
            cbm_tools=tools,
            tool_desc=None,
            max_steps=12,
            react_llm_model_id=16,
            react_example=tmp_react_example,
            handle_context_length_overflow=True,
            debug=True,
            apply_loop_detection_check=True,
            log_structured_messages=True,
            use_tool_cache=True,
            llm=selected_llm_family,
            enable_tool_partial_match=True,
            skip_token_counting=True,
        )

        result = ragent.run(name="test")
        agent_answer = ragent.answer

        # Clean and store result
        cleaned_answer = agent_answer.replace("Final Answer:", "").strip()
        all_results[unique_key] = cleaned_answer or "EMPTY"

        # Save to file
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        try:
          traj = ragent.export_trajectory()
          with open("../traj_store/" + unique_key + "_traj_output.json", "w", encoding="utf-8") as f:
              json.dump(traj, f, ensure_ascii=False, indent=2)
        except:
            pass

        logging.info(f"Saved result for {unique_key}")

    except Exception as e:
        logging.error(f"Failed to process {unique_key}: {e}")

logging.info("All done.")
