import json
import logging
import os
from functools import partial
from dotenv import load_dotenv
import pandas as pd

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

# Load previous results if available
all_results = {}
results_file = "sampled_q.csv"
df = pd.read_csv(results_file)
data = df.to_dict(orient="records")

results_file = "sampled_q_answer.jsonl"
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
def generate_groundedness_check_prompt(asset_name, claim_text):
    prompt = """
      Task: Check the Relevance of an LLM-Generated Question for a Given Industrial Asset Class.

      Objective: Determine whether the generated question is relevant to the specified asset type by identifying relevant supporting or contradicting evidence from trusted technical or scientific sources.

      Inputs:
      - Asset Type: "{asset_name}"
      - Generated Question: "{claim_text}"

      Steps:
      1. Search trusted sources such as Wikipedia, arXiv, DuckDuckGo, Wikidata, and Semantic Scholar for technical or scientific information that describes the asset type and covers topics mentioned in the generated question.
      2. Extract the most relevant passages from these sources. Provide clear citations, and label them as either supporting (evidence that the question is related to the asset) or contradicting (evidence that the topic is unrelated or irrelevant).
      3. Evaluate all retrieved evidence and determine whether the generated question is relevant to the given asset type.
      4. Assign a relevancy score from 0.0 (completely unrelated) to 1.0 (highly relevant).
      5. Provide a concise justification explaining why the question is or isn’t relevant based on the evidence.
      6. End with a Finish action that includes a valid JSON object containing all extracted information.

      IMPORTANT:
      - Respond only with a single valid JSON object with exactly two fields: "action" and "action_input".
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
      - Use multiple tools during the search process.
      - If initial results do not yield strong evidence, reformulate queries and try alternative tools.
      - Use retrieved keywords or context from one tool to refine queries for another.
      - Iterate the search until sufficient coverage is achieved.

      Final Output Format (for Finish action only):

      {{
        "action": "Finish",
        "action_input": {{
          "asset_type": "{asset_name}",
          "question": "{claim_text}",
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
          "relevancy_score": 0.0,
          "justification": "..."
        }}
      }}
      """
    return prompt.format(asset_name=asset_name, claim_text=claim_text)

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
    asset_name = entry.get("asset", "unknown_asset")
    entry_id = entry.get("id", "unknown_id")
    unique_key = f"autoq_{entry_id}_{asset_name}"

    if unique_key in all_results:
        continue  # already processed

    question = entry.get("questions", "")

    # Final claim text includes question, options, and selected answer
    claim_text = question
    asset_claim = {asset_name: claim_text}
    print (claim_text)

    prompt = generate_groundedness_check_prompt(asset_name, claim_text)

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
