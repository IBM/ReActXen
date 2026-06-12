import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from reactxen.agents.react.agents import ReactAgent
from langchain_community.agent_toolkits.load_tools import load_tools
from dotenv import load_dotenv
from functools import partial
from reactxen.utils.model_inference import watsonx_llm
import json

load_dotenv()

"""
tools = load_tools(
    ["arxiv", "wikipedia", "ddg-search"],
)
"""

tools = load_tools(
    ["arxiv", "wikipedia"],
)


def generate_validation_prompt(asset_claim_dict):
    assert (
        isinstance(asset_claim_dict, dict) and len(asset_claim_dict) == 1
    ), "Input must be a dictionary with exactly one asset_type: description pair."

    asset_type, claim = list(asset_claim_dict.items())[0]

    prompt = f"""
Task: Validate the accuracy of the following LLM-generated summary related to industrial asset given as Asset Type.

Asset Type: "{asset_type}"

Claim: "{claim}"

Instructions:
1. Search reliable sources such as Wikipedia and arXiv to find passages that either support or contradict this claim.
2. Present at least one supporting and one contradicting passage (if available), along with source citations.
3. Evaluate the overall accuracy of the claim using a confidence score between 0.0 (inaccurate) and 1.0 (fully accurate).
4. Provide a justification for the score based on the content of the retrieved passages.

Output Format (JSON):
{{
  "asset_type": "{asset_type}",
  "claim": "{claim}",
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
  "confidence_score": 0.0,
  "justification": "..."
}}
"""
    return prompt.strip()


def generate_groundedness_check_prompt(asset_claim_dict):
    assert (
        isinstance(asset_claim_dict, dict) and len(asset_claim_dict) == 1
    ), "Input must be a dictionary with exactly one asset_type: description pair."

    asset_type, claim = list(asset_claim_dict.items())[0]

    prompt = f"""
Task: Assess whether the following LLM-generated summary about an industrial asset is **factually grounded** in trusted technical or scientific sources.

Asset Type: "{asset_type}"

Claim (LLM-generated summary): "{claim}"

1. Search reliable sources such as Wikipedia and arXiv to identify passages that either **support** or **contradict** the claim.  
2. Extract and present the most relevant **supporting** and (if available) **contradicting** passages, with clear **source citations**.  
3. Ensure **all available high-quality sources** are considered to provide the most accurate answer, and that the **best passages** are selected based on relevance and clarity.  
4. Assign a **groundedness score** between `0.0` (no support or contradicted) and `1.0` (strongly supported).  
5. Provide a **brief justification** that explains how the evidence supports or contradicts the claim.  
6. When your analysis is complete, **you must conclude with a `Finish` action** that returns a **fully filled**, **valid**, and **parseable JSON object** matching the exact structure shown below.  
   - **Do not use placeholders** like `[answer]`, `<fill here>`, `...`, or `"TBD"`.  
   - The process is **not complete** unless the `Finish` step contains a fully populated JSON object.

Output Format (JSON):
{{
  "asset_type": "{asset_type}",
  "claim": "{claim}",
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
"""
    return prompt.strip()


asset_claim = {
    "rotating_equipment": "Equipment that involves rotating components used for mechanical energy conversion or transmission."
}

asset_claim1 = {
    "rotating_equipment": "Monitors pressure within a compressor to ensure safe operation and detect potential compressor failures or leaks"
}

selected_llm_family = partial(watsonx_llm, max_tokens=5000)


# Load the JSON file
with open("../data/llm_generated_asset_summary.json", "r") as file:
    data = json.load(file)

all_results = {}

with open("../result/llm_generated_asset_summary_result.json", "r") as file:
    all_results = json.load(file)

for key in data.keys():

    if key in all_results.keys():
        continue

    asset_claim = {key: data[key]}
    question = generate_groundedness_check_prompt(asset_claim)

    try:
        print("---------")
        ragent = ReactAgent(
            # question="what sites are present?",
            question=question,
            key="",
            cbm_tools=tools,
            tool_desc=None,
            max_steps=12,
            react_llm_model_id=16,
            react_example="",
            handle_context_length_overflow=True,
            debug=True,
            apply_loop_detection_check=True,
            log_structured_messages=True,
            use_tool_cache=True,
            llm=selected_llm_family,
            enable_tool_partial_match=True,
        )

        aa = ragent.run(name="test")
        all_results[key] = ragent.answer

        #store trajectroy
        try:
          traj = ragent.export_trajectory()
          with open("../traj_store/" + key + "_traj_output.json", "w", encoding="utf-8") as f:
              json.dump(traj, f, ensure_ascii=False, indent=2)
        except Exception as ex1:
            print (ex1)
            exit(0)
            pass

        with open(
            "../result/llm_generated_asset_summary_result.json", "w", encoding="utf-8"
        ) as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
    except Exception as ex:
        print (ex)
        pass
