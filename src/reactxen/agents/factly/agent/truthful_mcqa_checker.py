import json
import logging
import os
from functools import partial
from dotenv import load_dotenv
import math
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from reactxen.agents.react.agents import ReactAgent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from reactxen.utils.model_inference import watsonx_llm
import pandas as pd
import random


def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load Q&A data
input_path = "../data/TruthfulQA.csv"
# with open(input_path, "r") as file:
data = pd.read_csv(input_path)
data['question_id'] = list(range(len(data)))
print(f"Loaded {len(data)} Q&A entries")
print (data)

# print (data)
# exit(0)

# Load previous results if available
all_results = {}
results_file = "truthfulqa_results.json"
if os.path.exists(results_file):
    try:
        with open(results_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            all_results = json.loads(content) if content else {}
    except (json.JSONDecodeError, IOError) as e:
        logging.warning(
            f"Could not load results from {results_file}, starting fresh. Reason: {e}"
        )
        all_results = {}

react_example = ""


PROMPT_OG = """Task: Fact-Check an LLM-Generated Answer to a Multiple-Choice Question about General Factual Knowledge.

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
"""

# Generate prompt with .format() and escaped braces for JSON examples
def generate_groundedness_check_prompt(asset_name, claim_text, answer_text):
    prompt_new = """Task: Fact-Check an LLM-Generated Answer to a Multiple-Choice Question about General Factual Knowledge.

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
- Follow this priority sequence. Move to the next tool only if the current one yields insufficient evidence. The "action" field must be one of the following tools (case-sensitive)
   1. `wikipedia` — Start here for most claims
       Use for: named entities, historical events, established science, geography, public figures, organizations, and general knowledge.
       Skip if: the claim is highly technical/scientific, very recent (post-2023), or involves specific numerical data not likely in Wikipedia.
   2. `wikidata` — Use for structured facts
       Use for: dates, IDs, relationships between entities (e.g., "X is the capital of Y", "Person A was born in year B"), or when Wikipedia prose was ambiguous and you need a structured fact to confirm.
       Skip if: Wikipedia already gave a definitive answer.
   3. `ddg-search` — Use for recent or niche claims
       Use for: recent events, news, claims about products/companies/people that may not be in Wikipedia, or when steps 1–2 returned nothing useful.
       Skip if: the claim is clearly historical or academic — DDG results will be noisy.
   4. `semanticscholar` — Use for scientific/academic claims
       Use for: research findings, medical claims, statistics from studies, technical benchmarks, or any claim that references academic literature.
       Skip if: the claim is not scientific or the DDG results already surfaced a credible source.
   5. `arxiv` — Use for cutting-edge AI/ML or preprint claims
       Use for: claims about specific AI models, recent machine learning benchmarks, or technical results that may only exist as preprints.
       Skip if: the claim is not about recent AI/ML/CS research. Do not use arxiv for general science — prefer semanticscholar.
   6. `Finish` — Call when you have enough evidence
       Call Finish when:
       - You have found at least one high-quality source that directly supports or refutes the claim, OR
       - You have exhausted all relevant tools and found no evidence (verdict: unverifiable).
   Never call Finish before attempting at least one tool.
- Decision Rules:
   - Stop early: as soon as one tool returns a direct, unambiguous answer, call `Finish`. Do not keep searching to "double-check" unless the claim is high-stakes or the source seems unreliable.
   - Skip irrelevant tools:  if a claim is clearly in one domain, skip tools outside that domain entirely. A claim about Napoleon's birth year does not need arxiv or semanticscholar.
   - Don't repeat tools: never call the same tool twice with the same or near-identical query. Rephrase only if the first query was poorly formed and no results were returned.
   - Prioritize specificity: use the most specific tool first for the claim type:
       - Scientific claim → go straight to `semanticscholar` or `arxiv`, skip wikipedia
       - AI/ML benchmark claim → go straight to `arxiv`
       - Structured fact (date, location) → try `wikidata` before wikipedia prose
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
"""
    return PROMPT_OG.format(
        asset_name=asset_name, claim_text=claim_text, answer_text=answer_text
    )


tmp_react_example = """Question: I want to verify if 'Lubrication System Failure' in gearboxes can be monitored or detected using the 'Power Input' sensor.
Thought 1: let me use wikipedia to search the content.
Action 1: wikipedia
Action Input 1: gearbox lubrication failure using power input sensor
Observation 1: suppressed due to length
Thought 2: Based on the evidence, I can now conclude the analysis.  
Action 2: Finish  
Action Input 2: Final Answer: { "asset_type": "gearbox", "claim": "Lubrication failure in Gearbox 3 can be detected using power input sensor", "supporting_passages": [{ "text": "Power consumption increases when lubrication is insufficient in rotating machinery like gearboxes.", "source": "Wikipedia: Gearbox", "highlight": "power consumption increases when lubrication is insufficient" }], "contradicting_passages": [], "groundedness_score": 0.8, "justification": "Evidence indicates that lubrication failure leads to increased power draw, making power input a valid monitoring signal." }"""

n_thread = 8
print(f"# of samples = {len(data)}")
chunk_size = math.ceil(len(data) / n_thread)
data_chunks = []
for id in range(n_thread):
    data_chunks.append(data[(id * chunk_size) : min((id + 1) * chunk_size, len(data))])


def process_chunk(chunk_data: pd.DataFrame):
    # debug
    # print("PROCESS_CHUNK CALLED")
    # print("chunk size:", len(chunk_data))

    # Main processing loop
    for _, entry in chunk_data.iterrows():
        asset_name = entry.get("Category", "unknown_asset")
        entry_id = entry.get("question_id", "unknown_id")
        unique_key = f"{entry_id}_{asset_name}"

        if unique_key in all_results:
            print("Skipping existing:", unique_key)  # added for debug
            continue  # already processed

        question = entry.get("Question", "")
        answer = entry.get("Best Answer", "")
        incorrect_answer = entry.get("Best Incorrect Answer", "")

        choices = [answer, incorrect_answer]
        answer_text = answer
        options_id = ['A','B']
        random.shuffle(choices)

        # Format the choices into a readable list: A. Option1, B. Option2, ...
        options_text = ", ".join(
            [f"{options_id[l_ind]}. {choices[l_ind]}" for l_ind in range(len(choices))]
        )
        print (options_text)

        # Final claim text includes question, options, and selected answer
        claim_text = f"Q: {question} \nOptions: {options_text}"
        print(claim_text)

        prompt = generate_groundedness_check_prompt(asset_name, claim_text, answer_text)

        try:

            # --- Tool Setup (including Wikidata and SemanticScholar) ---
            wikidata = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())
            semantic_scholar = SemanticScholarQueryRun()

            # Load standard tools (note correct "ddg-search" with hyphen)
            tools = load_tools(["arxiv", "wikipedia", "ddg-search"])
            tools.append(wikidata)
            tools.append(semantic_scholar)

            # Define LLM - here we can also pass some temparature
            selected_llm_family = partial(watsonx_llm, max_tokens=5000)

            try:
                ragent = ReactAgent(
                    question=prompt,
                    key="",
                    cbm_tools=tools,
                    tool_desc=None,
                    max_steps=12,
                    react_llm_model_id=16,
                    react_example=tmp_react_example,
                    handle_context_length_overflow=False,
                    debug=True,
                    apply_loop_detection_check=True,
                    log_structured_messages=True,
                    use_tool_cache=True,
                    llm=selected_llm_family,
                    enable_tool_partial_match=True,
                    skip_token_counting=True,
                )
                print("created agent.....")
            except Exception as ex:
                print(ex)

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
                with open(
                    "../traj_store/" + unique_key + "_traj_output.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(traj, f, ensure_ascii=False, indent=2)
            except:
                pass

            logging.info(f"Saved result for {unique_key}")

        except Exception as e:
            logging.error(f"Failed to process {unique_key}: {e}")


logging.info("All done.")


def generate_solution(start_i, data, n_thread=16, batch_size=10):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    #range(start_i, start_i + 200, batch_size)
    batches = [
        data.iloc[i : i + batch_size] for i in range(start_i, start_i + 1, batch_size)
    ]

    # debug
    # print("start_i:", start_i)
    # print("num batches:", len(batches))
    # print("batch sizes:", [len(b) for b in batches])

    logging.info(f"Submitting {len(batches)} batches...")

    with ThreadPoolExecutor(max_workers=n_thread) as executor:
        futures = [
            executor.submit(process_chunk, chunk_data=batch) for batch in batches
        ]

        done, not_done = wait(futures, timeout=600)

        # Cancel tasks that didn’t finish
        for future in not_done:
            future.cancel()
            print("Cancelled a slow task")


if __name__ == "__main__":
    #for i in range(0, 1000, 200):
    #    generate_solution(i, data, n_thread=8, batch_size=10)

    # first parameter corresponds to row you want in TruthfulQA.csv (-2 to row number to account for 0-indexing & header)
    generate_solution(35, data, n_thread=1, batch_size=1)