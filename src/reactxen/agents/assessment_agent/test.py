from cbm_gen.agents.assessment_agent.agent import TaskAssessmentAgent
from cbm_gen.utils.model_inference import watsonx_llm
from cbm_gen.demo.agents.IoT.utils import getTools
from cbm_gen.agents.react.prompts.skysparkfewshots import SKYSPARK1
import json

A = TaskAssessmentAgent(llm=watsonx_llm, model_id=6)
samples = SKYSPARK1.split("\n\n")
tasks = []

for i in range(len(samples)):
    splitlines = samples[i].split("\n")
    task = splitlines[0]
    task = task.replace("Question: ", "")
    tasks.append(task)

tools, desc = getTools()

def generate_confidence(input_file, output_file):

    with open(input_file, "r") as file:
        data = json.load(file)

    all_res = []
    for item_index, item in enumerate(data):
        ans = A.evaluate_response(
            item["instruction"],
            "IoT",
            desc,
            "Download Sensor Related Data",
            "",
        )
        ans["question"] = item["instruction"]
        all_res.append(ans)
        print(item_index, ans)

    with open(output_file, "w") as file:
        json.dump(all_res, file, indent=4)


dir = "/Users/dhaval/Documents/GitHub/cbm-generative-ai/src/cbm_gen/agents/sdg/evol/"
file_paths = [
    #"iot_utterance_Feb1_two_instruction.json",
    #"iot_evol_Jan_31.json",
    #"iot_utterance_rchland_Jan_31.json",
    #"iot_utterance_rchland_Feb1_two_instruction.json",
    "iot_utterance_feb8_synthetic.json"
]

output_files = [
    #"sdg_confidence_iot_utterance_Feb1_two_instruction.json",
    #"sdg_confidence_iot_evol_Jan_31.json",
    #"sdg_confidence_iot_utterance_rchland_Jan_31.json",
    #"sdg_confidence_iot_utterance_rchland_Feb1_two_instruction.json",
    "sdg_confidence_iot_utterance_feb8_synthetic.json"
]

for i in range(1):
    generate_confidence(dir + file_paths[i], output_files[i])
