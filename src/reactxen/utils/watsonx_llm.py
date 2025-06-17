from typing import Any, Optional
from uuid import UUID
from reactxen.utils.utils import get_watsonx_llm
from dotenv import load_dotenv
load_dotenv()

watsonx_param = {
    "decoding_method": "greedy",
    "max_new_tokens": 2000,
    "min_new_tokens": 1,
    "temperature": 0.5,
    "top_k": 50,
    "top_p": 1,
    "stop_sequences": ["OBSERVATION"],
}
granite_llm = get_watsonx_llm(
    model_id="ibm/granite-13b-chat-v2", watsonx_params=watsonx_param
)

watsonx_param = {
    "decoding_method": "greedy",
    "max_new_tokens": 1024,
    "min_new_tokens": 1,
    "temperature": 0.5,
    "top_k": 50,
    "top_p": 1,
    "stop_sequences": ["OBSERVATION", "<|endoftext|>"],
}

granite_34b_code_instruct = get_watsonx_llm(
    model_id="ibm/granite-34b-code-instruct", watsonx_params=watsonx_param
)

watsonx_param = {
    "decoding_method": "greedy",
    "max_new_tokens": 2048,
    "min_new_tokens": 0,
    "temperature": 0.5,
    "top_k": 50,
    "top_p": 1,
    "stop_sequences": ["OBSERVATION"],
}
llama_3 = get_watsonx_llm(
    model_id="meta-llama/llama-3-70b-instruct", watsonx_params=watsonx_param
)

watsonx_param["max_new_tokens"] = 4000

mistral_large = get_watsonx_llm(
    model_id="mistralai/mistral-large", watsonx_params=watsonx_param
)

llama_3_405b = get_watsonx_llm(
    model_id="meta-llama/llama-3-405b-instruct", watsonx_params=watsonx_param
)

LLMName2ModelMapping = {}
LLMName2ModelMapping["granite-13b-chat-v2"] = granite_llm
LLMName2ModelMapping["granite-34b-code-instruct"] = granite_34b_code_instruct
LLMName2ModelMapping["llama-3-70b-instruct"] = llama_3
LLMName2ModelMapping["mistral-large"] = mistral_large
LLMName2ModelMapping["llama-3-405b-instruct"] = llama_3_405b

def get_llm(llm_name="granite-13b-chat-v2"):
    if llm_name in LLMName2ModelMapping.keys():
        return LLMName2ModelMapping[llm_name]
    else:
        raise Exception("Model {0} not found", llm_name)


def get_llm_names():
    return list(LLMName2ModelMapping.keys())