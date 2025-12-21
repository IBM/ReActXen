import os
import openai
from reactxen.experimental.wrapper.utils.prepare_chat_message import get_chat_message
from dotenv import load_dotenv

load_dotenv()

rits_ids = {
    "deepseek-ai/DeepSeek-V2.5": "deepseek-v2-5",
    "deepseek-ai/DeepSeek-V3.2": "deepseek-v3-2",
    "deepseek-ai/DeepSeek-V3": "deepseek-v3-h200",
    "Qwen/Qwen2.5-72B-Instruct": "qwen2-5-72b-instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct": "Qwen2.5-Coder-32B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct": "llama-3-1-8b-instruct",
    "mistralai/mixtral-8x7B-instruct-v0.1": "mixtral-8x7b-instruct-v01",
    "mistralai/mixtral-8x22B-instruct-v0.1": "mixtral-8x22b-instruct-v01",
    "openai/gpt-oss-20b": "gpt-oss-20b",
    "Qwen/Qwen3-8B": "qwen3-8b",
    "Qwen/Qwen2.5-72B-Instruct": "qwen2-5-72b-instruct",
    "Qwen/Qwen3-30B-A3B-Thinking-2507": "qwen3-30b-a3b-thinking-2507",
}

def get_rits_client(hf_model_id="meta-llama/llama-3-1-70b-instruct", num_retries=3):
    """
    Returns a client instance for interacting with the RITS API.

    Args:
        hf_model_id (str): The model ID used in the RITS API.

    Returns:
        openai.Client: A configured client to interact with the RITS API.
    """
    api_base = os.environ.get("RITS_BASE_URL")
    model_id = rits_ids.get(
        hf_model_id, hf_model_id
    )  # Default to provided ID if not in `rits_ids`
    model_id = model_id.split("/")[-1]
    rits_api_base = api_base + f"{model_id}/v1"
    print (rits_api_base)

    client = openai.Client(
        api_key=os.environ.get("RITS_API_KEY"),
        base_url=rits_api_base,
        default_headers={"RITS_API_KEY": os.environ["RITS_API_KEY"]},
        timeout=120.0,
        max_retries=num_retries,
    )
    print (client)
    return client


def get_chat_response(
    messages,
    model_id="meta-llama/llama-3-3-70b-instruct",
    max_tokens=2000,
    temperature=0,
    stop=None,
    num_retries=3,
    seed=20,
    is_system_prompt=False,
    **kwargs,  # Accept any additional parameters
):
    if stop is None:
        stop = ["<>", "Note:"]
    elif isinstance(stop, str):
        stop = [stop]
    else:
        pass

    client = get_rits_client(hf_model_id=model_id, num_retries=num_retries)

    if "mixtral-8x7B-instruct-v0.1" in model_id:
        if is_system_prompt:
            is_system_prompt = False
            messages = messages[1:]

    c_messages = get_chat_message(messages, is_system_prompt)
    try:
        chat_completion = client.chat.completions.create(
            messages=c_messages,
            model=model_id.replace("rits/", ""),
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            seed=seed,
            **kwargs,  # Accept any additional parameters
        )
    except Exception as ex:
        print (ex)

    return chat_completion.choices[0].message.content


def get_completion_response(
    message,
    model_id="meta-llama/llama-3-1-70b-instruct",
    max_tokens=2000,
    temperature=0,
    stop=None,
    num_retries=3,
    seed=20,
    is_system_prompt=False,
    **kwargs,  # Accept any additional parameters
):
    if stop is None:
        stop = ["<>", "Note:"]
    elif isinstance(stop, str):
        stop = [stop]
    else:
        pass

    # https://github.com/openai/openai-python/blob/main/src/openai/resources/completions.py
    client = get_rits_client(hf_model_id=model_id, num_retries=num_retries)
    text_completion = client.completions.create(
        prompt=message,
        model=model_id.replace("rits/", ""),
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
        seed=seed,
        **kwargs,  # Accept any additional parameters
    )
    return text_completion.choices[0].text


def count_tokens(text, model_id="rits/mistralai/mistral-large-instruct-2407"):

    c_messages = get_chat_message(text, False)
    client = get_rits_client(hf_model_id=model_id, num_retries=1)
    response = client.chat.completions.create(
        messages=c_messages,
        model=model_id.replace("rits/", ""),
        max_tokens=1,
        temperature=0,
    )
    return response.usage.prompt_tokens
