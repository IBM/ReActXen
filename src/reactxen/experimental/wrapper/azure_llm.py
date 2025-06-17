import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from reactxen.experimental.wrapper.utils.prepare_chat_message import get_chat_message
import tiktoken
load_dotenv()

# Fetch environment variables
azure_endpoint = os.getenv("AZURE_ENDPOINT", "")
api_key = os.getenv("AZURE_APIKEY", "")
api_version = os.getenv("API_VERSION", "")


def get_chat_response(
    messages,
    model_id="o1-preview",
    max_tokens=2000,
    temperature=1,
    stop=None,
    num_retries=2,
    seed=22,
    is_system_prompt=False,
    **kwargs,  # Accept any additional parameters
):

    client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
    )
    c_messages = get_chat_message(
        messages, is_system_prompt, replace_system_by_assistant=True
    )
    print (c_messages)

    request_params = {
        "model": model_id,  # Use direct model ID without modifications
        "messages": c_messages,
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
    }

    # Only include `stop` if it's not None
    if stop is not None:
        request_params["stop"] = stop

    response = client.chat.completions.create(**request_params)
    generated_text = response.choices[0].message.content
    print (response)
    return generated_text


def get_response(
    messages,
    model_id="o1-preview",
    max_tokens=2000,
    temperature=0,
    stop=None,
    num_retries=2,
    seed=20,
    is_system_prompt=False,
    **kwargs,  # Accept any additional parameters
):

    client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
    )

    request_params = {
        "model": model_id,  # Use direct model ID without modifications
        "prompt": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
    }

    # Only include `stop` if it's not None
    if stop is not None:
        request_params["stop"] = stop

    # Call Completion API
    response = client.completions.create(**request_params)
    generated_text = response.choices[0].text.strip()
    return generated_text


def openai_count_tokens(text, model="o1-preview", is_chat=False):
    total_token = None
    try:
        enc = tiktoken.encoding_for_model(model)
        if not is_chat:
            tokens = enc.encode(text)
            total_token = len(tokens)
        else:
            total_token = 0
            for message in text:
                total_token += 4  # Base metadata overhead
                total_token += len(enc.encode(message))
            total_token += 2  # Assistant reply overhead
    except KeyError:
        print(f"Could not find encoding for {model}.")

    return total_token