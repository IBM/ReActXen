import tempfile
import uuid
import re
import os
import json
from langchain_ibm import ChatWatsonx


def generate_multi_choices_sensor_list(available_sensor_list):
    LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    letters = LETTERS[: len(available_sensor_list)]
    multi_choices = ""
    multi_choices_dict = {}
    for letter, sensor in zip(list(letters), available_sensor_list):
        multi_choices += f"- {letter}. {sensor}\n"
        multi_choices_dict[letter] = sensor
    return multi_choices, multi_choices_dict

"""
def get_llm_model():
    client = Client(credentials=Credentials.from_env())
    model_id = "meta-llama/llama-3-70b-instruct"
    parameters = TextGenerationParameters(
        decoding_method="greedy",
        max_new_tokens=1000,
        min_new_tokens=10,
        temperature=0.5,
        top_k=50,
        top_p=1,
        stop_sequences=["(TOKENSTOP)"],
        return_options=TextGenerationReturnOptions(input_text=False, input_tokens=True),
    )
    model = LangChainInterface(model_id=model_id, client=client, parameters=parameters)
    return model
"""

def get_watsonx_llm(
    model_id="meta-llama/llama-3-70b-instruct", watsonx_params=None
):
    api_key = os.environ["WATSONX_APIKEY"]
    project_id = os.environ["WATSONX_PROJECT_ID"]

    if not watsonx_params:
        watsonx_params = {
            "decoding_method": "greedy",
            "max_new_tokens": 1000,
            "min_new_tokens": 0,
        }

    model = ChatWatsonx(
        model_id=model_id,
        url=os.environ["WATSONX_URL"],
        project_id=project_id,  # CBM project
        params=watsonx_params,
        api_key=api_key,
    )
    return model


def write_json_to_temp(json_data):
    tempDir = tempfile.gettempdir()
    tempFileUUID = uuid.uuid4().hex
    tempFilePath = f"{tempDir}/{tempFileUUID}.json"
    with open(tempFilePath, "w") as tempFile:
        tempFile.write(json_data)
    return tempFilePath


def normalize_string(input_string):
    """
    Normalize the input string by removing surrounding quotes and escape characters.
    """
    # Remove leading and trailing quotes if present
    if input_string.startswith('"') and input_string.endswith('"'):
        input_string = input_string[1:-1]

    # Remove leading and trailing escaped quotes if present
    input_string = re.sub(r'^\\?"(.*?)\\?"$', r"\1", input_string)

    return input_string


def is_valid_json_path(path):
    """
    Check if the given string ends with .json and represents a meaningful path.
    """
    if not path.endswith(".json"):
        return False

    if os.path.exists(path) or os.path.isdir(os.path.dirname(path)):
        return True

    return False


def get_value_from_json(file_path, key):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)

        # Retrieve the value for the given key
        value = data.get(key)
        return value

    except FileNotFoundError:
        return f"Error: The file {file_path} does not exist."
    except json.JSONDecodeError:
        return f"Error: The file {file_path} is not a valid JSON file."
    except Exception as e:
        return f"An unexpected error occurred while opening the file: {e}"

    return None


import re

def starts_with_true_or_false(stringFromLLM: str) -> bool:
    """
    Returns True if the cleaned input string starts with 'TRUE', 
    and False if it starts with 'FALSE'. The function is case insensitive.
    """
    # Remove blanks and special characters, then convert to uppercase
    cleaned_string = re.sub(r'[^a-zA-Z]', '', stringFromLLM).upper()
    
    # Check if it starts with 'TRUE' or 'FALSE'
    if cleaned_string.startswith("TRUE"):
    # if "TRUE" in cleaned_string:
        return True
    elif cleaned_string.startswith("FALSE"):
    # elif "FALSE" in cleaned_string:
        return False
    else:
        return False    # FIXME: this is to be fixed
        raise ValueError("The input string does not start with 'TRUE' or 'FALSE'.")

# Example usage
# print(check_starts_with_true_or_false("  tRuE---123"))  # Output: True
# print(check_starts_with_true_or_false("FALSE!"))        # Output: False
# print(check_starts_with_true_or_false("unknown"))       # Raises ValueError

from reactxen.utils.model_inference import (
    watsonx_llm
)

def is_matching_statement(statement_one, statement_two, matchingCriteria="similarMeaning"):
    """
    Checks if two statements are matching, using LLM (mistral-large) to decide based 
    on selected matching criteria:
    - "similarMeaning": the statements convey semantically similar meaning, one supports another 
        and do not disagree or contradict
    - "equivalent": the provided statements are conveying similar meaning.
    """
    
    if statement_one.strip()=="" or statement_two.strip()=="":
        return "FALSE"
    
    is_the_same_prompt = ""

#         2. The statements do not contradict each other.

    if matchingCriteria=="similarMeaning":
        is_the_same_prompt = f"""Answer "TRUE" for the following two statements if at least one of the following conditions is satisfied:
        1. The statements are conveying similar meaning. 
        2. Statement 1 supports claims of Statement 2.
        3. Statement 2 supports claims of Statement 1.
        Answer "FALSE" if they disagree or contradict.
        You must start the answers with the single word TRUE or FALSE, nothing else, followed by a detailed explanation.
            STATEMENT 1: {statement_one}
            STATEMENT 2: {statement_two}"""

    if matchingCriteria=="equivalent":
        is_the_same_prompt = f"""Answer "TRUE" for the following two statements if at least one of the following conditions is satisfied:
        1. The statements are conveying the same meaning. 
        Answer "FALSE" if they disagree.
        You must start the answers with the single word TRUE or FALSE, nothing else, followed by a detailed explanation.
            STATEMENT 1: {statement_one}
            STATEMENT 2: {statement_two}"""

    print(is_the_same_prompt)

    return watsonx_llm(is_the_same_prompt, 6)