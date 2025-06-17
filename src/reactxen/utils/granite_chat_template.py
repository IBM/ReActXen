import json
from reactxen.tools.skyspark.tool import (
    SkySparkRun,
    SkySparkAssets,
    SkySparkSites,
    SkySparkSensors,
    SkySparkHistory,
)
from reactxen.tools.skyspark.SkySparkWrapper import SkySparkWrapper, SkySparkFunctions
from reactxen.utils.tool_description import get_tool_description
from reactxen.agents.react.prompts.skysparkfewshots import SKYSPARK1
from reactxen.agents.react.agents import ReactAgent
import json
from reactxen.tools.jsonreader.jsonreader import JSONReader, JSONFileMerge
from reactxen.tools.jsonreader.jsonwrapper import JSONWrapperFunctions
from reactxen.tools.time.timewrapper import TimeWrapperFunctions
from reactxen.tools.time.timetool import CurrentTime
from transformers.utils import get_json_schema
from reactxen.utils.tool_description import get_tool_description_for_chat_template
from reactxen.utils.model_inference import watsonx_llm

# Constants for role names
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL_RESPONSE = "tool_response"
ROLE_AVAILABLE_TOOLS = "available_tools"
ROLE_TOOL_CALL = "tool call"


# Generalized function for generating templates
def get_template(role, message, debug=False, is_prefix=False):
    """
    Constructs a role-based template message.
    :param role: The role (e.g., "system", "user", "assistant", etc.)
    :param message: The message content for the template.
    :param debug: If True, prints the constructed template.
    :return: A formatted template string.
    """
    result = f"<|start of role|>{role}<|end of role|>{message}<|end of text|>"
    if is_prefix:
        result = f"<|start of role|>{role}<|end of role|>"
    if debug:
        print(result)
    return result


# Specialized functions for common roles
def get_system_template(system_message, debug=False):
    return get_template(ROLE_SYSTEM, system_message, debug)


def get_user_template(user_message, debug=False):
    return get_template(ROLE_USER, user_message, debug)


def get_assistant_template(assistant_message, debug=False, is_prefix=False):
    return get_template(ROLE_ASSISTANT, assistant_message, debug, is_prefix)


def get_assistant_template_for_tool_calling(tool_call_message, debug=False):
    return get_template(ROLE_TOOL_CALL, f"[{tool_call_message}]", debug)


def get_tool_definition_template(tool_definitions, debug=False):
    """
    Constructs the available tools template.
    :param tool_definitions: A JSON string or dictionary of tool definitions.
    :param debug: If True, prints the constructed template.
    :return: A formatted available tools template.
    """
    if isinstance(tool_definitions, dict):
        tool_definitions = json.dumps(tool_definitions)
    return get_template(ROLE_AVAILABLE_TOOLS, f"{tool_definitions}", debug)


def get_assistant_template_for_tool_calling(tool_call_message, debug=False):
    """
    Constructs a tool call message for the assistant role.
    :param tool_call_message: A JSON string or dictionary representing the tool call.
    :param debug: If True, prints the constructed template.
    :return: A formatted tool call template.
    """
    if isinstance(tool_call_message, dict):
        # Convert dictionary to a JSON string
        tool_call_message = json.dumps(tool_call_message)
    # Properly format the tool call as a JSON object, not a list
    result = f"<|start of role|>{ROLE_ASSISTANT}<|end of role|><|tool call|>[{tool_call_message}]<|end of text|>"
    if debug:
        print(result)
    return result


def get_tool_response_template(tool_responses, debug=False):
    """
    Constructs the tool response template.
    :param tool_responses: A JSON string or dictionary of tool responses.
    :param debug: If True, prints the constructed template.
    :return: A formatted tool response template.
    """
    if isinstance(tool_responses, dict):
        tool_responses = json.dumps(tool_responses)
    return get_template(ROLE_TOOL_RESPONSE, tool_responses, debug)


# Example usage
if __name__ == "__main__":
    # Example templates
    from cbm_gen.agents.react.chat_prompts.systemprompts import (
        react_agent_prompt_for_chat,
    )

    functions = SkySparkFunctions()
    assets = SkySparkAssets(skyspark_functions=functions)
    functions = SkySparkFunctions()
    assets = SkySparkAssets(skyspark_functions=functions)
    sites = SkySparkSites(skyspark_functions=functions)
    sensors = SkySparkSensors(skyspark_functions=functions)
    history = SkySparkHistory(skyspark_functions=functions)
    fns = JSONWrapperFunctions()
    jsonReader = JSONReader(functions=fns)
    jsonMerge = JSONFileMerge(functions=fns)
    fns = TimeWrapperFunctions()
    currenttime = CurrentTime(functions=fns)
    tools = [assets, sites, sensors, history, jsonReader, jsonMerge, currenttime]
    tool_desc = get_tool_description_for_chat_template(tools)
    tool_pt = get_tool_definition_template(tool_desc)
    query_pt = get_user_template("What assets are at site POKMAIN?")

    final_prompt = react_agent_prompt_for_chat.format(
        question="What assets are at site POKMAIN?",
        scratchpad="",
        tool_desc=tool_desc,
    )
    ans = watsonx_llm(final_prompt, model_id=8, stop=["<|end of text|>", "Action:"])
    print(ans)
    exit(0)

    assistant_prefix = get_assistant_template("", is_prefix=True)

    final_prompt = (
        system_pt + "\n" + tool_pt + "\n" + query_pt + "\n" + assistant_prefix
    )
    print(final_prompt)
    ans = watsonx_llm(final_prompt, model_id=8, stop=["<|end of text|>", "Tool Name:"])
    print(ans)
    exit(0)

    print(get_user_template("What is the temperature in Boston?"))
    print(get_assistant_template("The temperature in Boston is 20°C."))
    print(
        get_assistant_template_for_tool_calling(
            '{"name": "get temp", "params": {"location": "Boston"}}'
        )
    )
    print(
        get_tool_definition_template(
            {"name": "get temp", "description": "Fetch temperature data."}
        )
    )
    print(get_tool_response_template({"temp": 20.5, "unit": "C"}))
