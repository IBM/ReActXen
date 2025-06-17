import json


def get_tool_names(tools):
    """_summary_

    :param tools: _description_
    :type tools: _type_
    :return: _description_
    :rtype: _type_
    """
    # print("tools", tools)
    tool_names = []
    for tool in tools:
        tool_names.append(tool.name)
    return tool_names


def get_tool_description(tools, detailed_desc=True):
    """_summary_

    :param tools: _description_
    :type tools: _type_
    :return: _description_
    :rtype: _type_
    """
    tool_desc = ""
    for tool_index, tool in enumerate(tools):
        json_string = tool.input_schema.schema_json()
        dictionary = json.loads(json_string)
        keys = dictionary["properties"].keys()
        keys_list = list(keys)
        comma_separated_string = ", ".join(keys_list)
        if comma_separated_string == "args, kwargs":
            comma_separated_string = tool.func.__code__.co_varnames[
                : tool.func.__code__.co_argcount
            ]
            cleaned_argument_names = [
                name.replace("_", " ") for name in comma_separated_string
            ]
            comma_separated_string = ", ".join(cleaned_argument_names)
        if detailed_desc:
            tool_desc += f"({tool_index+1}) {tool.name}[{comma_separated_string}] : {tool.description}"
        else:
            tool_desc += f"({tool_index+1}) {tool.name} : {tool.description}"
        tool_desc += "\n"

    if detailed_desc:
        tool_desc += (
            f"({len(tools)+1}) Finish[answer] : returns the answer and finishes the task."
        )
    else:
        tool_desc += (
            f"({len(tools)+1}) Finish : returns the answer and finishes the task."
        )

    return tool_desc


def get_tool_description_for_chat_template(tools):
    tool_desc = ""
    for _, tool in enumerate(tools):
        json_string = tool.input_schema.schema_json()
        json_obj = json.loads(json_string)
        new_json_format = {}
        new_json_format["name"] = tool.name
        new_json_format["description"] = tool.description
        for _, value in json_obj["properties"].items():
            if isinstance(value, dict) and "title" in value:
                del value["title"]
        new_json_format["parameters"] = {
            "type": "object",
            "properties": json_obj["properties"],
        }
        if "required" in json_obj.keys():
            new_json_format["required"] = json_obj["required"]
        tool_desc += str(new_json_format) + "\n"
        #print(tool_desc.strip())

    finishtool = {
        "name": "Finish",
        "description": "returns the answer in json and finishes the task.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "the final generated answer",
                }
            },
            "required": ['answer'],
        }
    }
    tool_desc += str(finishtool)
    return tool_desc.strip()
