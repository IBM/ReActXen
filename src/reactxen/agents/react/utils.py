import re, string
from reactxen.agents.react.prompts.systemprompts import (
    REFLECTION_HEADER,
    LAST_TRIAL_HEADER,
)
from reactxen.utils.model_inference import count_tokens
from typing import List, Union, Literal
from mdextractor import extract_md_blocks
import json

# from cbm_gen.applications.event_forecasting.apis.api_implementation_cbm import *
from enum import Enum
from typing import List, Literal, Union
from reactxen.utils.model_inference import count_tokens


# This class is used only when we have understanding of
# what is a answer
class ActionStyle(Enum):
    """
    Enum representing different styles of actions that an agent can perform.

    Attributes:
        SINGLE_LINE_FUNCTION: Represents an action executed as a single-line Python function.
        BLOCK_FUNCTION: Represents an action executed as a multi-line Python function.
        SINGLE_LINE_TOOL_CALL: Represents an action that involves a single-line call to an external tool or command.
        MULTI_LINE_PARAGRAPH: Represents an action that requires multiple lines of execution to complete.
    """

    SINGLE_LINE_FUNCTION = "single_line_function"
    BLOCK_FUNCTION = "block_function"
    SINGLE_LINE_TOOL_CALL = "single_line_tool_call"
    MULTI_LINE_PARAGRAPH = "multi_line_paragraph"


# This class is used only when we have an understanding of what an answer is.
class ReActStyle(Enum):
    """
    Enum representing different styles of ReAct that an agent can perform.

    Attributes:
        ThoughtActTogether: Represents an action where thinking and acting happen in a single step.
        ThoughtThenAct: Represents an action where thought is completed before execution, requiring multiple steps.
        OnlyAct: Represents a action part only.
    """

    ThoughtActTogether = "thought_and_act_together"
    ThoughtThenAct = "thought_then_act"
    OnlyAct = "only_act"

def parse_cot_action(step):
    pattern = r"(\w+)\[(\{.*?\})\]"
    matches = re.findall(pattern, step)
    result = [(tool, eval(arguments)) for tool, arguments in matches]
    return result


def extract_line():
    pass


def original_format_step(step: str, stop, prefix) -> str:
    # print (step)
    if prefix == "Action:":
        if "Finish[" in step:
            step = step.split("Finish[")[-1]
        if "]]" in step:
            step, _, _ = step.rpartition("]]")
            step = "Finish[" + step + "]]"

    step = step.split(prefix)[-1]
    for item in stop:
        step = step.split(item)[0]
    return step.strip("\n").strip().replace("\n", " ")


def extract_thought(step: str) -> str:
    """
    Extracts the 'thought' portion from a given step string before 'Action' or 'Final Answer'.

    Args:
        step (str): The input text containing the reasoning process.

    Returns:
        str: The extracted thought text, cleaned of triple backticks.
    """
    regex = r"(.*?)(?:\nAction|\nFinal Answer)"
    thought_match = re.search(regex, step, re.DOTALL)

    if thought_match:
        thought = thought_match.group(1).strip()
        return thought.replace("```", "").strip()

    return ""


def extract_action_and_input(step: str, stop: Union[str, list]) -> str:
    """
    Answer:
    """
    regex = r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
    action_match = re.search(regex, step, re.DOTALL | re.IGNORECASE)
    if action_match:
        action = action_match.group(1).strip()
        action_input = action_match.group(2)
        if ActionStyle.SINGLE_LINE_TOOL_CALL:
            action_input = action_input.strip("\n").replace("\n", " ")
        tool_input = action_input.strip().strip('"')
        return {
            "action": action,
            "action_input": tool_input,
            "error": False,
            "error_feedback": None,
        }
    else:
        regex = "(.*)\s*Action\s*Input\s*\d*\s*:\s*(.*)"
        action_match = re.search(regex, step, re.DOTALL | re.IGNORECASE)
        if action_match:
            action = action_match.group(1).strip()
            action_input = action_match.group(2)
            if ActionStyle.SINGLE_LINE_TOOL_CALL:
                action_input = action_input.strip("\n").replace("\n", " ")
            tool_input = action_input.strip().strip('"')
            return {
                "action": action,
                "action_input": tool_input,
                "error": False,
                "error_feedback": None,
            }
        else:
            # in some cases finish is not as good
            action = None
            action_input = None
            # Provide custom error message in case of failure
            # final check
            if (
                "finish" in step.lower()
            ):  # Check if "finish" is in the string (case insensitive)
                finish_index = step.lower().index(
                    "finish"
                )  # Find the index of "finish" in the string
                action = "Finish"
                action_input = step[
                    finish_index + len("finish") :
                ].strip()  # Extract everything after "finish" and remove leading/trailing spaces

                ## adding more check
                if '\nQuestion' in stop and action_input.endswith('\nQuestion'):
                        action_input = action_input[:-9].strip()
                return {
                    "action": action,
                    "action_input": action_input,
                    "error": False,
                    "error_feedback": None,
                }
            # finally reporting error
            action = "ErrorMessage:MPEAgent:ActionDecoding:TextBlock"  # Custom error message for action
            feedback = (
                "The generated Action and/or Action Input are incorrectly formatted. "
                "Please ensure the following: "
                "1. The Action should be a valid, well-structured string that represents a specific operation or tool name. "
                "2. The Action Input should be a valid and coherent string that corresponds to the requirements of the Action. "
                "3. If the Action is 'Finish', the Action Input must contain the final answer explicitly. "
                "Review your output and correct the formatting or logic as necessary."
            )
            return {
                "action": action,
                "action_input": None,
                "error": True,
                "error_feedback": feedback,
            }


def format_step_for_thinkact_together(step: str, stop: Union[str, list]):
    original_step = step
    step = step.rstrip("\n\r").strip()
    thought = extract_thought(step)
    answer = extract_action_and_input(step, stop)
    answer["thought"] = thought
    answer["llm_output"] = original_step
    return answer


def format_step(
    step: str,
    stop: Union[str, list],
    prefix: str,
    debug: False,
    action_style: str,
    react_style: str,
) -> str:
    if (
        action_style == ActionStyle.SINGLE_LINE_TOOL_CALL
        or action_style == ActionStyle.MULTI_LINE_PARAGRAPH
    ) and react_style == ReActStyle.ThoughtActTogether:
        return format_step_for_thinkact_together(step=step,stop=stop)
    original_step = step
    # print ('I am in format = ', stop)
    """
    if debug:
        print("=======Raw output start========")
        print(step)
        print("Prefix:- ", prefix)
        print("Step:- ", stop)
        print("=======Raw output end======")
    """

    if "Action" in prefix:
        # this indicate we are output of an Action Generation
        # what happen it remove legitimate Observation
        # step = step.replace("Observation", "").strip("\n").strip()
        # following is a new code
        # Check if the step ends with "Observation" and remove it if true
        step = step.rstrip("\n\r").strip()  # Remove newlines at the end first
        if step.endswith("Observation"):
            step = step[: -len("Observation")].rstrip("\n\r").strip()

        if (
            action_style == ActionStyle.SINGLE_LINE_FUNCTION
            or action_style == ActionStyle.BLOCK_FUNCTION
        ):

            action = None
            action_input = None
            feedback = None
            is_error = False

            if "final answer" in step.lower():
                # this is a final action block so it will have json end
                action = "Final Answer"
                action_input = step.strip()
                pattern = r"```json(.*?)```"
                if "```json" in step and not step.endswith("```"):
                    step += "```"
                matches = re.findall(pattern, step, re.DOTALL)
                if matches:
                    bblocks = extract_md_blocks(step.strip())
                    for bblock in bblocks:
                        try:
                            json.loads(bblock)
                        except Exception as ex:
                            is_error = True
                            feedback = f"Invalid JSON block detected: {ex}"
                            break
                else:
                    is_error = True
                    feedback = (
                        "The response does not contain a valid or complete JSON block enclosed in triple backticks "
                        "(```json ... ```). Please ensure the output strictly adheres to this format. "
                        "If the response is expected to produce a JSON object with more than 50 keys, modify the code generating the JSON to store it into a file. "
                        "Provide a pointer to the file location for better usability. Additionally, refer to the File Handling guidelines for long responses mentioned in the system prompt "
                        "to ensure the output is accurate, complete, and manageable."
                    )
                return {
                    "action": action,
                    "action_input": action_input,  # this is error feedback or result
                    "llm_output": original_step,
                    "error": is_error,
                    "error_feedback": feedback,
                }

            # this can be a code block (python)
            action = "Code Block"
            action_input = None
            is_error = False
            feedback = None

            if "```python" in step and not step.endswith("```"):
                step += "```"
            pattern = r"```python(.*?)```"
            # pattern = r"```python(.*?)```|```python(.*)"
            matches = re.findall(pattern, step, re.DOTALL)
            if matches:
                action_input = extract_md_blocks(step.strip())
            else:
                try:
                    compile(step.strip(), "<string>", "exec")
                    action_input = [step.strip()]
                except Exception as e:
                    feedback = (
                        "There was an issue with extracting or compiling the Python code generated in the most recent action. "
                        "This could be due to a formatting error, missing code block, or invalid syntax. "
                        "Please check the code formatting and ensure it adheres to Python standards. "
                        "Error details: " + str(e)
                    )
                    action = "ErrorMessage:MPEAgent:ActionDecoding:CodeBlock"
                    action_input = None
                    is_error = True

            return {
                "action": action,
                "action_input": action_input,  # this is action input
                "llm_output": original_step,
                "error": is_error,
                "error_feedback": feedback,
            }

        # At this point we will now do action generation for the Text related stuff
        # ActionStyle. Text

        regex = (
            r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        )
        action_match = re.search(regex, step, re.DOTALL | re.IGNORECASE)
        if action_match:
            action = action_match.group(1).strip()
            action_input = action_match.group(2)
            if action_style == ActionStyle.SINGLE_LINE_TOOL_CALL:
                action_input = action_input.strip("\n").replace("\n", " ")
            tool_input = action_input.strip().strip('"')
            return {
                "action": action,
                "action_input": tool_input,
                "llm_output": original_step,
                "error": False,
                "error_feedback": None,
            }
        else:
            # print('came here.... ----->')
            regex = "(.*)\s*Action\s*Input\s*\d*\s*:\s*(.*)"
            action_match = re.search(regex, step, re.DOTALL | re.IGNORECASE)
            if action_match:
                action = action_match.group(1).strip()
                action_input = action_match.group(2)
                if action_style == ActionStyle.SINGLE_LINE_TOOL_CALL:
                    action_input = action_input.strip("\n").replace("\n", " ")
                tool_input = action_input.strip().strip('"')
                return {
                    "action": action,
                    "action_input": tool_input,
                    "llm_output": original_step,
                    "error": False,
                    "error_feedback": None,
                }
            else:
                # in some cases finish is not as good
                action = None
                action_input = None
                # Provide custom error message in case of failure
                # final check
                if (
                    "finish" in step.lower()
                ):  # Check if "finish" is in the string (case insensitive)
                    finish_index = step.lower().index(
                        "finish"
                    )  # Find the index of "finish" in the string
                    action = "Finish"
                    action_input = step[
                        finish_index + len("finish") :
                    ].strip()  # Extract everything after "finish" and remove leading/trailing spaces
                    
                    ## adding more check
                    if '\nQuestion' in stop and action_input.endswith('\nQuestion'):
                         action_input = action_input[:-9].strip()

                    return {
                        "action": action,
                        "action_input": action_input,
                        "llm_output": original_step,
                        "error": False,
                        "error_feedback": None,
                    }
                # finally reporting error
                action = "ErrorMessage:MPEAgent:ActionDecoding:TextBlock"  # Custom error message for action
                feedback = (
                    "The generated Action and/or Action Input are incorrectly formatted. "
                    "Please ensure the following: "
                    "1. The Action should be a valid, well-structured string that represents a specific operation or tool name. "
                    "2. The Action Input should be a valid and coherent string that corresponds to the requirements of the Action. "
                    "3. If the Action is 'Finish', the Action Input must contain the final answer explicitly. "
                    "Review your output and correct the formatting or logic as necessary."
                )
                return {
                    "action": action,
                    "action_input": None,
                    "llm_output": original_step,
                    "error": True,
                    "error_feedback": feedback,
                }
    elif "Thought" in prefix:
        # this is now about thougth generation.
        # step = step.replace("Action", "").strip("\n").strip()
        step = step.rstrip("\n\r").strip()  # Remove newlines at the end first
        if step.endswith("Action"):
            step = step[: -len("Action")].rstrip("\n\r").strip()

        # in case of prefix repetation, we trim it
        if prefix in step:
            step = step.split(prefix, 1)[-1].split("\n", 1)[0]

        # if stop is present we trim from end
        if isinstance(stop, str):
            if stop in step:
                step = step.split(stop, 1)[0]
        else:
            for stopword in stop:
                if stopword in step:
                    step = step.split(stopword, 1)[0]

        # thought part should be single line?
        step = step.replace("\n", " ").strip()
        step = re.sub(r"Thought \d+:", "", step).strip()
        pattern = r"^\d+:\s*(.*)"
        match = re.match(pattern, step)
        if match:
            step = match.group(1)
        return {"thought": step, "llm_output": original_step}
    elif "Reflection:" in prefix:
        step = step.split(prefix)[-1]
        for item in stop:
            step = step.split(item)[0]
        return step.strip("\n").strip().replace("\n", " ")
    elif "Observation " in prefix:
        step = step.rstrip("\n\r").strip()  # Remove newlines at the end first
        if step.endswith("Thought"):
            step = step[: -len("Thought")].rstrip("\n\r").strip()

        if isinstance(stop, str):
            return {
                "observation": step.replace(stop, "").replace("\n", " "),
                "llm_output": original_step,
            }
        else:
            for stopword in stop:
                step.replace(stopword, "")
            return {
                "observation": step.replace("\n", " ").strip(),
                "llm_output": original_step,
            }

    else:
        for stopword in stop:
            step.split(prefix)[-1].replace(stop, "")
        return {
            # step.split(prefix)[-1].replace(stop, "").strip("\n").strip().replace("\n", "")
            "step": step.strip("\n").strip(),
            "llm_output": original_step,
        }


def format_review(reviews: List[str], header: str = "") -> str:
    if reviews == []:
        return ""
    else:
        return header + "Review:\n- " + "\n- ".join([r.strip() for r in reviews])


def format_reflections(reflections: List[str], header: str = REFLECTION_HEADER) -> str:
    if reflections == []:
        return ""
    else:
        return (
            header + "Reflections:\n- " + "\n- ".join([r.strip() for r in reflections])
        )


def format_last_attempt(
    question: str, scratchpad: str, header: str = LAST_TRIAL_HEADER, model_id: int = 0
):
    # print(header)
    # print(scratchpad)
    return (
        header
        + f"Question: {question}\n"
        + truncate_scratchpad(scratchpad, tokenizer=count_tokens, model_id=model_id)
        .strip("\n")
        .strip()
        + "\n(END PREVIOUS TRIAL)\n"
    )


def truncate_scratchpad(
    scratchpad: str, n_tokens: int = 1600, tokenizer=count_tokens, model_id=0
) -> str:
    if count_tokens(scratchpad, model_id=model_id, upper_limit=n_tokens) > n_tokens:
        lines = scratchpad.split("\n")
        observations = filter(lambda x: x.startswith("Observation"), lines)
        observations_by_tokens = sorted(
            observations,
            key=lambda x: tokenizer(x, model_id=model_id, upper_limit=n_tokens),
        )
        while (
            count_tokens("\n".join(lines), model_id=model_id, upper_limit=n_tokens)
            > n_tokens
        ):
            largest_observation = observations_by_tokens.pop(-1)
            ind = lines.index(largest_observation)
            lines[ind] = largest_observation.split(":")[0] + ": [truncated excerpt]"
        new_scratchpad = "\n".join(lines)
        return new_scratchpad
    else:
        return scratchpad


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def EM(answer, key) -> bool:
    return normalize_answer(answer) == normalize_answer(key)

def apply_agent_controls(prompt, stop):
    stop_str = " ".join(stop) if isinstance(stop, (list, tuple)) else str(stop)

    has_obs = "Observation" in stop_str
    has_action = "Action" in stop_str
    has_thought = "Thought" in stop_str

    prompt += "\n\n--- AGENT CONTROL OVERRIDE (LATEST ONLY) ---\n"

    if has_obs and not has_action:
        prompt += (
            "[INSTRUCTION: GENERATION_PHASE_1]\n"
            "Generate ONLY: Thought, Action, Action Input.\n"
            "DO NOT generate Observation.\n"
            "Stop immediately after Action Input.\n"
            "End with <END_OF_TURN>."
        )

    elif has_thought:
        prompt += (
            "[INSTRUCTION: ACT_ONLY]\n"
            "Provide ONLY Action and Action Input.\n"
            "Do NOT repeat reasoning.\n"
            "End with <END_OF_TURN>."
        )

    elif has_action:
        prompt += (
            "[INSTRUCTION: PLAN_ONLY]\n"
            "Provide ONLY a Thought block.\n"
            "Do NOT call tools.\n"
            "End with <END_OF_TURN>."
        )

    else:
        prompt += (
            "Your response must conclude before reaching: "
            f"{stop}\nEnd with <END_OF_TURN>."
        )

    return prompt
