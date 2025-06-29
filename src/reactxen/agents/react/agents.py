import io
import re
import signal
import sys
import warnings
import importlib
import json
from enum import Enum
from colorama import Fore, Style
from langchain.prompts import PromptTemplate
import time
from reactxen.agents.react.prompts.metaplanprompts import (
    metaplan_prompt,
    METAPLAN_HEADER,
)
from reactxen.agents.reviewer_agent.agent import (
    ReviewerAgent,
    REVIEW_HEADER,
)
from reactxen.agents.react.prompts.fewshots import (
    COT,
    COT_REFLECT,
    COT_SIMPLE_REFLECTION2,
    COTQA_SIMPLE4,
    MPE_SIMPLE4,
    MPE_REFLECTIONS,
    MetaPlanSample,
)
from reactxen.agents.react.prompts.systemprompts import (
    COT_INSTRUCTION,
    COT_REFLECT_INSTRUCTION,
    LAST_TRIAL_HEADER,
    REFLECTION_AFTER_LAST_TRIAL_HEADER,
    REFLECTION_HEADER,
    cot_agent_prompt,
    cot_reflect_agent_prompt,
    cot_reflect_prompt,
    react_agent_prompt,
    react_reflect_agent_prompt,
    reflect_prompt,
)
from jinja2 import Template
from reactxen.agents.react.prompts.systemprompts import (
    LAST_TRIAL_HEADER,
    REFLECTION_AFTER_LAST_TRIAL_HEADER,
    REFLECTION_HEADER,
    react_agent_prompt,
    react_reflect_agent_prompt,
    reflect_prompt,
)
from reactxen.agents.react.prompts.agent_scratchpad_probe import scratchpad_probe_prompt
from reactxen.agents.react.utils import *
from reactxen.utils.model_inference import (
    count_tokens,
    get_context_length,
    watsonx_llm,
)
from reactxen.utils.tool_description import get_tool_description, get_tool_names
from reactxen.utils.tool_cache import ToolInvocationCache
from datetime import datetime
from functools import partial
from reactxen.agents.react.utils import ReActStyle
from reactxen.agents.assessment_agent.agent import TaskAssessmentAgent


def handler(signum, frame):
    print("Code execution took too long")


signal.signal(signal.SIGALRM, handler)


# This class is used only when we have understanding of
# what is a answer
class ReflexionStrategy(Enum):
    """
    NONE: No reflection
    LAST_ATTEMPT: Use last reasoning trace in context
    REFLEXION: Apply reflexion to the next reasoning trace
    LAST_ATTEMPT_AND_REFLEXION: Use last reasoning trace in context and apply reflexion to the next reasoning trace
    """

    NONE = "base"
    LAST_ATTEMPT = "last_trial"
    REFLEXION = "reflexion"
    LAST_ATTEMPT_AND_REFLEXION = "last_trial_and_reflexion"


# This is a chain of thought agent
class CoTAgent:
    """
    question: input question text
    key: answer to input text
    agent_prompt: system prompt for react agent
    reflect_prompt: system prompt to be used for reflextion agent
    cot_examples: in context example to be used for agent_prompt
    reflect_examples: in context example to be used for reflect_prompt
    self_reflect_llm: LLM model for doing reflexion
    action_llm: LLM model for doint react
    cbm_tools: list of all CBM tools
    tool_desc: list of description for all CBM tools cbm_tools
    """

    def __init__(
        self,
        question: str,
        key: str,
        agent_prompt: PromptTemplate = cot_reflect_agent_prompt,
        reflect_prompt: PromptTemplate = cot_reflect_prompt,
        cot_examples: str = COTQA_SIMPLE4,
        reflect_examples: str = COT_SIMPLE_REFLECTION2,
        self_reflect_llm=watsonx_llm,
        action_llm=watsonx_llm,
        action_llm_model_id=0,
        self_reflect_model_id=0,
        cbm_tools: list = [],
        tool_names: list = [],
        tool_desc: str = "",
        agent_name: str = "MPE",
    ) -> None:
        self.question = question
        self.key = key
        self.agent_prompt = agent_prompt
        self.reflect_prompt = reflect_prompt
        self.cot_examples = cot_examples
        self.reflect_examples = reflect_examples
        self.self_reflect_llm = self_reflect_llm
        self.action_llm = action_llm
        self.cbm_tools = cbm_tools
        self.tool_desc = tool_desc
        self.tool_names = tool_names
        if not self.tool_names:
            self.tool_names = get_tool_names(self.cbm_tools)
        if not self.tool_desc:
            self.tool_desc = get_tool_description(self.cbm_tools)

        self.action_llm_model_id = action_llm_model_id
        self.self_reflect_model_id = self_reflect_model_id

        # Storing Information
        self.reflections: List[str] = []
        self.reflections_str = ""
        self.answer = ""
        self.step_n: int = 0
        self.agent_name = agent_name

        self.reviewagt = ReviewerAgent(model_id=self.action_llm_model_id)

        self.reset()

    def run(
        self, reflexion_strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION
    ) -> None:
        print(
            f"{Style.BRIGHT}{Fore.MAGENTA}I am {self.agent_name} Agent with CoT-w/Reflexion - Ietration@{self.step_n}{Style.RESET_ALL}"
        )
        print(
            f"{Style.BRIGHT}{Fore.LIGHTMAGENTA_EX}Input Question: {self.question}{Style.RESET_ALL}"
        )

        if (
            self.step_n > 0
            and not self.is_correct()
            and reflexion_strategy != ReflexionStrategy.NONE
        ):
            self.reflect(reflexion_strategy)

        self.reset()
        self.step()

        # call review agents
        review_response = self.reviewagt.evaluate_response(
            self.question, self.scratchpad, self.answer
        )
        # print(review_response)

        self.step_n += 1

    def step(self) -> None:
        # Chain of Thought - In a two LLM call, the process will complete the answer generation
        # Our current understanding is - Thought is a textual and single line as well as Action is a single line
        self.scratchpad += f"\nThought: "
        let_me_think = " " + self.prompt_agent(
            stop=["\nAction", "Action:", "Action", "\\n", "\nAction:"],
            prefix="Thought:",
        )
        self.scratchpad += let_me_think.strip()
        msg = self.scratchpad.split("\n")[-1]
        print(
            f"{Style.BRIGHT}{Fore.BLUE}Think: {let_me_think.strip()}{Style.RESET_ALL}"
        )

        # Act
        self.scratchpad += f"\nAction: "
        let_me_act = self.prompt_agent(
            stop=["Thought", "\nThought:", "\nAnswer:", "Question:" "\nQuestion:"],
            prefix="Action:",
        )
        self.scratchpad += " " + let_me_act.strip()
        print(
            f"{Style.BRIGHT}{Fore.YELLOW}Action: {let_me_act.strip()}{Style.RESET_ALL}"
        )

        # Now we have thought and action and we will go over its execution - execute the tools
        tool_chains = parse_cot_action(let_me_act.strip())  # parse_cot_action(action)

        # print(tool_chains)
        # We may not need to do this for an observation (removal of \n)
        self.scratchpad += f"\nObservation: "
        self.scratchpad += (
            self.run_tool_chain(tool_chains).replace("\n", " ").strip("\n").strip()
        )
        msg = self.scratchpad.split("\n")[-1]
        print(f"{Style.BRIGHT}{Fore.GREEN}{msg}{Style.RESET_ALL}")
        msg = ""
        print(
            f"{Style.BRIGHT}{Fore.MAGENTA}CoT Process is completed now{Style.RESET_ALL}"
        )
        self.answer = msg

    def reflect(self, strategy: ReflexionStrategy) -> None:
        # print("Running Reflexion strategy...")
        if strategy == ReflexionStrategy.LAST_ATTEMPT:
            self.reflections = [self.scratchpad]
            self.reflections_str = format_last_attempt(
                self.question, self.reflections[0], model_id=self.self_reflect_model_id
            )
        elif strategy == ReflexionStrategy.REFLEXION:
            self.reflections += [self.prompt_reflection()]
            self.reflections_str = format_reflections(self.reflections)
        elif strategy == ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION:
            self.reflections_str = format_last_attempt(
                self.question, self.scratchpad, model_id=self.self_reflect_model_id
            )
            self.reflections = [self.prompt_reflection()]
            self.reflections_str += "\n" + format_reflections(
                self.reflections, header=REFLECTION_AFTER_LAST_TRIAL_HEADER
            )
        else:
            raise NotImplementedError(f"Unknown reflection strategy: {strategy}")
        print(
            f"{Style.BRIGHT}{Fore.LIGHTRED_EX}Reflexion: {self.reflections_str}{Style.RESET_ALL}"
        )

    def prompt_reflection(self) -> str:
        return original_format_step(
            self.self_reflect_llm(
                self._build_reflection_prompt(),
                model_id=self.self_reflect_model_id,
                stop=["Previous Trial:", "Previous Trial"],
            ),
            stop=["Previous Trial:", "Previous Trial"],
            prefix="Reflection:",
        )

    def reset(self) -> None:
        self.scratchpad: str = ""
        self.finished = False

    def prompt_agent(self, stop, prefix) -> str:
        llmResult = self.action_llm(
            self._build_agent_prompt(), model_id=self.action_llm_model_id, stop=stop
        )

        return original_format_step(
            llmResult["generated_text"],
            stop,
            prefix,
        )

    def _build_agent_prompt(self) -> str:
        t_prompt = self.agent_prompt.format(
            examples=self.cot_examples,
            reflections=self.reflections_str,
            question=self.question,
            scratchpad=self.scratchpad,
            tool_desc=self.tool_desc,
        )
        # print (t_prompt)
        return t_prompt

    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
            examples=self.reflect_examples,
            question=self.question,
            scratchpad=self.scratchpad,
            tool_desc=self.tool_desc,
        )

    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        return EM(self.answer, self.key)

    def run_tool_chain(self, tool_chain):
        # print (tool_chain)
        tool_outs = []
        is_error = False
        for action_type, argument in tool_chain:
            tool_not_find = True
            if action_type:
                for tool in self.cbm_tools:
                    if tool.name.lower() == action_type.lower():
                        tool_not_find = False
                        try:
                            parsed_dict = argument
                            tool_output = tool.run(tool_input=parsed_dict)
                            tool_output = (
                                tool_output.replace("\n", " ").strip("\n").strip()
                            )
                            tool_outs.append(tool_output)
                        except Exception as ex:
                            tool_outs.append(
                                f"Taking action: Execution Error while calling Tool {tool.name.lower()} with parameter {argument}. The reported error is : {str(ex)}"
                            )
                            is_error = True
                    if not tool_not_find:
                        break
                if tool_not_find:
                    is_error = True
                    tool_outs.append(
                        f"Tool {action_type} with parameter {argument} is not available. Action should be selected from avialable tools: {', '.join(self.tool_names)}"
                    )
            else:
                is_error = True
                tool_outs.append(
                    f"The formate of action for tool is not correct. Action should be selected from avialable tools: {', '.join(self.tool_names)}"
                )
            if is_error:
                break
        if is_error:
            return "Tool execution failed. " + ", ".join(tool_outs)
        if len(tool_chain) == 0:
            return "Tool is not executed. As response was not sufficient to execute any tool."
        return "Execution is completed. " + ", ".join(tool_outs)


class ReactAgent:
    def __init__(
        self,
        question: str,
        key: str,
        max_steps: int = 6,
        agent_prompt: PromptTemplate = react_agent_prompt,
        cbm_tools: list = [],
        tool_names: list = [],
        tool_desc: str = "",
        react_llm_model_id=0,
        react_example: str = MPE_SIMPLE4,
        llm=watsonx_llm,
        debug=False,
        actionstyle=ActionStyle.SINGLE_LINE_TOOL_CALL,
        log_structured_messages=False,
        handle_context_length_overflow=False,
        apply_chat_template=False,
        apply_loop_detection_check=False,
        max_retries: int = 1,  # support will be enabled as time goes
        early_stop=False,
        use_tool_cache=False,
        enable_tool_partial_match=False,
        reactstyle=ReActStyle.ThoughtThenAct,
        enable_meta_planning=False,
        enable_question_assement=False,
        enable_agent_ask=False,
        max_execution_time=-1,
        skip_token_counting=False,
    ) -> None:
        self.question = question
        self.answer = ""
        self.key = key
        self.max_steps = max_steps
        self.agent_prompt = agent_prompt
        self.react_examples = react_example
        self.tool_names = tool_names
        self.tool_desc = tool_desc
        self.debug = debug
        self.local_vars = {}
        self.log_structured_messages = log_structured_messages
        self.handle_context_length_overflow = handle_context_length_overflow
        self.apply_chat_template = apply_chat_template
        self._context_overflow_detected = False
        self.apply_loop_detection_check = apply_loop_detection_check
        self.early_stop = early_stop
        self.use_tool_cache = use_tool_cache
        self.enable_tool_partial_match = enable_tool_partial_match
        self.reactstyle = reactstyle
        self.enable_meta_planning = enable_meta_planning
        self.enable_question_assement = enable_question_assement
        self.enable_agent_ask = enable_agent_ask
        self.max_execution_time = max_execution_time
        self.skip_token_counting = skip_token_counting

        # tool invocation
        if self.use_tool_cache:
            self._tool_invocation_cache = ToolInvocationCache()

        if self.apply_loop_detection_check and not self.log_structured_messages:
            raise ValueError(
                "log_structured_messages must be True when loop detection is applied."
            )

        # storing log message into json
        self.json_log = []
        self.end_state = ""
        self.max_retries = max_retries

        self.cbm_tools = cbm_tools  # Search, Lookup
        if not self.tool_names:
            self.tool_names = get_tool_names(self.cbm_tools)
        if not self.tool_desc:
            self.tool_desc = get_tool_description(self.cbm_tools)

        self.react_llm_model_id = react_llm_model_id
        self.llm = llm
        if self.skip_token_counting:
            self.enc = partial(count_tokens, skip_token_counting=True)
        else:
            self.enc = count_tokens
        self.context_length = get_context_length(react_llm_model_id)
        self.actionstyle = actionstyle
        self.parent_agent = None

        # set the current object as parent of the each tool
        for tool in self.cbm_tools:
            if hasattr(
                tool, "parent_agent"
            ):  # Check if the 'parent_agent' attribute is present in the tool object
                tool.parent_agent = self  # Assign self as parent_agent
            if hasattr(tool, "parent_model_id"):
                tool.parent_model_id = self.react_llm_model_id

        # revise the prompt
        tmp_agent_prompt = self.agent_prompt.template
        jinja_template = Template(tmp_agent_prompt)
        revised_agent_prompt = jinja_template.render(
            enable_agent_ask=self.enable_agent_ask
        )
        self.agent_prompt.template = revised_agent_prompt

        self.__reset_agent()

    def add_step_trajectory(self, step_trajectory_file_name):
        if self.log_structured_messages:
            self.json_log[-1]["step_trajectory_file_name"] = step_trajectory_file_name

    def add_step_metric(self, step_metric_file_name):
        if self.log_structured_messages:
            self.json_log[-1]["step_metric_file_name"] = step_metric_file_name

    def add_step_trajectory_and_metric(self, atep_trajectory_json, step_metric_json):
        if self.log_structured_messages:
            self.json_log[-1]["step_trajectory_json"] = atep_trajectory_json
            self.json_log[-1]["step_metric_json"] = step_metric_json

    def add_module(self, module_path):
        """
        Dynamically imports all attributes from the given module and adds them to globals().

        :param module_path: Full module path as a string
        """

        try:

            module = importlib.import_module(module_path)  # Import module dynamically
            module_dict = vars(module)  # Get all attributes from the module

            # Add only functions and classes to globals()
            globals().update(
                {
                    name: obj
                    for name, obj in module_dict.items()
                    if not name.startswith("__")
                }
            )
        except ModuleNotFoundError:
            print(
                f"Error: Module '{module_path}' not found. Please check the module name and ensure it is installed."
            )

        except ImportError as e:
            print(f"Error importing '{module_path}': {e}")

        except Exception as e:
            print(f"Unexpected error while importing '{module_path}': {e}")

    def add_reference_to_parent(self, parent_agent):
        """Dynamically set the parent agent reference."""
        self.parent_agent = parent_agent

    def get_experiment_summary(self):
        # completion_tokens, prompt_tokens, api_calls = return_usage()
        ans = {
            "step": self.step_n,
            "info": {
                "model_stats": {
                    "tokens_sent": self.promptTokens,
                    "tokens_received": self.completionTokens,
                    "api_calls": self.llmCalls,
                    "total_cost": 0,
                    "instance_cost": 0,
                }
            },
        }
        return ans

    def export_trajectory(self):
        try:
            json_trajectory = {}
            json_trajectory["type"] = "mpe-agent"
            json_trajectory["task"] = self.question
            json_trajectory["environment"] = "mpe_main"
            json_trajectory["system_prompt"] = self._get_system_prompt()
            json_trajectory["demonstration"] = self.react_examples
            json_trajectory["scratchpad"] = self.scratchpad
            json_trajectory["endstate"] = self.end_state

            if self.log_structured_messages:
                json_trajectory["trajectroy_log"] = self.json_log

            history = [
                {
                    "role": "system",
                    "content": self._get_system_prompt(),
                    "agent": "primary",
                },
                {
                    "agent": "primary",
                    "content": self.react_examples,
                    "is_demo": True,
                    "role": "user",
                },
                {
                    "role": "user",
                    "content": "We're currently solving the condition based maintenance problem. Here's the problem description text:\nISSUE:\n"
                    + self.question
                    + "\n\n",
                    "agent": "primary",
                },
            ]
            json_trajectory["history"] = history

            # print (self.scratchpad)
            if (
                self.actionstyle == ActionStyle.BLOCK_FUNCTION
                or self.actionstyle == ActionStyle.SINGLE_LINE_FUNCTION
            ):
                # Regex for Thought blocks
                thought_pattern = re.compile(
                    r"Thought (\d+):\s(.*?)(?=Action \d+:|Thought \d+:|Observation \d+:|$)",
                    re.DOTALL,
                )
                thoughts = thought_pattern.findall(self.scratchpad)

                # Regex for Action blocks
                action_pattern = re.compile(
                    r"Action (\d+):\s(.*?)(?=Thought \d+:|Action \d+:|Observation \d+:|$)",
                    re.DOTALL,
                )
                actions = action_pattern.findall(self.scratchpad)

                # Regex for Observation blocks
                observation_pattern = re.compile(
                    r"Observation (\d+):\s(.*?)(?=Thought \d+:|Action \d+:|Observation \d+:|$)",
                    re.DOTALL,
                )
                observations = observation_pattern.findall(self.scratchpad)
            else:
                # split the scratchpad
                lines = self.scratchpad.split("\n")
                thoughts = list(filter(lambda x: x.startswith("Thought"), lines))
                actions = list(filter(lambda x: x.startswith("Action"), lines))
                observations = list(
                    filter(lambda x: x.startswith("Observation"), lines)
                )

            def get_part(text):
                pattern = r"\s*(?:Action|Thought|Observation)\s*\d+:\s*"
                cleaned_text = re.sub(pattern, "", text).strip()
                return cleaned_text

            total_round = max(len(thoughts), len(observations))
            # print(len(thoughts), len(actions), len(observations))
            trajectory = []

            for j in range(total_round):
                new_entry = {}
                if (
                    self.actionstyle == ActionStyle.BLOCK_FUNCTION
                    or self.actionstyle == ActionStyle.SINGLE_LINE_FUNCTION
                ):
                    try:
                        new_entry["thought"] = thoughts[j][1]
                        new_entry["action"] = actions[j][1]
                        if j < len(observations):
                            new_entry["observation"] = observations[j][1]
                    except:
                        pass
                else:
                    try:
                        new_entry["thought"] = get_part(thoughts[j])
                        new_entry["action"] = (
                            "Tool Name: "
                            + get_part(actions[j * 2])
                            + " , Tool Parameter: "
                            + get_part(actions[j * 2 + 1])
                            + ""
                        )
                        if j != total_round - 1:
                            new_entry["observation"] = get_part(observations[j])
                        else:
                            new_entry["observation"] = ""
                    except:
                        pass
                trajectory.append(new_entry)

            # print (json_trajectory)
            json_trajectory["trajectory"] = trajectory
            completion_tokens, prompt_tokens, api_calls = (
                self.completionTokens,
                self.promptTokens,
                self.llmCalls,
            )

            info = {
                "model_stats": {
                    "tokens_sent": prompt_tokens,
                    "tokens_received": completion_tokens,
                    "api_calls": api_calls,
                    "total_cost": 0,
                    "instance_cost": 0,
                }
            }
            json_trajectory["info"] = info
            json_trajectory["final_answer"] = self.answer
            return json_trajectory
        except Exception as ex:
            #print (ex)
            return {}

    def print_final_answer(self):
        """ """
        print(Fore.YELLOW + "-" * 50 + Style.RESET_ALL)
        print("Model Produced following final output:")
        print(self.answer)
        print(Fore.YELLOW + "*" * 50 + Style.RESET_ALL)

    def evaluate_question_for_agent_ability(self, name):
        taa = TaskAssessmentAgent(llm=watsonx_llm, model_id=self.react_llm_model_id)
        ans = taa.evaluate_response(
            question=self.question,
            agent_name=name,
            agent_expertise="AI Agent with tools",
            agent_task_history="",
            agent_tools=self.tool_desc,
        )

        return True

    def run(self, reset=True, name="ReActXen Agent") -> None:

        if reset:
            self.__reset_agent()

        start_time = time.time()

        if self.enable_question_assement:
            self.evaluate_question_for_agent_ability()
            pass

        if self.enable_meta_planning:
            plan = self.prompt_meta_agent()
            plan = re.search(r"1\..*", plan, re.DOTALL)
            plan_header = METAPLAN_HEADER.format(steps=str(plan.group()))
            self.scratchpad += plan_header + "\n"

        if self.debug:
            print(Fore.YELLOW + "-" * 50 + Style.RESET_ALL)
            print("Scratch Pad Content - At the Start of Running Agent")
            print(self.scratchpad)
            print(Fore.YELLOW + "*" * 50 + Style.RESET_ALL)

        print(f"{Style.BRIGHT}{Fore.MAGENTA}I am {name} with ReAct{Style.RESET_ALL}")
        print(
            f"{Style.BRIGHT}{Fore.LIGHTMAGENTA_EX}Input Question: {self.question}{Style.RESET_ALL}"
        )

        while not self.is_halted() and not self.is_finished():
            self.step()

            # check loop detection
            if self.apply_loop_detection_check:
                if self.step_n > 2:
                    loop_type = self.check_for_loop_type()
                    self.json_log[-1]["is_loop_detected"] = loop_type
                    if loop_type == "early_loop":
                        self.json_log[-1][
                            "state"
                        ] = f"Early stop due to repeated actions."
                        feedback = f"The same action has been executed over 3 times. Early stop due to repeated actions."
                        self.scratchpad += feedback
                        self.json_log[-1]["additional_scratchpad_feedback"] = feedback
                        if self.debug:
                            print(Fore.YELLOW + "-" * 50 + Style.RESET_ALL)
                            print("We early stop execution.")
                            print(feedback)
                            print(Fore.YELLOW + "*" * 50 + Style.RESET_ALL)
                        break
                    elif loop_type == "action_loop":
                        feedback = (
                            "\nIt appears that the previous steps were repetitive, as the same actions and inputs were used in the last few interactions. "
                            "This repetition indicates that the system is stuck in an action-input loop, producing similar outputs. "
                            "Please generate an alternative action or modify the input parameters to explore a new approach and break the loop."
                        )
                        self.scratchpad += feedback
                        self.json_log[-1]["additional_scratchpad_feedback"] = feedback

                        if self.debug:
                            print(Fore.YELLOW + "-" * 50 + Style.RESET_ALL)
                            print(
                                "Action Loop Detected. We provided following feedback to Agent."
                            )
                            print(feedback)
                            print(Fore.YELLOW + "*" * 50 + Style.RESET_ALL)

                    elif loop_type == "thought_loop":
                        feedback = (
                            "\nThe recent thought steps seem repetitive, as the same thought has been generated multiple times despite changing observations or inputs. "
                            "Please generate a better explanation of the recent observations and then generate a thought that adds value or evolves the reasoning process. "
                            "You may also consider using a different tool or parameter or approach if that is needed."
                        )
                        self.scratchpad += feedback
                        self.json_log[-1]["additional_scratchpad_feedback"] = feedback

                        if self.debug:
                            print(Fore.YELLOW + "-" * 50 + Style.RESET_ALL)
                            print(
                                "Thought Loop Detected. We provided following feedback to Agent."
                            )
                            print(feedback)
                            print(Fore.YELLOW + "*" * 50 + Style.RESET_ALL)

            # added to handle the context length issue
            if self.handle_context_length_overflow:
                # print (f'calling truncate_scratchpad in handle_context_length_overflow : {self.context_length}')
                self.scratchpad = truncate_scratchpad(
                    self.scratchpad,
                    n_tokens=self.context_length,
                    tokenizer=self.enc,
                    model_id=self.react_llm_model_id,
                )

            # Check the total execution time
            if self.max_execution_time != -1 and time.time() - start_time > self.max_execution_time:
                print(
                    f"{Style.BRIGHT}{Fore.MAGENTA}We cross the total alloted execution time{Style.RESET_ALL}"
                )
                break

        print(f"{Style.BRIGHT}{Fore.MAGENTA}Process is completed now{Style.RESET_ALL}")

    def probe(self, agent_question):
        """
        Handle an incoming question and provide a response.
        """
        llmResult = self.llm(
            self._build_agent_probe_prompt(agent_question=agent_question),
            model_id=self.react_llm_model_id,
            stop=["\nQuestion"],
        )
        return llmResult["generated_text"]

    def agent_skeleton(self, question):
        """
        Skeleton layer that calls `probe`.
        This is the core processing layer.
        """
        print(f"[Skeleton] Processing question: {question}")
        return self.probe(question)

    def agent_stub(self, agent_name, question):
        """
        Entry point for a request, forwards to skeleton or other agents.
        """
        print(f"[Stub] Received request for {agent_name} with question: {question}")

        if agent_name == "parent" and self.parent_agent:
            print("[Stub] Forwarding request to parent Agent")
            return self.parent_agent.agent_skeleton(question)
        else:
            print("[Stub] No matching agent found")
            return "Agent not found."
        """ 
        # TBI
        elif agent_name in self.Ttools:
            print(f"[Stub] Forwarding request to {agent_name}")
            return self.Ttools[agent_name].agent_skeleton(question)
        """

    def check_for_loop_type(self):
        # If there are less than 3 entries, can't detect a loop
        if len(self.json_log) < 3:
            return None

        # Get the last 3 log entries
        last_three = self.json_log[-3:]

        # Check for action and action_input loop
        action_input_matches = (
            last_three[0]["action"]
            == last_three[1]["action"]
            == last_three[2]["action"]
        ) and (
            last_three[0]["action_input"]
            == last_three[1]["action_input"]
            == last_three[2]["action_input"]
        )

        # Check for thought loop
        thought_matches = (
            last_three[0]["thought"]
            == last_three[1]["thought"]
            == last_three[2]["thought"]
        )
        observation_changes = (
            last_three[0]["observation"] != last_three[1]["observation"]
            or last_three[1]["observation"] != last_three[2]["observation"]
        )

        if (
            self.early_stop
            and (action_input_matches or thought_matches)
            and last_three[1]["is_loop_detected"] is not None
        ):
            return "early_loop"
        elif action_input_matches:
            return "action_loop"
        elif thought_matches and observation_changes:
            return "thought_loop"

        return None  # No loop detected

    def tool_pattern_match(self, tool_name, action_name):
        if self.enable_tool_partial_match:
            if action_name.startswith(tool_name):
                return True
        else:
            return False

    def step(self) -> None:

        if self.log_structured_messages:
            # start storing the information in more structured formate
            # these are the various field we store at each iteration
            self.json_log.append(
                {
                    "step": self.step_n,
                    "raw_llm_thought_output": "",
                    "raw_llm_action_output": "",
                    "raw_observation_output": "",
                    "raw_llm_output": "",
                    "thought": "",
                    "action": "",
                    "action_input": "",
                    "observation": "",
                    "state": "",  # invalid_action, invalid_thought, etc
                    # "include_in_extraction": False,
                    "is_loop_detected": None,
                    "additional_scratchpad_feedback": "",
                    "step_trajectory_file_name": None,
                    "step_metric_file_name": None,
                    "step_trajectory_json": None,
                    "step_metric_json": None,
                }
            )

        ################## This is a Agent thinking : Started ##########################
        # Think
        # print (type(self.scratchpad), type(self.step_n))
        if self.reactstyle == ReActStyle.OnlyAct:
            self.scratchpad += f"\nAction {self.step_n}:"
        else:
            self.scratchpad += f"\nThought {self.step_n}:"

        if (
            self.actionstyle == ActionStyle.SINGLE_LINE_TOOL_CALL
            or self.actionstyle == ActionStyle.MULTI_LINE_PARAGRAPH
        ) and self.reactstyle == ReActStyle.ThoughtActTogether:
            # call to think and stop prior to action
            # prompt_agent is a casecading function. it first call llm followed by formating of llm output
            let_me_think_dict = self.prompt_agent(
                stop=["\nObservation"], prefix=f"Thought {self.step_n}:"
            )
            if self.debug:
                print(
                    f"{Style.BRIGHT}{Fore.RED}Debug Info (Step {self.step_n}):{Style.RESET_ALL}"
                )
                print(
                    f"{Fore.RED}{json.dumps(let_me_think_dict, indent=4)}{Style.RESET_ALL}"
                )

            if self.log_structured_messages:
                self.json_log[-1]["thought"] = let_me_think_dict["thought"]
                self.json_log[-1]["raw_llm_output"] = let_me_think_dict["llm_output"]

            action_dict = let_me_think_dict
            let_me_think = action_dict["thought"]
            self.scratchpad += " " + action_dict["thought"]
            self.scratchpad += f"\nAction {self.step_n}:"

            print(
                f"{Style.BRIGHT}{Fore.BLUE}Thought {self.step_n}: {let_me_think}{Style.RESET_ALL}"
            )
        elif self.reactstyle == ReActStyle.OnlyAct:
            ################## This is a Agent Act Generation : Started ##########################
            action_dict = self.prompt_agent(
                stop=["\nObservation", "\nThought"], prefix=f"Action {self.step_n}:"
            )
            if self.debug:
                print(
                    f"{Style.BRIGHT}{Fore.RED}Debug Info (Step {self.step_n}):{Style.RESET_ALL}"
                )
                print(f"{Fore.RED}{json.dumps(action_dict, indent=4)}{Style.RESET_ALL}")
        else:
            # call to think and stop prior to action
            # prompt_agent is a casecading function. it first call llm followed by formating of llm output
            let_me_think_dict = self.prompt_agent(
                stop=["\nAction", "\nObservation"], prefix=f"Thought {self.step_n}:"
            )
            let_me_think = let_me_think_dict["thought"]
            if len(let_me_think) == 0:
                retry = 0
                self.original_scratchpad = self.scratchpad
                while len(let_me_think) != 0:
                    retry += 1
                    if retry > self.max_retries:
                        break
                    self.scratchpad += " Your thought is filtered due to empty content. Please make sure your thought content does not start with ['Thought', 'Action', 'Observation']."
                    let_me_think_dict = self.prompt_agent(
                        stop=["\nAction", "\nObservation"],
                        prefix=f"Thought {self.step_n}:",
                    )
                    let_me_think = let_me_think_dict["thought"]
                self.scratchpad = self.original_scratchpad

            if self.debug:
                print(
                    f"{Style.BRIGHT}{Fore.RED}Debug Info (Step {self.step_n}):{Style.RESET_ALL}"
                )
                print(
                    f"{Fore.RED}{json.dumps(let_me_think_dict, indent=4)}{Style.RESET_ALL}"
                )

            if self.log_structured_messages:
                self.json_log[-1]["thought"] = let_me_think_dict["thought"]
                self.json_log[-1]["raw_llm_thought_output"] = let_me_think_dict[
                    "llm_output"
                ]

            self.scratchpad += " " + let_me_think
            print(
                f"{Style.BRIGHT}{Fore.BLUE}Thought {self.step_n}: {let_me_think}{Style.RESET_ALL}"
            )

            ################## This is a Agent thinking : Completed ##########################

            # Additional check - TBA
            """ 
            if self.apply_chat_template:
                if 'Final Answer:'.lower() in let_me_think.lower():
                    print ('we find the final answer inside the thougth')
                    #exit(0)
            """

            ################## This is a Agent Act Generation : Started ##########################
            message = ""
            self.scratchpad += f"\nAction {self.step_n}:"
            action_dict = self.prompt_agent(
                stop=["\nObservation", "\nThought"], prefix=f"Action {self.step_n}:"
            )

            if self.debug:
                print(
                    f"{Style.BRIGHT}{Fore.RED}Debug Info (Step {self.step_n}):{Style.RESET_ALL}"
                )
                print(f"{Fore.RED}{json.dumps(action_dict, indent=4)}{Style.RESET_ALL}")

        ##### Error Handeling

        # There is many error an agent make while generating the Action (and Action Input)
        # detecting decoding error and providing feedback

        # Error 1 : The Code Agent
        if (
            action_dict["error"]
            and action_dict["action"]
            == "ErrorMessage:MPEAgent:ActionDecoding:CodeBlock"
        ):
            self.scratchpad += " " + action_dict["llm_output"]
            self.scratchpad += (
                f"\nObservation {self.step_n}: {action_dict['error_feedback']}"
            )
            if self.log_structured_messages:
                self.json_log[-1]["action"] = action_dict["llm_output"]
                self.json_log[-1]["observation"] = action_dict["error_feedback"]
                self.json_log[-1]["state"] = "Invalid Action Decoding"
            print(
                f"{Style.BRIGHT}{Fore.YELLOW}Action {self.step_n}: {action_dict['llm_output']} {Style.RESET_ALL}"
            )
            print(
                f"{Style.BRIGHT}{Fore.GREEN}Observation {self.step_n}: {action_dict['error_feedback']}{Style.RESET_ALL}"
            )
            self.step_n = self.step_n + 1
            return

        # Error 2 : The Text Agent
        if (
            action_dict["error"]
            and action_dict["action"]
            == "ErrorMessage:MPEAgent:ActionDecoding:TextBlock"
        ):
            self.scratchpad += " " + action_dict["llm_output"]
            # Some time action input is already part of action
            if f"Action Input {self.step_n}:" not in action_dict["llm_output"]:
                self.scratchpad += f"\nAction Input {self.step_n}:"
                self.scratchpad += " "
            self.scratchpad += (
                f"\nObservation {self.step_n}: {action_dict['error_feedback']}"
            )
            if self.log_structured_messages:
                self.json_log[-1]["action"] = action_dict["llm_output"]
                self.json_log[-1]["action_input"] = None
                self.json_log[-1]["observation"] = action_dict["error_feedback"]
                self.json_log[-1]["state"] = "Invalid Action Decoding"
            print(
                f"{Style.BRIGHT}{Fore.YELLOW}Action {self.step_n}: {action_dict['llm_output']} {Style.RESET_ALL}"
            )
            print(
                f"{Style.BRIGHT}{Fore.YELLOW}Action Input {self.step_n}: None {Style.RESET_ALL}"
            )
            print(
                f"{Style.BRIGHT}{Fore.GREEN}Observation {self.step_n}: {action_dict['error_feedback']}{Style.RESET_ALL}"
            )
            self.step_n = self.step_n + 1
            return

        # we find every thing is good.
        if self.actionstyle == ActionStyle.SINGLE_LINE_FUNCTION:
            self.scratchpad += " " + " ".join(action_dict["action_input"])

            if self.log_structured_messages:
                self.json_log[-1]["action"] = action_dict["action_input"]
                self.json_log[-1]["raw_llm_action_output"] = action_dict["llm_output"]

        elif self.actionstyle == ActionStyle.BLOCK_FUNCTION:
            self.scratchpad += " " + " ".join(action_dict["action_input"])

            if self.log_structured_messages:
                self.json_log[-1]["action"] = action_dict["action_input"]
                self.json_log[-1]["raw_llm_action_output"] = action_dict["llm_output"]

        elif self.actionstyle == ActionStyle.SINGLE_LINE_TOOL_CALL:
            if action_dict["action"] and action_dict["action_input"]:
                self.scratchpad += " " + action_dict["action"]
                self.scratchpad += f"\nAction Input {self.step_n}:"
                self.scratchpad += " " + action_dict["action_input"]

                if self.log_structured_messages:
                    self.json_log[-1]["action"] = action_dict["action"]
                    self.json_log[-1]["action_input"] = action_dict["action_input"]
                    self.json_log[-1]["raw_llm_action_output"] = action_dict[
                        "llm_output"
                    ]
            else:
                if action_dict["action"].lower() == "Finish".lower():
                    self.scratchpad += " " + action_dict["action"]
                    self.scratchpad += f"\nAction Input {self.step_n}:"
                    self.scratchpad += " " + action_dict["action_input"]
                    if self.log_structured_messages:
                        self.json_log[-1]["action"] = action_dict["action"]
                        self.json_log[-1]["action_input"] = action_dict["action_input"]
                        self.json_log[-1]["raw_llm_action_output"] = action_dict[
                            "llm_output"
                        ]
                    # we expect some thing
                    action_dict["error"] = True
                    action_dict["error_feedback"] = (
                        "Task marked as finished, but no final answer was provided in the action-input. The Finish action requires the final answer to complete the task. Please provide the answer in the action-input field before marking the task as finished. The final answer should be a clear response."
                    )
                elif action_dict["action"] and len(action_dict["action_input"]) == 0:
                    self.scratchpad += " " + action_dict["action"]
                    self.scratchpad += f"\nAction Input {self.step_n}:"
                    self.scratchpad += " " + action_dict["action_input"]

                    if self.log_structured_messages:
                        self.json_log[-1]["action"] = action_dict["action"]
                        self.json_log[-1]["action_input"] = action_dict["action_input"]
                        self.json_log[-1]["raw_llm_action_output"] = action_dict[
                            "llm_output"
                        ]
                else:
                    # Added an error handeling task
                    feedback = (
                        "The generated Action and/or Action Input are incorrectly formatted. "
                        "Please ensure the following: "
                        "1. The Action should be a valid, well-structured string that represents a specific operation or tool name. "
                        "2. The Action Input should be a valid and coherent string that corresponds to the requirements of the Action. "
                        "3. If the Action is 'Finish', the Action Input must contain the final answer explicitly. "
                        "Review your output and correct the formatting or logic as necessary."
                    )
                    self.scratchpad += ""
                    self.scratchpad += f"\nAction Input {self.step_n}:"
                    self.scratchpad += f"\nObservation {self.step_n}: {feedback}"
                    if self.log_structured_messages:
                        self.json_log[-1]["action"] = ""
                        self.json_log[-1]["action_input"] = ""
                        self.json_log[-1]["observation"] = feedback
                        self.json_log[-1]["state"] = "Invalid Action"
                    print(
                        f"{Style.BRIGHT}{Fore.YELLOW}Action {self.step_n}: None {Style.RESET_ALL}"
                    )
                    print(
                        f"{Style.BRIGHT}{Fore.YELLOW}Action Input {self.step_n}: None {Style.RESET_ALL}"
                    )
                    print(
                        f"{Style.BRIGHT}{Fore.GREEN}Observation {self.step_n}: No Action and Action Input is generated {Style.RESET_ALL}"
                    )
                    self.step_n = self.step_n + 1
                    return

        elif self.actionstyle == ActionStyle.MULTI_LINE_PARAGRAPH:
            raise NotImplementedError(
                f"Action Style {ActionStyle.MULTI_LINE_PARAGRAPH} is not yet implemented"
            )

        # print output
        if (
            self.actionstyle == ActionStyle.SINGLE_LINE_FUNCTION
            or self.actionstyle == ActionStyle.BLOCK_FUNCTION
        ):
            print(
                f"{Style.BRIGHT}{Fore.YELLOW}Action {self.step_n}: {action_dict['action_input']} {Style.RESET_ALL}"
            )
        else:
            print(
                f"{Style.BRIGHT}{Fore.YELLOW}Action {self.step_n}: {action_dict['action']} {Style.RESET_ALL}"
            )
            print(
                f"{Style.BRIGHT}{Fore.CYAN}Action Input {self.step_n}: {action_dict['action_input']} {Style.RESET_ALL}"
            )
        ################## This is a Agent Act Generation : Completed ##########################

        # Observe

        ################## This is a Agent Act Execution : Started ##########################
        # print ('you came here', self.actionstyle)
        # Here we store the results
        action_execution_output = None

        try:
            action_type, argument = action_dict["action"], action_dict["action_input"]
            # print(action_type, argument)
            # check is proces over
            # print(action_type)
            # Finish --> Text Agent
            # Final Answer --> Code Agent
            if isinstance(action_type, str) and (
                action_type.lower() == "Finish".lower()
                or "final answer" in action_type.lower()
            ):
                if (
                    "final answer" in action_type.lower()
                    or "finish" in action_type.lower()
                ) and action_dict["error"]:
                    # there is some error need to be handled before we call it off
                    self.scratchpad += f"\nObservation {self.step_n}: "
                    self.scratchpad += action_dict["error_feedback"]
                    self.step_n += 1

                    print(
                        f"{Style.BRIGHT}{Fore.GREEN}Observation {self.step_n}:  {action_dict['error_feedback']}{Style.RESET_ALL}"
                    )

                    if self.log_structured_messages:
                        self.json_log[-1]["observation"] = action_dict["error_feedback"]
                        self.json_log[-1]["state"] = "Invalid Final Answer"

                    return

                self.answer = argument
                self.finished = True
                self.step_n += 1
                return
            elif isinstance(action_type, str) and (
                action_type.lower() == "Self-Ask".lower()
                or "self-ask" in action_type.lower()
            ):
                # Special Tool
                self.scratchpad += f"\nObservation {self.step_n}: "
                action_dict = self.prompt_agent(
                    stop=["\nThought", "\nAction"],
                    prefix=f"\nObservation {self.step_n}:",
                )
                if len(action_dict["observation"]) > 0:
                    self.scratchpad += action_dict["observation"]
                    action_execution_output = action_dict["observation"]
                    if self.log_structured_messages:
                        self.json_log[-1]["raw_observation_output"] = action_dict[
                            "llm_output"
                        ]
                else:
                    self.scratchpad += "No relevant output or response was generated."
                    action_execution_output = (
                        "No relevant output or response was generated."
                    )
                    if self.log_structured_messages:
                        self.json_log[-1][
                            "raw_observation_output"
                        ] = "No relevant output or response was generated."
            elif isinstance(action_type, str) and (
                action_type.lower() == "Agent-Ask".lower()
                or "agent-ask" in action_type.lower()
            ):
                # Special Tool
                self.scratchpad += f"\nObservation {self.step_n}: "
                # call_parent(parent_id, action_input)
                agent_response = self.agent_stub(agent_name="parent", question=argument)
                self.scratchpad += agent_response
                action_execution_output = agent_response
                if self.log_structured_messages:
                    self.json_log[-1]["raw_observation_output"] = agent_response
            else:
                self.scratchpad += f"\nObservation {self.step_n}: "
                if self.actionstyle == ActionStyle.SINGLE_LINE_FUNCTION:
                    tool_not_find = False
                    try:
                        code_output = eval(argument[0], globals())
                        current_observation = ""
                        if (
                            type(code_output) == list or type(code_output) == dict
                        ) and (len(code_output) == 0):
                            current_observation = "The code executed successfully. However, the function call returned an empty output. "
                        else:
                            current_observation = str(code_output)
                        self.scratchpad += current_observation
                        action_execution_output = current_observation  # storing output
                        if self.log_structured_messages:
                            self.json_log[-1]["raw_observation_output"] = str(
                                code_output
                            )
                    except Exception as e:
                        current_observation = f"Invalid action: {e}. Please make sure your action is a valid and executable function call with correct arguments based on the API description."
                        self.scratchpad += current_observation
                        if self.log_structured_messages:
                            self.json_log[-1][
                                "raw_observation_output"
                            ] = current_observation
                            self.json_log[-1]["state"] = f"Invalid Action"
                elif (
                    self.actionstyle == ActionStyle.BLOCK_FUNCTION
                    or action_type == "Code Block"
                ):
                    # print("Came here -----------------------")
                    tool_not_find = False
                    err = None
                    output_buffer = io.StringIO()

                    current_stdout = sys.stdout  # Save the current standard output
                    current_stderr = sys.stderr  # Save the current standard error

                    sys.stdout = output_buffer  # Redirect standard output to the buffer
                    sys.stderr = output_buffer  # Redirect standard error to the buffer

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        signal.alarm(300)  # Set the alarm to 300 seconds
                        try:
                            exec(argument[0], globals(), self.local_vars)
                            # Reset the alarm
                            # signal.alarm(0)
                        except Exception as e:
                            # print(e)
                            err = e
                            # print(f"Illegal action: {e}. {self.api_error_note}")
                        finally:
                            signal.alarm(0)

                        sys.stdout = (
                            current_stdout  # Restore the original standard output
                        )
                        sys.stderr = (
                            current_stderr  # Restore the original standard error
                        )

                        code_output = (
                            output_buffer.getvalue()
                        )  # Get the output from the buffer
                        output_buffer.close()

                    api_error_note = "If you are collecting data with code, please make sure your action is a valid and executable block of code with correct syntax based on the API description, and use print() for outputs; If you are making the final answer, please start the action immediately with 'Final Answer:' without enclosing within triple backticks, for example, 'Action: Final Answer: {{}}'"

                    if err != None:
                        current_observation = f"Invalid action: {err}. {api_error_note}"
                        self.scratchpad += current_observation
                        if self.log_structured_messages:
                            self.json_log[-1][
                                "raw_observation_output"
                            ] = current_observation
                            self.json_log[-1]["state"] = f"Invalid Action"
                    elif len(code_output) == 0:
                        if "print(" in argument[0]:
                            current_observation = "No printed output from the action because you are printing an empty object. It is possible that there is no real output."
                            self.scratchpad += current_observation
                            if self.log_structured_messages:
                                self.json_log[-1][
                                    "raw_observation_output"
                                ] = current_observation
                        else:
                            current_observation = f"Invalid action: No print() statement in the action. {api_error_note}"
                            self.scratchpad += current_observation
                            if self.log_structured_messages:
                                self.json_log[-1][
                                    "raw_observation_output"
                                ] = current_observation
                                self.json_log[-1]["state"] = f"Invalid Action"
                    else:
                        current_observation = str(code_output)
                        current_observation = current_observation.strip()
                        # print(current_observation)
                        self.scratchpad += current_observation
                        action_execution_output = current_observation
                        if self.log_structured_messages:
                            self.json_log[-1]["raw_observation_output"] = str(
                                code_output
                            )

                elif self.apply_chat_template:
                    # This code need to be improved !!!
                    # print ('you came here ------')
                    try:
                        json_dict = json.loads(argument)
                        # print(json_dict)
                        tool_not_find = True
                        for tool in self.cbm_tools:
                            if tool.name.lower() == action_type.lower():
                                tool_not_find = False
                                tool_output = tool.run(tool_input=json_dict)
                                action_execution_output = tool_output
                                self.scratchpad += tool_output
                                # print(tool_output)
                    except Exception as ex:
                        # print(ex)
                        pass
                else:
                    # call toolsearch
                    tool_not_find = True
                    if self.use_tool_cache:
                        tool_cached_ans = self._tool_invocation_cache.get_from_cache(
                            tool_name=action_type, tool_parameters=argument
                        )
                        if tool_cached_ans:
                            tool_not_find = False
                            action_execution_output = tool_cached_ans
                            self.scratchpad += action_execution_output
                            if self.log_structured_messages:
                                self.json_log[-1][
                                    "raw_observation_output"
                                ] = action_execution_output
                    if tool_not_find:
                        for tool in self.cbm_tools:
                            if (
                                tool.name.lower() == action_type.lower()
                                or self.tool_pattern_match(
                                    tool.name.lower(), action_type.lower()
                                )
                            ):
                                tool_not_find = False
                                # parameter value cases
                                if "=" in argument:
                                    try:
                                        pattern = r"(\w+)=([^,]+)"
                                        # pattern = r"(\w+)=\[(.*?)\]|\b(\w+)=([^,]+)"
                                        pattern = (
                                            r"(\w+)=\[(.*?)\]|(\w+)=(.+?)(?=, \w+=|\Z)"
                                        )
                                        matches = re.findall(pattern, argument)
                                        dictionary = {}

                                        # for key, value in matches:
                                        for match in matches:
                                            # for match in matches:
                                            #    if match[0]:
                                            #        key = match[0]

                                            if match[0]:  # Case: metadata=[...]
                                                key = match[0]
                                                value = match[1].split(", ")
                                            else:
                                                key = match[2]
                                                value = match[3].strip()

                                            key = key.strip()
                                            value = (
                                                value.strip()
                                                if value != "None"
                                                else None
                                            )
                                            if key != "v__args":
                                                dictionary[key] = value

                                        """
                                        pairs = argument.split(", ")
                                        dictionary = {}
                                        for pair in pairs:
                                            key, value = pair.split("=")
                                            if value == "None":
                                                value = None
                                            else:
                                                value = value.strip()
                                            if key.strip() != "v__args":
                                                dictionary[key.strip()] = value
                                        """
                                        # print("trying to call tool...")
                                        try:
                                            tool_output = tool.run(
                                                tool_input=dictionary
                                            )

                                            if isinstance(tool_output, dict):
                                                tool_output = str(tool_output)

                                            action_execution_output = tool_output
                                            self.scratchpad += tool_output

                                            if self.use_tool_cache:
                                                self._tool_invocation_cache.add_to_cache(
                                                    tool_name=action_type,
                                                    tool_parameters=argument,
                                                    result=tool_output,
                                                )

                                            if self.log_structured_messages:
                                                self.json_log[-1][
                                                    "raw_observation_output"
                                                ] = tool_output

                                        except Exception as ex:
                                            self.scratchpad += (
                                                f"Error encountered: Execution failed while calling the Tool. Details: {str(ex)}"
                                            ).strip()
                                            if self.log_structured_messages:
                                                self.json_log[-1][
                                                    "raw_observation_output"
                                                ] = str(ex)
                                    except Exception as ex:
                                        self.scratchpad += (
                                            "Error encountered: Issue while unpacking the arguments provided as input to the tool. "
                                            f"Details: {str(ex)}"
                                        ).strip()
                                        if self.log_structured_messages:
                                            self.json_log[-1][
                                                "raw_observation_output"
                                            ] = str(ex)
                                elif argument.strip().startswith(
                                    "{"
                                ) and argument.strip().endswith("}"):
                                    try:
                                        dictionary = json.loads(
                                            argument
                                        )  # Convert JSON string to dictionary

                                        try:
                                            tool_output = tool.run(
                                                tool_input=dictionary
                                            )
                                            if isinstance(tool_output, dict):
                                                tool_output = str(tool_output)

                                            if self.log_structured_messages:
                                                self.json_log[-1][
                                                    "raw_observation_output"
                                                ] = tool_output  # Log raw output

                                            # Clean up tool output (only if it's a string)
                                            if isinstance(tool_output, str):
                                                tool_output = tool_output.replace(
                                                    "\n", " "
                                                ).strip()

                                            action_execution_output = tool_output
                                            self.scratchpad += tool_output  # Store output in scratchpad

                                            # Cache the result if caching is enabled
                                            if self.use_tool_cache:
                                                self._tool_invocation_cache.add_to_cache(
                                                    tool_name=action_type,
                                                    tool_parameters=dictionary,  # ✅ Store original dictionary
                                                    result=tool_output,
                                                )

                                        except TypeError as ex:
                                            error_message = f"Error: Invalid parameters for Tool. Details: {str(ex)}"
                                            self.scratchpad += error_message

                                            if self.log_structured_messages:
                                                self.json_log[-1][
                                                    "raw_observation_output"
                                                ] = error_message  # Log error

                                        except Exception as ex:
                                            error_message = f"Error: Execution failed while calling the Tool. Details: {str(ex)}"
                                            self.scratchpad += error_message

                                            if self.log_structured_messages:
                                                self.json_log[-1][
                                                    "raw_observation_output"
                                                ] = error_message  # Log error

                                    except json.JSONDecodeError as e:
                                        error_message = f"Error: Failed to parse JSON. Details: {str(e)}"
                                        self.scratchpad += error_message

                                        if self.log_structured_messages:
                                            self.json_log[-1][
                                                "raw_observation_output"
                                            ] = error_message  # Log JSON error

                                else:
                                    # this is for a thread which is for direct passing
                                    try:
                                        tool_output = tool.run(argument)
                                        if self.log_structured_messages:
                                            self.json_log[-1][
                                                "raw_observation_output"
                                            ] = tool_output
                                        tool_output = (
                                            tool_output.replace("\n", " ")
                                            .strip("\n")
                                            .strip()
                                        )
                                        action_execution_output = tool_output
                                        self.scratchpad += tool_output

                                        if self.use_tool_cache:
                                            self._tool_invocation_cache.add_to_cache(
                                                tool_name=action_type,
                                                tool_parameters=argument,
                                                result=tool_output,
                                            )

                                    except Exception as ex:
                                        self.scratchpad += (
                                            f"Error encountered: Execution failed while calling the Tool. Details: {str(ex)}"
                                        ).strip()
                                        if self.log_structured_messages:
                                            self.json_log[-1][
                                                "raw_observation_output"
                                            ] = str(ex)

                                # we find tool and now good
                                break

                if tool_not_find:
                    self.scratchpad += (
                        f"Invalid Action detected. The selected action - {action_type} does not match any valid options. "
                        f"Please choose an action from the following list: [{', '.join(self.tool_names)}]."
                    ).strip()
                    if self.log_structured_messages:
                        self.json_log[-1]["state"] = "Invalid Action: Tool Not Found"

        except Exception as ex:
            # print(ex)
            self.scratchpad += f"Error encountered: Issue while parsing/executing Action. Details: {str(ex)}"

        if action_execution_output:
            print(
                f"{Style.BRIGHT}{Fore.GREEN}Observation {self.step_n}: {action_execution_output}{Style.RESET_ALL}"
            )
            if self.log_structured_messages:
                self.json_log[-1]["observation"] = action_execution_output
                self.json_log[-1]["state"] = "Valid Action"
        else:
            # message = self.scratchpad.split("\n")[-1]
            observation_prefix = f"\nObservation {self.step_n}: "
            last_observation_index = self.scratchpad.rfind(observation_prefix)
            message = self.scratchpad[last_observation_index:]
            print(f"{Style.BRIGHT}{Fore.GREEN}{message}{Style.RESET_ALL}")
            if self.log_structured_messages:
                self.json_log[-1]["observation"] = message.split(observation_prefix)[-1]
                self.json_log[-1]["state"] = "Invalid Action"

        message = ""
        self.step_n += 1

    def prompt_meta_agent(self) -> str:
        llmResult = self.llm(
            self._build_meta_agent_prompt(),
            model_id=self.react_llm_model_id,
            stop=["\nFinish", "Finish."],
        )
        return llmResult["generated_text"]

    def prompt_agent(self, stop, prefix) -> str:
        if isinstance(stop, str):
            stop = [stop]
        llmResult = self.llm(
            self._build_agent_prompt(),
            model_id=self.react_llm_model_id,
            stop=stop + ["\nQuestion"],
        )
        self.completionTokens += llmResult["generated_token_count"]
        self.promptTokens += llmResult["input_token_count"]
        self.llmCalls += 1

        return format_step(
            llmResult["generated_text"],
            stop,
            prefix,
            self.debug,
            self.actionstyle,
            self.reactstyle,
        )

    def _get_system_prompt(self) -> str:
        return self.agent_prompt.format(
            examples=self.react_examples,
            question=self.question,
            scratchpad=self.scratchpad,
            tool_desc=self.tool_desc,
            tool_names=", ".join(self.tool_names),
        )

    """
    <|start of role|>system<|end of role|>Your name is Granite.<|end of text|>
    <|start of role|>available tools<|end of role|>[{“name”: “get temp”, ...}, ...]<|end of text|>
    <|start of role|>user<|end of role|>What is temperature in Boston?<|end of text|>
    <|start of role|>assistant<|end of role|><|tool call|>[{“name”: “get temp”, ...}]<|end of text|>
    <|start of role|>tool response<|end of role|>{“temp”: 20.5, “unit”: “C”}<|end of text|>
    """

    def _decorate_message_for_llm_using_chat_template(self):
        # final_prompt = ''
        # final_prompt +=
        pass

    def _build_agent_probe_prompt(self, agent_question) -> str:
        return scratchpad_probe_prompt.format(
            agent_question=agent_question,
            question=self.question,
            scratchpad=self.scratchpad,
        )

    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
            examples=self.react_examples,
            question=self.question,
            scratchpad=self.scratchpad,
            tool_desc=self.tool_desc,
            tool_names=", ".join(self.tool_names),
        )

    def _build_meta_agent_prompt(self) -> str:
        return metaplan_prompt.format(
            examples=MetaPlanSample,
            task=self.question,
            tool_desc=self.tool_desc,
        )

    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        return EM(self.answer, self.key)

    def is_halted(self) -> bool:
        return (
            (self.step_n > self.max_steps)
            or (
                self.enc(
                    self._build_agent_prompt(),
                    model_id=self.react_llm_model_id,
                    upper_limit=self.context_length,
                )
                > self.context_length
            )
        ) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.answer = ""
        self.scratchpad: str = ""
        self._context_overflow_detected = True
        self.end_state = ""
        self.json_log = []
        self.completionTokens = 0
        self.promptTokens = 0
        self.llmCalls = 0

    def set_qa(self, question: str, key: str) -> None:
        self.question = question
        self.key = key


class ReactReflectAgent(ReactAgent):
    def __init__(
        self,
        question: str,
        key: str,
        max_steps: int = 6,
        agent_prompt: PromptTemplate = react_reflect_agent_prompt,
        reflect_prompt: PromptTemplate = reflect_prompt,
        cbm_tools: list = [],
        tool_names: list = [],
        tool_desc: str = "",
        react_llm=watsonx_llm,
        reflect_llm=watsonx_llm,
        react_llm_model_id=0,
        review_llm_model_id=None,
        reflect_llm_model_id=0,
        debug=False,
        actionstyle=ActionStyle.SINGLE_LINE_TOOL_CALL,
        react_example: str = MPE_SIMPLE4,
        reflect_example: str = MPE_REFLECTIONS,
        handle_context_length_overflow=False,
        num_reflect_iteration=3,
        log_structured_messages=False,
        apply_chat_template=False,
        apply_loop_detection_check=False,
        max_retries: int = 1,  # support will be enabled as time goes
        early_stop=False,
        apply_adaptive_parameter_adjustment=False,
        parameter_configuration=None,
        use_tool_cache=False,
        enable_tool_partial_match=False,
        reactstyle=ReActStyle.ThoughtThenAct,
        enable_meta_planning=False,
        enable_question_assement=False,
        enable_agent_ask=False,
        max_execution_time=-1,
        skip_token_counting=False,
    ) -> None:

        super().__init__(
            question,
            key,
            max_steps,
            agent_prompt,
            react_llm_model_id=react_llm_model_id,
            cbm_tools=cbm_tools,
            tool_desc=tool_desc,
            debug=debug,
            actionstyle=actionstyle,
            react_example=react_example,
            handle_context_length_overflow=handle_context_length_overflow,
            log_structured_messages=log_structured_messages,
            apply_chat_template=apply_chat_template,
            apply_loop_detection_check=apply_loop_detection_check,
            max_retries=max_retries,  # support will be enabled as time goes
            llm=react_llm,
            tool_names=tool_names,
            early_stop=early_stop,
            use_tool_cache=use_tool_cache,
            enable_tool_partial_match=enable_tool_partial_match,
            reactstyle=reactstyle,
            enable_meta_planning=enable_meta_planning,
            enable_question_assement=enable_question_assement,
            enable_agent_ask=enable_agent_ask,
            max_execution_time=max_execution_time,
            skip_token_counting=skip_token_counting,
        )
        self.reflect_llm = reflect_llm
        self.reflect_prompt = reflect_prompt
        self.reflect_examples = MPE_REFLECTIONS
        self.reflections: List[str] = []
        self.reflections_str: str = ""
        self.reflect_llm_model_id = reflect_llm_model_id
        self.reflect_examples = reflect_example
        self.reviews: List[str] = []
        self.reviews_str: str = ""
        self.num_reflect_iteration = num_reflect_iteration
        self.review_llm_model_id = review_llm_model_id
        if review_llm_model_id:
            self.review_llm_model_id = review_llm_model_id
        else:
            self.review_llm_model_id = react_llm_model_id
        self.reviewagt = ReviewerAgent(model_id=self.review_llm_model_id)
        self.apply_adaptive_parameter_adjustment = apply_adaptive_parameter_adjustment
        self.parameter_configuration = parameter_configuration
        print("************ReactReflectAgent************")
        print(react_example)
        print("*****************************************")

    def export_trajectory(self):
        traj = super().export_trajectory()
        traj["reviews"] = self.reviews
        traj["reflections"] = self.reflections
        return traj

    def _adjust_ReAct_Agent_Parameter_Configuration(self, current_reflexion_round):
        if self.debug:
            print(
                f"{Fore.LIGHTGREEN_EX} {Style.BRIGHT} Updating Parameter {self.parameter_configuration['temperature'][current_reflexion_round]} in {current_reflexion_round} {Style.RESET_ALL}"
            )

        if self.parameter_configuration:
            if self.parameter_configuration["temperature"]:
                if (
                    self.parameter_configuration["temperature"][current_reflexion_round]
                    != 0
                ):
                    self.llm = partial(
                        self.llm,
                        temperature=self.parameter_configuration["temperature"][
                            current_reflexion_round
                        ],
                    )

    def run(
        self,
        reset=True,
        reflect_strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION,
    ) -> None:
        review = None

        startTime = datetime.now()
        self.total_reflection_round = 0
        self.per_round_info = []

        print(
            f"{Style.BRIGHT}{Fore.MAGENTA}Agent is Enabled with Reflexion{Style.RESET_ALL}"
        )

        skip_review = False
        start_time = time.time()
        for i in range(self.num_reflect_iteration):

            if self.apply_adaptive_parameter_adjustment:
                self._adjust_ReAct_Agent_Parameter_Configuration(i)

            if self.debug:
                # Descriptive text with corresponding colors
                print(
                    f"{Fore.RED} {Style.BRIGHT} Task Execution Status (Finished): {str(self.is_finished())} {Style.RESET_ALL}"
                )
            if i > 0:
                if not self.is_finished():
                    # this is for a case where system run till final step (exhusted)
                    self.reflect(reflect_strategy)
                    self.total_reflection_round += 1
                else:
                    # print("----calling review----")
                    # this is a case where system tells I am done but we review the work
                    review = self.reviewagt.evaluate_response(
                        question=self.question,
                        agent_think=self.scratchpad,
                        agent_response=self.answer,
                    )

                    print(
                        f"{Fore.LIGHTGREEN_EX} {Style.BRIGHT} Review Agent Feedback: {review} {Style.RESET_ALL}"
                    )

                    if "Not" in review["status"] or "Partially" in review["status"]:
                        status = review.get("status", "Unknown Status")
                        reasoning = review.get("reasoning", "No reasoning provided.")
                        suggestions = review.get(
                            "suggestions", "No suggestions provided."
                        )
                        # Build the reflection string
                        review_str = f"Task Status: {status}\n"
                        review_str += f"Reasoning: {reasoning}\n"
                        review_str += f"Suggestions for Improvement: {suggestions}\n"
                        self.reviews.append(review_str)
                        self.reviews_str = format_review([review_str], REVIEW_HEADER)
                        self.reflect(reflect_strategy)
                        self.total_reflection_round += 1
                    else:
                        status = review.get("status", "Unknown Status")
                        reasoning = review.get("reasoning", "No reasoning provided.")
                        suggestions = review.get(
                            "suggestions", "No suggestions provided."
                        )
                        # Build the reflection string
                        review_str = f"Task Status: {status}\n"
                        review_str += f"Reasoning: {reasoning}\n"
                        review_str += f"Suggestions for Improvement: {suggestions}\n"
                        self.reviews.append(review_str)
                        skip_review = True
                        break

            # call parent class run method
            ReactAgent.run(self, reset)
            self.per_round_info.append(self.get_experiment_summary())

            # Check the total execution time
            if self.max_execution_time != -1 and time.time() - start_time > self.max_execution_time:
                print(
                    f"{Style.BRIGHT}{Fore.MAGENTA}We cross the total alloted execution time{Style.RESET_ALL}"
                )
                break

        if not skip_review:
            # final pass
            review = self.reviewagt.evaluate_response(
                question=self.question,
                agent_think=self.scratchpad,
                agent_response=self.answer,
            )

            print(
                f"{Fore.LIGHTGREEN_EX} {Style.BRIGHT} Review Agent: {review} {Style.RESET_ALL}"
            )

            status = review.get("status", "Unknown Status")
            reasoning = review.get("reasoning", "No reasoning provided.")
            suggestions = review.get("suggestions", "No suggestions provided.")
            # Build the reflection string
            review_str = f"Task Status: {status}\n"
            review_str += f"Reasoning: {reasoning}\n"
            review_str += f"Suggestions for Improvement: {suggestions}\n"
            self.reviews.append(review_str)
            self.reviews_str = format_review([review_str], REVIEW_HEADER)

        endTime = datetime.now()
        runTime = endTime - startTime
        runMinutes = runTime.total_seconds() / 60.0
        self.total_execution_time = runMinutes

        self.final_review = review

        self.metric = self.export_benchmark_metric()
        self.trajectory = self.export_trajectory()

        return review

    def reflect(self, strategy: ReflexionStrategy) -> None:

        if strategy == ReflexionStrategy.LAST_ATTEMPT:
            self.reflections = [self.scratchpad]
            self.reflections_str = format_last_attempt(
                self.question, self.reflections[0]
            )
        elif strategy == ReflexionStrategy.REFLEXION:
            self.reflections += [self.prompt_reflection()]

            print(
                f"{Fore.BLUE} {Style.BRIGHT} Reflect Agent Feedback : {self.reflections[-1]} {Style.RESET_ALL}"
            )

            self.reflections_str = format_reflections(self.reflections)
            # combine review and reflexion
            if len(self.reviews_str) > 0:
                self.reflections_str = self.reviews_str + "\n\n" + self.reflections_str
                # print(self.reviews_str)
        elif strategy == ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION:
            self.reflections_str = format_last_attempt(
                self.question, self.scratchpad, model_id=self.reflect_llm_model_id
            )
            self.reflections = [self.prompt_reflection()]
            self.reflections_str += format_reflections(
                self.reflections, header=REFLECTION_AFTER_LAST_TRIAL_HEADER
            )
        else:
            raise NotImplementedError(f"Unknown reflection strategy: {strategy}")
        """
        print(
            f"{Style.BRIGHT}{Fore.LIGHTRED_EX}Reflexion: {self.reflections_str}{Style.RESET_ALL}"
        )
        """

    def prompt_agent(self, stop, prefix) -> str:
        if isinstance(stop, str):
            stop = [stop]
        llmResult = self.llm(
            self._build_agent_prompt(),
            model_id=self.react_llm_model_id,
            stop=stop + ["\nQuestion"],
        )

        self.completionTokens += llmResult["generated_token_count"]
        self.promptTokens += llmResult["input_token_count"]
        self.llmCalls += 1

        return format_step(
            llmResult["generated_text"],
            stop,
            prefix,
            debug=self.debug,
            action_style=self.actionstyle,
            react_style=self.reactstyle,
        )

    def prompt_reflection(self) -> str:
        llmResult = self.reflect_llm(
            self._build_reflection_prompt(),
            model_id=self.reflect_llm_model_id,
            stop=["Previous Trial:", "Previous Trial", "\n\n"],
        )

        self.completionTokens += llmResult["generated_token_count"]
        self.promptTokens += llmResult["input_token_count"]
        self.llmCalls += 1

        return format_step(
            llmResult["generated_text"],
            stop=["Previous Trial:", "Previous Trial"],
            prefix="Reflection:",
            debug=self.debug,
            action_style=self.actionstyle,
            react_style="Reflection:",
        )

    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
            examples=self.reflect_examples,
            question=self.question,
            scratchpad=truncate_scratchpad(
                self.scratchpad,
                tokenizer=self.enc,
                model_id=self.reflect_llm_model_id,
                n_tokens=self.context_length,
            ),
        )

    def _get_system_prompt(self) -> str:
        return self.agent_prompt.format(
            examples="",
            reflections=self.reflections_str,
            question="",
            scratchpad="",
            tool_desc=self.tool_desc,
            tool_names=", ".join(self.tool_names),
        )

    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
            examples=self.react_examples,
            reflections=self.reflections_str,
            question=self.question,
            scratchpad=self.scratchpad,
            tool_desc=self.tool_desc,
            tool_names=", ".join(self.tool_names),
        )

    def count_questions(self, text):

        # Initialize the counter
        count = 0

        if text and len(text) > 0:
            # Split the text into lines
            lines = text.splitlines()

            # Iterate over each line and check if it starts with "Question"
            for line in lines:
                if line.strip().startswith("Question"):
                    count += 1

        return count

    def export_benchmark_metric(self):
        """
        This method will call and export benchmark metrics to a JSON file.
        Metrics include:
        - number of reflections
        - total execution time
        - steps per round (as an array)
        - configuration parameters like # of shots in reflect, review, react
        """
        # Define the metrics
        metrics = {
            "questions": self.question,
            "number_of_reflections": self.total_reflection_round,
            "total_execution_time": self.total_execution_time,  # in minutes
            "per_round_info": self.per_round_info,  # This is an array now
            "configuration_parameters": {
                "shots_in_reflect": self.count_questions(self.reflect_examples),
                "reflect_llm_model_id": self.reflect_llm_model_id,
                "react_llm_model_id": self.react_llm_model_id,
                "shots_in_react": self.count_questions(self.react_examples),
            },
            "status": self.final_review.get("status", "Unknown Status"),
            "max_steps": self.max_steps,
            "num_reflect_iteration": self.num_reflect_iteration,
            "max_retries": self.max_retries,
            "num_tools": len(self.tool_names),
        }

        return metrics
        # print("Benchmark metrics have been exported.")
