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
from reactxen.agents.react.utils import format_step_for_thinkact_together
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
from reactxen.agents.rafa.utility import *
from reactxen.agents.rafa.prompts.systemprompts import (
    thinkact_system_agent_prompt,
    observation_system_agent_prompt,
    critic_system_agent_prompt,
)
from reactxen.agents.reviewer_agent.agent import ReviewerAgent

# the HyperParameterCheck become important
# The observation is correct.


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


class RAFAAgent:
    def __init__(
        self,
        question: str,
        key: str,
        max_steps: int = 6,
        agent_prompt: PromptTemplate = react_agent_prompt,
        thinkact_system_agent_prompt: PromptTemplate = thinkact_system_agent_prompt,
        observation_system_agent_prompt: PromptTemplate = observation_system_agent_prompt,
        critic_system_agent_prompt: PromptTemplate = critic_system_agent_prompt,
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
        self.critic_system_agent_prompt = critic_system_agent_prompt
        self.thinkact_system_agent_prompt = thinkact_system_agent_prompt
        self.observation_system_agent_prompt = observation_system_agent_prompt

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

        self.__reset_agent()

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
        except:
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

    def run(
        self, reset=True, name="RAFA Agent", k=5, search_width=2, search_depth=1
    ) -> None:

        if reset:
            self.__reset_agent()

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

        total_mem_updates = 0
        while not self.is_halted() and not self.is_finished() and total_mem_updates < k:
            self.step(search_width, search_depth)

        print(f"{Style.BRIGHT}{Fore.MAGENTA}Process is completed now{Style.RESET_ALL}")
        self.reviewagt = ReviewerAgent(model_id=self.react_llm_model_id)
        self.review_str = self.reviewagt.evaluate_response(
            question=self.question,
            agent_think=self.scratchpad,
            agent_response=self.answer,
        )
        #print (self.review_str)

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

    def generate_future_thinkact(
        self, cur_step, future_step, current_scratchpad, bandwidth
    ):
        """
        Generates multiple future think-act plans based on the current scratchpad.

        Args:
        - current_scratchpad (dict): The current state or trajectory (could be a dictionary or other object).
        - bandwidth (int): The number of future plans to generate at this step.

        Returns:
        - plans (list): List of generated future plans based on the current scratchpad.
        """
        plans = []
        for i in range(bandwidth):
            # print(">>>>>>------ calling think and act...")
            if not current_scratchpad.endswith("\nThought"):
                new_scratchpad = (
                    current_scratchpad + f"\nThought {cur_step + future_step}:"
                )
            else:
                new_scratchpad = current_scratchpad + f" {cur_step + future_step}:"
            ans = self.prompt_think_act_action(new_scratchpad)
            new_scratchpad += ans
            plans.append(new_scratchpad)
            # print(new_scratchpad)
            # print("                    -------------")
        return plans

    def generate_future_observation(self, current_scratchpad):
        """
        Generates an observation from the current scratchpad.
        For now, it simply returns the scratchpad as is, but this can include observations
        based on environment feedback.

        Args:
        - current_scratchpad (dict): The current state or trajectory.

        Returns:
        - observation (dict): The current state as an observation.
        """
        ans = self.prompt_observation_action(current_scratchpad)
        new_scratchpad = current_scratchpad + ans
        return new_scratchpad  # Placeholder for future observation logic

    def generate_best_action(
        self, recent_observation, cur_step, current_scratchpad, bandwidth, depth
    ):
        """
        Generates the best action by exploring future plans at each step, considering the given depth.

        Args:
        - current_scratchpad (dict): The current state or trajectory.
        - bandwidth (int): The number of candidate plans to explore at each step.
        - depth (int): The depth of planning (how many steps into the future to consider).

        Returns:
        - best_action (dict): The best action from the final evaluated plans.
        """
        if len(recent_observation) > 0:
            current_scratchpad += f"\nOnservation {cur_step}: {recent_observation}"
        plans = [
            current_scratchpad
        ]  # Start with the initial state (current scratchpad)
        all_plans = []

        # Iteratively generate future plans
        for i in range(depth):
            plan_in_step = []  # List of plans generated in the current step

            for p in plans:
                # Generate future think-act sequences for each plan
                _thinkact = self.generate_future_thinkact(cur_step, i, p, bandwidth)
                _thingactobserves = []
                for _onethinkact in _thinkact:
                    obs = ""
                    if (
                        "Final Answer:" not in _onethinkact
                        and ": Finish" not in _onethinkact
                    ):
                        if not _onethinkact.endswith("\nObservation"):
                            _onethinkact += f"\nObservation {cur_step + i}: "
                        else:
                            _onethinkact += f" {cur_step + i}"
                        obs = self.generate_future_observation(_onethinkact).strip()
                    else:
                        obs = _onethinkact
                    if obs.endswith("Question"):
                        obs = obs[: -len("Question")].strip()

                    # print("----")
                    # print(obs)
                    # print("----")

                    _thingactobserves.append(obs)
                # format_step_for_thinkact_together(cc)
                plan_in_step.extend(
                    _thingactobserves
                )  # Add new plans to the current step

            # After considering all plans at the current depth, update the plan list
            plans = []
            if i == depth - 1:
                all_plans.extend(plan_in_step)
            else:
                for plan in plan_in_step:
                    if "Final Answer:" not in plan:
                        plans.append(plan)
                    else:
                        all_plans.append(plan)

        # Evaluate the plans (In this case, we just return the first plan, but you can add a selection criterion)
        # For simplicity, let's assume the first plan is the best (you can expand this with evaluation logic)
        # print("started")
        all_ans = []
        for plan in all_plans:
            pp = self.prompt_critical_value_action(current_scratchpad, plan)
            ans = extract_and_parse_json(pp)
            ans["plan"] = plan
            all_ans.append(ans)

        # Sort the list of dictionaries by 'value_function_score' in descending order
        sorted_plans = sorted(
            all_ans, key=lambda x: x["value_function_score"], reverse=True
        )

        best_plan = sorted_plans[0]["plan"]
        current_scratchpad = current_scratchpad.strip()
        best_plan = best_plan.strip()
        # print('+++++++++++++')
        # print(best_plan)
        # print('-------------')
        # print(current_scratchpad)
        # print('~~~~~~~~~~~~~')
        if best_plan.startswith(current_scratchpad):
            best_plan = best_plan[len(current_scratchpad) :]
        thought_match = re.search(r"Thought \d+: (.+?)(?=\n|$)", best_plan)
        action_match = re.search(r"Action \d+: (.+?)(?=\n|$)", best_plan)
        action_input_match = re.search(r"Action Input \d+: (.+?)(?=\n|$)", best_plan)
        observation_match = re.search(r"Observation \d+: (.+?)(?=\n|$)", best_plan)

        # Extract the values and store them in a dictionary
        extracted_data = {}
        extracted_data["thought"] = ""
        extracted_data["action"] = ""
        extracted_data["action_input"] = "{}"
        extracted_data["expected_observation"] = ""

        if thought_match:
            extracted_data["thought"] = thought_match.group(1).strip()
        if action_match:
            extracted_data["action"] = action_match.group(1).strip()
        if action_input_match:
            extracted_data["action_input"] = action_input_match.group(1).strip()
        if observation_match:
            extracted_data["expected_observation"] = observation_match.group(1).strip()

        extracted_data["error"] = False
        extracted_data["error_feedback"] = None
        extracted_data["llm_output"] = best_plan

        # print(extracted_data)

        return extracted_data

    def reasoning_learning_step(
        self, recent_observation, cur_step, search_width=3, search_depth=3
    ):
        # print("calling reasoning model...")
        ans = self.generate_best_action(
            recent_observation,
            cur_step,
            self.global_scratchpad,
            search_width,
            search_depth,
        )
        return ans

    def step(self, search_width=1, search_depth=1) -> None:

        current_t = self.step_n
        recent_observation = ""
        buffer_limit = 4
        current_limit = 0

        while current_limit < buffer_limit:
            current_limit += 1
            let_me_think_dict = self.reasoning_learning_step(
                recent_observation, current_t, search_width, search_depth
            )
            if self.debug:
                print(
                    f"{Style.BRIGHT}{Fore.RED}Debug Info (Step {self.step_n}):{Style.RESET_ALL}"
                )
                print(
                    f"{Fore.RED}{json.dumps(let_me_think_dict, indent=4)}{Style.RESET_ALL}"
                )

            action_dict = let_me_think_dict
            let_me_think = action_dict["thought"]
            self.scratchpad += f"\nThought {self.step_n}:"
            self.scratchpad += " " + action_dict["thought"]
            self.scratchpad += f"\nAction {self.step_n}:"
            print(
                f"{Style.BRIGHT}{Fore.BLUE}Thought {self.step_n}: {let_me_think}{Style.RESET_ALL}"
            )

            tool_execution_success = False
            if self.actionstyle == ActionStyle.SINGLE_LINE_TOOL_CALL:
                if action_dict["action"] and action_dict["action_input"]:
                    self.scratchpad += " " + action_dict["action"]
                    self.scratchpad += f"\nAction Input {self.step_n}:"
                    self.scratchpad += " " + action_dict["action_input"]
                else:
                    error_msg = ''
                    if action_dict["action"].lower() == "Finish".lower():
                        self.scratchpad += " " + action_dict["action"]
                        self.scratchpad += f"\nAction Input {self.step_n}:"
                        self.scratchpad += " " + action_dict["action_input"]
                        error_msg = f"Task marked as finished, but no final answer was provided in the action-input. The Finish action requires the final answer to complete the task. Please provide the answer in the action-input field before marking the task as finished. The final answer should be a clear response."
                        self.scratchpad += f"\nObservation {self.step_n}: Task marked as finished, but no final answer was provided in the action-input. The Finish action requires the final answer to complete the task. Please provide the answer in the action-input field before marking the task as finished. The final answer should be a clear response."
                    elif (
                        action_dict["action"] and len(action_dict["action_input"]) == 0
                    ):
                        self.scratchpad += " " + action_dict["action"]
                        self.scratchpad += f"\nAction Input {self.step_n}:"
                        self.scratchpad += " " + action_dict["action_input"]
                        self.scratchpad += f"\nObservation {self.step_n}: You did not provide the action_input"
                        error_msg = f"You did not provide the action_input"
                    else:
                        self.scratchpad += " " + action_dict["action"]
                        self.scratchpad += f"\nAction Input {self.step_n}:"
                        self.scratchpad += " " + action_dict["action_input"]
                        error_msg = f"You did not provide the action and action_input"
                        self.scratchpad += f"\nObservation {self.step_n}: You did not provide the action and action_input"
                    print(
                        f"{Style.BRIGHT}{Fore.YELLOW}Action {self.step_n}: {action_dict['action']} {Style.RESET_ALL}"
                    )
                    print(
                        f"{Style.BRIGHT}{Fore.CYAN}Action Input {self.step_n}: {action_dict['action_input']} {Style.RESET_ALL}"
                    )
                    print(
                        f"{Style.BRIGHT}{Fore.CYAN}Observation {self.step_n}: {error_msg} {Style.RESET_ALL}"
                    )
                    break

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
                action_type, argument = (
                    action_dict["action"],
                    action_dict["action_input"],
                )
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
                    action_dict_internal = self.prompt_agent(
                        stop=["\nThought", "\nAction"],
                        prefix=f"\nObservation {self.step_n}:",
                    )
                    if len(action_dict_internal["observation"]) > 0:
                        self.scratchpad += action_dict_internal["observation"]
                        action_execution_output = action_dict_internal["observation"]
                    else:
                        self.scratchpad += (
                            "No relevant output or response was generated."
                        )
                        action_execution_output = (
                            "No relevant output or response was generated."
                        )
                else:
                    self.scratchpad += f"\nObservation {self.step_n}: "
                    # call toolsearch
                    tool_not_find = True
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

                                        # print("trying to call tool...")
                                        try:
                                            tool_output = tool.run(
                                                tool_input=dictionary
                                            )

                                            if isinstance(tool_output, dict):
                                                tool_output = str(tool_output)

                                            action_execution_output = tool_output
                                            self.scratchpad += tool_output
                                            tool_execution_success = True

                                        except Exception as ex:
                                            self.scratchpad += (
                                                f"Error encountered: Execution failed while calling the Tool. Details: {str(ex)}"
                                            ).strip()
                                    except Exception as ex:
                                        self.scratchpad += (
                                            "Error encountered: Issue while unpacking the arguments provided as input to the tool. "
                                            f"Details: {str(ex)}"
                                        ).strip()
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

                                            # Clean up tool output (only if it's a string)
                                            if isinstance(tool_output, str):
                                                tool_output = tool_output.replace(
                                                    "\n", " "
                                                ).strip()

                                            action_execution_output = tool_output
                                            self.scratchpad += tool_output  # Store output in scratchpad
                                            tool_execution_success = True

                                        except TypeError as ex:
                                            error_message = f"Error: Invalid parameters for Tool. Details: {str(ex)}"
                                            self.scratchpad += error_message

                                        except Exception as ex:
                                            error_message = f"Error: Execution failed while calling the Tool. Details: {str(ex)}"
                                            self.scratchpad += error_message

                                    except json.JSONDecodeError as e:
                                        error_message = f"Error: Failed to parse JSON. Details: {str(e)}"
                                        self.scratchpad += error_message

                                else:
                                    # this is for a thread which is for direct passing
                                    try:
                                        tool_output = tool.run(argument)
                                        tool_output = (
                                            tool_output.replace("\n", " ")
                                            .strip("\n")
                                            .strip()
                                        )
                                        action_execution_output = tool_output
                                        self.scratchpad += tool_output
                                        tool_execution_success = True

                                    except Exception as ex:
                                        self.scratchpad += (
                                            f"Error encountered: Execution failed while calling the Tool. Details: {str(ex)}"
                                        ).strip()
                                # we find tool and now good
                                break

                    if tool_not_find:
                        self.scratchpad += (
                            f"Invalid Action detected. The selected action - {action_type} does not match any valid options. "
                            f"Please choose an action from the following list: [{', '.join(self.tool_names)}]."
                        ).strip()

            except Exception as ex:
                # print(ex)
                self.scratchpad += f"Error encountered: Issue while parsing/executing Action. Details: {str(ex)}"

            if action_execution_output:
                print(
                    f"{Style.BRIGHT}{Fore.GREEN}Observation {self.step_n}: {action_execution_output}{Style.RESET_ALL}"
                )
                recent_observation = action_execution_output
            else:
                # message = self.scratchpad.split("\n")[-1]
                observation_prefix = f"\nObservation {self.step_n}: "
                last_observation_index = self.scratchpad.rfind(observation_prefix)
                message = self.scratchpad[last_observation_index:]
                recent_observation = message
                print(f"{Style.BRIGHT}{Fore.GREEN}{message}{Style.RESET_ALL}")

            message = ""
            current_t += 1
            self.step_n += 1

            if (
                action_execution_output == action_dict["expected_observation"]
                and tool_execution_success
            ):
                pass
                # print(recent_observation)
                # pass
            else:
                break

        # its time to update the scratch pad now.
        self.global_scratchpad = self.scratchpad
        # print("revised golbal scratch pad")
        # print(self.global_scratchpad)

    def prompt_think_act_action(self, current_scratchpad) -> str:
        llmResult = self.llm(
            self._build_think_act_action_prompt(current_scratchpad),
            model_id=self.react_llm_model_id,
            stop=[
                "\nObservation",
                "\n\nObservation",
                "\nThought",
                "\n\nThought",
                "\nQuestion",
                "\n\nQuestion",
            ],
            temperature=0.7,
        )
        return llmResult["generated_text"]

    def _build_think_act_action_prompt(self, current_scratchpad) -> str:

        return self.thinkact_system_agent_prompt.format(
            examples=self.react_examples,
            question=self.question,
            scratchpad=current_scratchpad,
            tool_desc=self.tool_desc,
            tool_names=", ".join(self.tool_names),
        )

    def prompt_observation_action(self, current_scratchpad) -> str:
        llmResult = self.llm(
            self._build_observation_action_prompt(current_scratchpad),
            model_id=self.react_llm_model_id,
            stop=[
                "\nObservation",
                "\n\nObservation",
                "\nThought",
                "\n\nThought",
                "\nQuestion",
                "\n\nQuestion",
            ],
        )
        return llmResult["generated_text"]

    def _build_observation_action_prompt(self, current_scratchpad) -> str:

        return self.observation_system_agent_prompt.format(
            examples=self.react_examples,
            question=self.question,
            scratchpad=current_scratchpad,
            tool_desc=self.tool_desc,
            tool_names=", ".join(self.tool_names),
        )

    def prompt_critical_value_action(
        self, current_scratchpad, agent_generated_plan
    ) -> str:
        llmResult = self.llm(
            self._build_critical_value_action_prompt(
                current_scratchpad, agent_generated_plan
            ),
            model_id=self.react_llm_model_id,
            stop=["\n(END OF RESPONSE)", "(END OF RESPONSE)"],
            max_tokens=2000,
        )
        return llmResult["generated_text"]

    def _build_critical_value_action_prompt(
        self, current_scratchpad, agent_generated_plan
    ) -> str:
        return self.critic_system_agent_prompt.format(
            examples=self.react_examples,
            question=self.question,
            scratchpad=current_scratchpad,
            tool_desc=self.tool_desc,
            agent_generated_plan=agent_generated_plan,
        )

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
            examples="",
            question="",
            scratchpad="",
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
        self.global_scratchpad: str = ""
        self._context_overflow_detected = True
        self.end_state = ""
        self.json_log = []
        self.completionTokens = 0
        self.promptTokens = 0
        self.llmCalls = 0

    def set_qa(self, question: str, key: str) -> None:
        self.question = question
        self.key = key

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
            "status": self.review_str.get("status", "Unknown Status"),
            "max_steps": self.max_steps,
            "max_retries": self.max_retries,
            "num_tools": len(self.tool_names),
        }

        return metrics
        # print("Benchmark metrics have been exported.")
