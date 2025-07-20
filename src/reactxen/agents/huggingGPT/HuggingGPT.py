from typing import List
import json
from langchain.tools.base import BaseTool
from reactxen.agents.huggingGPT.utils import (
    prepare_task_planning_prompt,
    load_response_generator,
)
from reactxen.agents.huggingGPT.utils import cTaskExecutor
from reactxen.agents.huggingGPT.prompts.task_template import final_system_prompt
from reactxen.agents.huggingGPT.prompts.task_planning_demonstration_iot import (
    DEMONSTRATIONS,
)
from reactxen.utils.model_inference import (
    watsonx_llm_chat,
)
from reactxen.utils.tool_description import get_tool_description, get_tool_names
from colorama import Fore, Style


class HuggingGPTCBMAgent:
    """Agent for interacting with Maximo CBM."""

    def __init__(
        self,
        question: str,
        key: str,
        max_iter: int = 6,
        agent_prompt: str = final_system_prompt,
        tools: List[BaseTool] = [],
        tool_names: List[str] = [],
        tool_desc: str = "",
        llm_model_id: int = 0,
        examples: str = DEMONSTRATIONS,
        llm=watsonx_llm_chat,
        debug: bool = False,
        log_structured_messages: bool = False,
    ) -> None:

        self.question = question
        self.key = key
        self.agent_prompt = agent_prompt
        self.examples = examples
        self.max_iter = max_iter
        self.tools = tools
        self.tool_desc = tool_desc
        self.tool_names = tool_names
        self.llm = llm
        self.llm_model_id = llm_model_id
        self.run_status = None
        if not self.tool_names:
            self.tool_names = get_tool_names(self.tools)
        if not self.tool_desc:
            self.tool_desc = get_tool_description(self.tools)

        self.debug = debug
        self.log_structured_messages = log_structured_messages
        self.chat_planner = prepare_task_planning_prompt(
            model_id=llm_model_id,
            llm_call=self.llm,
            system_template=agent_prompt,
            demos=examples,
        )
        self.response_generator = load_response_generator(
            model_id=llm_model_id, llm_call=self.llm
        )
        self.task_executor: cTaskExecutor
        self.agent_name = "MPE"

    def update_chat_planner(self, last_agent_trial, last_agent_response):
        self.chat_planner = prepare_task_planning_prompt(
            model_id=self.llm_model_id,
            llm_call=self.llm,
            system_template=self.agent_prompt,
            demos=self.examples,
            last_agent_trial=last_agent_trial,
            last_agent_response=last_agent_response,
        )

    def run(self) -> str:
        """Turn this into iteration"""
        """Plan, Execute, Report"""
        trace_plans = []
        trace_responses = []
        self.run_status = None

        print(
            f"{Style.BRIGHT}{Fore.MAGENTA}I am {self.agent_name} Agent with Plan-Execute{Style.RESET_ALL}"
        )
        print(
            f"{Style.BRIGHT}{Fore.LIGHTMAGENTA_EX}Input Question: {self.question}{Style.RESET_ALL}"
        )

        for _ in range(self.max_iter):

            if len(trace_plans) > 0:
                self.update_chat_planner(trace_plans[-1], trace_responses[-1])

            # Generate Plan
            plan = self.chat_planner.plan(
                inputs={
                    "input": self.question,
                    "tool_desc": self.tool_desc,
                    "hf_tools": self.tools,
                }
            )

            # Print Plan
            print(Fore.YELLOW + "-" * 50 + Style.RESET_ALL)
            print(Fore.GREEN + "Model Produced following plan:")
            print(plan)
            print(Style.RESET_ALL)
            print(Fore.YELLOW + "*" * 50 + Style.RESET_ALL)

            # Create an executor
            try:
                self.task_executor = cTaskExecutor(plan)
            except Exception as e:
                print("No plan generated due to error: ", e)
                return "failed"

            # this place we store the plans
            plans = []
            for item in self.task_executor.plan.steps:
                params_str = ", ".join(
                    f"{key}={json.dumps(value)}" for key, value in item.args.items()
                )
                complete_func_call = f"{item.task}({params_str})"
                plans.append(complete_func_call)

            trace_plans.append(
                "\n".join(f"{i + 1}. {string}" for i, string in enumerate(plans))
            )

            # We should pass now additional argument such as message passing to get the
            run_status = self.task_executor.run()
            self.run_status = run_status
            print(Fore.BLUE + "Execution Produced following result:")
            print(run_status)
            print(Style.RESET_ALL)
            print(Fore.YELLOW + "*" * 50 + Style.RESET_ALL)

            response = self.response_generator.generate(
                {"task_execution": self.task_executor}
            )
            print(Fore.CYAN + "Final Response:")
            print(response)
            print(Style.RESET_ALL)
            print(Fore.YELLOW + "*" * 50 + Style.RESET_ALL)

            trace_responses.append(response)

            if run_status == "completed":
                return response
            else:
                trace_responses.append(
                    response + "\n\n" + run_status.replace("{", "").replace("}", "")
                )
        # final answer
        return "failed"
