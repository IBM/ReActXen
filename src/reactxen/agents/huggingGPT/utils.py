## The part of the code is motivated from original paper and then langchain implementation
## We added a reflexion part when the code get error. 

from langchain_core.callbacks.manager import Callbacks
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from cbm_gen.agents.huggingGPT.prompts.task_planning_demonstrate import DEMONSTRATIONS
from typing import Optional, List, Dict, Callable
import copy
from reactxen.agents.huggingGPT.prompts.task_template import (
    modified_system_prompt_template,
)
from typing import Dict, List
import json
from langchain.tools.base import BaseTool
from typing import List
import re


def get_chat_message(system_prompt, demonstrations):
    c_messages = []
    c_messages.append({"content": system_prompt, "role": "system"})
    for inputputput in demonstrations:
        c_messages.append({"content": inputputput["Question"], "role": "user"})
        c_messages.append({"content": inputputput["Answer"], "role": "assistant"})
    return c_messages


class Step:
    """A step in the plan."""

    def __init__(
        self, task: str, id: int, dep: List[int], args: Dict[str, str], tool: BaseTool
    ):
        self.task = task
        self.id = id
        self.dep = dep
        self.args = args
        self.tool = tool

    def __str__(self) -> str:
        # Formatting the step's details in a more readable way
        dep_str = ", ".join(map(str, self.dep)) if self.dep else "None"
        args_str = ", ".join(f"{key}: {value}" for key, value in self.args.items())
        return (
            f"Step {self.id}: Task = '{self.task}'\n"
            f"  Dependencies: {dep_str}\n"
            f"  Arguments: {args_str}\n"
            f"  Tool: {self.tool.name if self.tool else 'None'}"
        )

    def __repr__(self) -> str:
        return f"Step(id={self.id}, task={self.task}, dep={self.dep}, args={self.args}, tool={self.tool.name if self.tool else 'None'})"


class Plan:
    """A plan to execute."""

    def __init__(self, steps: List[Step]):
        self.steps = steps

    def __str__(self) -> str:
        # Presenting each step of the plan in a more readable format
        steps_str = "\n\n".join(str(step) for step in self.steps)
        return f"Plan:\n{steps_str}"

    def __repr__(self) -> str:
        # Representing the plan as a string (useful for debugging)
        return f"Plan(steps={len(self.steps)} steps)"


class PlanningOutputParser(BaseModel):
    """Parses the output of the planning stage."""

    def parse(self, text: str, hf_tools: List[BaseTool]) -> Plan:
        """Parse the output of the planning stage.

        Args:
            text: The output of the planning stage.
            hf_tools: The tools available.

        Returns:
            The plan or an error message if parsing fails.
        """
        steps = []
        # print(text)
        text = text.replace("\n", "").replace("\r", "").replace("\\", "")
        text = auto_correct_json(
            text
        )  # Assuming this is a helper function for cleaning JSON.
        # print("text", text)
        # Step 1: Clean the input by replacing the double curly braces with single curly braces
        cleaned_text = re.sub(r"\{(\{)", "{", text)  # Replace '{{' with '{'
        text = re.sub(r"(\})\}", "}", cleaned_text)  # Replace '}}' with '}'

        try:
            for v in json.loads(re.findall(r"\[.*\]", text)[0]):
                # print("each step:", v)
                choose_tool = None
                for tool in hf_tools:
                    if tool.name == v["task"]:
                        choose_tool = tool
                        break
                if choose_tool:
                    steps.append(Step(v["task"], v["id"], v["dep"], v["args"], tool))
            return Plan(steps=steps)
        except Exception as e:
            # Return the error message as part of a structured response.
            return f"Plan parsing error: {str(e)} - {text}"


class TaskPlanner:
    """Planner for tasks."""

    def __init__(
        self,
        prompt_llm: Callable,  # Assuming prompt_llm is callable
        task_planning_prompt: List[Dict],
        llm_model_id: int,
        output_parser: PlanningOutputParser,
        stop: Optional[List] = None,
    ) -> None:
        self.prompt_llm = prompt_llm
        self.task_planning_prompt = task_planning_prompt
        self.llm_model_id = llm_model_id
        self.output_parser = output_parser
        self.stop = stop

    def plan(self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any) -> Plan:

        # print (self.task_planning_prompt)
        # tools = [f"{tool.name}: {tool.description}" for tool in inputs["hf_tools"]]
        # tools_value = "[" + ", ".join(f'"{tool}"' for tool in tools) + "]"
        tools_value = inputs["tool_desc"]
        ans = self.task_planning_prompt[0]["content"].replace("{tools}", tools_value)
        self.task_planning_prompt[0]["content"] = ans
        for item_index, item in enumerate(self.task_planning_prompt):
            if "{input}" in item["content"]:
                ans = item["content"].replace("{input}", inputs["input"])
                self.task_planning_prompt[item_index]["content"] = ans
        # print (type(self.prompt_llm))
        llm_response = self.prompt_llm(
            prompt=self.task_planning_prompt, model_id=self.llm_model_id, stop=self.stop
        )
        return self.output_parser.parse(llm_response, inputs["hf_tools"])

    async def aplan(
        self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any
    ) -> Plan:
        """Asynchronous Given input, decided what to do."""
        inputs["hf_tools"] = [
            f"{tool.name}: {tool.description}" for tool in inputs["hf_tools"]
        ]
        llm_response = await self.llm_chain.arun(
            **inputs, stop=self.stop, callbacks=callbacks
        )
        return self.output_parser.parse(llm_response, inputs["hf_tools"])


def auto_correct_json(json_string):
    # Try to load the JSON string
    try:
        data = json.loads(json_string)
        return json_string  # JSON is already valid
    except json.JSONDecodeError as e:
        # print("Initial JSONDecodeError:", e)
        pass

    # Fix common issues

    # Fix missing commas between JSON objects
    json_string = re.sub(r"}\s*{", "},{", json_string)

    # Fix unclosed strings (if any)
    json_string = re.sub(r'(?<!\\)"\s*:', '":', json_string)
    json_string = re.sub(r':\s*(?<!\\)"', ': "', json_string)

    # Fix unclosed braces or brackets
    open_braces = json_string.count("{")
    close_braces = json_string.count("}")
    open_brackets = json_string.count("[")
    close_brackets = json_string.count("]")

    if open_braces > close_braces:
        json_string += "}" * (open_braces - close_braces)
    if close_braces > open_braces:
        json_string = "{" * (close_braces - open_braces) + json_string

    if open_brackets > close_brackets:
        json_string += "]" * (open_brackets - close_brackets)
    if close_brackets > open_brackets:
        json_string = "[" * (close_brackets - open_brackets) + json_string

    # Attempt to load the corrected JSON string
    try:
        data = json.loads(json_string)
        return json.dumps(data, indent=4)  # Pretty-print the JSON for readability
    except json.JSONDecodeError as e:
        # print("JSONDecodeError after correction:", e)
        return json_string


def prepare_task_planning_prompt(
    llm_call,
    model_id: int = 8,
    system_template=modified_system_prompt_template,
    demos: List[Dict] = DEMONSTRATIONS,
    last_agent_trial: str = None,
    last_agent_response: str = None,
) -> TaskPlanner:
    """Load the chat planner."""

    task_planning_prompt = get_chat_message(system_template, demos)
    task_planning_prompt.append({"content": "Now I input: {input}.", "role": "user"})
    if last_agent_trial:
        task_planning_prompt.append(
            {
                "role": "assistant",
                "content": "Previous failed trial for {input} was : "
                + last_agent_trial,
            }
        )
        task_planning_prompt.append(
            {
                "role": "user",
                "content": "Please consider following error while regenerating plan for {input} : "
                + last_agent_response,
            }
        )
    # print(task_planning_prompt)

    # Here we directly instantiate the PlanningOutputParser as expected
    output_parser = PlanningOutputParser()

    # Return the TaskPlanner with the llm_chain and output_parser instances
    return TaskPlanner(
        prompt_llm=llm_call,
        llm_model_id=model_id,
        task_planning_prompt=task_planning_prompt,
        output_parser=output_parser,
    )


class Task:
    """Task to be executed."""

    def __init__(self, task: str, id: int, dep: List[int], args: Dict, tool: BaseTool):
        self.task = task
        self.id = id
        self.dep = dep
        self.args = args
        self.tool = tool
        self.status = "pending"
        self.message = ""
        self.result = ""

    def __str__(self) -> str:
        return f"{self.task}({self.args})"

    def save_product(self) -> None:
        pass

    def completed(self) -> bool:
        return self.status == "completed"

    def failed(self) -> bool:
        return self.status == "failed"

    def pending(self) -> bool:
        return self.status == "pending"

    def run(self) -> str:
        try:
            new_args = copy.deepcopy(self.args)
            self.result = self.tool.run(tool_input=new_args)
        except Exception as e:
            self.status = "failed"
            self.message = str(e)
            return self.message

        self.status = "completed"
        self.save_product()

        return self.result


class TaskExecutor:
    """Load tools and execute tasks."""

    def __init__(self, plan: Plan):
        self.plan = plan
        self.tasks = []
        self.id_task_map = {}
        self.status = "pending"
        for step in self.plan.steps:
            task = Task(step.task, step.id, step.dep, step.args, step.tool)
            self.tasks.append(task)
            self.id_task_map[step.id] = task

    def completed(self) -> bool:
        return all(task.completed() for task in self.tasks)

    def failed(self) -> bool:
        return any(task.failed() for task in self.tasks)

    def pending(self) -> bool:
        return any(task.pending() for task in self.tasks)

    def check_dependency(self, task: Task) -> bool:
        for dep_id in task.dep:
            if dep_id == -1:
                continue
            # temporary hack to avode planing mistake
            if dep_id in self.id_task_map:
                dep_task = self.id_task_map[dep_id]
            else:
                continue
            if dep_task.failed() or dep_task.pending():
                return False
        return True

    def update_args(self, task: Task) -> None:
        for dep_id in task.dep:
            if dep_id == -1:
                continue
            dep_task = self.id_task_map[dep_id]
            for k, v in task.args.items():
                if f"<resource-{dep_id}>" in v:
                    task.args[k] = task.args[k].replace(
                        f"<resource-{dep_id}>", dep_task.result
                    )

    def run(self) -> str:
        for task in self.tasks:
            print(f"running {task}")  # noqa: T201
            if task.pending() and self.check_dependency(task):
                self.update_args(task)
                task.run()
        if self.completed():
            self.status = "completed"
        elif self.failed():
            self.status = "failed"
        else:
            self.status = "pending"
        return self.status

    def __str__(self) -> str:
        result = ""
        for task in self.tasks:
            result += f"{task}\n"
            result += f"status: {task.status}\n"
            if task.failed():
                result += f"message: {task.message}\n"
            if task.completed():
                result += f"result: {task.result}\n"
        return result

    def __repr__(self) -> str:
        return self.__str__()

    def describe(self) -> str:
        return self.__str__()


class cTaskExecutor(TaskExecutor):
    """Load tools and execute tasks."""

    def _add_status_message_into_queue(self, task, progress_queue):
        progress_queue.put(
            {
                "id": task.id,
                "task": task.task,
                "result": task.result,
                "message": task.message,
                "task_execution": 1,
            }
        )

    def run(self, progress_queue=None) -> str:
        error_message = ""
        for task in self.tasks:
            # print(f"Running {task}")  # This is useful for debugging

            # Check if the task is pending and meets the dependency check
            if task.pending() and self.check_dependency(task):
                self.update_args(task)
                task.run()

            if progress_queue:
                self._add_status_message_into_queue(task, progress_queue)

            if len(task.message) > 0:
                error_message = f"Error Message from previous trial, where task {task} failed with the following error message: {task.message}. Avoid this mistake next time and generate complete plan."
                break

        if len(error_message) > 0:
            return error_message
        if self.completed():
            self.status = "completed"
        elif self.failed():
            self.status = "failed"
        else:
            self.status = "pending"

        return self.status


from typing import Optional, Callable, List, Any


class ResponseGenerator:
    """Generates a response based on the input."""

    prompt_llm: Optional[Callable] = None
    response_generation_prompt: str
    llm_model_id: int
    stop: Optional[List] = None

    def __init__(
        self,
        response_generation_prompt: str,
        llm_model_id: int,
        prompt_llm: Optional[Callable] = None,
        stop: Optional[List] = None,
    ) -> None:
        """
        Initialize the ResponseGenerator with required and optional parameters.

        :param response_generation_prompt: The prompt template used for generating responses.
        :param llm_model_id: The ID for the LLM model to be used.
        :param prompt_llm: Optional callable function for interacting with an LLM.
        :param stop: Optional list of stopping conditions.
        """
        self.response_generation_prompt = response_generation_prompt
        self.llm_model_id = llm_model_id
        self.prompt_llm = prompt_llm
        self.stop = stop

    def generate(
        self, inputs: dict, callbacks: Optional[Callable] = None, **kwargs: Any
    ) -> str:
        """Given input, decide what to do."""

        llm_response = ""

        if self.prompt_llm:
            prompt = self.response_generation_prompt.replace(
                "{task_execution}", str(inputs["task_execution"])
            )
            prompt = [{"content": prompt, "role": "user"}]
            # Use the provided callable function (e.g., LLM) to generate a response
            llm_response = self.prompt_llm(prompt=prompt, model_id=self.llm_model_id)

        # Handle the stopping conditions or callbacks if necessary
        if self.stop:
            # Add logic to handle stop conditions if needed
            pass

        return llm_response


def prepare_response_generation_prompt():
    execution_template = (
        "The AI assistant has parsed the user input into several tasks"
        "and executed them. The results are as follows:\n"
        "{task_execution}"
        "\nPlease summarize the results and generate a response."
    )
    return execution_template


def load_response_generator(llm_call, model_id) -> ResponseGenerator:

    return ResponseGenerator(
        prompt_llm=llm_call,
        llm_model_id=model_id,
        response_generation_prompt=prepare_response_generation_prompt(),
    )
