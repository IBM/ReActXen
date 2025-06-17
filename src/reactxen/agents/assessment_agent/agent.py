from reactxen.agents.assessment_agent.assessment_prompt import (
    task_assessment_system_prompt,
)
from reactxen.utils.model_inference import watsonx_llm
import json
import re


class TaskAssessmentAgent:
    """
    A class to encapsulate the logic for the TaskAssement, which evaluates the success or failure
    of an AI agent's response based on given criteria.
    """

    def __init__(self, llm=watsonx_llm, model_id=6, max_retries=3):
        """
        Initialize the TaskAssessmentAgent.

        Args:
            llm: An instance of the language model (e.g., OpenAI, LangChain's LLM wrapper).
            model_id: Identifier for the LLM to use.
            max_retries: The maximum number of retry attempts for parsing valid JSON.
        """
        self.llm = llm
        self.model_id = model_id
        self.max_retries = max_retries

    def extract_and_parse_json(self, response):
        """
        Extract and parse JSON from the response.

        Args:
            response (str): The raw response from the LLM.

        Returns:
            dict: Parsed JSON object or an error report.
        """
        try:
            # Extract JSON block enclosed in ```json ... ```
            # match = re.search(r"```json(.*?)```", response, re.DOTALL)
            match = re.search(r"\{.*\}", response.strip(), re.DOTALL)
            if match:
                json_block = match.group(0).strip()  # Extract and clean the JSON block
            else:
                json_block = response.strip()

            if not json_block:
                raise ValueError("Extracted JSON block is empty.")

            parsed_json = json.loads(json_block)
            # print (parsed_json)
            return parsed_json

        except json.JSONDecodeError as ex:
            # print(f'came here : {ex}')
            # print(f'{response}')
            # Return error information if parsing fails
            return {
                "status": "Error",
                "reasoning": f"The extracted JSON block could not be parsed. {ex}",
                "suggestions": "Ensure the LLM outputs valid JSON inside the ```json``` block.",
            }

        except ValueError as ex:
            # print(f"Value Error: {ex}")
            return {
                "status": "Error",
                "reasoning": str(ex),
                "suggestions": "Check if the extracted JSON block is empty or improperly formatted.",
            }

    def refine_response(
        self,
        question,
        agent_name,
        agent_tools,
        agent_expertise,
        agent_task_history,
        error_details,
    ):
        """
        Generate a refined prompt to request the LLM to fix JSON issues.

        Args:
            question (str): The original question or task.
            agent_think (str): The agent's explanation of its approach.
            agent_response (str): The agent's final response.
            error_details (dict): Details about the JSON decoding error.

        Returns:
            str: Refined LLM response.
        """
        refinement_prompt = (
            "Your previous response contained errors in the JSON formatting. "
            "Please ensure that your output is a valid JSON object enclosed in ```json``` blocks. "
            "Here are the error details:\n"
            f"{json.dumps(error_details, indent=2)}\n"
            "\nRegenerate your response in the requested JSON format."
        )
        prompt = task_assessment_system_prompt.format(
            input_task=question,
            agent_name=agent_name,
            agent_tools=agent_tools,
            agent_expertise=agent_expertise,
            agent_task_history=agent_task_history,
        )
        combined_prompt = f"{prompt}\n\n{refinement_prompt}"
        fullResponse = self.llm(combined_prompt, model_id=self.model_id)
        # print (fullResponse['generated_text'])
        return fullResponse["generated_text"]

    def evaluate_response(
        self, question, agent_name, agent_tools, agent_expertise, agent_task_history
    ):
        """
        Evaluate the agent's response to a given question.

        Args:
            question (str): The original question or task.
            agent_think (str): The agent's explanation of its approach.
            agent_response (str): The agent's final response.

        Returns:
            dict: A JSON-like dictionary with the evaluation result.
        """
        prompt = task_assessment_system_prompt.format(
            input_task=question,
            agent_name=agent_name,
            agent_tools=agent_tools,
            agent_expertise=agent_expertise,
            agent_task_history=agent_task_history,
        )

        # Retry mechanism
        # print(f'INITIAL PROMPT = {prompt}')
        for _ in range(self.max_retries):
            review_resultFull = self.llm(prompt, model_id=self.model_id)
            # print(review_resultFull["generated_text"])
            review_result = review_resultFull["generated_text"]
            parsed_result = self.extract_and_parse_json(review_result)
            # print(parsed_result)

            # Check if parsing succeeded
            if parsed_result.get("status") != "Error":
                return parsed_result

            # Refine response on failure
            prompt = self.refine_response(
                question,
                agent_name,
                agent_tools,
                agent_expertise,
                agent_task_history,
                parsed_result,
            )

        # Return error after exceeding retries
        return {
            "status": "Error",
            "reasoning": f"Failed to produce valid JSON after {self.max_retries} attempts.",
            "suggestions": "Review the prompt and refine the LLM response strategy.",
        }
