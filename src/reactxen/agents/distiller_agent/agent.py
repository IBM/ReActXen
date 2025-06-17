from reactxen.agents.distiller_agent.distiller_prompt import (    system_prompt_template,
    system_prompt_template_with_example,
)
from reactxen.utils.model_inference import watsonx_llm



class QueryDistillerAgent:
    """
    A class to encapsulate the logic for the QueryDistillerAgent, which evaluates the success or failure
    of an AI agent's response based on given criteria.
    """

    def __init__(self, llm=watsonx_llm, model_id=6, max_retries=3, with_example=None):
        """
        Initialize the QueryDistillerAgent.

        Args:
            llm: An instance of the language model (e.g., OpenAI, LangChain's LLM wrapper).
            model_id: Identifier for the LLM to use.
            max_retries: The maximum number of retry attempts for parsing valid JSON.
            with_example: few in-context examples (just a problem statement only)
        """
        self.llm = llm
        self.model_id = model_id
        self.max_retries = max_retries
        self.with_example = with_example

    def evaluate_response(self, question):
        """
        Evaluate the agent's response to a given question.

        Args:
            question (str): The original question or task.

        """
        # print ('called here')
        if self.with_example:
            prompt = system_prompt_template_with_example.format(
                question=question,
                examples=self.with_example,
            )
        else:
            prompt = system_prompt_template.format(
                question=question,
            )

        # Retry mechanism
        for _ in range(self.max_retries):
            try:
                distillation_result = self.llm(
                    prompt, max_tokens=2000, model_id=self.model_id, stop=["\nQuery:", "\n(END OF RESPONSE)"]
                )
                ans_text = distillation_result["generated_text"]
                ans_text = ans_text.removeprefix("Meta distiller Respond:")
                ans_text = ans_text.removesuffix("(END OF RESPONSE)").strip()
                return ans_text
            except:
                pass
        
        return question
