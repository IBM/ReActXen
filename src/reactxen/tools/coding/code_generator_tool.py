from langchain.tools import BaseTool
from pydantic import Field
from reactxen.utils.model_inference import watsonx_llm


class CodeGeneratorTool(BaseTool):
    """
    A LangChain-compatible tool that uses an LLM to generate Python code from natural-language tasks.
    Designed for integration within ReAct-style agent frameworks.
    """

    name: str = Field(default="code_generator", description="Tool name.")
    description: str = Field(
        default=(
            "Generates Python code to solve a given task. "
            "Input: a natural language task description. "
            "Output: Python source code as a string."
        )
    )

    model_id: int = Field(default=8, description="Identifier for the LLM model used.")

    def _run(self, query: str) -> str:
        """
        Generate Python code from a task description using the LLM.
        """

        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string describing the coding task.")

        # Strict instruction prompt for reliable code output
        prompt = f"""
You are a highly skilled Python programmer.
Generate a complete, executable Python script for the following task:

Task:
{query}

Rules:
- Use only Python standard library modules.
- The output must be valid Python code, ready to execute.
- Do NOT include explanations, comments, or markdown formatting.
- The script must print the final result as JSON with a top-level key 'result'.
Example: print({{"result": "some output"}})
"""

        try:
            response = watsonx_llm(prompt, model_id=self.model_id)
            response = response['generated_text']
            print (response)
            code = response if isinstance(response, str) else getattr(response, "text", "")

            # Sanitize output (remove markdown code fences or LLM chatter)
            lines = [line for line in code.splitlines() if not line.strip().startswith("```")]
            clean_code = "\n".join(lines).strip()

            if not clean_code:
                raise ValueError("Generated code is empty or invalid.")

            return clean_code

        except Exception as e:
            return f"# Error during code generation: {str(e)}"

    async def _arun(self, query: str) -> str:
        """Async version (not implemented)."""
        raise NotImplementedError("Async execution is not implemented yet.")
