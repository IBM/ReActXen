import asyncio
from typing import Dict, Any
from pydantic import Field, PrivateAttr
from langchain.tools import BaseTool
from langchain_sandbox import PyodideSandbox

# install deno
# brew install deno

class SandboxExecutorTool(BaseTool):
    """
    Executes Python code safely using LangChain Sandbox (Pyodide backend).
    Suitable for ReAct-style agents to evaluate generated code securely.
    """

    name: str = Field(default="code_executor_sandbox", description="Tool name.")
    description: str = Field(
        default=(
            "Executes Python code safely using LangChain Sandbox (Pyodide). "
            "Input: Python source code as a string. "
            "Output: A JSON-like dict with execution status, stdout, stderr, and timing."
        )
    )

    allow_net: bool = Field(default=True, description="Allow network access in the sandbox.")
    stateful: bool = Field(default=True, description="Persist sandbox state between executions.")

    # Use PrivateAttr for runtime-only attributes
    _sandbox: PyodideSandbox = PrivateAttr()

    def __init__(self, allow_net: bool = True, stateful: bool = True, **kwargs):
        super().__init__(allow_net=allow_net, stateful=stateful, **kwargs)
        # Assign to private attribute
        self._sandbox = PyodideSandbox(allow_net=allow_net, stateful=stateful)

    def _run(self, code: str) -> Dict[str, Any]:
        """Execute Python code safely and return only JSON output."""
        import asyncio
        import json

        async def execute():
            # Wrap user code to capture only printed JSON
            wrapper_code = f"""
    import sys
    import io
    from contextlib import redirect_stdout

    stdout_buffer = io.StringIO()
    with redirect_stdout(stdout_buffer):
        {code}

    output = stdout_buffer.getvalue()
    """
            result = await self._sandbox.execute(wrapper_code)

            # Extract JSON lines from stdout safely
            lines = result.stdout.splitlines()
            json_lines = [line for line in lines if line.strip().startswith("{")]
            json_output = "\n".join(json_lines)

            return {
                "status": result.status,
                "stdout": json_output,
                "stderr": result.stderr.strip() if result.stderr else "",
                "execution_time": getattr(result, "execution_time", None),
            }

        # Run the async function synchronously
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return asyncio.run_coroutine_threadsafe(execute(), loop).result()
        else:
            return loop.run_until_complete(execute())


    def _run(self, code: str) -> Dict[str, Any]:
        """Synchronously execute Python code inside the sandbox."""
        if not code or not isinstance(code, str):
            raise ValueError("Input code must be a non-empty string.")

        async def execute():
            result = await self._sandbox.execute(code)
            return {
                "status": result.status,
                "stdout": result.stdout.strip() if result.stdout else "",
                "stderr": result.stderr.strip() if result.stderr else "",
                "execution_time": getattr(result, "execution_time", None),
            }

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return asyncio.run_coroutine_threadsafe(execute(), loop).result()
            else:
                return loop.run_until_complete(execute())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(execute())
        except Exception as e:
            return {"status": "error", "stderr": str(e), "stdout": "", "execution_time": None}

    async def _arun(self, code: str) -> Dict[str, Any]:
        """Asynchronously execute Python code inside the sandbox."""
        if not code or not isinstance(code, str):
            raise ValueError("Input code must be a non-empty string.")

        try:
            result = await self._sandbox.execute(code)
            return {
                "status": result.status,
                "stdout": result.stdout.strip() if result.stdout else "",
                "stderr": result.stderr.strip() if result.stderr else "",
                "execution_time": getattr(result, "execution_time", None),
            }
        except Exception as e:
            return {"status": "error", "stderr": str(e), "stdout": "", "execution_time": None}
