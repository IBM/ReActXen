from typing import List, Dict, Any
from langchain.tools import BaseTool

"""
Plan
- ControlFlowTool (Base class for all control logic tools)
    - LoopTool (Base class for loops)
        - ForEachLoopTool
        - WhileLoopTool
    - ConditionalTool (Base class for conditionals)
        - IfElseTool
        - SwitchTool
    - WorkflowTool (Sequences multiple tools together)
        - SequentialWorkflowTool
"""


class ControlFlowTool(BaseTool):
    """Base class for all control flow-related tools like loops and conditionals."""

    def _run(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Subclasses must implement _run() method.")


class LoopTool(ControlFlowTool):
    """Base class for looping constructs like ForEachLoop and WhileLoop."""

    pass


class ConditionalTool(ControlFlowTool):
    """Base class for conditional constructs like IfElseTool."""

    pass


class WorkflowTool(ControlFlowTool):
    """Executes a sequence of tools where outputs of one tool are used as inputs for the next."""

    def _run(self, steps: List[Dict], initial_args: Dict[str, Any] = {}) -> Any:
        """
        Executes tools in sequence while passing results as inputs.

        Args:
            steps (List[Dict]): List of steps, each defining a tool and how to handle inputs/outputs.
            initial_args (Dict[str, Any]): Initial input arguments for the first tool.

        Returns:
            Any: The final result of the workflow.
        """
        data_store = initial_args.copy()  # Stores intermediate results

        for step in steps:
            tool = step["tool"]  # Tool instance
            input_map = step.get(
                "input_map", {}
            )  # Maps data_store keys to tool input args
            output_key = step.get("output_key")  # Key to store tool output

            # Prepare tool arguments using mapping
            tool_args = {
                arg: data_store[src]
                for arg, src in input_map.items()
                if src in data_store
            }

            # Check if required inputs are present
            if not all(arg in data_store for arg in input_map.values()):
                raise ValueError(f"Missing inputs for tool {tool}")

            try:
                # Run tool
                result = tool.run(**tool_args)
            except Exception as e:
                # Handle or log the error
                print(f"Error executing tool {tool}: {e}")
                continue  # Optionally continue with the next tool in the workflow

            # Store result in data_store
            if output_key:
                data_store[output_key] = result

        return data_store  # Final state of the workflow
