import logging
from typing import List, Type, Dict, Any
from langchain.tools import BaseTool
from reactxen.tools.controlflow.base import LoopTool

# Setup a logger for the tool
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)


class ForEachLoopTool(LoopTool):
    """
    This tool allows the agent to call another tool (e.g., SkySparkHistory) multiple times for different items.
    The parameter for which the loop is conducted is passed as `item_name`, and `items` is the list of values for that parameter.
    """

    def _run(
        self,
        items: List[Any],  # List of values to loop over (e.g., asset names)
        item_name: str,  # The parameter name in the tool's signature to loop over
        tool_to_call: Type[
            BaseTool
        ],  # The tool to call for each item (e.g., SkySparkHistory)
        static_params: Dict[
            str, Any
        ],  # Static configuration for the tool passed dynamically during each run
    ) -> List[Any]:
        """
        Calls the specified tool for each item in the list, passing both the static configuration and dynamic parameters.

        Args:
            items (List[Any]): List of items to iterate over (e.g., asset names).
            item_name (str): The parameter name in the tool's signature to loop over (e.g., "asset_name_list").
            tool_to_call (BaseTool): The tool to call for each item (e.g., SkySparkHistory).
            static_params (Dict[str, Any]): Static configuration for the tool passed dynamically during each run.

        Returns:
            List[Any]: List of results from calling the tool for each item.
        """
        results = []

        # Iterate through each item in the list
        for item in items:
            # Prepare the arguments for the tool dynamically
            tool_args = {**static_params}  # Start with the static parameters

            # Add the current item to the tool arguments with the specified item_name
            tool_args[item_name] = item

            try:
                # Call the tool (e.g., SkySparkHistory) for the current item, passing the arguments dynamically
                result = tool_to_call.run(**tool_args)

                # Append the result to the results list
                results.append(result)

            except Exception as e:
                logger.error(f"Error calling tool for item '{item}': {e}")
                results.append(
                    None
                )  # Optionally append None or handle the error in another way

        return results
