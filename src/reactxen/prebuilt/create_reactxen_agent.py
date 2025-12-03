from reactxen.agents.react.agents import ReactReflectAgent
from reactxen.agents.react.utils import ActionStyle, ReActStyle
from reactxen.agents.react.prompts.codesystemprompts import code_react_reflect_prompt, Block_python_code
from typing import List, Optional, Callable, Literal
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool

def create_reactxen_agent(
    question: str,
    key: str,
    max_steps: Optional[int] = 6,
    agent_prompt: Optional[PromptTemplate] = None,
    reflect_prompt: Optional[PromptTemplate] = None,
    tools: List[BaseTool] = [],  # List of BaseTool objects
    tool_names: List[str] = [],  # List of strings (tool names)
    tool_desc: str = "",         # Tool description (string)
    
    # LLM Model Parameters
    react_llm: Optional[Callable] = None,  # react_llm can be a callable or None
    reflect_llm: Optional[Callable] = None,  # reflect_llm can be a callable or None
    react_llm_model_id: int = 8,
    reflect_llm_model_id: int = 8,
    review_llm_model_id: Optional[int] = None,
    
    # Action Style and Examples
    actionstyle: Literal["Code", "Text"] = "Text",
    react_example: Optional[str] = None,
    reflect_example: Optional[str] = None,
    reactstyle: str = "thought_and_act_together",
    
    # Context and Iteration Control
    max_retries: int = 1,
    num_reflect_iteration: int = 10,
    early_stop: bool = False,
    handle_context_length_overflow: bool = False,
    
    # Additional Configurations
    debug: bool = True,
    log_structured_messages: bool = False,
    apply_chat_template: bool = False,
    apply_loop_detection_check: bool = False,
    
    # Adaptive Parameter Adjustments and Caching
    apply_adaptive_parameter_adjustment: bool = False,
    parameter_configuration: Optional[dict] = None,
    use_tool_cache: bool = False,
    enable_tool_partial_match: bool = False
) -> ReactReflectAgent:
    """
    Create a ReactReflectAgent instance with various configuration parameters.

    Parameters:
    - question (str): The main question to process.
    - key (str): A unique key related to the task.
    - max_steps (int): Maximum number of steps the agent should perform.
    - agent_prompt (PromptTemplate): The template for the agent's prompt.
    - reflect_prompt (PromptTemplate): The template for the reflection phase.
    - cbm_tools (list): A list of tools for context-based model execution.
    - tool_names (list): List of tool names used in the workflow.
    - tool_desc (str): Description for tools.
    - react_llm (LLM): LLM instance for React agent.
    - reflect_llm (LLM): LLM instance for Reflect agent.
    - react_llm_model_id (int): Model ID for React LLM.
    - reflect_llm_model_id (int): Model ID for Reflect LLM.
    - review_llm_model_id (Optional[int]): Model ID for Review LLM.
    - debug (bool): Whether to enable debugging.
    - actionstyle (ActionStyle): Style for action handling (e.g., single-line tool calls).
    - react_example (str): Example prompt for the React agent.
    - reflect_example (str): Example prompt for the Reflect agent.
    - handle_context_length_overflow (bool): Whether to handle context overflow.
    - num_reflect_iteration (int): Number of reflection iterations.
    - log_structured_messages (bool): Whether to log structured messages.
    - apply_chat_template (bool): Whether to apply chat templates.
    - apply_loop_detection_check (bool): Whether to check for loop detection.
    - max_retries (int): Number of retries for failed actions.
    - early_stop (bool): If set, will stop early under certain conditions.
    - apply_adaptive_parameter_adjustment (bool): If set, adaptive parameter adjustment is enabled.
    - parameter_configuration (Optional[dict]): Configuration for parameters.
    - use_tool_cache (bool): Whether to use a tool cache.
    - enable_tool_partial_match (bool): Whether partial matches for tools are enabled.

    Returns:
    - ReactReflectAgent: A configured instance of the ReactReflectAgent.
    """

    agent_args = {key: value for key, value in locals().items() if value is not None and key != "agent_args"}

    # Copy the value of 'tools' to 'cbm_tools'
    agent_args['cbm_tools'] = agent_args.get('tools')

    if actionstyle == 'Code':
        agent_args['actionstyle'] = ActionStyle.BLOCK_FUNCTION
        if react_example is None:
            agent_args['react_example'] = ''

        initial_code_prompt = code_react_reflect_prompt.partial(
            max_iterations=max_steps,
            api_description=tool_desc,
            react_option=Block_python_code,
        )
        agent_args['agent_prompt'] = initial_code_prompt
        #print (initial_code_prompt)

    elif actionstyle == 'Text':
        agent_args['actionstyle'] = ActionStyle.SINGLE_LINE_TOOL_CALL

    if reactstyle == 'thought_and_act_together':
        agent_args['reactstyle'] = ReActStyle.ThoughtActTogether
    elif reactstyle == "only_act":
        agent_args['reactstyle'] = ReActStyle.OnlyAct

    # Remove 'tools' from agent_args to prevent duplication
    if 'tools' in agent_args:
        del agent_args['tools']

    print (agent_args)

    # Initialize and configure the agent
    agent = ReactReflectAgent(**agent_args)
    
    # Return the configured agent
    return agent