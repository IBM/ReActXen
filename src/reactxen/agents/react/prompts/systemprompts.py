from langchain.prompts import PromptTemplate

COT_INSTRUCTION = """Solve a given task by having a Thought, then Finish with your answer. Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task. You have access to the following tools that you should use to help you answer the question:
 
{tool_desc}

Here are some examples:
{examples}
(END OF EXAMPLES)

Use example to solve following question:
Question: {question}
{scratchpad}"""

COT_AGENT_REFLECT_INSTRUCTION = """Solve a given task by having a Thought, then Finish with your answer. Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task. You have access to the following tools that you should use to help you answer the question:

{tool_desc}

Here are some examples:
{examples}
(END OF EXAMPLES)

Here is feedback:
{reflections}

Use example and feedback to solve following question:
Question: {question}
{scratchpad}"""

COT_REFLECT_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to relevant context, tools and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>] or there is a phrasing discrepancy with your provided answer and the answer key. In a few sentences, Diagnose a possible reason for failure or phrasing discrepancy and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences and only use tools as given below.  

You have access to the following tools while solving problem: 

{tool_desc}

Here are some examples:
{examples}
(END OF EXAMPLES)

Previous trial:
Question: {question}
{scratchpad}

Reflection:"""

cot_agent_prompt = PromptTemplate(
    input_variables=["examples", "question", "scratchpad", "tool_desc"],
    template=COT_INSTRUCTION,
)

cot_reflect_agent_prompt = PromptTemplate(
    input_variables=["examples", "reflections", "question", "scratchpad", "tool_desc"],
    template=COT_AGENT_REFLECT_INSTRUCTION,
)

cot_reflect_prompt = PromptTemplate(
    input_variables=["examples", "question", "scratchpad", "tool_desc"],
    template=COT_REFLECT_INSTRUCTION,
)

BASE_REACT_INSTRUCTION = """Answer the following questions as best you can. You have access to the following tools:

{tool_desc}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {question}
{scratchpad}
"""

REACT_INSTRUCTION = """Answer the following questions as best you can. You have access to the following tools: 

{tool_desc}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

## Here are Guidance on Tool Usage:
### 1. Default to Self-Ask When Tools Are Missing: Fallback to Self-Ask when tools are unavailable, using logical problem-solving, knowledge or recent interaction (e.g., datetime calculations, math reasoning).
### 2. Prioritize Step-by-Step Explanations: Provide step-by-step explanations to ensure clarity and transparency in reasoning.
### 3. Fallback for Common Operations: Manually solve common operations like arithmetic, string manipulations, or date handling when necessary, and validate the solution.
### 4. Clearly Identify the Steps: Explicitly state when reasoning is used and solve problems step-by-step.
{% if enable_agent_ask -%}
### 5. Utilize Agent-Ask for Clarifications: When additional information is needed to resolve the question, use Agent-Ask to query another agent before taking action. Ensure the clarification request is specific and directly addresses missing details in the input question.
{%- endif %}

Here are some examples:
{examples}
(END OF EXAMPLES)

Question: {question}
{scratchpad}"""

REACT_REFLECT_INSTRUCTION = """Answer the following questions as best you can. You have access to the following tools: 

{tool_desc}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

## Here are Guidance on Tool Usage:
### 1. Default to Self-Ask When Tools Are Missing: Fallback to Self-Ask when tools are unavailable, using logical problem-solving, knowledge or recent interaction (e.g., datetime calculations, math reasoning).
### 2. Prioritize Step-by-Step Explanations: Provide step-by-step explanations to ensure clarity and transparency in reasoning.
### 3. Fallback for Common Operations: Manually solve common operations like arithmetic, string manipulations, or date handling when necessary, and validate the solution.
### 4. Clearly Identify the Steps: Explicitly state when reasoning is used and solve problems step-by-step.
{% if enable_agent_ask -%}
### 5. Utilize Agent-Ask for Clarifications: When additional information is needed to resolve the question, use Agent-Ask to query another agent before taking action. Ensure the clarification request is specific and directly addresses missing details in the input question.
{%- endif %}

Here are some examples:
{examples}
(END OF EXAMPLES)

Here is feedback:
{reflections}

Question: {question}
{scratchpad}"""


REFLECTION_HEADER = "You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\n"
REFLECTION_AFTER_LAST_TRIAL_HEADER = "The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\n"
LAST_TRIAL_HEADER = "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\n"

REFLECT_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an several tools and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  

Here are some examples:
{examples}

Previous trial:
Question: {question}
{scratchpad}

Reflection:"""

base_react_agent_prompt = PromptTemplate(
    input_variables=["question", "scratchpad", "tool_desc"],
    template=BASE_REACT_INSTRUCTION,
)


react_agent_prompt = PromptTemplate(
    input_variables=["examples", "question", "scratchpad", "tool_desc"],
    template=REACT_INSTRUCTION,
)

react_reflect_agent_prompt = PromptTemplate(
    input_variables=["examples", "reflections", "question", "scratchpad", "tool_desc"],
    template=REACT_REFLECT_INSTRUCTION,
)

reflect_prompt = PromptTemplate(
    input_variables=["examples", "question", "scratchpad"],
    template=REFLECT_INSTRUCTION,
)

