from langchain.prompts import PromptTemplate

MetaPlan_INSTRUCTION = """Decompose the given task into steps as an ordered list. Each step should be a single action that can be performed by one of the available tools.

You have access to the following tools: 

{tool_desc}

Use the following format:

Question: Decompose the task: {{Here is the user defined task}} 
Thought: Think about how to break down the task into smaller, manageable steps using the available tools.
Final Answer: Here are the steps to decompose the task:
1. Step 1 description.
2. Step 2 description.
3. Step 3 description.
...
N. Step N description.
Finish.

## Guidance on Tool Usage:
### 1. Identify the Required Tool: Determine which available tool is most appropriate for each step based on the task requirements. Select the tool that matches the task's nature (e.g., forecasting, anomaly detection, data querying).
### 2. Use Tools According to Their Strengths: Choose tools that excel at specific actions. For instance:
   - Use forecasting tools when performing time series analysis.
   - Use data querying tools to fetch datasets or models.
   - Use anomaly detection tools to assess outliers or anomalies in data.
### 3. Sequential Tool Usage: Some tasks may require the use of multiple tools in sequence (e.g., first query available models, then perform a forecast using one of them). 
   - Ensure that each step logically leads to the next, utilizing appropriate tools in the correct order.
### 4. Reassess and Adjust Tools When Needed: If a tool's action does not fully resolve the task, you may need to revisit the previous steps and choose different tools, or perform additional actions to refine the results.
### 5. Use Tools Efficiently: Optimize tool usage by limiting unnecessary steps. For example, avoid querying or using a tool for the same task multiple times unless needed for accuracy or clarification.
### 6. Handle Missing or Ambiguous Data: When the necessary data is not available or the task is ambiguous, use clarifications, assumptions, or fallback methods to ensure progress without waiting for additional resources.

Here are some examples:
{examples}
(END OF EXAMPLES)

Here are given task:
Question: Decompose the task: {task}
"""

metaplan_prompt = PromptTemplate(
    input_variables=["examples", "tool_desc", "task"],
    template=MetaPlan_INSTRUCTION,
)

METAPLAN_HEADER = """### Meta-Plan:
To solve the task, I think I need to execute the following high-level steps: 
{steps}
Deviation from these steps might be needed based on the following observations."""