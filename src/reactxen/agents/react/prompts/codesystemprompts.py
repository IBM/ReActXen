from langchain_core.prompts import PromptTemplate


Block_python_code = """
- 'Thought': Analyze the current information and reason about the current situation, and predicts which API you want to use (try to use different APIs to collect diverse information) or make a decision that you want to make a final answer.
- 'Action': Use the API to gather more information or provide the final answer.
    - If gathering more data: the action must be an executable Python code snippet that starts with '```python' and ends with '```'. It can contain multiple lines of codes and function calls using the defined API or Python libraries. You must use print() to output the results, and only the printed output will be returned in the observation step.
    - If making the final result: the action must start immediately with 'Final Answer:', and follow with the answer in the expected JSON format. This should not be enclosed within triple backticks. If the result size is very large (e.g., contains more than 50 keys), store it in a file instead of directly including it in the output and provide a pointer to the file.
- 'Observation': Return the printed output of the executed code snippet.
"""

Function_python_code = """
- 'Thought': Analyze the current information and reason about the current situation, and predicts which API you want to use (try to use different APIs to collect diverse information) or make a decision that you want to make a final answer.
- 'Action': Use the API to gather more information or provide the final forecast.
    - If using the API: the action must be only one single line of exactly one function call from the API with appropriate inputs, without additional code, explanations, or natural language descriptions.
    - If making the final forecast: the action must start immediately with 'Final Answer:', and follow with the results in the expected JSON format.
- 'Observation': Return the output of the called function.
"""

Code_REACT_INSTRUCTION = """Answer the following questions as best you can. You have access to the following tools/APIs:

The defined APIs are described as follows:
```python
{api_description}
```

You will use an iterative approach, interleaving 'Thought', 'Action', and 'Observation' steps to collect information and perform the analysis. You may perform up to {max_iterations} iterations. The steps are as follows:

{react_option}

Read the input question carefully and generate results based on the input request. When you are fully confident that you have accumulated enough information to make the final result, you should start the 'Thought' with your reasoning using the collected information to make the prediction, and then start the 'Action' step with 'Final Answer:' followed by the answer in the expected JSON format. The answer should be a JSON dictionary where the keys are the forecasted equipment or component codes and the values are lists of forecasted failure or alert codes.

**Important:** If your reasoning in the 'Thought' step already includes a **'Final Answer'**, then in the 'Action' step, you should repeat or echo that **Final Answer** exactly as it was presented in the previous 'Thought'. Do not generate a new answer in 'Action' unless additional information is provided or the situation changes.

**File Handling for Long Responses:**
If the final result is too large (e.g., a large JSON dictionary say 50 keys), the 'Action' step should produce a python code to save the response into a JSON file. You can create a file in current folder or temporary folder if file path is not provided explict in the input query. Provide a file path and name for the saved JSON file in the final response. Ensure that the file path is clearly specified, and include the file extension `.json`. The file will store the answer for later retrieval.
        
Here are some examples:
{examples}
(END OF EXAMPLES)

Use example to solve following question:
Question: {question}
{scratchpad}
"""

react_agent_prompt = PromptTemplate(
    input_variables=[
        "examples",
        "question",
        "scratchpad",
        "api_description",
        "max_iterations",
        "react_option",
    ],
    template=Code_REACT_INSTRUCTION,
)

Code_REACT_Reflect_INSTRUCTION = """Answer the following questions as best you can. You have access to the following tools/APIs:

The defined APIs are described as follows:
```python
{api_description}


You will use an iterative approach, interleaving 'Thought', 'Action', and 'Observation' steps to collect information and perform the analysis. You may perform up to {max_iterations} iterations. The steps are as follows:

{react_option}

Read the input question carefully and generate results based on the input request. When you are fully confident that you have accumulated enough information to make the final result, you should start the 'Thought' with your reasoning using the collected information to make the prediction, and then start the 'Action' step with 'Final Answer:' followed by the answer in the expected JSON format. The final answer should be a JSON dictionary with following format:

Final Answer: 
```json
{{
"write your final answer here in json form"
}}
```

**Important:** If your reasoning in the 'Thought' step already includes a **'Final Answer'**, then in the 'Action' step, you should repeat or echo that **Final Answer** exactly as it was presented in the previous 'Thought'. Do not generate a new answer in 'Action' unless additional information is provided or the situation changes.

**File Handling for Long Responses:**
If the final result is too large (e.g., a large JSON dictionary say 50 keys), the 'Action' step should produce a python code to save the response into a JSON file. You can create a file in current folder or temporary folder if file path is not provided explict in the input query. Provide a file path and name for the saved JSON file in the final response. Ensure that the file path is clearly specified, and include the file extension `.json`. The file will store the answer for later retrieval.
        
Here are some examples:
{examples}
(END OF EXAMPLES)

Here is feedback:
{reflections}

Use example to solve following question:
Question: {question}
{scratchpad}
"""

code_react_reflect_prompt = PromptTemplate(
    input_variables=[
        "examples",
        "reflections",
        "question",
        "scratchpad",
        "react_option",
        "api_description",
    ],
    template=Code_REACT_Reflect_INSTRUCTION,
)
