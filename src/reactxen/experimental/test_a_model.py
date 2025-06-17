from reactxen.experimental.wrapper.watsonx_llm import get_completion_response
from reactxen.experimental.wrapper.utils.prepare_chat_message import get_decorated_chat_template

model_name = "watsonx/mistralai/mistral-large"
model_name = "rits/mistralai/mixtral-8x7B-instruct-v0.1"
model_name = "rits/mistralai/mistral-large-instruct-2407"
model_name = "azureopenai/o1-preview-2024-09-12"
# model_name = 'watsonx/mistralai/mixtral-8x7b-instruct-v01'
model_name = "ibm/granite-3-2-8b-instruct-preview-rc"

params = {
    "max_tokens": 500,
    "stop": ["<>", "Question:", "Note:", "Human:"],
    "temperature": 0.1,
    "seed": 20,
    "num_retries": 2,
    "text_generation_choice": "text",
    "is_system_prompt": False,
}

"""
ans = generate_final_answer(
    "What are the important component of Pump? (A) Roller (B) Test. Answer the question",
    model_name=model_name,
    params=params
)
"""

d_message = get_decorated_chat_template(user_message="Hello, what is todays date?")

# ans = get_chat_response(messages=["Good Morning"])
ans = get_completion_response(
    message=d_message, model_id=model_name
)
print(ans)
