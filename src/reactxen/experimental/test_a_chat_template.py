from transformers import AutoTokenizer

model_path = "ibm-granite/granite-3.2-8b-instruct-preview"
tokenizer = AutoTokenizer.from_pretrained(model_path)

conv = [{"role": "user", "content": "How do I develop a skill?"}]

ans = tokenizer.apply_chat_template(
    conv, tokenize=False, thinking=True, add_generation_prompt=True
)
print(ans)

print("------------")
ans = tokenizer.apply_chat_template(
    conv, tokenize=False, thinking=False, add_generation_prompt=True
)
print(ans)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Retrieves.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": 'The stock ticker symbol, e.g., "IBM".',
                    },
                    "date": {
                        "type": "string",
                        "description": 'The date in "YYYY-MM-DD".',
                    },
                },
                "required": ["ticker", "date"],
            },
            "return": {
                "type": "object",
                "description": "A dictionary containing the low and high stock prices on the given date.",
            },
        },
    },
]

print("------------")
ans = tokenizer.apply_chat_template(
    conv, tokenize=False, tools=tools, thinking=True, add_generation_prompt=True
)
print(ans)
