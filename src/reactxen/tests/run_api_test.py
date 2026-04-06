import openai
client = openai.OpenAI(
    api_key="sk-NjV_SjZ3WJiPayqc0SsGy",
    base_url="https://ete-litellm.ai-models.vpc-int.res.ibm.com" # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
)

# azure/gpt-5.4
# gcp/gemini-3.1-pro-preview
# aws/claude-sonnet-4-6
# aws/claude-sonnet-4-6

response = client.chat.completions.create(
    model="aws/claude-sonnet-4-6", # model to send to the proxy
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ]
)

print(response)