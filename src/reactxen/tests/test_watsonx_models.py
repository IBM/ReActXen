import time
from reactxen.utils.model_inference import watsonx_llm, watsonx_llm_chat

# List of models exposed through LiteLLM Proxy — update as needed
LITELLM_MODELS = [
    "meta-llama/llama-3-405b-instruct",  # 7
    "meta-llama/llama-3-3-70b-instruct",  # 12
    "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",  # 16
    "ibm/granite-3-3-8b-instruct",  # 19
    "openai/gpt-oss-120b",  # 20
    #"openai/gpt-oss-20b",  # 21
    "mistralai/mistral-medium-2505",  # 22
    "mistralai/mistral-small-3-1-24b-instruct-2503",  # 23
    "ibm/granite-4-h-small", #38
    "mistral-large-2512", #39
]

prompt = "Write a short paragraph explaining predictive maintenance in industrial machines."

results = {}

print("\n================= LiteLLM Model Test Runner =================\n")

for model in LITELLM_MODELS:
    print(f"\n--- Testing model: {model} ---")
    start = time.time()

    try:
        out = watsonx_llm_chat(
            prompt=prompt,
            model_id=model,
            temperature=0.0,
            max_tokens=500,
        )
        elapsed = round(time.time() - start, 2)

        print(f"Response ({elapsed}s): {out}")
        results[model] = {"status": "PASS", "time": elapsed, "output": out}

    except Exception as e:
        elapsed = round(time.time() - start, 2)
        print(f"❌ FAILED ({elapsed}s) — {e}")
        results[model] = {"status": "FAIL", "time": elapsed, "error": str(e)}

print("\n==================== Summary ====================\n")
for model, resp in results.items():
    status = "✔" if resp["status"] == "PASS" else "❌"
    print(f"{status} {model:<30}  {resp['status']}   {resp['time']}s")
