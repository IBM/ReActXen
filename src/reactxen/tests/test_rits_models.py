import time
from reactxen.utils.model_inference import watsonx_llm

# List of models exposed through LiteLLM Proxy — update as needed
LITELLM_MODELS = [
    #"rits/deepseek-ai/DeepSeek-V2.5",
    #"rits/deepseek-ai/DeepSeek-V3.2",
    #"rits/deepseek-ai/DeepSeek-V3",
    "rits/openai/gpt-oss-20b",
    "rits/Qwen/Qwen3-8B",
    "rits/Qwen/Qwen2.5-72B-Instruct",
    "rits/Qwen/Qwen3-30B-A3B-Thinking-2507",
    "rits/meta-llama/Llama-3.1-8B-Instruct"
]

prompt = (
    "Write a short paragraph explaining predictive maintenance in industrial machines."
)

results = {}

print("\n================= LiteLLM Model Test Runner =================\n")

for model in LITELLM_MODELS:
    print(f"\n--- Testing model: {model} ---")
    start = time.time()

    try:
        out = watsonx_llm(
            prompt=prompt,
            model_id=model,
            temperature=0.0,
            max_tokens=1500,
            reasoning_effort="medium",  # low
            # stop=["maintenance", "Maintenance", " maintenance", " Maintenance"],
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
