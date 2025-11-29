import time
from reactxen.utils.model_inference import litellm_call

# List of models exposed through LiteLLM Proxy — update as needed
LITELLM_MODELS = [
    "GCP/gemini-2.0-flash",           # 24
    "GCP/gemini-2.0-flash-lite",      # 25
    #"GCP/gemini-1.5-pro",             # 26
    "GCP/claude-3-5-haiku",           # 27
    "GCP/claude-3-7-sonnet",          # 28
    "Azure/gpt-5-2025-08-07",         # 29
    "Azure/gpt-5-mini-2025-08-07",    # 30
    "Azure/gpt-5-nano-2025-08-07",    # 31
    #"Azure/gpt-5-chat-2025-08-07",    # 32
    "GCP/gemini-2.5-flash",           # 33
    "GCP/gemini-2.5-pro",             # 34
    "GCP/gemini-2.5-flash-lite",      # 35
    "GCP/claude-4-sonnet",            # 36
    "GCP/claude-opus-4",              # 37
]

prompt = "Write a short paragraph explaining predictive maintenance in industrial machines."

results = {}

print("\n================= LiteLLM Model Test Runner =================\n")

for model in LITELLM_MODELS:
    print(f"\n--- Testing model: {model} ---")
    start = time.time()

    try:
        out = litellm_call(
            prompt=prompt,
            model_id=model,
            temperature=0.0,
            max_tokens=500,
            reasoning_effort=None,
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
