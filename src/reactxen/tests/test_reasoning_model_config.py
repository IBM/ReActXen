import time
from reactxen.utils.model_inference import litellm_call
import json

# List of models exposed through LiteLLM Proxy — update as needed
LITELLM_MODELS = [
    "GCP/gemini-2.0-flash",  # 24
    "GCP/gemini-2.0-flash-lite",  # 25
    # "GCP/gemini-1.5-pro",             # 26
    "GCP/claude-3-5-haiku",  # 27
    "GCP/claude-3-7-sonnet",  # 28
    "Azure/gpt-5-2025-08-07",  # 29
    "Azure/gpt-5-mini-2025-08-07",  # 30
    "Azure/gpt-5-nano-2025-08-07",  # 31
    # "Azure/gpt-5-chat-2025-08-07",    # 32
    "GCP/gemini-2.5-flash",  # 33
    "GCP/gemini-2.5-pro",  # 34
    "GCP/gemini-2.5-flash-lite",  # 35
    "GCP/claude-4-sonnet",  # 36
    "GCP/claude-opus-4",  # 37
]

model_configs = [
    {
        "model": "GCP/claude-4-sonnet",
        "name": "claude-4-sonnet-low",
        "reasoning_effort": "low",
        "max_tokens": 4096,
    },
    {
        "model": "GCP/claude-4-sonnet",
        "name": "claude-4-sonnet-medium",
        "reasoning_effort": "medium",
        "max_tokens": 4096,
    },
    {
        "model": "GCP/claude-4-sonnet",
        "name": "claude-4-sonnet-high",
        "reasoning_effort": "high",
        "max_tokens": 4096,
    },
    {
        "model": "GCP/claude-4-sonnet",
        "name": "claude-4-sonnet-nothink",
        "reasoning_effort": None,
        "max_tokens": 4096,
    },
    {
        "model": "GCP/gemini-2.5-pro",
        "name": "gemini-2.5-pro-low",
        "reasoning_effort": "low",
        "max_tokens": 4096,
    },
    {
        "model": "GCP/gemini-2.5-pro",
        "name": "gemini-2.5-pro-medium",
        "reasoning_effort": "medium",
        "max_tokens": 4096,
    },
    {
        "model": "GCP/gemini-2.5-pro",
        "name": "gemini-2.5-pro-high",
        "reasoning_effort": "high",
        "max_tokens": 4096,
    },
    {
        "model": "Azure/gpt-5-2025-08-07",
        "name": "gpt-5-2025-08-07-low",
        "reasoning_effort": "low",
        "max_tokens": 4096,
    },
    {
        "model": "Azure/gpt-5-2025-08-07",
        "name": "gpt-5-2025-08-07-medium",
        "reasoning_effort": "medium",
        "max_tokens": 4096,
    },
    {
        "model": "Azure/gpt-5-2025-08-07",
        "name": "gpt-5-2025-08-07-high",
        "reasoning_effort": "high",
        "max_tokens": 4096,
    },
    {
        "model": "Azure/gpt-5-2025-08-07",
        "name": "gpt-5-2025-08-07-nothink",
        "reasoning_effort": None,
        "max_tokens": 4096,
    },
]

prompt = (
    "Write a short paragraph explaining predictive maintenance in industrial machines."
)

results = {}

print("\n================= LiteLLM Model Test Runner =================\n")

for model_config in model_configs:
    print(f"\n--- Testing model: {model_config["name"]} ---")
    start = time.time()

    try:
        out = litellm_call(
            prompt=prompt,
            model_id=model_config["model"],
            temperature=0.0,
            max_tokens=model_config["max_tokens"],
            reasoning_effort=model_config["reasoning_effort"],  # low
        )
        elapsed = round(time.time() - start, 2)

        print(json.dumps(out, indent=2, ensure_ascii=False))
        results[model_config["name"]] = {"status": "PASS", "time": elapsed, "output": out}

    except Exception as e:
        elapsed = round(time.time() - start, 2)
        print(f"❌ FAILED ({elapsed}s) — {e}")
        results[model_config["name"]] = {"status": "FAIL", "time": elapsed, "error": str(e)}

print("\n==================== Summary ====================\n")
for model, resp in results.items():
    status = "✔" if resp["status"] == "PASS" else "❌"
    print(f"{status} {model:<30}  {resp['status']}   {resp['time']}s")
