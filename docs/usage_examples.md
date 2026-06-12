# Usage Examples and Demonstrations

This guide walks through common ways to use the ReActXen framework, from the minimal hello-world setup to advanced configurations and experiment workflows.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Hello World: Math Problem (Code Mode)](#hello-world-math-problem-code-mode)
- [Hello World: Math Problem (Text Mode)](#hello-world-math-problem-text-mode)
- [Using Custom Tools](#using-custom-tools)
- [Exporting Results](#exporting-results)
- [Advanced Agent Configuration](#advanced-agent-configuration)
- [Running the Factly Experiments](#running-the-factly-experiments)
- [Understanding Output Formats](#understanding-output-formats)
- [Model ID Reference](#model-id-reference)

---

## Prerequisites

Activate your virtual environment and confirm the package is installed:

```bash
source reactxen/bin/activate
pip install .
```

Copy and fill in your credentials:

```bash
cp env/.env_template .env
# Edit .env with your WATSONX_APIKEY, WATSONX_URL, WATSONX_PROJECT_ID
```

---

## Hello World: Math Problem (Code Mode)

The demo script in `src/reactxen/demo/hello_world_math.py` is the quickest way to verify your setup. It solves a logarithmic equation using the agent in code-generation mode.

```bash
cd src/reactxen/demo
python hello_world_math.py --mode code --model_id 15
```

`model_id 15` = `ibm/granite-3-2-8b-instruct`. See the [Model ID Reference](#model-id-reference) for all options.

**Equivalent Python:**

```python
from reactxen.prebuilt.create_reactxen_agent import create_reactxen_agent

agent = create_reactxen_agent(
    question="Find the $r$ that satisfies $\\log_{16} (r+16) = \\frac{5}{4}$.",
    key="",
    react_llm_model_id=15,
    reflect_llm_model_id=15,
    actionstyle="Code",
    max_steps=6,
    num_reflect_iteration=3,
    early_stop=False,
    debug=False,
)

result = agent.run()
print(result)
```

**Expected result structure:**

```json
{
  "status": "Accomplished",
  "reasoning": "The agent successfully converted the logarithmic equation to its exponential form and solved for 'r'.",
  "suggestions": "None"
}
```

---

## Hello World: Math Problem (Text Mode)

Text mode uses single-line tool calls instead of code blocks. Useful for models that perform better with natural-language actions.

```bash
python hello_world_math.py --mode text --model_id 15
```

**Equivalent Python:**

```python
agent = create_reactxen_agent(
    question="Find the $r$ that satisfies $\\log_{16} (r+16) = \\frac{5}{4}$.",
    key="",
    react_llm_model_id=15,
    reflect_llm_model_id=15,
    actionstyle="Text",
    max_steps=6,
)

result = agent.run()
```

---

## Using Custom Tools

Pass `langchain` `BaseTool` subclasses to give the agent access to external APIs or functions.

```python
from langchain_core.tools import BaseTool
from reactxen.prebuilt.create_reactxen_agent import create_reactxen_agent

class TemperatureTool(BaseTool):
    name: str = "get_temperature"
    description: str = "Returns the current temperature for a given city."

    def _run(self, city: str) -> str:
        # Replace with a real API call
        return f"The temperature in {city} is 22°C."

    async def _arun(self, city: str) -> str:
        raise NotImplementedError

tool = TemperatureTool()

agent = create_reactxen_agent(
    question="What is the temperature in Tokyo?",
    key="",
    react_llm_model_id=15,
    tools=[tool],
    tool_names=["get_temperature"],
    tool_desc="get_temperature(city: str) -> str: Returns the current temperature for a city.",
    actionstyle="Text",
    max_steps=4,
)

result = agent.run()
print(result)
```

---

## Exporting Results

After running the agent, you can export three types of output.

### Review

A structured assessment of whether the agent accomplished the task:

```python
result = agent.run()
# result["status"] is "Accomplished" or "Failed"
# result["reasoning"] explains the outcome
# result["suggestions"] provides improvement hints

import json
with open("review.json", "w") as f:
    json.dump(result, f, indent=4)
```

### Benchmark Metrics

Token usage, API call count, execution time, and configuration snapshot:

```python
metric = agent.export_benchmark_metric()
print(metric)

with open("metric.json", "w") as f:
    json.dump(metric, f, indent=4)
```

**Example output:**

```json
{
  "questions": "Find the $r$ ...",
  "number_of_reflections": 0,
  "total_execution_time": 0.41,
  "per_round_info": [
    {
      "step": 4,
      "info": {
        "model_stats": {
          "tokens_sent": 5140,
          "tokens_received": 259,
          "api_calls": 6,
          "total_cost": 0
        }
      }
    }
  ],
  "status": "Accomplished",
  "max_steps": 6,
  "num_reflect_iteration": 3
}
```

### Trajectory

The full step-by-step reasoning trace (`Thought → Action → Observation`):

```python
traj = agent.export_trajectory()

with open("trajectory.json", "w") as f:
    json.dump(traj, f, indent=4)
```

The trajectory includes `thought`, `action`, and `observation` for each step, plus the full message history and final answer.

---

## Advanced Agent Configuration

### Enable Loop Detection

Prevents the agent from repeating the same action indefinitely:

```python
agent = create_reactxen_agent(
    question="...",
    key="",
    react_llm_model_id=15,
    apply_loop_detection_check=True,
)
```

### Handle Context Length Overflow

Automatically trims the scratchpad when the prompt approaches the model's context limit:

```python
agent = create_reactxen_agent(
    question="...",
    key="",
    react_llm_model_id=15,
    handle_context_length_overflow=True,
)
```

### Enable Tool Caching

Caches tool call results to `tool_cache.json` so repeated identical calls are skipped:

```python
agent = create_reactxen_agent(
    question="...",
    key="",
    react_llm_model_id=15,
    tools=[my_tool],
    use_tool_cache=True,
    enable_tool_partial_match=True,
)
```

### Adaptive Parameter Adjustment

Lets the agent dynamically tune generation parameters between reflection iterations:

```python
param_config = {
    "temperature": [0.1, 0.3, 0.5],
    "max_new_tokens": [256, 512, 1024],
}

agent = create_reactxen_agent(
    question="...",
    key="",
    react_llm_model_id=15,
    apply_adaptive_parameter_adjustment=True,
    parameter_configuration=param_config,
)
```

### Using a Granite Model with Chat Template

```python
agent = create_reactxen_agent(
    question="...",
    key="",
    react_llm_model_id=15,   # ibm/granite-3-2-8b-instruct
    apply_chat_template=True,
    actionstyle="Code",
)
```

### Using a Different Model for Reflection

You can assign separate models to the react and reflect phases:

```python
agent = create_reactxen_agent(
    question="...",
    key="",
    react_llm_model_id=15,   # granite-3-2-8b-instruct for react
    reflect_llm_model_id=5,  # gpt-4o for reflection
    num_reflect_iteration=5,
)
```

---

## Running the Factly Experiments

All experiment scripts live in `src/reactxen/agents/factly/agent/`. Navigate there before running any of them:

```bash
cd src/reactxen/agents/factly/agent
```

### Research Question 1 — Credit Assignment

Computes step-wise credit scores across agent trajectories using either semantic (embedding-based) or lexical (word-overlap) similarity.

```bash
python credit_assignment.py semantic
python credit_assignment.py lexical
```

Results are saved in a `credit_assignment_results/` folder.

To benchmark across all trajectories in a directory:

```bash
python benchmark_credit_assignment.py \
  --traj_dir ../traj_store \
  --csv_dir . \
  --output_dir benchmark_outputs
```

### Research Question 2 — Prompt Optimizer Comparison (Baseline vs V1)

```bash
python benchmark_prompt_optimizers.py --n 5 --methods baseline,v1
```

### Research Question 3 — Full Prompt Optimizer Comparison

```bash
python benchmark_prompt_optimizers.py --n 5 --methods v1,v2_lexical,v2_semantic
```

`--n` controls how many TruthfulQA questions are sampled for the benchmark run.

---

## Understanding Output Formats

### Review JSON (`sample_review_math_problem.json`)

```json
{
  "status": "Accomplished",
  "reasoning": "...",
  "suggestions": "None"
}
```

| Field | Description |
|---|---|
| `status` | `"Accomplished"` or `"Failed"` |
| `reasoning` | Reviewer agent's explanation of the outcome |
| `suggestions` | Improvement hints if the task was not fully solved |

### Trajectory JSON (`sample_traj_math_problem.json`)

```json
{
  "type": "mpe-agent",
  "task": "...",
  "scratchpad": "...",
  "trajectory": [
    {
      "thought": "To solve the equation ...",
      "action": "import math\nbase = 16\n...",
      "observation": "32.0\n"
    }
  ],
  "final_answer": "Final Answer:\n```json\n{\"r\": 16}\n```",
  "reflections": [],
  "reviews": []
}
```

| Field | Description |
|---|---|
| `trajectory` | List of `thought / action / observation` steps |
| `scratchpad` | Full concatenated reasoning trace |
| `final_answer` | Raw final answer string from the agent |
| `reflections` | Reflection outputs if `num_reflect_iteration > 0` |
| `reviews` | Reviewer agent outputs |

### Benchmark Metric JSON (`sample_metric_math_problem.json`)

| Field | Description |
|---|---|
| `number_of_reflections` | How many reflection rounds were needed |
| `total_execution_time` | Wall-clock time in minutes |
| `per_round_info` | Per-step token and API call counts |
| `status` | Final task status |
| `configuration_parameters` | Snapshot of model IDs and shot counts used |

---

## Model ID Reference

| ID | Model |
|---|---|
| 0 | `meta-llama/llama-3-70b-instruct` |
| 1 | `ibm/granite-13b-chat-v2` |
| 4 | `openai/gpt-3.5-turbo` |
| 5 | `openai/gpt-4o` |
| 8 | `ibm/granite-3-8b-instruct` *(default)* |
| 12 | `meta-llama/llama-3-3-70b-instruct` |
| 15 | `ibm/granite-3-2-8b-instruct` |
| 16 | `meta-llama/llama-4-maverick-17b-128e-instruct-fp8` |
| 19 | `ibm/granite-3-3-8b-instruct` |
| 24 | `litellm/GCP/gemini-2.0-flash` |
| 33 | `litellm/GCP/gemini-2.5-flash` |
| 40 | `rits/deepseek-ai/DeepSeek-V2.5` |
| 44 | `rits/Qwen/Qwen3-8B` |

See the full list in [src/reactxen/utils/model_inference.py](../src/reactxen/utils/model_inference.py).
