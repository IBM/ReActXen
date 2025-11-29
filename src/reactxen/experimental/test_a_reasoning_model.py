import os
from openai import OpenAI
from typing import List, Dict, Any, Optional

# --- 1. Model Configuration ---
# Group the models by how their reasoning is controlled.
MODEL_CONFIG: Dict[str, List[Optional[str]]] = {
    # Reasoning Model: Supports the 'reasoning' parameter.
    # We will test it with two different effort levels.
    "o4-mini-2025-04-16": ["high", "minimal"],

    # General-Purpose Models: Do NOT support the 'reasoning' parameter.
    # We pass 'None' to indicate a standard call with no extra body.
    "gpt-4.1-2025-04-14": [None],
    "gpt-4.1-mini-2025-04-14": [None],
    "gpt-4.1-nano-2025-04-14": [None],
}

# --- 2. Test Prompt ---
# A complex, multi-step problem to highlight the difference in reasoning effort.
TEST_PROMPT = (
    "I have three boxes. Box A has 5 red and 3 blue balls. "
    "Box B has 4 red and 6 blue balls. Box C has 2 red and 8 blue balls. "
    "If I pick one ball from each box, what is the exact probability of picking "
    "exactly two red balls and one blue ball?"
)

def run_test_case(model_name: str, effort: Optional[str], client: OpenAI):
    """Executes a single chat completion test case."""
    
    # 1. Define the parameters for the API call
    params: Dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "temperature": 0.0,  # Set to 0 for deterministic testing
    }

    # 2. Add the reasoning control for Reasoning Models
    if effort is not None:
        params["extra_body"] = {
            "reasoning": {
                "effort": effort  # e.g., "high" or "minimal"
            }
        }
        effort_label = f" (Reasoning Effort: {effort.upper()})"
    else:
        effort_label = " (General-Purpose)"

    print(f"\n--- Testing: {model_name}{effort_label} ---")
    
    try:
        # 3. Make the API Call
        response = client.chat.completions.create(**params)
        
        # 4. Extract and print the response
        content = response.choices[0].message.content
        
        # Display the crucial part of the probability calculation
        print(f"✅ Status: Success.")
        print(f"   Response Start: {content.split('.')[0]}")
        print(f"   Usage: P: {response.usage.prompt_tokens}, C: {response.usage.completion_tokens}")
        # Note: Reasoning tokens are included in completion_tokens for the o4-mini model
        
    except Exception as e:
        # 5. Handle errors (e.g., trying to use 'reasoning' on a non-reasoning model)
        print(f"❌ Status: Failed.")
        print(f"   Error: {e}")
        # If you run into an error for the General-Purpose models, 
        # it usually means the 'extra_body' parameter is not supported for that model.

def main():
    """Main function to run all model tests."""
    try:
        # Initialize the OpenAI client
        # It automatically picks up the API key from the environment variable
        client = OpenAI()
    except Exception as e:
        print("Error: Could not initialize OpenAI client.")
        print("Please ensure your OPENAI_API_KEY environment variable is set.")
        print(f"Details: {e}")
        return

    # Iterate through the models and their configurations
    for model, efforts in MODEL_CONFIG.items():
        for effort in efforts:
            run_test_case(model, effort, client)

if __name__ == "__main__":
    main()