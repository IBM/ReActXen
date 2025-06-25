from transformers import AutoTokenizer
import tiktoken
from typing import Optional

# Define the model list
model_names = [
    "meta-llama/llama-3-70b-instruct",                  # 0
    "ibm/granite-13b-chat-v2",                          # 1
    "mistralai/mixtral-8x7b-instruct-v01",              # 2
    "ibm-meta/llama-2-70b-chat-q",                      # 3
    "openai/gpt-3.5-turbo",                             # 4
    "openai/gpt-4o",                                    # 5
    "mistralai/mistral-large",                          # 6
    "meta-llama/llama-3-405b-instruct",                 # 7
    "ibm/granite-3-8b-instruct",                        # 8
    "ibm/granite-3-3b-instruct",                        # 9
    "meta-llama/llama-3-1-8b-instruct",                 #10
    "mistralai/mixtral-8x7b-instruct-v01",              #11 (duplicate of 2)
    "meta-llama/llama-3-3-70b-instruct",                #12
    "openai-azure/o1-preview",                          #13
    "ibm/granite-3-2-8b-instruct-preview-rc",           #14
    "ibm/granite-3-2-8b-instruct",                      #15
    "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",#16
    "meta-llama/llama-4-scout-17b-16e-instruct",        #17
    "openai-azure/gpt-4.1-2025-04-14",                  #18
    "ibm/granite-3-3-8b-instruct",                      #19
]

# Sample input prompt
text = "Explain how condition-based maintenance differs from preventive maintenance."

# OpenAI models fallback using tiktoken
openai_or_azure_models = {
    "openai/gpt-3.5-turbo",
    "openai/gpt-4o",
    "openai-azure/o1-preview",
    "openai-azure/gpt-4.1-2025-04-14"
}

# Fallback for all LLaMA 3 and LLaMA 4 variants
#llama_fallback_tokenizer = "meta-llama/llama-3-70b-instruct"
FALLBACK_LLAMA_TOKENIZER = "NousResearch/Llama-2-7b-hf"
FALLBACK_MISTRAL_TOKENIZER = "mistralai/Mistral-7B-Instruct-v0.1"

def get_token_count(model_name: str, text: str, token=None) -> Optional[int]:
    try:
        if model_name in openai_or_azure_models:
            enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
            return len(enc.encode(text))

        if "llama" in model_name:
            fallback = FALLBACK_LLAMA_TOKENIZER
        elif "mistral" in model_name or "mixtral" in model_name:
            fallback = FALLBACK_MISTRAL_TOKENIZER
        elif "granite" in model_name:
            fallback = "ibm/granite-13b-chat-v2"  # if available in your HF space
        else:
            fallback = model_name  # try default

        tokenizer = AutoTokenizer.from_pretrained(
            fallback,
            token=token,
            trust_remote_code=True
        )

        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    except Exception as e:
        print(f"[Warning] Tokenizer failed for {model_name} via {fallback}: {e}")
        return None

# Function to get token count
def get_token_count_1(model_name: str, text: str) -> Optional[int]:
    try:
        # Fallback for OpenAI models using tiktoken
        if model_name in openai_or_azure_models:
            enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
            return len(enc.encode(text))

        # Fallback for LLaMA 3/4 variants
        if "llama-3" in model_name or "llama-4" in model_name:
            tokenizer = AutoTokenizer.from_pretrained(llama_fallback_tokenizer)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_auth_token=True,         # Needed for IBM models
                trust_remote_code=True       # Needed for Mistral/Mixtral
            )

        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)

    except Exception as e:
        print(f"[Warning] Tokenizer not available for {model_name}: {e}")
        return None

# Run the tokenizer test on all models
print(f"\nToken counts for prompt:\n  \"{text}\"\n")
print("=" * 80)
for idx, model in enumerate(model_names):
    token_count = get_token_count(model, text)
    if token_count is not None:
        print(f"[{idx:02}] {model:<55} => {token_count:3} tokens")
    else:
        print(f"[{idx:02}] {model:<55} => ❌ Tokenizer unavailable or fallback failed")
