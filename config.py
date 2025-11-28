"""
Multi-LLM Provider Configuration
================================

INSTRUCTIONS FOR USERS:
1. Choose your provider by changing CURRENT_PROVIDER below
2. Customize the model and parameters in the MODELS dictionary
3. Edit SYSTEM_PROMPT to change AI behavior
4. Edit DEFAULT_USER_PROMPT for your default question
5. Make sure your API key is set in the .env file

Available providers: openai, anthropic, google, cohere, mistral, groq, together, replicate
"""

import os
import math
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# MAIN CONFIGURATION - EDIT THESE VALUES
# =============================================================================

# Choose your LLM provider (change this to switch providers)
CURRENT_PROVIDER = "openai"  # Options: openai, anthropic, google, cohere, mistral, groq, together, replicate

# System prompt - This sets the AI's behavior and personality
SYSTEM_PROMPT = """You are a helpful, knowledgeable, and friendly AI assistant. 
Provide clear, accurate, and concise responses. If you're unsure about something, 
say so rather than guessing. Be conversational but professional."""

# Default user prompt - Change this to your common use case
DEFAULT_USER_PROMPT = "Hello! Please introduce yourself and explain what you can help me with."

# Enable streaming responses (where supported)
ENABLE_STREAMING = False

# Request settings
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3

# =============================================================================
# MODEL CONFIGURATIONS - CUSTOMIZE PARAMETERS HERE
# =============================================================================

MODELS = {
    "openai": {
        # Model Selection - Choose one of these models:
        # - gpt-4o (Latest GPT-4 model with improved capabilities)
        # - gpt-4-turbo (Faster version of GPT-4)
        # - gpt-4 (Original GPT-4 model)
        # - gpt-3.5-turbo (Fast and cost-effective)
        # - gpt-4-vision-preview (Supports image inputs)
        # - gpt-4-32k (Extended context window)
        # Full list: https://platform.openai.com/docs/models/overview
        "model": "gpt-4.1-nano",
        
        # Parameters
        "temperature": 0.2,            # Lower for deterministic JSON
        "max_tokens": 2000,            # Maximum response length
        "top_p": 1.0,                  # 0.0-1.0, nucleus sampling
        "frequency_penalty": 0.0,      # -2.0 to 2.0, positive values penalize repetition
        "presence_penalty": 0.0,       # -2.0 to 2.0, positive values penalize topic repetition
        "response_format": "json_object",  # Ask OpenAI to return valid JSON
    },
    
    "anthropic": {
        # Model Selection - Choose one of these models:
        # - claude-3-opus-20240229 (Most capable, best for complex tasks)
        # - claude-3-sonnet-20240229 (Balanced performance and speed)
        # - claude-3-haiku-20240307 (Fastest, good for simple tasks)
        # - claude-2.1 (Previous generation, 200k context)
        # - claude-2.0 (Previous generation, 100k context)
        # - claude-instant-1.2 (Fast and affordable)
        # Full list: https://docs.anthropic.com/claude/docs/models-overview
        "model": "claude-3-sonnet-20240229",
        
        # Parameters
        "temperature": 0.7,            # 0.0-1.0, higher = more creative
        "max_tokens": 2000,            # Maximum response length
        "top_p": 0.9,                  # 0.0-1.0, nucleus sampling
        "top_k": 40,                   # 1-500, top-k sampling
    },
    
    "google": {
        # Model Selection - Choose one of these models:
        # - gemini-1.5-pro-latest (Latest Gemini 1.5 Pro)
        # - gemini-1.5-pro (Gemini 1.5 Pro)
        # - gemini-1.5-flash-latest (Latest Gemini 1.5 Flash)
        # - gemini-1.5-flash (Gemini 1.5 Flash - faster)
        # - gemini-pro (Gemini Pro - text only)
        # - gemini-pro-vision (Gemini Pro with vision)
        # - gemini-1.0-pro (Gemini 1.0 Pro)
        # Full list: https://ai.google.dev/models/gemini
        "model": "gemini-pro",
        
        # Parameters
        "temperature": 0.7,            # 0.0-2.0, higher = more creative
        "max_output_tokens": 2000,     # Maximum response length
        "top_p": 1.0,                  # 0.0-1.0, nucleus sampling
        "top_k": 40,                   # 1-40, top-k sampling
        "candidate_count": 1,          # Number of responses to generate
    },
    
    "cohere": {
        # Model Selection - Choose one of these models:
        # - command (Latest Command model)
        # - command-light (Faster, more efficient version)
        # - command-nightly (Experimental version with latest improvements)
        # - command-r (Enhanced reasoning capabilities)
        # - command-r-plus (Most capable reasoning model)
        # Full list: https://docs.cohere.com/docs/models
        "model": "command",
        
        # Parameters
        "temperature": 0.7,            # 0.0-5.0, higher = more creative
        "max_tokens": 2000,            # Maximum response length
        "p": 1.0,                      # 0.0-1.0, nucleus sampling (called 'p' in Cohere)
        "k": 0,                        # 0-500, top-k sampling
        "frequency_penalty": 0.0,      # 0.0-1.0, higher = less repetition
        "presence_penalty": 0.0,       # 0.0-1.0, higher = less topic repetition
    },
    
    "mistral": {
        # Model Selection - Choose one of these models:
        # - mistral-large-latest (Most capable model)
        # - mistral-large-2402 (Specific version of Large)
        # - mistral-medium-latest (Balanced performance and cost)
        # - mistral-medium-2312 (Specific version of Medium)
        # - mistral-small-latest (Fast and cost-effective)
        # - mistral-small-2402 (Specific version of Small)
        # - open-mistral-7b (Open-source 7B model)
        # - open-mixtral-8x7b (Open-source mixture of experts)
        # Full list: https://docs.mistral.ai/platform/models/
        "model": "mistral-large-latest",
        
        # Parameters
        "temperature": 0.7,            # 0.0-1.0, higher = more creative
        "max_tokens": 2000,            # Maximum response length
        "top_p": 1.0,                  # 0.0-1.0, nucleus sampling
        "safe_prompt": False,          # Enable/disable content filtering
        "random_seed": None,           # Integer for deterministic outputs
    },
    
    "groq": {
        # Model Selection - Choose one of these models:
        # - mixtral-8x7b-32768 (Mixtral with 32k context)
        # - llama2-70b-4096 (LLaMA 2 70B with 4k context)
        # - gemma-7b-it (Gemma 7B instruction-tuned)
        # - claude-3-opus-20240229 (Claude 3 Opus)
        # - claude-3-sonnet-20240229 (Claude 3 Sonnet)
        # - claude-3-haiku-20240307 (Claude 3 Haiku)
        # Full list: https://console.groq.com/docs/models
        "model": "mixtral-8x7b-32768",
        
        # Parameters
        "temperature": 0.7,            # 0.0-1.0, higher = more creative
        "max_tokens": 2000,            # Maximum response length
        "top_p": 1.0,                  # 0.0-1.0, nucleus sampling
        "top_k": 40,                   # 1-100, top-k sampling
    },
    
    "together": {
        # Model Selection - Choose one of these models:
        # - meta-llama/Llama-2-70b-chat-hf (LLaMA 2 70B)
        # - meta-llama/Llama-2-13b-chat-hf (LLaMA 2 13B)
        # - meta-llama/Llama-2-7b-chat-hf (LLaMA 2 7B)
        # - mistralai/Mixtral-8x7B-Instruct-v0.1 (Mixtral 8x7B)
        # - mistralai/Mistral-7B-Instruct-v0.2 (Mistral 7B)
        # - Qwen/Qwen1.5-72B-Chat (Qwen 72B)
        # - Qwen/Qwen1.5-14B-Chat (Qwen 14B)
        # - google/gemma-7b-it (Gemma 7B)
        # Full list: https://docs.together.ai/docs/inference-models
        "model": "meta-llama/Llama-2-70b-chat-hf",
        
        # Parameters
        "temperature": 0.7,            # 0.0-1.0, higher = more creative
        "max_tokens": 2000,            # Maximum response length
        "top_p": 1.0,                  # 0.0-1.0, nucleus sampling
        "top_k": 50,                   # 1-100, top-k sampling
        "repetition_penalty": 1.0,     # 1.0-2.0, higher = less repetition
    },
    
    "replicate": {
        # Model Selection - Choose one of these models:
        # - meta/llama-2-70b-chat (LLaMA 2 70B)
        # - meta/llama-2-13b-chat (LLaMA 2 13B)
        # - meta/llama-2-7b-chat (LLaMA 2 7B)
        # - mistralai/mixtral-8x7b-instruct-v0.1 (Mixtral 8x7B)
        # - mistralai/mistral-7b-instruct-v0.2 (Mistral 7B)
        # - anthropic/claude-3-sonnet-20240229 (Claude 3 Sonnet)
        # - anthropic/claude-3-haiku-20240307 (Claude 3 Haiku)
        # Full list: https://replicate.com/collections/language-models
        "model": "meta/llama-2-70b-chat",
        
        # Parameters
        "temperature": 0.7,            # 0.0-1.0, higher = more creative
        "max_tokens": 2000,            # Maximum response length (note: some models use max_length instead)
        "top_p": 1.0,                  # 0.0-1.0, nucleus sampling
        "repetition_penalty": 1.0,     # 1.0-2.0, higher = less repetition
    },
    
    "huggingface": {
        # Model Selection - Choose one of these models:
        # - microsoft/DialoGPT-large (Conversational model)
        # - facebook/blenderbot-400M-distill (Conversational model)
        # - gpt2 (Text generation)
        # - gpt2-xl (Larger GPT-2)
        # - EleutherAI/gpt-j-6B (6B parameter model)
        # - EleutherAI/gpt-neox-20b (20B parameter model)
        # - google/flan-t5-xxl (Text-to-text model)
        # - google/flan-ul2 (Instruction-tuned model)
        # - stabilityai/stablelm-tuned-alpha-7b (StableLM 7B)
        # Full list: https://huggingface.co/models
        "model": "microsoft/DialoGPT-large",
        
        # Parameters
        "temperature": 0.7,            # 0.0-1.0, higher = more creative
        "max_tokens": 2000,            # Maximum response length
        "top_p": 1.0,                  # 0.0-1.0, nucleus sampling
        "top_k": 50,                   # 1-100, top-k sampling
        "repetition_penalty": 1.0,     # 1.0-2.0, higher = less repetition
        "do_sample": True,             # Whether to use sampling
        "wait_for_model": True,        # Wait if model is loading
    },
}

# DISPLAY LLM NAME
# =============================================================================
# Displays the LLM Provider that is currently in use.
#  The LLM Provider can be change in line 26

def display_current_llm():
    """
    Returns a user-friendly string showing the current provider and model.
    Example: "Current LLM: OpenAI - gpt-4.1-nano"
    """
    from config import CURRENT_PROVIDER, MODELS
    import os

    provider = CURRENT_PROVIDER
    model_name = MODELS.get(provider, {}).get("model", "unknown")

    return f"Current LLM: {provider.capitalize()} - {model_name}"

# =============================================================================
# VALIDATION AND HELPER FUNCTIONS
# =============================================================================

def get_current_config():
    """Get the configuration for the current provider"""
    if CURRENT_PROVIDER not in MODELS:
        raise ValueError(f"Provider '{CURRENT_PROVIDER}' not found in MODELS configuration")
    
    return MODELS[CURRENT_PROVIDER]

def validate_api_key(provider: str) -> bool:
    """Check if API key exists for the given provider"""
    key_mapping = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY", 
        "google": "GOOGLE_API_KEY",
        "cohere": "COHERE_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "groq": "GROQ_API_KEY",
        "together": "TOGETHER_API_KEY",
        "replicate": "REPLICATE_API_TOKEN",
    }
    
    env_key = key_mapping.get(provider)
    if not env_key:
        return False
        
    api_key = os.getenv(env_key)
    return api_key is not None and api_key.strip() != "" and not api_key.startswith("your_")

def list_available_providers():
    """List all configured providers"""
    return list(MODELS.keys())

def get_provider_info():
    """Get information about the current provider"""
    config = get_current_config()
    has_key = validate_api_key(CURRENT_PROVIDER)
    
    return {
        "provider": CURRENT_PROVIDER,
        "model": config.get("model", "Unknown"),
        "has_api_key": has_key,
        "config": config
    }

# Validate current configuration on import
if __name__ == "__main__":
    print("Current Configuration:")
    print(f"Provider: {CURRENT_PROVIDER}")
    print(f"Model: {get_current_config().get('model')}")
    print(f"API Key Available: {validate_api_key(CURRENT_PROVIDER)}")



# =============================================================================
# COST REPORTING (From User Submitted File/Hard Code for testing) 
# For file use: calculate_cost_from_file(path, provider=None, model=None)
# For text use: calculate_cost(input_text, output_text="", provider=None, model=None)
# =============================================================================

# Turn cost reporting on/off
ENABLE_COST_REPORT = True

COST_REPORT_OUTPUT_DIR = "cost_reports"

# To track cost per 1K tokens for each provider/model
COST_PER_1K_TOKENS = {
    "openai": {
        "gpt-4.1-nano": 0.0005,
        "gpt-4o": 0.03,
        "gpt-4-turbo": 0.015,
        "gpt-4": 0.06,
        "gpt-3.5-turbo": 0.002,
        "gpt-4-vision-preview": 0.04,
        "gpt-4-32k": 0.12,
    },
    "anthropic": {
        "claude-3-opus-20240229": 0.09,
        "claude-3-sonnet-20240229": 0.06,
        "claude-3-haiku-20240307": 0.03,
        "claude-2.1": 0.045,
        "claude-2.0": 0.03,
        "claude-instant-1.2": 0.015,
    },
    "google": {
        "gemini-1.5-pro-latest": 0.05,
        "gemini-1.5-flash-latest": 0.03,
        "gemini-pro": 0.04,
        "gemini-1.0-pro": 0.025,
    },
    "cohere": {
        "command": 0.02,
        "command-r": 0.03,
    },
    "mistral": {
        "mistral-large-latest": 0.04,
        "mistral-medium-latest": 0.025,
        "mistral-small-latest": 0.015,
    },
    "groq": {
        "mixtral-8x7b-32768": 0.03,
        "llama2-70b-4096": 0.05,
    },
    "together": {
        "meta-llama/Llama-2-70b-chat-hf": 0.04,
        "mistralai/Mixtral-8x7B-Instruct-v0.1": 0.03,
    },
    "replicate": {
        "meta/llama-2-70b-chat": 0.04,
        "mistralai/mixtral-8x7b-instruct-v0.03": 0.03,
    }
}

DEFAULT_CHARS_PER_TOKEN = 4  # Approximate

# 2 Options for token estimating - hard coded for testing purposes or from file
def estimate_tokens_from_text(text, chars_per_token=DEFAULT_CHARS_PER_TOKEN):
    if not text:
        return 0
    return int(math.ceil(len(text) / float(chars_per_token)))


def estimate_tokens_from_file(path, chars_per_token=DEFAULT_CHARS_PER_TOKEN):
    p = Path(path)
    try:
        raw_bytes = p.read_bytes()
        text = raw_bytes.decode("utf-8", errors="replace")
    except Exception:
        text = ""
    return estimate_tokens_from_text(text, chars_per_token)


def calculate_cost(input_text, output_text="", provider=None, model=None):
    provider = provider or CURRENT_PROVIDER
    model = model or MODELS.get(provider, {}).get("model", "unknown")

    input_tokens = estimate_tokens_from_text(input_text)
    output_tokens = estimate_tokens_from_text(output_text)
    total_tokens = input_tokens + output_tokens

    cost_per_1k = COST_PER_1K_TOKENS.get(provider, {}).get(model, 0.0)
    cost_available = cost_per_1k > 0.0
    estimated_cost = (total_tokens / 1000.0) * cost_per_1k if cost_available else 0.0

    return {
        "provider": provider,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_per_1k": cost_per_1k,
        "estimated_cost_usd": round(estimated_cost, 6),
        "cost_available": cost_available,
    }


def calculate_cost_from_file(path, provider=None, model=None, chars_per_token=DEFAULT_CHARS_PER_TOKEN):
    p = Path(path)
    try:
        raw_bytes = p.read_bytes()
        text = raw_bytes.decode("utf-8", errors="replace")
    except Exception:
        text = ""

    report = calculate_cost(text, "", provider=provider, model=model)
    report["source_file"] = str(p.resolve())
    return report


def calculate_cost_auto(input_source, output_text="", provider=None, model=None):
    """Calculate cost from either a text input or a file path."""
    if isinstance(input_source, str) and os.path.isfile(input_source):
        return calculate_cost_from_file(input_source, provider=provider, model=model)
    return calculate_cost(input_source, output_text, provider=provider, model=model)


def display_cost(input_text, output_text=""):
    report = calculate_cost(input_text, output_text)
    suffix = "" if report["cost_available"] else " (cost config not found)"
    print(
        f"💰 Input: {report['input_tokens']} | "
        f"Output: {report['output_tokens']} | "
        f"Total: {report['total_tokens']} | "
        f"Cost: ${report['estimated_cost_usd']:.6f}{suffix}"
    )

    
