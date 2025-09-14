"""
Multi-LLM Provider Manager
Main interface for interacting with multiple LLM providers
"""

import os
import sys
import time
import json
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from importlib import import_module

# Import configuration
from .config import (
    CURRENT_PROVIDER, MODELS, SYSTEM_PROMPT, DEFAULT_USER_PROMPT,
    REQUEST_TIMEOUT, MAX_RETRIES, ENABLE_STREAMING
)

# NOTE: Providers are lazily imported to avoid requiring all optional
# dependencies at import time. This allows using OpenAI without having to
# install packages for Anthropic, Google, etc.

class LLMManager:
    """Main class for managing multiple LLM providers"""
    
    def __init__(self, provider: Optional[str] = None):
        load_dotenv()
        
        self.current_provider = provider or CURRENT_PROVIDER
        # Map provider keys to their provider class import path
        self.providers = {
            "openai": "SecureBYTE_AI.providers.openai_provider.OpenAIProvider",
            "anthropic": "SecureBYTE_AI.providers.anthropic_provider.AnthropicProvider",
            "google": "SecureBYTE_AI.providers.google_provider.GoogleProvider",
            "cohere": "SecureBYTE_AI.providers.cohere_provider.CohereProvider",
            "mistral": "SecureBYTE_AI.providers.mistral_provider.MistralProvider",
            "groq": "SecureBYTE_AI.providers.groq_provider.GroqProvider",
            "together": "SecureBYTE_AI.providers.together_provider.TogetherProvider",
            "replicate": "SecureBYTE_AI.providers.replicate_provider.ReplicateProvider",
            "huggingface": "SecureBYTE_AI.providers.huggingface_provider.HuggingFaceProvider",
        }
        
        self.provider_instance = None
        self._initialize_provider()
    
    def _initialize_provider(self):
        """Initialize the current provider"""
        if self.current_provider not in self.providers:
            raise ValueError(f"Provider '{self.current_provider}' not supported")
        
        try:
            provider_class = self._load_provider_class(self.providers[self.current_provider])
            self.provider_instance = provider_class()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize {self.current_provider}: {str(e)}")
    
    def _load_provider_class(self, dotted_path: str):
        """Dynamically import a class from a dotted module path."""
        module_path, class_name = dotted_path.rsplit('.', 1)
        module = import_module(module_path)
        return getattr(module, class_name)
    
    def switch_provider(self, provider: str):
        """Switch to a different provider"""
        if provider not in self.providers:
            raise ValueError(f"Provider '{provider}' not supported")
        
        self.current_provider = provider
        self._initialize_provider()
        print(f"✅ Switched to {provider}")
    
    def get_model_config(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """Get model configuration for the current or specified provider"""
        provider = provider or self.current_provider
        return MODELS.get(provider, {})
    
    def generate_response(self, 
                         user_prompt: str = DEFAULT_USER_PROMPT,
                         system_prompt: str = SYSTEM_PROMPT,
                         custom_config: Optional[Dict[str, Any]] = None) -> str:
        """Generate a response using the current provider"""
        
        config = self.get_model_config()
        if custom_config:
            config.update(custom_config)
        
        return self.provider_instance.generate_response(
            system_prompt, user_prompt, config
        )
    
    def stream_response(self, 
                       user_prompt: str = DEFAULT_USER_PROMPT,
                       system_prompt: str = SYSTEM_PROMPT,
                       custom_config: Optional[Dict[str, Any]] = None):
        """Stream a response using the current provider"""
        
        config = self.get_model_config()
        if custom_config:
            config.update(custom_config)
        
        return self.provider_instance.stream_response(
            system_prompt, user_prompt, config
        )
    
    def benchmark_provider(self, 
                          test_prompts: List[str],
                          provider: Optional[str] = None) -> Dict[str, Any]:
        """Benchmark a specific provider with multiple prompts"""
        
        original_provider = self.current_provider
        if provider and provider != self.current_provider:
            self.switch_provider(provider)
        
        results = {
            "provider": self.current_provider,
            "model": self.get_model_config().get("model", "unknown"),
            "tests": [],
            "total_time": 0,
            "average_time": 0,
            "total_characters": 0,
            "average_characters": 0
        }
        
        print(f"\n🚀 Benchmarking {self.current_provider}...")
        print("=" * 50)
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"Test {i}/{len(test_prompts)}: {prompt[:50]}...")
            
            start_time = time.time()
            response = self.generate_response(prompt)
            end_time = time.time()
            
            test_result = {
                "prompt": prompt,
                "response": response,
                "time": end_time - start_time,
                "characters": len(response),
                "success": not response.startswith("Error")
            }
            
            results["tests"].append(test_result)
            results["total_time"] += test_result["time"]
            results["total_characters"] += test_result["characters"]
            
            print(f"  ✅ {test_result['time']:.2f}s, {test_result['characters']} chars")
        
        results["average_time"] = results["total_time"] / len(test_prompts)
        results["average_characters"] = results["total_characters"] / len(test_prompts)
        
        print(f"\n📊 Results for {self.current_provider}:")
        print(f"  Average time: {results['average_time']:.2f}s")
        print(f"  Average response length: {results['average_characters']} characters")
        print(f"  Total time: {results['total_time']:.2f}s")
        
        # Switch back to original provider
        if provider and provider != original_provider:
            self.switch_provider(original_provider)
        
        return results
    
    def compare_providers(self, 
                         providers: List[str],
                         test_prompts: List[str]) -> Dict[str, Any]:
        """Compare multiple providers with the same test prompts"""
        
        comparison_results = {
            "test_prompts": test_prompts,
            "providers": {},
            "summary": {}
        }
        
        print("\n🏁 Starting provider comparison...")
        print("=" * 60)
        
        for provider in providers:
            if provider not in self.providers:
                print(f"⚠️  Skipping unsupported provider: {provider}")
                continue
            
            try:
                results = self.benchmark_provider(test_prompts, provider)
                comparison_results["providers"][provider] = results
            except Exception as e:
                print(f"❌ Error testing {provider}: {str(e)}")
                comparison_results["providers"][provider] = {"error": str(e)}
        
        # Generate summary
        successful_providers = {
            name: data for name, data in comparison_results["providers"].items()
            if "error" not in data
        }
        
        if successful_providers:
            fastest = min(successful_providers.items(), 
                         key=lambda x: x[1]["average_time"])
            longest_responses = max(successful_providers.items(), 
                                  key=lambda x: x[1]["average_characters"])
            
            comparison_results["summary"] = {
                "fastest_provider": fastest[0],
                "fastest_time": fastest[1]["average_time"],
                "most_verbose_provider": longest_responses[0],
                "most_verbose_length": longest_responses[1]["average_characters"]
            }
            
            print("\n🏆 Comparison Summary:")
            print(f"  Fastest: {fastest[0]} ({fastest[1]['average_time']:.2f}s avg)")
            print(f"  Most verbose: {longest_responses[0]} ({longest_responses[1]['average_characters']} chars avg)")
        
        return comparison_results
    
    def save_benchmark_results(self, results: Dict[str, Any], filename: str):
        """Save benchmark results to a JSON file"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Results saved to {filename}")

def interactive_mode():
    """Interactive mode for testing providers"""
    llm = LLMManager()
    
    print("🤖 Multi-LLM Interactive Mode")
    print("=" * 40)
    print("Commands:")
    print("  /switch <provider> - Switch provider")
    print("  /providers - List available providers")
    print("  /config - Show current configuration")
    print("  /benchmark - Run benchmark tests")
    print("  /quit - Exit")
    print("=" * 40)
    
    while True:
        try:
            user_input = input(f"\n[{llm.current_provider}] > ").strip()
            
            if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            elif user_input.startswith('/switch '):
                provider = user_input.split(' ', 1)[1]
                try:
                    llm.switch_provider(provider)
                except Exception as e:
                    print(f"❌ Error: {e}")
            
            elif user_input == '/providers':
                print("Available providers:")
                for provider in llm.providers.keys():
                    status = "✅" if provider == llm.current_provider else "  "
                    print(f"  {status} {provider}")
            
            elif user_input == '/config':
                config = llm.get_model_config()
                print(f"Current provider: {llm.current_provider}")
                print(f"Model configuration: {json.dumps(config, indent=2)}")
            
            elif user_input == '/benchmark':
                test_prompts = [
                    "What is artificial intelligence?",
                    "Explain quantum computing in simple terms.",
                    "Write a short poem about technology."
                ]
                results = llm.benchmark_provider(test_prompts)
                
            elif user_input:
                print(f"\n🤖 {llm.current_provider} response:")
                print("-" * 40)
                
                if ENABLE_STREAMING:
                    for chunk in llm.stream_response(user_input):
                        print(chunk, end="", flush=True)
                    print()  # New line after streaming
                else:
                    response = llm.generate_response(user_input)
                    print(response)
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "interactive":
            interactive_mode()
            return
        elif sys.argv[1] == "benchmark":
            # Run benchmark on all available providers
            llm = LLMManager()
            
            # Comprehensive test prompts covering different types of tasks
            test_prompts = [
                "What is machine learning?",  # Simple definition
                "Explain the concept of neural networks in 3 sentences.",  # Concise explanation
                "What are the benefits of cloud computing for small businesses?",  # Targeted question
                "Write a haiku about programming.",  # Creative task
                "Summarize the key features of transformer models in AI.",  # Technical summary
                "How would you explain quantum computing to a 10-year-old?",  # Simplification
                "List 5 best practices for secure password management.",  # List generation
                "Compare and contrast REST and GraphQL APIs.",  # Comparison
                "What are the ethical considerations of using AI in healthcare?",  # Ethical analysis
                "Debug this Python code: def factorial(n): if n == 0: return 1 else: return n*factorial(n)"  # Technical task
            ]
            
            # Select providers to benchmark based on available API keys
            available_providers = []
            for provider in llm.providers.keys():
                env_var = f"{provider.upper()}_API_KEY"
                if provider == "replicate":
                    env_var = "REPLICATE_API_TOKEN"
                if os.getenv(env_var):
                    available_providers.append(provider)
            
            if not available_providers:
                print("❌ No API keys found. Please add at least one API key to your .env file.")
                return
                
            print(f"🔍 Found API keys for: {', '.join(available_providers)}")
            
            # Allow selecting specific providers via command line
            if len(sys.argv) > 2:
                requested_providers = sys.argv[2].split(',')
                benchmark_providers = [p for p in requested_providers if p in available_providers]
                if not benchmark_providers:
                    print(f"❌ None of the requested providers ({', '.join(requested_providers)}) have API keys configured.")
                    return
            else:
                benchmark_providers = available_providers
            
            # Run the benchmark
            results = llm.compare_providers(benchmark_providers, test_prompts)
            
            # Save results with timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
            llm.save_benchmark_results(results, filename)
            return
    
    # Default: Simple test
    print("🚀 Testing Multi-LLM System")
    print("=" * 40)
    
    llm = LLMManager()
    
    test_prompt = "Hello! Tell me an interesting fact about space in 2 sentences."
    
    print(f"Provider: {llm.current_provider}")
    print(f"Model: {llm.get_model_config().get('model', 'unknown')}")
    print(f"Prompt: {test_prompt}")
    print("-" * 40)
    
    start_time = time.time()
    response = llm.generate_response(test_prompt)
    end_time = time.time()
    
    print(f"Response: {response}")
    print(f"Time: {end_time - start_time:.2f} seconds")
    print("\n💡 Try: python main.py interactive")
    print("💡 Try: python main.py benchmark")

if __name__ == "__main__":
    main()