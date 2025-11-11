"""
OpenAI Provider
Documentation: https://platform.openai.com/docs/models
API Reference: https://platform.openai.com/docs/api-reference/chat

Supported Models:
- gpt-4-turbo-preview (Latest GPT-4 Turbo)
- gpt-4-turbo (GPT-4 Turbo with vision)
- gpt-4-turbo-2024-04-09 (Specific version)
- gpt-4-0125-preview (GPT-4 Turbo preview)
- gpt-4-1106-preview (GPT-4 Turbo preview)
- gpt-4 (GPT-4 base model)
- gpt-4-0613 (GPT-4 snapshot)
- gpt-4-32k (GPT-4 with 32k context)
- gpt-4-32k-0613 (GPT-4 32k snapshot)
- gpt-3.5-turbo (Latest GPT-3.5 Turbo)
- gpt-3.5-turbo-0125 (GPT-3.5 Turbo snapshot)
- gpt-3.5-turbo-1106 (GPT-3.5 Turbo snapshot)
- gpt-3.5-turbo-16k (GPT-3.5 with 16k context)
- gpt-3.5-turbo-instruct (Instruction-following model)

Default Parameters:
- temperature: 0.7 (0.0-2.0, higher = more creative)
- max_tokens: 2000 (1-4096 for most models)
- top_p: 1.0 (0.0-1.0, nucleus sampling)
- frequency_penalty: 0.0 (-2.0 to 2.0)
- presence_penalty: 0.0 (-2.0 to 2.0)
"""

import os
from openai import OpenAI
from typing import Dict, Any, Optional
import json

class OpenAIProvider:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def generate_response(self, 
                         system_prompt: str, 
                         user_prompt: str, 
                         model_config: Dict[str, Any]) -> str:
        """Generate response using OpenAI API"""
        try:
            response_format = model_config.get("response_format")
            response = self.client.chat.completions.create(
                model=model_config.get("model", "gpt-4"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=model_config.get("temperature", 0.7),
                max_tokens=model_config.get("max_tokens", 2000),
                top_p=model_config.get("top_p", 1.0),
                frequency_penalty=model_config.get("frequency_penalty", 0.0),
                presence_penalty=model_config.get("presence_penalty", 0.0),
                response_format={"type": "json_object"} if response_format == "json_object" else None
            )
            content = response.choices[0].message.content
            try: 
                parsed = json.loads(content)
                with open("model_output.json", "w", encoding="utf-8") as f:
                    json.dump(parsed, f, indent=4)
            except json.JSONDecodeError:
                print("Model output not a valid JSON, skipping save")
            return content
        
        except Exception as e:
            return f"Error with OpenAI: {str(e)}"
    
    def stream_response(self, 
                       system_prompt: str, 
                       user_prompt: str, 
                       model_config: Dict[str, Any]):
        """Stream response using OpenAI API"""
        try:
            stream = self.client.chat.completions.create(
                model=model_config.get("model", "gpt-4"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=model_config.get("temperature", 0.7),
                max_tokens=model_config.get("max_tokens", 2000),
                top_p=model_config.get("top_p", 1.0),
                frequency_penalty=model_config.get("frequency_penalty", 0.0),
                presence_penalty=model_config.get("presence_penalty", 0.0),
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"Error with OpenAI streaming: {str(e)}"

# Test this provider individually
if __name__ == "__main__":
    import sys
    import time
    from dotenv import load_dotenv
    
    load_dotenv()
    
    print("🤖 Testing OpenAI Provider...")
    print("=" * 50)
    
    # Test configuration
    test_config = {
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 100,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
    }
    
    system_prompt = "You are a helpful AI assistant. Be concise."
    user_prompt = "What is artificial intelligence in one sentence?"
    
    print(f"Model: {test_config['model']}")
    print(f"System: {system_prompt}")
    print(f"User: {user_prompt}")
    print("-" * 50)
    
    try:
        provider = OpenAIProvider()
        
        # Test regular response
        start_time = time.time()
        response = provider.generate_response(system_prompt, user_prompt, test_config)
        end_time = time.time()
        
        print(f"Response: {response}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print(f"Characters: {len(response)}")
        
        # Test streaming response
        print("\n🔄 Testing streaming...")
        start_time = time.time()
        full_response = ""
        for chunk in provider.stream_response(system_prompt, user_prompt, test_config):
            print(chunk, end="", flush=True)
            full_response += chunk
        end_time = time.time()
        
        print(f"\nStreaming time: {end_time - start_time:.2f} seconds")
        print("✅ OpenAI test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Make sure your OPENAI_API_KEY is set in the .env file")
        sys.exit(1)