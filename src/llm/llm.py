from src.llm.config import MODELS
from openai import OpenAI
import json

class LLMClient:
    def __init__(self, model_name):
        self.config = MODELS[model_name]
        self.client = OpenAI(
            api_key=self.config['api_key'],
            base_url=self.config['base_url'],
        )
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        
    def generate_response(self, prompt, n=1):
        try:
            generate_params = {
                "model": self.config['model'],
                "messages": [{"role": "user", "content": prompt}],
                "top_p": self.config.get('top_p', 0.8),  
                "temperature": self.config.get('temperature', 0.7),
                "n": n
            }
            completion = self.client.chat.completions.create(**generate_params)
            self.token_usage["prompt_tokens"] += completion.usage.prompt_tokens
            self.token_usage["completion_tokens"] += completion.usage.completion_tokens
            self.token_usage["total_tokens"] += completion.usage.total_tokens
            
            results = []
            for choice in completion.choices:
                result = {
                    "content": "",
                    "reasoning_content": ""
                }
                if self.config.get('is_inference', True):
                    result["content"] = choice.message.content
                    result["reasoning_content"] = getattr(choice.message, "reasoning_content", "")
                else:
                    result["content"] = choice.message.content
                results.append(result)
            return results, False
        except Exception as e:
            return [str(e)], True
        
    def reset_token_usage(self):
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
if __name__ == "__main__":
    llm = LLMClient("qwen")
    response, error = llm.generate_response("hello")
    print(response)