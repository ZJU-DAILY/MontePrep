qwen_api = ""

MODELS = {
    'qwen': {
        'api_key': qwen_api,
        'base_url': "https://dashscope.aliyuncs.com/compatible-mode/v1",
        'model': "qwen2.5-coder-32b-instruct",
        'is_inference': False,
        'top_p': 0.8,
        'temperature': 0.1    
    },
    'qwen_7B': {
        'api_key': qwen_api,
        'base_url': "https://dashscope.aliyuncs.com/compatible-mode/v1",
        'model': "qwen2.5-coder-7b-instruct",
        'is_inference': False,
        'top_p': 0.8,
        'temperature': 0.1    
    },
    'qwen_14B': {
        'api_key': qwen_api,
        'base_url': "https://dashscope.aliyuncs.com/compatible-mode/v1",
        'model': "qwen2.5-coder-14b-instruct",
        'is_inference': False,
        'top_p': 0.8,
        'temperature': 0.1    
    }
}