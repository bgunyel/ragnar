from .agents import BusinessIntelligenceAgent
from .agents import Table as DatabaseTable
from config import settings
from ai_common import LlmServers


def get_llm_config():
    llm_config = {
        'language_model': {
            'model': 'llama-3.3-70b-versatile',
            'model_provider': LlmServers.GROQ.value,
            'api_key': settings.GROQ_API_KEY,
            'model_args': {
                'service_tier': "auto",
                'temperature': 0,
                'max_retries': 5,
                'max_tokens': 32768,
                'model_kwargs': {
                    'top_p': 0.95,
                }
            }
        },
        'reasoning_model': {
            'model': 'openai/gpt-oss-120b', #'qwen/qwen3-32b',  # 'deepseek-r1-distill-llama-70b',
            'model_provider': LlmServers.GROQ.value,
            'api_key': settings.GROQ_API_KEY,
            'model_args': {
                'service_tier': "auto",
                'temperature': 0,
                'max_retries': 5,
                'max_tokens': 65536, # for deepseek and qwen3: 32768,
                'reasoning_effort': 'medium', # only for gpt-oss models: ['high', 'medium', 'low']
                'model_kwargs': {
                    'top_p': 0.95,
                }
            }
        }
    }

    return llm_config

__all__ = [
    'BusinessIntelligenceAgent',
    'DatabaseTable',
    'get_llm_config',
]