from .agents import BusinessIntelligenceAgent
from .agents import Table as DatabaseTable
from config import settings
from ai_common import LlmServers, ModelNames


def get_llm_config():
    llm_config = {
        'language_model': {
            'model': 'llama-3.3-70b-versatile',
            'model_provider': LlmServers.GROQ,
            'api_key': settings.GROQ_API_KEY,
            'max_llm_retries': 3,
            'model_args': {
                'temperature': 0,
                'max_tokens': 131_072,
                'top_p': 0.95,
                }
            },
        'reasoning_model': {
            'model': ModelNames.GPT_OSS_120B,
            'model_provider': LlmServers.OLLAMA,
            'api_key': settings.OLLAMA_API_KEY,
            'max_llm_retries': 3,
            'model_args': {
                'temperature': 0,
                #'max_tokens': 131_072,
                'reasoning_effort': 'high', # only for gpt-oss models: ['high', 'medium', 'low']
                'top_p': 0.95,
                }
            }
        }

    return llm_config

__all__ = [
    'BusinessIntelligenceAgent',
    'DatabaseTable',
    'get_llm_config',
]