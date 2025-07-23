import asyncio
import datetime
import time
import rich
from uuid import uuid4

from config import settings
from ragnar import BusinessIntelligenceAgent
from ai_common import LlmServers, calculate_token_cost


def main():
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
            'model': 'qwen/qwen3-32b', #'deepseek-r1-distill-llama-70b',
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
        }
    }

    bia = BusinessIntelligenceAgent(llm_config=llm_config, web_search_api_key=settings.TAVILY_API_KEY)


    print('\n')
    print('Welcome! Type "exit" to quit.')
    while True:
        print('')
        user_input = input('You: ')
        if user_input.lower() == 'exit':
            break

        print(f'Ragnar: ', end='')

        response = bia.run(query=user_input)

        for chunk in response.split():
            print(chunk, end=' ')
            if chunk.endswith('.'):
                print('')
            time.sleep(0.05)

    dummy = -32


if __name__ == '__main__':
    time_now = datetime.datetime.now().replace(microsecond=0).astimezone(
        tz=datetime.timezone(offset=datetime.timedelta(hours=3), name='UTC+3'))

    print(f'{settings.APPLICATION_NAME} started at {time_now}')
    time1 = time.time()
    main()
    time2 = time.time()

    time_now = datetime.datetime.now().replace(microsecond=0).astimezone(
        tz=datetime.timezone(offset=datetime.timedelta(hours=3), name='UTC+3'))
    print(f'{settings.APPLICATION_NAME} finished at {time_now}')
    print(f'{settings.APPLICATION_NAME} took {(time2 - time1):.2f} seconds')
