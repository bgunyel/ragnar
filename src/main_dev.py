import datetime
import os
import time
import rich

from config import settings
from ragnar import BusinessIntelligenceAgent, get_llm_config


def main():
    os.environ['LANGSMITH_API_KEY'] = settings.LANGSMITH_API_KEY
    os.environ['LANGSMITH_TRACING'] = settings.LANGSMITH_TRACING

    llm_config = get_llm_config()

    bia = BusinessIntelligenceAgent(llm_config=llm_config,
                                    web_search_api_key=settings.TAVILY_API_KEY,
                                    database_url=settings.SUPABASE_URL,
                                    database_key=settings.SUPABASE_SECRET_KEY)
    print('\n')
    print('Welcome! Type "exit" to quit.')
    while True:
        print('')
        user_input = input('You: ')
        if user_input.lower() == 'exit':
            break

        print(f'Ragnar: ', end='')

        out_dict = bia.run(query=user_input)
        rich.print(out_dict['content'])


if __name__ == '__main__':
    time_now = datetime.datetime.now().astimezone(tz=settings.TIME_ZONE)
    print(f"{settings.APPLICATION_NAME} started at {time_now.isoformat(timespec='seconds')}")

    time1 = time.time()
    main()
    time2 = time.time()

    time_now = datetime.datetime.now().astimezone(tz=settings.TIME_ZONE)
    print(f"{settings.APPLICATION_NAME} finished at {time_now.isoformat(timespec='seconds')}")
    print(f"{settings.APPLICATION_NAME} took {(time2 - time1):.2f} seconds")
