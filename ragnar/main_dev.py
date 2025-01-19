import datetime
import time

from backend.rag_engine import RagEngine
from config import settings


def main():

    # print(f"OLLAMA HOST: {os.environ['OLLAMA_HOST']}")
    rag_engine = RagEngine()
    # rag_engine.insert_web_doc_to_db(url='https://en.wikipedia.org/wiki/Syria')
    # rag_engine.insert_web_doc_to_db(url='https://en.wikipedia.org/wiki/Syrian_civil_war')
    # rag_engine.insert_web_doc_to_db(url='https://en.wikipedia.org/wiki/Fall_of_the_Assad_regime')
    # rag_engine.insert_web_doc_to_db(url='https://en.wikipedia.org/wiki/President_of_Syria')


    print('\n')
    print('Welcome! Type "exit" to quit.')
    while True:
        print('')
        user_input = input('You: ')
        if user_input.lower() == 'exit':
            break

        time_now = datetime.datetime.now().replace(microsecond=0).astimezone(
            tz=datetime.timezone(offset=datetime.timedelta(hours=3), name='UTC+3'))
        print(f'{time_now}\n')
        print(f'Ragnar ({settings.MODEL}) ', end='')

        response = rag_engine.get_response(user_message=user_input)

        time_now = datetime.datetime.now().replace(microsecond=0).astimezone(
            tz=datetime.timezone(offset=datetime.timedelta(hours=3), name='UTC+3'))

        print(f'({time_now}) ', end='')

        for chunk in response.split():
            print(chunk, end=' ')
            if chunk.endswith('.'):
                print('')
            time.sleep(0.05)



if __name__ == '__main__':
    print(f'{settings.APPLICATION_NAME} started at {datetime.datetime.now()}')
    time1 = time.time()
    main()
    time2 = time.time()
    print(f'{settings.APPLICATION_NAME} finished at {datetime.datetime.now()}')
    print(f'{settings.APPLICATION_NAME} took {(time2 - time1):.2f} seconds')
