import datetime
import time

from backend.rag_engine import RagEngine
from config import settings


def main():

    # print(f"OLLAMA HOST: {os.environ['OLLAMA_HOST']}")
    rag_engine = RagEngine()


    print('\n')
    print('Welcome! Type "exit" to quit.')
    while True:
        print('')
        user_input = input('You: ')
        if user_input.lower() == 'exit':
            break

        response = rag_engine.get_response(user_message=user_input)
        print(f'Ragnar ({settings.MODEL}): ', end='')

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
