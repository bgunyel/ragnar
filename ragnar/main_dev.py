import os
import time
import datetime

from config import settings
from backend.rag_engine import RagEngine

def main():

    rag_engine = RagEngine()

    print('Welcome! Type "exit" to quit.')
    while True:
        print('')
        user_input = input('You: ')
        if user_input.lower() == 'exit':
            break

        response = rag_engine.stream_response(user_message=user_input)
        print('Ragnar: ')
        chunk_id = 0
        for chunk in response:
            print(chunk.content, end='')
            chunk_id += 1
            if chunk_id % 50 == 0:
                print('')
            time.sleep(0.05)



if __name__ == '__main__':
    print(f'{settings.APPLICATION_NAME} started at {datetime.datetime.now()}')
    time1 = time.time()
    main()
    time2 = time.time()
    print(f'{settings.APPLICATION_NAME} finished at {datetime.datetime.now()}')
    print(f'{settings.APPLICATION_NAME} took {(time2 - time1):.2f} seconds')
