import time
import os
import datetime

from ragnar.config import settings
from ragnar.backend.utils import load_ollama_model, get_flow_chart
from ragnar.backend.researcher.summarizer import Summarizer


def save_response(user_message: str, response: str):
    time_now = datetime.datetime.now().replace(microsecond=0).astimezone(
        tz=datetime.timezone(offset=datetime.timedelta(hours=3), name='UTC+3'))
    file_name = os.path.join(settings.OUT_FOLDER, f'response-{time_now.isoformat()}.md')
    with open(file_name, 'w', encoding='utf-8') as f:
        # f.write(f'Topic: {user_message}\n\n')
        # f.write('Response: \n\n')
        f.write(f'{response}')


class RagEngine:
    def __init__(self):

        load_ollama_model(model_name=settings.LANGUAGE_MODEL, ollama_url=f'{settings.OLLAMA_URL}')
        load_ollama_model(model_name=settings.REASONING_MODEL, ollama_url=f'{settings.OLLAMA_URL}')
        self.history = []
        self.responder = Summarizer()

        flow_chart = get_flow_chart(rag_model=self.responder)
        flow_chart.save(os.path.join(settings.OUT_FOLDER, 'flow_chart.png'))


    def get_response(self, user_message: str):
        self.history.append({"role": "user", "content": user_message})
        response = self.responder.get_response(topic=user_message, verbose=False)
        self.history.append({"role": "assistant", "content": response})
        save_response(user_message=user_message, response=response)
        return response

    def stream_response(self, user_message: str):
        response = self.get_response(user_message=user_message)

        for chunk in response.split():
            chunk += ' '
            yield chunk
            time.sleep(0.05)
