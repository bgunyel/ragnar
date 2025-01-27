import time

from ragnar.backend.researcher.researcher import Researcher
from ragnar.backend.utils import load_ollama_model
from ragnar.config import settings


class RagEngine:
    def __init__(self):

        load_ollama_model(model_name=settings.MODEL, ollama_url=f'{settings.OLLAMA_URL}')

        self.history = []

        self.responder = Researcher()

        dummy = -43


    def get_response(self, user_message: str):
        self.history.append({"role": "user", "content": user_message})
        response = self.responder.get_response(question=user_message, verbose=False)
        self.history.append({"role": "assistant", "content": response})
        return response

    def stream_response(self, user_message: str):
        response = self.get_response(user_message=user_message)

        for chunk in response.split():
            chunk += ' '
            yield chunk
            time.sleep(0.05)
