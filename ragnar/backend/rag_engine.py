from langchain_ollama import ChatOllama

from ragnar.config import settings


class RagEngine:
    def __init__(self):

        self.model = ChatOllama(model=settings.MODEL, temperature=0.1)
        self.history = []

        self.system_message = (
            "System: This is a chat between a user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. "
            "The assistant should also indicate when the answer cannot be found in the context. "
            "The assistant is named Ragnar."
        )

        self.instruction = "Please give a full and complete answer for the question."

        print(self.system_message)


    def get_formatted_input(self, context:str = None):

        conversation = '\n\n'.join(
            ["User: " + item["content"] if item["role"] == "user" else "Assistant: " + item["content"]
             for item in self.history]
        ) + "\n\nAssistant:"

        formatted_input = (
            self.system_message + "\n\n" + context + "\n\n" + conversation if context is not None
            else self.system_message + "\n\n" + conversation
        )

        return formatted_input

    def stream_response(self, user_message: str):
        content = user_message if len(self.history) > 0 else self.instruction + ' ' + user_message
        self.history.append({"role": "user", "content": content})
        formatted_input = self.get_formatted_input()
        response = self.model.stream(formatted_input)
        content = ''
        for chunk in response:
            content += chunk.content
            yield chunk

        self.history.append({"role": "assistant", "content": content})
