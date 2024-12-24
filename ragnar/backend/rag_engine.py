from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from ragnar.config import settings


class RagEngine:
    def __init__(self):
        template = """
        Answer the following question:

        Here is the conversation history: {context}

        Question: {question}

        Answer:    
        """

        model = ChatOllama(model=settings.MODEL, temperature=0.5, num_predict=256)
        prompt = ChatPromptTemplate.from_template(template=template)
        self.chain = prompt | model
        self.history = [
            SystemMessage(
                content=  """
                            You are Ragnar, a RAG based AI assistant. 
                            You are named after the Viking leader Ragnar Lothbrok.
                            Keep your messages concise. 
                            """
            )
        ]

    def stream_response(self, user_message: str):
        self.history.append(HumanMessage(content=user_message))
        response = self.chain.stream({'context': self.history, 'question': user_message})
        content = ''
        for chunk in response:
            content += chunk.content
            yield chunk

        self.history.append(AIMessage(content=content))
