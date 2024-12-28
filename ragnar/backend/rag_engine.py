import os

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from typing_extensions import List, TypedDict

from ragnar.config import settings


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


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

        #######
        loader = PyPDFLoader(os.path.join(settings.INPUT_FOLDER, 'Syria-short.pdf'))
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        all_splits = text_splitter.split_documents(docs)

        embeddings = OllamaEmbeddings(model=settings.MODEL)
        self.vector_store = InMemoryVectorStore(embeddings)
        _ = self.vector_store.add_documents(documents=all_splits)

        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 1},
        )


        dummy = -32


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

        retrieved_docs = self.retrieve(query=user_message)
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)

        formatted_input = self.get_formatted_input(context=context)
        response = self.model.stream(formatted_input)
        content = ''
        for chunk in response:
            content += chunk.content
            yield chunk

        self.history.append({"role": "assistant", "content": content})

    def retrieve(self, query: str) -> list[Document]:
        retrieved_docs = self.vector_store.similarity_search(query)
        return retrieved_docs
