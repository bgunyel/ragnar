import time

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import List, TypedDict

from ragnar.config import settings
from ragnar.backend.utils import check_and_pull_ollama_model
from ragnar.backend.rag import RAG


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


class RagEngine:
    def __init__(self):

        check_and_pull_ollama_model(model_name=settings.MODEL, ollama_url=f'{settings.OLLAMA_URL}')

        self.history = []

        self.vector_store = Chroma(
            collection_name='ragnar',
            embedding_function=GPT4AllEmbeddings(),
            client=chromadb.HttpClient(host='localhost', port=8000),
        )

        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )

        self.rag = RAG(retriever=retriever)

        dummy = -32

    def insert_web_doc_to_db(self, url: str):
        doc = WebBaseLoader(url).load()
        docs_list = [item for item in doc]

        # Initialize a text splitter with specified chunk size and overlap
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=100)

        # Split the documents into chunks
        doc_splits = text_splitter.split_documents(docs_list)
        self.vector_store.add_documents(documents=doc_splits)


    def get_response(self, user_message: str):
        self.history.append({"role": "user", "content": user_message})
        response = self.rag.get_response(question=user_message)
        self.history.append({"role": "assistant", "content": response})
        return response

    def stream_response(self, user_message: str):
        response = self.get_response(user_message=user_message)

        for chunk in response.split():
            chunk += ' '
            yield chunk
            time.sleep(0.05)
