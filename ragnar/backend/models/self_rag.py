from typing import TypedDict
from uuid import uuid4
from pprint import pprint

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ragnar.backend.components.answer_generator import AnswerGenerator
from ragnar.backend.components.question_rewriter import QuestionRewriter
from ragnar.backend.components.retrieval_grader import RetrievalGrader
from ragnar.backend.components.retriever import Retriever
from ragnar.backend.components.router import Router
from ragnar.backend.models.enums import Nodes
from ragnar.config import settings


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        documents: list of documents
        generation: LLM generation
        documents_grade: grade of documents

    """
    question: str
    documents: list[Document]
    generation: str
    documents_grade: str
    steps: list[str]
    iteration: int
    good_documents: list[Document]
    original_question: str


class SelfRAG:

    """
    Self-RAG - Learning to Retrieve, Generate, and Critique through Self-Reflection (arxiv: 2310.11511)
    """

    def __init__(self, retriever: VectorStoreRetriever):

        self.K = retriever.search_kwargs['k'] # store the K parameter

        self.router = Router(model_name=settings.MODEL)
        self.retriever = Retriever(retriever=retriever)
        self.retrieval_grader = RetrievalGrader(model_name=settings.MODEL)
        self.question_rewriter = QuestionRewriter(model_name=settings.MODEL)
        self.answer_generator = AnswerGenerator(model_name=settings.MODEL)
        self.graph = self.build_graph()

    def get_response(self):
        pass

    def build_graph(self):
        pass