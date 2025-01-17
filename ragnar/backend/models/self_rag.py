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
from ragnar.backend.components.hallucination_grader import HallucinationGrader
from ragnar.backend.components.answer_grader import AnswerGrader
from ragnar.backend.components.utility_components import (
    increment_iteration,
    are_documents_relevant,
    is_answer_grounded,
    is_answer_useful
)
from ragnar.backend.enums import Node, StateField
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
    datasource: str


def route_query(state: GraphState) -> str:
    """
    Determines the path after query routing

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    if state[StateField.DATA_SOURCE.value] == 'vectorstore':
        out = Node.RETRIEVE.value
    else:
        out = Node.INTERNAL_ANSWER_GENERATOR.value

    return out


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
        self.answer_generator = AnswerGenerator(model_name=settings.MODEL, is_rag=True)
        self.internal_answer_generator = AnswerGenerator(model_name=settings.MODEL, is_rag=False)
        self.hallucination_grader = HallucinationGrader(model_name=settings.MODEL)
        self.answer_grader = AnswerGrader(model_name=settings.MODEL)
        self.graph = self.build_graph()

    def get_response(self):
        pass

    def build_graph(self):
        workflow = StateGraph(GraphState)

        # Nodes
        workflow.add_node(node=Node.ROUTER.value, action=self.router.run)
        workflow.add_node(node=Node.INTERNAL_ANSWER_GENERATOR.value, action=self.internal_answer_generator.run)
        workflow.add_node(node=Node.RETRIEVE.value, action=self.retriever.run)
        workflow.add_node(node=Node.DOCUMENT_GRADER.value, action=self.retrieval_grader.run)
        workflow.add_node(node=Node.ANSWER_GENERATOR.value, action=self.answer_generator.run)
        workflow.add_node(node=Node.HALLUCINATION_GRADER.value, action=self.hallucination_grader.run)
        workflow.add_node(node=Node.ANSWER_GRADER.value, action=self.answer_grader.run)
        workflow.add_node(node=Node.REWRITE_QUESTION.value, action=self.question_rewriter.run)
        workflow.add_node(node=Node.INCREMENT_ITERATION.value, action=increment_iteration)

        # Edges
        workflow.add_edge(start_key=START, end_key=Node.ROUTER.value)
        workflow.add_conditional_edges(source=Node.ROUTER.value, path=route_query)
        workflow.add_edge(start_key=Node.INTERNAL_ANSWER_GENERATOR.value, end_key=END)

        workflow.add_edge(start_key=Node.RETRIEVE.value, end_key=Node.INCREMENT_ITERATION.value)
        workflow.add_edge(start_key=Node.INCREMENT_ITERATION.value, end_key=Node.DOCUMENT_GRADER.value)
        workflow.add_conditional_edges(source=Node.DOCUMENT_GRADER.value, path=are_documents_relevant)

        workflow.add_edge(start_key=Node.REWRITE_QUESTION.value, end_key=Node.RETRIEVE.value)
        workflow.add_edge(start_key=Node.ANSWER_GENERATOR.value, end_key=Node.HALLUCINATION_GRADER.value)
        workflow.add_conditional_edges(source=Node.HALLUCINATION_GRADER.value, path=is_answer_grounded)
        workflow.add_conditional_edges(source=Node.ANSWER_GRADER.value, path=is_answer_useful)

        compiled_graph = workflow.compile()
        return compiled_graph

