from typing import TypedDict
from uuid import uuid4
from pprint import pprint

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ragnar.backend.components.answer_generator import AnswerGenerator
from ragnar.backend.components.retriever import Retriever
from ragnar.backend.enums import Node
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


class BaseRAG:
    def __init__(self, retriever: VectorStoreRetriever):
        self.retriever = Retriever(retriever=retriever)
        self.answer_generator = AnswerGenerator(model_name=settings.MODEL)
        self.graph = self.build_graph()

    def get_response(self, question: str) -> str:

        ##
        for output in self.graph.stream({
            "question": question,
            'documents': [],
            'generation': '',
            'documents_grade': '',
            "steps": [],
        }):
            for key, value in output.items():
                # Node
                pprint(f"Node '{key}':")
                # Optional: print full state at each node
                pprint(value, indent=2, width=80, depth=None)
            pprint("\n---\n")

        ##

        config = {"configurable": {"thread_id": str(uuid4())}}
        state_dict = self.graph.invoke(
            {
                'question': question,
                'documents': [],
                'generation': '',
                'documents_grade': '',
                "steps": []
             },
            config
        )
        return state_dict["generation"]

    def build_graph(self) -> CompiledStateGraph:

        workflow = StateGraph(GraphState)

        # Nodes
        workflow.add_node(node=Node.RETRIEVE.value, action=self.retriever.run)
        workflow.add_node(node=Node.ANSWER_GENERATOR.value, action=self.answer_generator.run)

        # Edges
        workflow.add_edge(start_key=START, end_key=Node.RETRIEVE.value)
        workflow.add_edge(start_key=Node.RETRIEVE.value, end_key=Node.ANSWER_GENERATOR.value)
        workflow.add_edge(start_key=Node.ANSWER_GENERATOR.value, end_key=END)

        compiled_graph = workflow.compile()
        return compiled_graph
