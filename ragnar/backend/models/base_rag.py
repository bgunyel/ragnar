from io import BytesIO
from typing import TypedDict
from uuid import uuid4
from pprint import pprint

from PIL import Image
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ragnar.backend.components.answer_generator import AnswerGenerator
from ragnar.backend.components.retriever import Retriever
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

    def get_flow_chart(self):
        img_bytes = BytesIO(self.graph.get_graph(xray=True).draw_mermaid_png())
        img = Image.open(img_bytes).convert("RGB")
        return img

    def build_graph(self) -> CompiledStateGraph:

        workflow = StateGraph(GraphState)

        # Nodes
        workflow.add_node(node=Nodes.RETRIEVE.value, action=self.retriever.run)
        workflow.add_node(node=Nodes.ANSWER_GENERATOR.value, action=self.answer_generator.run)

        # Edges
        workflow.add_edge(start_key=START, end_key=Nodes.RETRIEVE.value)
        workflow.add_edge(start_key=Nodes.RETRIEVE.value, end_key=Nodes.ANSWER_GENERATOR.value)
        workflow.add_edge(start_key=Nodes.ANSWER_GENERATOR.value, end_key=END)

        compiled_graph = workflow.compile()
        return compiled_graph
