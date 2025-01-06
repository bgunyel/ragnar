from enum import Enum
from typing import TypedDict
from uuid import uuid4

from PIL import Image
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ragnar.backend.components.answer_generator import get_answer_generator
from ragnar.backend.components.retrieval_grader import get_retrieval_grader
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


class Nodes(Enum):
    RETRIEVE = 'retrieve'
    GENERATE = 'generate'


class BaseRAG:
    def __init__(self, retriever: VectorStoreRetriever):
        self.retriever = retriever
        self.rag_generator = get_answer_generator(model_name=settings.MODEL)
        self.graph = self.build_graph()

    def get_flow_chart(self):
        img_bytes = self.graph.get_graph(xray=True).draw_mermaid_png()
        img = Image.open(img_bytes).convert("RGB")
        return img

    def get_response(self, question: str) -> str:
        config = {"configurable": {"thread_id": str(uuid4())}}
        state_dict = self.graph.invoke({"question": question, "steps": []}, config)
        return state_dict["generation"]

    def retrieve(self, state:GraphState) -> GraphState:
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """

        question = state["question"]
        documents = self.retriever.invoke(question)
        steps = state["steps"]
        steps.append("retrieve_documents")

        return {
            "question": question,
            "documents": documents,
            "generation": '',
            'documents_grade': '',
            "steps": steps
        }


    def generate(self, state: GraphState) -> GraphState:
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        question = state["question"]
        documents = state["documents"]
        context = "\n\n".join(doc.page_content for doc in documents)
        generation = self.rag_generator.invoke({"documents": context, "question": question})
        steps = state["steps"]
        steps.append("generate_answer")

        return {
            "question": question,
            "documents": documents,
            "generation": generation,
            "documents_grade": state["documents_grade"],
            "steps": steps,
        }

    def build_graph(self) -> CompiledStateGraph:

        workflow = StateGraph(GraphState)

        # Nodes
        workflow.add_node(node=Nodes.RETRIEVE.value, action=self.retrieve)
        workflow.add_node(node=Nodes.GENERATE.value, action=self.generate)

        # Edges
        workflow.add_edge(start_key=START, end_key=Nodes.RETRIEVE.value)
        workflow.add_edge(start_key=Nodes.RETRIEVE.value, end_key=Nodes.GENERATE.value)
        workflow.add_edge(start_key=Nodes.GENERATE.value, end_key=END)

        compiled_graph = workflow.compile()
        return compiled_graph
