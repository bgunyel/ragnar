from enum import Enum
from typing import TypedDict
from uuid import uuid4

from PIL import Image
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ragnar.backend.models.base_rag import BaseRAG, GraphState
from ragnar.backend.models.enums import Nodes
from ragnar.backend.components.answer_generator import get_answer_generator
from ragnar.backend.components.retrieval_grader import get_retrieval_grader
from ragnar.backend.components.question_rewriter import get_question_rewriter
from ragnar.config import settings


class State(GraphState):
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


class FeedbackBaseRAG(BaseRAG):
    def __init__(self, retriever: VectorStoreRetriever):
        super().__init__(retriever=retriever)
        self.retrieval_grader = get_retrieval_grader(model_name=settings.MODEL)
        self.question_rewriter = get_question_rewriter(model_name=settings.MODEL)

    def get_response(self, question: str) -> str:
        config = {"configurable": {"thread_id": str(uuid4())}}
        state_dict = self.graph.invoke({"question": question, "steps": []}, config)
        return state_dict["generation"]

    def grade_documents(self, state: GraphState) -> GraphState:
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        state["steps"].append("grade_document_retrieval")
        filtered_docs = []
        state['documents_grade'] = 'Good'

        for d in state['documents']:
            score = self.retrieval_grader.invoke({"question": state['question'], "documents": d.page_content})
            grade = score["score"]
            if grade == "yes":
                filtered_docs.append(d)
            else:
                state['documents_grade'] = "Bad"
                continue

        state['documents'] = filtered_docs
        return state


    def rewrite_question(self, state: GraphState) -> GraphState:
        # Re-write question
        new_question = self.question_rewriter.invoke({"question": state['question']})
        state['question'] = new_question
        return state

    def decide_to_generate(self, state: GraphState) -> str:
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        if state['documents_grade'] == 'Good':
            out = Nodes.GENERATE.value
        elif state['iteration'] < 3:
            out = Nodes.REWRITE_QUESTION.value
            state['iteration'] += 1
        else:
            out = Nodes.GENERATE.value

        return out


    def build_graph(self) -> CompiledStateGraph:

        workflow = StateGraph(GraphState)

        # Nodes
        workflow.add_node(node=Nodes.RETRIEVE.value, action=self.retrieve)
        workflow.add_node(node=Nodes.GRADE_DOCS.value, action=self.grade_documents)
        workflow.add_node(node=Nodes.REWRITE_QUESTION.value, action=self.rewrite_question)
        workflow.add_node(node=Nodes.GENERATE.value, action=self.generate)

        # Edges
        workflow.add_edge(start_key=START, end_key=Nodes.RETRIEVE.value)
        workflow.add_edge(start_key=Nodes.RETRIEVE.value, end_key=Nodes.GRADE_DOCS.value)
        workflow.add_conditional_edges(
            source=Nodes.GRADE_DOCS.value,
            path=self.decide_to_generate,
        )

        workflow.add_edge(start_key=Nodes.REWRITE_QUESTION.value, end_key=Nodes.RETRIEVE.value)
        workflow.add_edge(start_key=Nodes.GENERATE.value, end_key=END)

        compiled_graph = workflow.compile()
        return compiled_graph

