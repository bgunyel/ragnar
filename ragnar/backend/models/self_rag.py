from typing import Literal
from uuid import uuid4
from pprint import pprint

from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.graph import START, END, StateGraph

from ragnar.backend.components.answer_generator import AnswerGenerator
from ragnar.backend.components.question_rewriter import QuestionRewriter
from ragnar.backend.components.retrieval_grader import RetrievalGrader
from ragnar.backend.components.retriever import Retriever
from ragnar.backend.components.router import Router
from ragnar.backend.components.hallucination_grader import HallucinationGrader
from ragnar.backend.components.answer_grader import AnswerGrader
from ragnar.backend.components.utility_components import (
    are_documents_relevant,
    is_answer_grounded,
    is_answer_useful,
    reset_generation
)
from ragnar.backend.enums import Node
from ragnar.backend.models_config import Configuration
from ragnar.backend.state import GraphState
from ragnar.config import settings


def route_query(state: GraphState) -> Literal['vectorstore', 'internal']:
    """
    Determines the path after query routing

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    if state.datasource == 'vectorstore':
        return 'vectorstore'
    else:
        return 'internal'


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

    def get_response(self, question: str, verbose: bool = False) -> str:

        if verbose:
            in_state = GraphState(question=question,
                                  query=question,
                                  documents=[],
                                  good_documents=[],
                                  generation='',
                                  documents_grade='',
                                  steps=[],
                                  retrieval_iteration=0,
                                  generation_iteration=0,
                                  datasource='',
                                  answer_useful='',
                                  answer_grounded='')

            for output in self.graph.stream(in_state):
                for key, value in output.items():
                    # Node
                    pprint(f"Node '{key}':")
                    # Optional: print full state at each node
                    pprint(value, indent=2, width=80, depth=None)
                pprint("\n---\n")

        config = {"configurable": {"thread_id": str(uuid4())}}
        in_state = GraphState(question=question,
                              query=question,
                              documents=[],
                              good_documents= [],
                              generation= '',
                              documents_grade= '',
                              steps= [],
                              retrieval_iteration= 0,
                              generation_iteration= 0,
                              datasource= '',
                              answer_useful= '',
                              answer_grounded= '')
        out_state = self.graph.invoke(in_state, config)
        return out_state['generation']


    def build_graph(self):
        workflow = StateGraph(GraphState, config_schema=Configuration)

        # Nodes
        workflow.add_node(node=Node.ROUTER.value, action=self.router.run)
        workflow.add_node(node=Node.INTERNAL_ANSWER_GENERATOR.value, action=self.internal_answer_generator.run)
        workflow.add_node(node=Node.RETRIEVE.value, action=self.retriever.run)
        workflow.add_node(node=Node.DOCUMENT_GRADER.value, action=self.retrieval_grader.run)
        workflow.add_node(node=Node.ANSWER_GENERATOR.value, action=self.answer_generator.run)
        workflow.add_node(node=Node.HALLUCINATION_GRADER.value, action=self.hallucination_grader.run)
        workflow.add_node(node=Node.ANSWER_GRADER.value, action=self.answer_grader.run)
        workflow.add_node(node=Node.REWRITE_QUESTION.value, action=self.question_rewriter.run)
        workflow.add_node(node=Node.RESET.value, action=reset_generation)

        # Edges
        workflow.add_edge(start_key=START, end_key=Node.ROUTER.value)
        workflow.add_conditional_edges(
            source=Node.ROUTER.value,
            path=route_query,
            path_map={
                'vectorstore': Node.RETRIEVE.value,
                'internal': Node.INTERNAL_ANSWER_GENERATOR.value,
            }
        )
        workflow.add_edge(start_key=Node.INTERNAL_ANSWER_GENERATOR.value, end_key=END)
        workflow.add_edge(start_key=Node.RETRIEVE.value, end_key=Node.DOCUMENT_GRADER.value)
        workflow.add_conditional_edges(
            source=Node.DOCUMENT_GRADER.value,
            path=are_documents_relevant,
            path_map={
                'relevant': Node.ANSWER_GENERATOR.value,
                'not relevant': Node.REWRITE_QUESTION.value,
                'max_iter':Node.ANSWER_GENERATOR.value
            }
        )

        workflow.add_edge(start_key=Node.REWRITE_QUESTION.value, end_key=Node.RETRIEVE.value)
        workflow.add_edge(start_key=Node.ANSWER_GENERATOR.value, end_key=Node.HALLUCINATION_GRADER.value)
        workflow.add_conditional_edges(
            source=Node.HALLUCINATION_GRADER.value,
            path=is_answer_grounded,
            path_map={
                'grounded': Node.ANSWER_GRADER.value,
                'not grounded': Node.ANSWER_GENERATOR.value,
                'max_iter': Node.ANSWER_GRADER.value,
            }
        )
        workflow.add_conditional_edges(
            source=Node.ANSWER_GRADER.value,
            path=is_answer_useful,
            path_map={
                'useful': END,
                'not useful': Node.REWRITE_QUESTION.value,
                'max_iter': Node.RESET.value,
            }
        )

        workflow.add_edge(start_key=Node.RESET.value, end_key=END)

        compiled_graph = workflow.compile()
        return compiled_graph

