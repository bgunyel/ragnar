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


class FeedbackBaseRAG:
    def __init__(self, retriever: VectorStoreRetriever):

        self.K = retriever.search_kwargs['k'] # store the K parameter
        self.retriever = Retriever(retriever=retriever)
        self.retrieval_grader = RetrievalGrader(model_name=settings.MODEL)
        self.question_rewriter = QuestionRewriter(model_name=settings.MODEL)
        self.answer_generator = AnswerGenerator(model_name=settings.MODEL)
        self.graph = self.build_graph()

    def get_response(self, question: str, verbose: bool = False) -> str:

        if verbose:
            for output in self.graph.stream({
                "question": question,
                'original_question': question,
                'documents': [],
                'good_documents': [],
                'generation': '',
                'documents_grade': '',
                "steps": [],
                "iteration": 0,
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
                "question": question,
                'original_question': question,
                'documents': [],
                'good_documents': [],
                'generation': '',
                'documents_grade': '',
                "steps": [],
                "iteration": 0,
            }, config
        )
        return state_dict["generation"]

    def increment_iteration(self, state: GraphState):
        state["iteration"] += 1
        return state

    def decide_to_generate(self, state: GraphState) -> str:
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        if (state['iteration'] > 3) or (len(state['good_documents']) >= self.K):
            out = Nodes.GENERATE.value
        else:
            out = Nodes.REWRITE_QUESTION.value

        return out


    def build_graph(self) -> CompiledStateGraph:

        workflow = StateGraph(GraphState)

        # Nodes
        workflow.add_node(node=Nodes.RETRIEVE.value, action=self.retriever.run)
        workflow.add_node(node=Nodes.GRADE_DOCS.value, action=self.retrieval_grader.run)
        workflow.add_node(node=Nodes.REWRITE_QUESTION.value, action=self.question_rewriter.run)
        workflow.add_node(node=Nodes.INCREMENT_ITERATION.value, action=self.increment_iteration)
        workflow.add_node(node=Nodes.GENERATE.value, action=self.answer_generator.run)

        # Edges
        workflow.add_edge(start_key=START, end_key=Nodes.RETRIEVE.value)
        workflow.add_edge(start_key=Nodes.RETRIEVE.value, end_key=Nodes.INCREMENT_ITERATION.value)
        workflow.add_edge(start_key=Nodes.INCREMENT_ITERATION.value, end_key=Nodes.GRADE_DOCS.value)
        workflow.add_conditional_edges(
            source=Nodes.GRADE_DOCS.value,
            path=self.decide_to_generate,
        )

        workflow.add_edge(start_key=Nodes.REWRITE_QUESTION.value, end_key=Nodes.RETRIEVE.value)
        workflow.add_edge(start_key=Nodes.GENERATE.value, end_key=END)

        compiled_graph = workflow.compile()
        return compiled_graph
