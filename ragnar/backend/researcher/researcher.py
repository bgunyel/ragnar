from uuid import uuid4
from pprint import pprint

from langgraph.graph import START, END, StateGraph

from ragnar.config import settings
from ragnar.backend.researcher.state import ReportState
from ragnar.backend.researcher.configuration import Configuration
from ragnar.backend.researcher.enums import Node
from ragnar.backend.researcher.components.planner import Planner


class Researcher:

    """
    Reproduction of https://github.com/langchain-ai/report-mAIstro with Ollama
    """

    def __init__(self):
        self.planner = Planner(model_name=settings.MODEL)
        self.graph = self.build_graph()

    def get_response(self, question: str, verbose: bool = False) -> str:
        config = {"configurable": {"thread_id": str(uuid4())}}

        in_state = ReportState(
            topic=question,
            steps=[],
        )
        out_state = self.graph.invoke(in_state, config)

        return out_state


    def build_graph(self):
        workflow = StateGraph(ReportState, config_schema=Configuration)

        ## Nodes
        workflow.add_node(node=Node.PLANNER.value, action=self.planner.run)

        ## Edges
        workflow.add_edge(start_key=START, end_key=Node.PLANNER.value)
        workflow.add_edge(start_key=Node.PLANNER.value, end_key=END)

        compiled_graph = workflow.compile()
        return compiled_graph

