from uuid import uuid4
from pprint import pprint

from langgraph.graph import START, END, StateGraph

from ragnar.config import settings
from ragnar.backend.researcher.state import SummaryState
from ragnar.backend.researcher.configuration import Configuration
from ragnar.backend.researcher.enums import Node
from ragnar.backend.researcher.components.query_writer import QueryWriter
from ragnar.backend.researcher.components.web_search import WebSearch
from ragnar.backend.researcher.components.summary_writer import SummaryWriter


class Summarizer:

    """
    Reproduction from https://github.com/langchain-ai/report-mAIstro with Ollama
    """

    def __init__(self):
        self.query_writer = QueryWriter(model_name=settings.REASONING_MODEL)
        self.web_search = WebSearch()
        self.summary_writer = SummaryWriter(model_name=settings.LANGUAGE_MODEL)

        self.graph = self.build_graph()

    def get_response(self, topic: str, verbose: bool = False) -> str:
        config = {"configurable": {"thread_id": str(uuid4())}}

        in_state = SummaryState(
            topic = topic,
            search_queries = [],
            source_str = '',
            content = '',
            steps = [],
        )
        out_state = self.graph.invoke(in_state, config)
        return out_state['content']


    def build_graph(self):
        workflow = StateGraph(SummaryState, config_schema=Configuration)

        ## Nodes
        workflow.add_node(node=Node.QUERY_WRITER.value, action=self.query_writer.run)
        workflow.add_node(node=Node.WEB_SEARCH.value, action=self.web_search.run)
        workflow.add_node(node=Node.SUMMARY_WRITER.value, action=self.summary_writer.run)

        ## Edges
        workflow.add_edge(start_key=START, end_key=Node.QUERY_WRITER.value)
        workflow.add_edge(start_key=Node.QUERY_WRITER.value, end_key=Node.WEB_SEARCH.value)
        workflow.add_edge(start_key=Node.WEB_SEARCH.value, end_key=Node.SUMMARY_WRITER.value)
        workflow.add_edge(start_key=Node.SUMMARY_WRITER.value, end_key=END)

        compiled_graph = workflow.compile()
        return compiled_graph
