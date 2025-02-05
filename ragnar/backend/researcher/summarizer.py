from uuid import uuid4
from pprint import pprint
from typing import Literal

from langgraph.graph import START, END, StateGraph
from langchain_core.runnables import RunnableConfig
from opentelemetry import trace

from ragnar.config import settings
from ragnar.backend.tools import tracer
from ragnar.backend.researcher.state import SummaryState
from ragnar.backend.researcher.configuration import Configuration
from ragnar.backend.researcher.enums import Node
from ragnar.backend.researcher.components.query_writer import QueryWriter
from ragnar.backend.researcher.components.web_search import WebSearch
from ragnar.backend.researcher.components.summary_writer import SummaryWriter
from ragnar.backend.researcher.components.summary_reviewer import SummaryReviewer


def route_research(state: SummaryState, config: RunnableConfig) -> Literal["continue_research", "end_research"]:
    """ Route the research based on the follow-up query """

    configurable = Configuration.from_runnable_config(config=config)
    if state.iteration <= configurable.research_iterations:
        return "continue_research"
    else:
        return "end_research"


class Summarizer:

    """
    Reproduction from https://github.com/langchain-ai/report-mAIstro with Ollama
    """

    def __init__(self):
        config = Configuration()
        self.query_writer = QueryWriter(model_name=settings.REASONING_MODEL, context_window_length=config.context_window_length)
        self.web_search = WebSearch()
        self.summary_writer = SummaryWriter(model_name=settings.LANGUAGE_MODEL, context_window_length=config.context_window_length)
        self.summary_reviewer = SummaryReviewer(model_name=settings.REASONING_MODEL, context_window_length=config.context_window_length)

        self.graph = self.build_graph()

    @tracer.start_as_current_span('summarizer')
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
        workflow.add_node(node=Node.SUMMARY_REVIEWER.value, action=self.summary_reviewer.run)

        ## Edges
        workflow.add_edge(start_key=START, end_key=Node.QUERY_WRITER.value)
        workflow.add_edge(start_key=Node.QUERY_WRITER.value, end_key=Node.WEB_SEARCH.value)
        workflow.add_edge(start_key=Node.WEB_SEARCH.value, end_key=Node.SUMMARY_WRITER.value)
        workflow.add_edge(start_key=Node.SUMMARY_WRITER.value, end_key=Node.SUMMARY_REVIEWER.value)

        workflow.add_conditional_edges(
            source=Node.SUMMARY_REVIEWER.value,
            path=route_research,
            path_map={
                'continue_research': Node.WEB_SEARCH.value,
                'end_research': END,
            }
        )

        compiled_graph = workflow.compile()
        return compiled_graph
