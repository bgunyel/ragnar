import asyncio

from langchain_core.runnables import RunnableConfig
from tavily import AsyncTavilyClient
from opentelemetry import trace

from ragnar.config import settings
from ragnar.backend.tools import tracer
from ragnar.backend.researcher.state import SummaryState
from ragnar.backend.tools import tavily_search_async
from ragnar.backend.utils import deduplicate_and_format_sources
from ragnar.backend.researcher.enums import Node
from ragnar.backend.researcher.configuration import Configuration

class WebSearch:
    def __init__(self):
        self.event_loop = asyncio.get_event_loop()

    @tracer.start_as_current_span('web_search')
    def run(self, state: SummaryState, config: RunnableConfig) -> SummaryState:
        configurable = Configuration.from_runnable_config(config=config)

        search_docs = self.event_loop.run_until_complete(
            tavily_search_async(
                client=AsyncTavilyClient(api_key=settings.TAVILY_API_KEY),
                search_queries=state.search_queries,
                search_category=configurable.search_category,
                number_of_days_back=configurable.number_of_days_back
            )
        )

        source_str = deduplicate_and_format_sources(search_response=search_docs,
                                                    max_tokens_per_source=5000,
                                                    include_raw_content=False)

        state.steps.append(Node.WEB_SEARCH.value)
        state.source_str = source_str

        span = trace.get_current_span()
        span.set_status(trace.StatusCode.OK)
        span.set_attributes(
            attributes={
                'topic': state.topic,
                'search_queries': state.search_queries,
                'iteration': state.iteration,
                'source_string': state.source_str,
            }
        )

        return state
