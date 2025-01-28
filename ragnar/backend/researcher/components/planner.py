import asyncio

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableConfig
from tavily import AsyncTavilyClient

from ragnar.config import settings
from ragnar.backend.base import Queries
from ragnar.backend.researcher.enums import Node
from ragnar.backend.researcher.state import Sections, ReportState
from ragnar.backend.researcher.configuration import Configuration
from ragnar.backend.tools import tavily_search_async


QUERY_WRITER_INSTRUCTIONS = """You are an expert technical writer, helping to plan a report. 

The report will be focused on the following topic:

{topic}

The report structure will follow these guidelines:

{report_organization}

Your goal is to generate {number_of_queries} search queries that will help gather comprehensive information for planning the report sections. 

Each query should:

1. Be related to the topic 
2. Help satisfy the requirements specified in the report organization

Make the queries specific enough to find high-quality, relevant sources while covering the breadth needed for the report structure.
"""

PLANNER_INSTRUCTIONS = """You are an expert technical writer, helping to plan a report.

Your goal is to generate the outline of the sections of the report. 

The overall topic of the report is:

{topic}

The report should follow this organization: 

{report_organization}

You should reflect on this information to plan the sections of the report: 

{context}

Now, generate the sections of the report. Each section should have the following fields:

- Name - Name for this section of the report.
- Description - Brief overview of the main topics and concepts to be covered in this section.
- Research - Whether to perform web research for this section of the report.
- Content - The content of the section, which you will leave blank for now.

Consider which sections require web research. 
For example, introduction and conclusion will not require research because they will distill information from other parts of the report."""


class Planner:
    def __init__(self, model_name: str):
        # self.query_writer_llm = ChatOllama(model=model_name, temperature=0, base_url=settings.OLLAMA_URL, format='json')
        self.query_writer_llm = ChatOllama(model=model_name, temperature=0,
                                           base_url=settings.OLLAMA_URL).with_structured_output(schema=Queries)
        self.planner_llm = ChatOllama(model=model_name, temperature=0, base_url=settings.OLLAMA_URL).with_structured_output(schema=Sections)

        self.loop = asyncio.get_event_loop()

    def run(self, state: ReportState, config: RunnableConfig) -> ReportState:
        """
        Does the research planning.
            :param state: The current flow state
            :param config: The configuration
        """

        configurable = Configuration.from_runnable_config(config=config)
        state.steps.append(Node.PLANNER.value)

        query_writer_query = QUERY_WRITER_INSTRUCTIONS.format(topic=state.topic,
                                                              report_organization=configurable.report_structure,
                                                              number_of_queries=configurable.number_of_queries)

        results = self.query_writer_llm.invoke(
            [
                SystemMessage(content=query_writer_query),
                HumanMessage(content="Generate search queries that will help with planning the sections of the report.")
            ]
        )

        client = AsyncTavilyClient(api_key=settings.TAVILY_API_KEY)

        search_docs = self.loop.run_until_complete(
            tavily_search_async(client=client,
                                search_queries=results.queries,
                                search_category=configurable.search_category,
                                number_of_days_back=configurable.number_of_days_back)
        )

        return state
