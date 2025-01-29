import asyncio

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableConfig
from tavily import AsyncTavilyClient

from ragnar.config import settings
from ragnar.backend.tools import tavily_search_async
from ragnar.backend.utils import deduplicate_and_format_sources
from ragnar.backend.researcher.enums import Node
from ragnar.backend.researcher.state import ReportState
from ragnar.backend.researcher.configuration import Configuration


QUERY_WRITER_INSTRUCTIONS = """You are an expert technical writer, helping to plan a report. 

The report will be focused on the following topic:

{topic}

The report structure will follow these guidelines:

{report_organization}

Your goal is to generate exactly {number_of_queries} search queries that will help gather comprehensive information for planning the report sections. 

Each query should:

1. Be related to the topic 
2. Help satisfy the requirements specified in the report organization

Make the queries specific enough to find high-quality, relevant sources while covering the breadth needed for the report structure.

Return the queries as a JSON object:

{{
    queries: [
            {{
                "query": "string",
                "aspect": "string",
                "rationale": "string"
            }}
    ]
}}
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
- Research - Whether to perform web research for this section of the report (binary score 'yes' or 'no').
- Content - The content of the section, which you will leave blank for now.

Consider which sections require web research. 
For example, introduction and conclusion will not require research because they will distill information from other parts of the report.

Return the sections of the report as a JSON object:

{{
    sections: [
            {{
                "name": "string",
                "description": "string",
                "research": "string",
                "content": "string",
            }}
    ]
}}

"""


class Planner:
    def __init__(self, model_name: str):
        self.query_writer_llm = ChatOllama(
            model=model_name,
            temperature=0,
            base_url=settings.OLLAMA_URL,
            format='json'
        ) | JsonOutputParser()

        self.planner_llm = ChatOllama(
            model=model_name,
            temperature=0,
            base_url=settings.OLLAMA_URL,
            format='json'
        ) | JsonOutputParser()

        self.event_loop = asyncio.get_event_loop()

    def query_related_sources(self, topic: str, configurable: Configuration):
        query_writer_instructions = QUERY_WRITER_INSTRUCTIONS.format(topic=topic,
                                                              report_organization=configurable.report_structure,
                                                              number_of_queries=configurable.number_of_queries)

        results = self.query_writer_llm.invoke(
            [
                SystemMessage(content=query_writer_instructions),
                HumanMessage(content="Generate search queries that will help with planning the sections of the report.")
            ]
        )

        search_docs = self.event_loop.run_until_complete(
            tavily_search_async(
                client = AsyncTavilyClient(api_key=settings.TAVILY_API_KEY),
                search_queries = [x['query'] for x in results['queries']],
                search_category = configurable.search_category,
                number_of_days_back = configurable.number_of_days_back
            )
        )

        source_str = deduplicate_and_format_sources(search_response=search_docs,
                                                    max_tokens_per_source=1000,
                                                    include_raw_content=False)

        return source_str


    def run(self, state: ReportState, config: RunnableConfig) -> ReportState:
        """
        Does the research planning.
            :param state: The current flow state
            :param config: The configuration
        """

        configurable = Configuration.from_runnable_config(config=config)
        state.steps.append(Node.PLANNER.value)

        source_str = self.query_related_sources(topic=state.topic, configurable=configurable)

        planning_instructions = PLANNER_INSTRUCTIONS.format(topic=state.topic,
                                                     report_organization=configurable.report_structure,
                                                     context=source_str)

        report_sections = self.planner_llm.invoke(
            [
                SystemMessage(content=planning_instructions),
                HumanMessage(content=("Generate the sections of the report. "
                         "Your response must include a 'sections' field containing a list of sections. "
                         "Each section must have: name, description, plan, research, and content fields."))
            ]
        )




        return state
