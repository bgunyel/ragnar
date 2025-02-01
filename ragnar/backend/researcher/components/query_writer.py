from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableConfig

from ragnar.config import settings
from ragnar.backend.researcher.enums import Node
from ragnar.backend.researcher.state import SummaryState
from ragnar.backend.researcher.configuration import Configuration


QUERY_WRITER_INSTRUCTIONS = """
Your goal is to generate targeted web search queries that will gather comprehensive information for writing a summary about the following topic:

{topic}

When generating the search queries, ensure they:
1. Cover different aspects of the topic (e.g., core features, real-world applications, technical architecture)
2. Include specific technical terms related to the topic
3. Target recent information by including year markers where relevant (e.g., "2024")
4. Look for comparisons or differentiators from similar technologies/approaches
5. Search for both official documentation and practical implementation examples

Your queries should be:
- Specific enough to avoid generic results
- Technical enough to capture detailed implementation information
- Diverse enough to cover all aspects of the summary plan
- Focused on authoritative sources (documentation, technical blogs, academic papers, reputable news sources, etc)

You will generate exactly {number_of_queries} queries.

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


class QueryWriter:
    def __init__(self, model_name: str, context_window_length: int):
        self.query_writer_llm = ChatOllama(
            model=model_name,
            temperature=0,
            base_url=settings.OLLAMA_URL,
            format='json',
            num_ctx=context_window_length,
        ) | JsonOutputParser()

    def run(self, state: SummaryState, config: RunnableConfig) -> SummaryState:
        """
        Writes queries for comprehensive web search.
            :param state: The current flow state
            :param config: The configuration
        """
        configurable = Configuration.from_runnable_config(config=config)
        state.steps.append(Node.QUERY_WRITER.value)

        query_writer_instructions = QUERY_WRITER_INSTRUCTIONS.format(topic=state.topic,
                                                                     number_of_queries=configurable.number_of_queries)
        results = self.query_writer_llm.invoke(
            [
                SystemMessage(content=query_writer_instructions),
                HumanMessage(content="Generate search queries that will help with writing the summary.")
            ]
        )

        state.search_queries = [x['query'] for x in results['queries']]
        return state
