import asyncio
from pprint import pprint

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableConfig
from tavily import AsyncTavilyClient

from ragnar.backend.qa_rag.components.utility_components import reset_generation
from ragnar.config import settings
from ragnar.backend.tools import tavily_search_async
from ragnar.backend.utils import deduplicate_and_format_sources
from ragnar.backend.researcher.enums import Node
from ragnar.backend.researcher.state import SummaryState
from ragnar.backend.researcher.configuration import Configuration


REVIEW_INSTRUCTIONS = """You are an expert research assistant analyzing a summary about {topic}.

Here is the summary:
{summary}

Your tasks:
1. Identify knowledge gaps or areas that need deeper exploration in the summary
2. Generate follow-up questions that would help expand your understanding
3. Focus on details about the topic, or emerging trends that weren't fully covered
4. Generate stand-alone web queries from the follow-up questions

Ensure the follow-up questions are self-contained and include necessary context for web search.
Convert each follow-up question to a stand-alone web search query

Format your response as a JSON object with three fields:
- reasoning: all the reasoning you do
- knowledge_gap: Describe what information is missing or needs clarification
- follow-up questions: Write specific questions to address this gap
- queries: Convert the follow-up questions to stand-alone web search queries

Provide your analysis in JSON format:

{{
    "reasoning": "string"
    "knowledge_gap": "string",
    "follow-up questions": [
            {{
                "question": "string",                
            }}
    ]
    "queries": [
            {{
                "query": "string",                
            }}
    ]
}}
"""


class SummaryReviewer:
    def __init__(self, model_name: str, context_window_length: int):
        self.reviewer_llm = ChatOllama(
            model=model_name,
            temperature=0,
            base_url=settings.OLLAMA_URL,
            format='json',
            num_ctx=context_window_length
        ) | JsonOutputParser()

    def run(self, state: SummaryState) -> SummaryState:

        instructions = REVIEW_INSTRUCTIONS.format(topic = state.topic, summary=state.content)

        result = self.reviewer_llm.invoke(
            [
                SystemMessage(content=instructions),
            ]
        )

        state.steps.append(Node.SUMMARY_REVIEWER.value)
        state.search_queries = [x['query'] for x in result['queries']]
        state.iteration += 1

        print(f'Iteration: {state.iteration}')
        pprint(state.content)
        pprint(result)

        return state
