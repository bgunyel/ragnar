import asyncio
from uuid import uuid4
from typing import Any

from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from business_researcher import BusinessResearcher, SearchType


CONFIG = {
    "configurable": {
        'thread_id': str(uuid4()),
        'max_iterations': 3,
        'max_results_per_query': 4,
        'max_tokens_per_source': 10000,
        'number_of_days_back': 1e6,
        'number_of_queries': 3,
        }
    }

AGENT_INSTRUCTIONS = """
You are a smart and helpful business intelligence assistant.
"""

class BusinessIntelligenceAgent:
    def __init__(self, llm_config: dict[str, Any], web_search_api_key: str):
        self.business_researcher = BusinessResearcher(llm_config = llm_config, web_search_api_key = web_search_api_key)
        self.agent = self.build_agent(llm_config=llm_config)

    def run(self, query: str):
        config = {"configurable": {"thread_id": '1'}}
        out = self.agent.invoke(
            input={"messages": [{"role": "user", "content": query}]},
            config=config
        )
        dummy = -32
        return out['messages'][-1].content

    def run_event_loop(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        event_loop = asyncio.new_event_loop()
        out_dict = event_loop.run_until_complete(
            self.business_researcher.run(input_dict=input_dict, config=CONFIG)
        )
        event_loop.close()
        return out_dict

    def build_agent(self, llm_config: dict[str, Any]):

        @tool("research_person")
        def research_person(name: str, company: str) -> dict[str, Any]:
            """
            Research a specific person within a company using web search and AI analysis.

            Args:
                name (str): The full name of the person to research. Should be as specific
                           as possible to ensure accurate search results.
                company (str): The name of the company where the person works or is associated
                              with. This helps narrow the search scope and improve result relevance.
            """

            input_dict = {
                "name": name,
                "company": company,
                'search_type': SearchType.PERSON
            }
            return self.run_event_loop(input_dict = input_dict)

        @tool("research_company")
        def research_company(company_name: str) -> dict[str, Any]:
            """
            Research a company using comprehensive web search and AI analysis.

            Args:
                company_name (str): The name of the company to research. Should be the
                                   official company name or commonly recognized brand name
                                   to ensure accurate and comprehensive search results.
            """

            input_dict = {
                "name": company_name,
                'search_type': SearchType.COMPANY
            }
            return self.run_event_loop(input_dict = input_dict)

        # Agent Configuration
        model_params = llm_config['reasoning_model']
        base_llm = init_chat_model(
            model=model_params['model'],
            model_provider=model_params['model_provider'],
            api_key=model_params['api_key'],
            **model_params['model_args']
        )

        agent = create_react_agent(
            model=base_llm,
            tools=[research_person, research_company],
            checkpointer=MemorySaver(),
            prompt=AGENT_INSTRUCTIONS,
        )
        return agent
