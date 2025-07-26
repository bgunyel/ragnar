import asyncio
from uuid import uuid4
from typing import Any, Literal
import json
import copy

from langchain.chat_models import init_chat_model
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.callbacks import get_usage_metadata_callback
from supabase import create_client, Client

from business_researcher import BusinessResearcher, SearchType
from .state import AgentState
from .configuration import Configuration
from .enums import Node, Table
from .tools import (
    ResearchPerson,
    ResearchCompany,
    InsertCompanyToDataBase,
    InsertPersonToDataBase,
    FetchCompanyFromDataBase,
    FetchPersonFromDataBase,
)

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
You are a smart and helpful business intelligence assistant. Your name is Bia.
"""

def should_continue(state: AgentState) -> Literal['continue', 'end']:
    # If the last message is not a tool call, then we finish
    if len(state.messages[-1].tool_calls) == 0:
        return "end"
    else:
        return "continue"

class BusinessIntelligenceAgent:
    def __init__(self,
                 llm_config: dict[str, Any],
                 web_search_api_key: str,
                 database_url: str,
                 database_key: str):
        self.memory_saver = MemorySaver()
        self.models = list({*[v['model'] for k, v in llm_config.items()]})
        self.business_researcher = BusinessResearcher(llm_config = llm_config, web_search_api_key = web_search_api_key)
        self.db_client: Client = create_client(supabase_url=database_url, supabase_key=database_key)
        self.message_memory = []

        model_params = llm_config['reasoning_model']
        base_llm = init_chat_model(
            model=model_params['model'],
            model_provider=model_params['model_provider'],
            api_key=model_params['api_key'],
            **model_params['model_args']
        )
        self.model_name = model_params['model']
        self.structured_llm = base_llm.bind_tools(tools = [ResearchPerson, ResearchCompany,
                                                           InsertCompanyToDataBase, InsertPersonToDataBase,
                                                           FetchCompanyFromDataBase, FetchPersonFromDataBase])

        self.graph = self.build_graph()
        self.message_memory.append(SystemMessage(content=AGENT_INSTRUCTIONS))


    def run(self, query: str):
        self.message_memory.append(HumanMessage(content=query))
        in_state = AgentState(
            messages = self.message_memory,
            token_usage = {m: {'input_tokens': 0, 'output_tokens': 0} for m in self.models},
        )

        config = {"configurable": {"thread_id": '1'}}

        out_state = self.graph.invoke(in_state, config)
        self.message_memory = out_state['messages']
        dummy = -32

    def llm_call(self, state: AgentState) -> AgentState:
        with get_usage_metadata_callback() as cb:
            response = self.structured_llm.invoke(state.messages)
            state.token_usage[self.model_name]['input_tokens'] += cb.usage_metadata[self.model_name]['input_tokens']
            state.token_usage[self.model_name]['output_tokens'] += cb.usage_metadata[self.model_name]['output_tokens']
            state.messages.extend([response])
        return state

    def tools_call(self, state: AgentState) -> AgentState:

        for tool_call in state.messages[-1].tool_calls:

            tool_message_content = f"Tool message content NOT INITIALIZED: {tool_call['name']}"

            match tool_call['name']:
                case 'ResearchPerson':
                    research_required = False

                    # First fetch the current company of the person
                    name = tool_call['args']['name']
                    company_name = tool_call['args']['company']
                    response = self.fetch_company_from_db(company_name=company_name)
                    if len(response) > 0:
                        company = response[0]
                        current_company_id = company['id']
                        response = self.fetch_person_from_db(name=name, current_company_id=current_company_id)
                        if len(response) > 0:
                            person = response[0]
                            tool_message_content = json.dumps(person, indent=2)
                        else:
                            research_required = True
                    else:
                        research_required = True

                    if research_required:
                        state, out_dict = self.research_person(name=tool_call['args']['name'], company=tool_call['args']['company'], state=state)
                        tool_message_content = json.dumps(out_dict['content'], indent=2)

                case 'ResearchCompany':
                    response = self.fetch_company_from_db(company_name=tool_call['args']['company_name'])
                    if len(response) > 0:
                        company = response[0]
                        tool_message_content = "Information fetched from database:\n\n" + json.dumps(company, indent=2)
                    else:
                        state, out_dict = self.research_company(company_name=tool_call['args']['company_name'], state=state)
                        tool_message_content = json.dumps(out_dict['content'], indent=2)

                case 'FetchCompanyFromDataBase':
                    response = self.fetch_company_from_db(company_name=tool_call['args']['company_name'])
                    if len(response) > 0:
                        company = response[0]
                        tool_message_content = json.dumps(company, indent=2)
                    else:
                        tool_message_content = f"There is no record for {tool_call['args']['company_name']} in database."

                case 'FetchPersonFromDataBase':
                    # First fetch the current company of the person
                    name = tool_call['args']['name']
                    company_name = tool_call['args']['company']
                    response = self.fetch_company_from_db(company_name = company_name)
                    if len(response) > 0:
                        company = response[0]
                        current_company_id = company['id']
                        response = self.fetch_person_from_db(name = name, current_company_id = current_company_id)
                        if len(response) > 0:
                            person = response[0]
                            tool_message_content = json.dumps(person, indent=2)
                        else:
                            tool_message_content = f"There is no record for {name} from {company_name} in database."
                    else:
                        tool_message_content = (
                                f"There is no record for {tool_call['args']['company']} in database.\n\n" +
                                f"For a person to be successfully inserted into the database, first his/her current company should be inserted."
                        )

                case 'InsertCompanyToDataBase':
                    # Check whether the company record exists in the DB
                    response = self.fetch_company_from_db(company_name=tool_call['args']['company_name'])
                    if len(response) > 0:
                        company = response[0]
                        tool_message_content = f"Company {tool_call['args']['company_name']} already exists in database with id: {company['id']}"
                    else:
                        id = self.insert_company_to_db(input_dict=tool_call['args'])
                        tool_message_content = f"{tool_call['args']['name']} successfully inserted into database {Table.COMPANIES} table with id {id}"

                case 'InsertPersonToDataBase':

                    name = tool_call['args']['name']
                    current_company = tool_call['args']['current_company']

                    # First check whether the company of the person exists in database
                    response = self.fetch_company_from_db(company_name=current_company)
                    if len(response) > 0:
                        company = response[0]
                        current_company_id = company['id'] # Company exists in the database

                        response = self.fetch_person_from_db(name=name, current_company_id=current_company_id)
                        if len(response) > 0:
                            person = response[0]  # Person exists in the database
                            tool_message_content = f"{name} from {current_company} already exist in the database with id: {person['id']}."
                        else:
                            id = self.insert_person_to_db(input_dict=tool_call['args'],
                                                          current_company_id=current_company_id)
                            tool_message_content = f"{name} from {current_company} successfully inserted into database {Table.PERSONS} table with id {id}"

                    else: # No company, no person in the database
                        state, out_dict = self.research_company(company_name=tool_call['args']['company_name'], state=state)
                        current_company_id = self.insert_company_to_db(input_dict=out_dict['content'])
                        id = self.insert_person_to_db(input_dict=tool_call['args'], current_company_id=current_company_id)
                        tool_message_content = f"{tool_call['args']['name']} successfully inserted into database {Table.PERSONS} table with id {id}"

                case _:
                    tool_message_content = f"Unknown tool call: {tool_call['name']}"
                    # raise RuntimeError('Unknown tool call')

            state.messages.append(
                ToolMessage(
                    content=tool_message_content,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
            ))

        return state

    def research_person(self, name: str, company: str, state: AgentState) -> (AgentState, dict[str, Any]):
        input_dict = {
            "name": name,
            "company": company,
            'search_type': SearchType.PERSON
        }
        out_dict = self.run_research_loop(input_dict=input_dict)
        state = self.update_token_usage(state=state, token_usage=out_dict['token_usage'])
        return state, out_dict

    def research_company(self, company_name: str, state: AgentState) -> (AgentState, dict[str, Any]):
        input_dict = {
            "name": company_name,
            'search_type': SearchType.COMPANY
        }
        out_dict = self.run_research_loop(input_dict=input_dict)
        state = self.update_token_usage(state=state, token_usage=out_dict['token_usage'])
        return state, out_dict

    def update_token_usage(self, state: AgentState, token_usage: dict[str, Any]) -> AgentState:
        for m in self.models:
            state.token_usage[m]['input_tokens'] += token_usage[m]['input_tokens']
            state.token_usage[m]['output_tokens'] += token_usage[m]['output_tokens']
        return state

    def run_research_loop(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        event_loop = asyncio.new_event_loop()
        out_dict = event_loop.run_until_complete(self.business_researcher.run(input_dict=input_dict, config=CONFIG))
        event_loop.close()
        return out_dict

    def insert_company_to_db(self, input_dict: dict[str, Any]):
        row_dict = copy.deepcopy(input_dict)
        row_dict['created_by_id'] = 1
        row_dict['updated_by_id'] = 1

        response = (
            self.db_client.table(Table.COMPANIES)
            .insert(row_dict)
            .execute()
        )
        id = response.data[0]['id']
        return id

    def fetch_company_from_db(self, company_name: str) -> list[dict[str, Any]]:
        response = (
            self.db_client.table(Table.COMPANIES)
            .select("*")
            .eq("name", company_name)
            .execute()
        )
        return response.data

    def insert_person_to_db(self, input_dict: dict[str, Any], current_company_id: int):
        row_dict = copy.deepcopy(input_dict)
        row_dict['created_by_id'] = 1
        row_dict['updated_by_id'] = 1
        row_dict['current_company_id'] = current_company_id
        row_dict.pop('current_company')

        response = (
            self.db_client.table(Table.PERSONS)
            .insert(row_dict)
            .execute()
        )
        id = response.data[0]['id']
        return id

    def fetch_person_from_db(self, name: str, current_company_id: int | None) -> list[dict[str, Any]]:

        if current_company_id is None:
            response = (
                self.db_client.table(Table.PERSONS)
                .select("*")
                .eq("name", name)
                .execute()
            )
        else:
            response = (
                self.db_client.table(Table.PERSONS)
                .select("*")
                .eq("name", name)
                .eq("current_company_id", current_company_id)
                .execute()
            )
        return response.data

    def build_graph(self):
        workflow = StateGraph(AgentState, config_schema=Configuration)

        ## Nodes
        workflow.add_node(node=Node.LLM_CALL, action=self.llm_call)
        workflow.add_node(node=Node.TOOLS_CALL, action=self.tools_call)

        ## Edges
        workflow.add_edge(start_key=START, end_key=Node.LLM_CALL)
        workflow.add_edge(start_key=Node.TOOLS_CALL, end_key=Node.LLM_CALL)
        workflow.add_conditional_edges(
            source=Node.LLM_CALL,
            path=should_continue,
            path_map={
                "continue": Node.TOOLS_CALL,
                "end": END,
            },
        )

        ## Compile graph
        compiled_graph = workflow.compile(checkpointer=self.memory_saver)
        return compiled_graph
