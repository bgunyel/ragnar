import asyncio
import datetime
import time
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
from .enums import Node, Table, ColumnsBase, CompaniesColumns, PersonsColumns
from .tools import (
    ResearchPerson,
    ResearchCompany,
    InsertCompanyToDataBase,
    InsertPersonToDataBase,
    UpdateCompanyInDatabase,
    FetchCompanyFromDataBase,
    FetchPersonFromDataBase,
    ListAllPersonNamesFromDataBase,
    ListAllCompanyNamesFromDataBase,
    ListPersonsFromCompanyId,
)

AGENT_INSTRUCTIONS = """
You are a smart and helpful business intelligence assistant. Your name is Bia. You are a member of Team Ragnar.

<Task>
Your job is to use tools to perform user's commands and find information to answer user's questions.
You can use any of the tools provided to you. 
You can call these tools in series or in parallel, your functionality is conducted in a tool-calling loop.
</Task>

<Available Tools>
You have access to 6 main tools:
1. **ResearchPerson**: To research a specific person within a company using web search.
2. **ResearchCompany**: To research a company using web search.
3. **InsertCompanyToDataBase**: To insert company information to database.
4. **InsertPersonToDataBase**: To insert person information to database.
5. **FetchCompanyFromDataBase**: To get information about a company from the database.
6. **FetchPersonFromDataBase**: To get information about a person from the database.
7. **UpdateCompanyInDatabase**: To update already existing information about a company in the database. 
8. **ListAllPersonNamesFromDataBase**: To get the list of all person names in the database.
9. **ListAllCompanyNamesFromDataBase**: To get the list of all company names in the database.
10. **ListPersonsFromCompanyId**: To get the list of all persons in a given company.

**CRITICAL**:
* There are 2 tables in the database: persons and companies.
    - The "current_company_id" column of the persons table is linked to the "id" column of the companies table.
    - For example, if the company LangChain has id of 2 in the companies table, then the people working at LangChain have current_company_id equal to 2. 
* When the user wants to get information about a company:
    - First check whether there is an entry about the company in the database (by using FetchCompanyFromDataBase).
    - If there is information about the company in the database, return the information.
    - If there is not any information about the company in the database, perform a research about the company (by using ResearchCompany).
* When the user wants to get information about a person:
    - First check whether there is an entry about the person in the database (by using FetchPersonFromDataBase).
    - If there is information about the person in the database, return the information.
    - If there is not any information about the person in the database, perform a research about the person (by using ResearchPerson).
* When the user wants you to update the information about a specific company in the database:
    - Be sure to include the union of the alternative names information both from the existing database record, and the fresh information about the company.
    - If the name of the company is found as an alternative name in the new information, add the name of the company in the old record to the union of alternative names.
    - Be sure to include the union of key executives both from the existing database record, and the fresh information about the company.
    - Be sure to include the union of similar companies both from the existing database record, and the fresh information about the company.
    - Example: 
        + old record in the database: {
            name: 'Perplexity AI', 
            alternative_names: ['Perplexity', 'Perplexity Inc'], 
            key_executives: ['Aravind Srinivas'], 
            similar_companies: ['Anthropic', 'OpenAI']
        }
        + new record for update: {
            name: 'Perplexity', 
            alternative_names: ['Perplexity Inc.'], 
            key_executives: ['Denis Yarats'],
            similar_companies: ['Hugging Face', 'Glean']
        }
        + updated record should be: {
            name: 'Perplexity', 
            alternative_names: ['Perplexity Inc.', 'Perplexity', 'Perplexity Inc', 'Perplexity AI'], 
            key_executives: ['Aravind Srinivas', 'Denis Yarats'],
            similar_companies: ['Anthropic', 'OpenAI', 'Hugging Face', 'Glean']
        }        
</Available Tools>
"""

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
        self.structured_llm = base_llm.bind_tools(
            tools = [
                ResearchPerson,
                ResearchCompany,
                InsertCompanyToDataBase,
                InsertPersonToDataBase,
                UpdateCompanyInDatabase,
                FetchCompanyFromDataBase,
                FetchPersonFromDataBase,
                ListAllPersonNamesFromDataBase,
                ListAllCompanyNamesFromDataBase,
                ListPersonsFromCompanyId
            ]
        )

        self.graph = self.build_graph()
        self.message_memory.append(SystemMessage(content=AGENT_INSTRUCTIONS))

    def stream_response(self, user_message: str):
        out_dict = self.run(query=user_message)

        for chunk in out_dict['content'].split(sep=' '):
            chunk += ' '
            yield chunk
            time.sleep(0.05)

    def run(self, query: str) -> dict[str, Any]:
        self.message_memory.append(HumanMessage(content=query))
        in_state = AgentState(
            messages = self.message_memory,
            token_usage = {m: {'input_tokens': 0, 'output_tokens': 0} for m in self.models},
        )

        config = {"configurable": {"thread_id": '1'}}

        out_state = self.graph.invoke(in_state, config)
        self.message_memory = out_state['messages']

        out_dict = {
            'content':out_state['messages'][-1].content,
            'token_usage': out_state['token_usage'],
        }

        return out_dict

    def llm_call(self, state: AgentState) -> AgentState:
        with get_usage_metadata_callback() as cb:
            response = self.structured_llm.invoke(state.messages)
            state.token_usage[self.model_name]['input_tokens'] += cb.usage_metadata[self.model_name]['input_tokens']
            state.token_usage[self.model_name]['output_tokens'] += cb.usage_metadata[self.model_name]['output_tokens']
            state.messages.extend([response])
        return state

    def tools_call(self, state: AgentState) -> AgentState:

        for tool_call in state.messages[-1].tool_calls:

            match tool_call['name']:
                case 'ResearchPerson':
                    state, out_dict = self.research_person(name=tool_call['args']['name'], company=tool_call['args']['company'], state=state)
                    tool_message_content = json.dumps(out_dict['content'], indent=2)

                case 'ResearchCompany':
                    state, out_dict = self.research_company(company_name=tool_call['args']['company_name'], state=state)
                    tool_message_content = json.dumps(out_dict['content'], indent=2)

                case 'FetchCompanyFromDataBase':
                    response = self.fetch_company_by_name(company_name=tool_call['args']['company_name'])
                    if len(response) > 0:
                        company = response[0]
                        tool_message_content = json.dumps(company, indent=2)
                    else:
                        tool_message_content = f"There is no record for {tool_call['args']['company_name']} in database."

                case 'FetchPersonFromDataBase':
                    # First fetch the current company of the person
                    name = tool_call['args']['name']
                    company_name = tool_call['args']['company']
                    response = self.fetch_company_by_name(company_name = company_name)

                    if len(response) > 0: # Company of the person exists in the database
                        company = response[0]
                        current_company_id = company['id']
                        response = self.fetch_person_from_db(name = name, current_company_id = current_company_id)
                        if len(response) > 0: # Person exists in the database
                            person = response[0]
                            tool_message_content = json.dumps(person, indent=2)
                        else:
                            tool_message_content = f"There is no record for {name} from {company_name} in database."
                    else:
                        tool_message_content = (
                                f"There is no record for {company_name} in database.\n\n" +
                                f"For a person to be successfully inserted into the database, first his/her current company should be inserted."
                        )

                case 'InsertCompanyToDataBase':
                    # Check whether the company record exists in the DB
                    company_name = tool_call['args']['name']
                    response = self.fetch_company_by_name(company_name=company_name)
                    if len(response) > 0:
                        company = response[0]
                        tool_message_content = f"Company {company_name} already exists in database with id: {company['id']}"
                    else:
                        id = self.insert_company_to_db(input_dict=tool_call['args'])
                        tool_message_content = f"{company_name} successfully inserted into database {Table.COMPANIES} table with id {id}"

                case 'InsertPersonToDataBase':

                    name = tool_call['args']['name']
                    current_company = tool_call['args']['current_company']

                    # First check whether the company of the person exists in database
                    response = self.fetch_company_by_name(company_name=current_company)

                    if len(response) > 0: # Company exists in the database
                        company = response[0]
                        current_company_id = company['id']

                        response = self.fetch_person_from_db(name=name, current_company_id=current_company_id)
                        if len(response) > 0:
                            person = response[0]  # Person exists in the database
                            tool_message_content = f"{name} from {current_company} already exist in the database with id: {person['id']}."
                        else:
                            id = self.insert_person_to_db(input_dict=tool_call['args'],
                                                          current_company_id=current_company_id)
                            tool_message_content = f"{name} from {current_company} successfully inserted into database {Table.PERSONS} table with id {id}"

                    else: # No company, no person in the database
                        # First, research the company and insert to the database.
                        # Then insert the person (with link to the company entry).
                        state, out_dict = self.research_company(company_name=tool_call['args']['current_company'], state=state)
                        current_company_id = self.insert_company_to_db(input_dict=out_dict['content'])
                        id = self.insert_person_to_db(input_dict=tool_call['args'], current_company_id=current_company_id)
                        tool_message_content = f"{tool_call['args']['name']} successfully inserted into database {Table.PERSONS} table with id {id}"

                case 'UpdateCompanyInDatabase':
                    id = self.update_company_in_db(input_dict=tool_call['args'])
                    tool_message_content = f"{tool_call['args']['name']} in database {Table.COMPANIES} table with id {id} is successfully updated."
                case 'ListAllPersonNamesFromDataBase':
                    response = self.list_all_names(table_name=Table.PERSONS)
                    tool_message_content = json.dumps(response, indent=2)
                case 'ListAllCompanyNamesFromDataBase':
                    response = self.list_all_names(table_name=Table.COMPANIES)
                    tool_message_content = json.dumps(response, indent=2)
                case 'ListPersonsFromCompanyId':
                    response = self.list_persons_from_company_id(company_id=tool_call['args']['company_id'])
                    tool_message_content = json.dumps(response, indent=2)
                case _:
                    tool_message_content = f"Unknown tool call: {tool_call['name']}"

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
        time_now = datetime.datetime.now().replace(microsecond=0).astimezone(
            tz=datetime.timezone(offset=datetime.timedelta(hours=3), name='UTC+3'))

        row_dict = copy.deepcopy(input_dict)
        row_dict[CompaniesColumns.CREATED_AT] = str(time_now)
        row_dict[CompaniesColumns.UPDATED_AT] = str(time_now)
        # These two will be inputs after the system supports multiple users
        row_dict[CompaniesColumns.CREATED_BY_ID] = 1
        row_dict[CompaniesColumns.UPDATED_BY_ID] = 1

        response = (
            self.db_client.table(Table.COMPANIES)
            .insert(row_dict)
            .execute()
        )
        id = response.data[0]['id']
        return id

    def update_company_in_db(self, input_dict: dict[str, Any]):
        time_now = datetime.datetime.now().replace(microsecond=0).astimezone(
            tz=datetime.timezone(offset=datetime.timedelta(hours=3), name='UTC+3'))

        row_dict = copy.deepcopy(input_dict)
        id = row_dict.pop(CompaniesColumns.ID)
        row_dict[CompaniesColumns.UPDATED_BY_ID] = 1 # This will be an input after the system supports multiple users
        row_dict[CompaniesColumns.UPDATED_AT] = str(time_now)

        response = (
            self.db_client.table(Table.COMPANIES)
            .update(row_dict)
            .eq(CompaniesColumns.ID, id)
            .execute()
        )
        id = response.data[0]['id']
        return id

    def fetch_company_by_name(self, company_name: str) -> list[dict[str, Any]]:
        response = (
            self.db_client.table(Table.COMPANIES)
            .select("*")
            .eq(CompaniesColumns.NAME, company_name)
            .execute()
        )

        if len(response.data) == 0:
            response = (
                self.db_client.table(Table.COMPANIES)
                .select("*")
                .contains(CompaniesColumns.ALTERNATIVE_NAMES, [company_name])
                .execute()
            )

        return response.data

    def fetch_company_by_id(self, company_id: int) -> list[dict[str, Any]]:
        response = (
            self.db_client.table(Table.COMPANIES)
            .select("*")
            .eq(CompaniesColumns.ID, company_id)
            .execute()
        )
        return response.data

    def insert_person_to_db(self, input_dict: dict[str, Any], current_company_id: int):
        time_now = datetime.datetime.now().replace(microsecond=0).astimezone(
            tz=datetime.timezone(offset=datetime.timedelta(hours=3), name='UTC+3'))

        row_dict = copy.deepcopy(input_dict)
        row_dict[PersonsColumns.CREATED_AT] = str(time_now)
        row_dict[PersonsColumns.UPDATED_AT] = str(time_now)
        # These two will be inputs after the system supports multiple users
        row_dict[PersonsColumns.CREATED_BY_ID] = 1
        row_dict[PersonsColumns.UPDATED_BY_ID] = 1
        row_dict[PersonsColumns.CURRENT_COMPANY_ID] = current_company_id
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
                .eq(PersonsColumns.NAME, name)
                .execute()
            )
        else:
            response = (
                self.db_client.table(Table.PERSONS)
                .select("*")
                .eq(PersonsColumns.NAME, name)
                .eq(PersonsColumns.CURRENT_COMPANY_ID, current_company_id)
                .execute()
            )
        return response.data

    def list_persons_from_company_id(self, company_id: int) -> list[dict[str, Any]]:
        response = (
            self.db_client.table(Table.PERSONS)
            .select(PersonsColumns.NAME)
            .eq(PersonsColumns.CURRENT_COMPANY_ID, company_id)
            .execute()
        )
        return response.data

    def list_all_names(self, table_name: str) -> list[dict[str, Any]]:
        response = (
            self.db_client.table(table_name)
            .select(ColumnsBase.NAME)
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
