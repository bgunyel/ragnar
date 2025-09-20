import asyncio
import json
from typing import Any
from uuid import uuid4

from business_researcher import BusinessResearcher, SearchType
from langchain_core.runnables import RunnableConfig
from supabase import create_client, Client

from .base_agent import BaseAgent
from .enums import Table, ColumnsBase, CompaniesColumns, PersonsColumns
from .state import AgentState
from .tools import (
    ResearchPerson,
    ResearchCompany,
    InsertCompanyToDataBase,
    InsertPersonToDataBase,
    UpdateCompanyInDatabase,
    UpdatePersonInDatabase,
    FetchCompanyFromDataBase,
    FetchPersonFromDataBase,
    ListAllPersonNamesFromDataBase,
    ListAllCompanyNamesFromDataBase,
    ListPersonsFromCompanyId,
)
from .utils import insert_entity_to_db, update_entity_in_db, fetch_entity_by_id, fetch_entity_by_name

AGENT_INSTRUCTIONS = """
You are a smart and helpful business intelligence assistant. Your name is Bia. You are a member of King Ragnar's team.

<Task>
Your job is to use tools to perform user's commands and find information to answer user's questions.
You can use any of the tools provided to you. 
You can call these tools in series or in parallel, your functionality is conducted in a tool-calling loop.
</Task>

<Available Tools>
You have access to the following main tools:
1. **ResearchPerson**: To research a specific person within a company using web search.
2. **ResearchCompany**: To research a company using web search.
3. **InsertCompanyToDataBase**: To insert company information to database.
4. **InsertPersonToDataBase**: To insert person information to database.
5. **FetchCompanyFromDataBase**: To get information about a company from the database.
6. **FetchPersonFromDataBase**: To get information about a person from the database.
7. **UpdateCompanyInDatabase**: To update already existing information about a company in the database.
8. **UpdatePersonInDatabase**: To update already existing information about a person in the database. 
9. **ListAllPersonNamesFromDataBase**: To get the list of all person names in the database.
10. **ListAllCompanyNamesFromDataBase**: To get the list of all company names in the database.
11. **ListPersonsFromCompanyId**: To get the list of all persons in a given company.

**CRITICAL**:
* There are 2 tables in the database: persons and companies.
    - The "current_company_id" column of the persons table is linked to the "id" column of the companies table.
    - For example, if the company LangChain has id of 2 in the companies table, then the people working at LangChain have current_company_id equal to 2. 
* When the user wants to get information about a company:
    - First check whether there is an entry about the company in the database (by using FetchCompanyFromDataBase).
    - If there is information about the company in the database, return the information.
    - If there is not any information about the company in the database, perform a research about the company (by using ResearchCompany). 
    - If you perform a research about a company (by using ResearchCompany), ask the user for confirmation to save the new information to the database.
* When the user wants to get information about a person:
    - First check whether there is an entry about the person in the database (by using FetchPersonFromDataBase).
    - If there is information about the person in the database, return the information.
    - If there is not any information about the person in the database, perform a research about the person (by using ResearchPerson).
    - If you perform a research about a person (by using ResearchPerson), ask the user for confirmation to save the new information to the database.
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

CONFIG = RunnableConfig(
    configurable={
        'thread_id': str(uuid4()),
        'max_iterations': 3,
        'max_results_per_query': 4,
        'max_tokens_per_source': 10000,
        'number_of_days_back': 1e6,
        'number_of_queries': 3,
        }
)

TOOLS = [
            ResearchPerson,
            ResearchCompany,
            InsertCompanyToDataBase,
            InsertPersonToDataBase,
            UpdateCompanyInDatabase,
            UpdatePersonInDatabase,
            FetchCompanyFromDataBase,
            FetchPersonFromDataBase,
            ListAllPersonNamesFromDataBase,
            ListAllCompanyNamesFromDataBase,
            ListPersonsFromCompanyId
        ]

class BusinessIntelligenceAgent(BaseAgent):
    def __init__(self,
                 llm_config: dict[str, Any],
                 web_search_api_key: str,
                 database_url: str,
                 database_key: str):

        super().__init__(llm_config=llm_config, tools=TOOLS, agent_instructions=AGENT_INSTRUCTIONS)
        self.business_researcher = BusinessResearcher(llm_config = llm_config, web_search_api_key = web_search_api_key)
        self.db_client: Client = create_client(supabase_url=database_url, supabase_key=database_key)

        # Tool dispatcher mapping
        self.tool_handlers = {
            'ResearchPerson': self._handle_research_person,
            'ResearchCompany': self._handle_research_company,
            'FetchCompanyFromDataBase': self._handle_fetch_company,
            'FetchPersonFromDataBase': self._handle_fetch_person,
            'InsertCompanyToDataBase': self._handle_insert_company,
            'InsertPersonToDataBase': self._handle_insert_person,
            'UpdateCompanyInDatabase': self._handle_update_company,
            'UpdatePersonInDatabase': self._handle_update_person,
            'ListAllPersonNamesFromDataBase': self._handle_list_persons,
            'ListAllCompanyNamesFromDataBase': self._handle_list_companies,
            'ListPersonsFromCompanyId': self._handle_list_persons_from_company,
        }

    def research_person(self, name: str, company: str, state: AgentState) -> tuple[AgentState, dict[str, Any]]:
        input_dict = {
            "name": name,
            "company": company,
            'search_type': SearchType.PERSON
        }
        out_dict = self.run_research_loop(input_dict=input_dict)
        state = self.update_token_usage(state=state, token_usage=out_dict['token_usage'])
        return state, out_dict

    def research_company(self, company_name: str, state: AgentState) -> tuple[AgentState, dict[str, Any]]:
        input_dict = {
            "name": company_name,
            'search_type': SearchType.COMPANY
        }
        out_dict = self.run_research_loop(input_dict=input_dict)
        state = self.update_token_usage(state=state, token_usage=out_dict['token_usage'])
        return state, out_dict

    def run_research_loop(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        event_loop = asyncio.new_event_loop()
        out_dict = event_loop.run_until_complete(self.business_researcher.run(input_dict=input_dict, config=CONFIG))
        event_loop.close()
        return out_dict

    def insert_company_to_db(self, input_dict: dict[str, Any]):
        idx = insert_entity_to_db(db_client=self.db_client, input_dict=input_dict, table_name=Table.COMPANIES)
        return idx

    def insert_person_to_db(self, input_dict: dict[str, Any], current_company_id: int):
        input_dict[PersonsColumns.CURRENT_COMPANY_ID] = current_company_id
        input_dict.pop('current_company')
        idx = insert_entity_to_db(db_client=self.db_client, input_dict=input_dict, table_name=Table.PERSONS)
        return idx

    def update_company_in_db(self, input_dict: dict[str, Any]):
        idx = update_entity_in_db(db_client=self.db_client, input_dict=input_dict, table_name=Table.COMPANIES)
        return idx

    def update_person_in_db(self, input_dict: dict[str, Any], new_company_id: int):
        input_dict[PersonsColumns.CURRENT_COMPANY_ID] = new_company_id
        input_dict.pop('current_company')
        idx = update_entity_in_db(db_client=self.db_client, input_dict=input_dict, table_name=Table.PERSONS)
        return idx

    def fetch_company_by_name(self, company_name: str) -> list[dict[str, Any]]:
        data = fetch_entity_by_name(db_client=self.db_client, entity_name=company_name, table_name=Table.COMPANIES)
        if len(data) == 0:
            response = (
                self.db_client.table(Table.COMPANIES)
                .select("*")
                .contains(CompaniesColumns.ALTERNATIVE_NAMES, [company_name])
                .execute()
            )
            data = response.data
        return data

    def fetch_company_by_id(self, company_id: int) -> list[dict[str, Any]]:
        data = fetch_entity_by_id(db_client=self.db_client, table_name=Table.COMPANIES, entity_id=company_id)
        return data

    def fetch_person_from_db(self, name: str, current_company_id: int | None) -> list[dict[str, Any]]:
        if current_company_id is None:
            data = fetch_entity_by_name(db_client=self.db_client, entity_name=name, table_name=Table.PERSONS)
        else:
            response = (
                self.db_client.table(Table.PERSONS)
                .select("*")
                .eq(PersonsColumns.NAME, name)
                .eq(PersonsColumns.CURRENT_COMPANY_ID, current_company_id)
                .execute()
            )
            data = response.data
        return data

    def list_persons_from_company_id(self, company_id: int) -> list[dict[str, Any]]:
        response = (
            self.db_client.table(Table.PERSONS)
            .select(PersonsColumns.NAME)
            .eq(PersonsColumns.CURRENT_COMPANY_ID, company_id)
            .execute()
        )
        return response.data

    def list_all_names(self, table_name: str) -> list[dict[str, Any]]:
        out = []
        match table_name:
            case Table.COMPANIES:
                response = (
                    self.db_client.table(table_name)
                    .select(ColumnsBase.NAME)
                    .execute()
                )
                out = response.data
            case Table.PERSONS:
                response = (
                    self.db_client.table(table_name)
                    .select(f"{ColumnsBase.NAME}, {PersonsColumns.CURRENT_COMPANY_ID}, {Table.COMPANIES}!inner({ColumnsBase.NAME})")
                    .execute()
                )
                out = [{'name': x['name'], 'current_company': x['companies']['name']} for x in response.data]
            case _: # noinspection PyUnreachableCode
                raise ValueError(f'Invalid table name! - Can be either {Table.COMPANIES} or {Table.PERSONS}')

        return out

    def _handle_research_person(self, tool_call: dict, state: AgentState) -> tuple[AgentState, str]:
        state, out_dict = self.research_person(
            name=tool_call['args']['name'],
            company=tool_call['args']['company'],
            state=state
        )
        return state, json.dumps(out_dict['content'], indent=2)

    def _handle_research_company(self, tool_call: dict, state: AgentState) -> tuple[AgentState, str]:
        state, out_dict = self.research_company(
            company_name=tool_call['args']['company_name'],
            state=state
        )
        return state, json.dumps(out_dict['content'], indent=2)

    def _handle_fetch_company(self, tool_call: dict, state: AgentState) -> tuple[AgentState, str]:
        response = self.fetch_company_by_name(company_name=tool_call['args']['company_name'])
        if len(response) > 0:
            company = response[0]
            message = json.dumps(company, indent=2)
        else:
            message = f"There is no record for {tool_call['args']['company_name']} in database."
        return state, message

    def _handle_fetch_person(self, tool_call: dict, state: AgentState) -> tuple[AgentState, str]:
        name = tool_call['args']['name']
        company_name = tool_call['args']['company']
        response = self.fetch_company_by_name(company_name=company_name)

        if len(response) > 0:
            company = response[0]
            current_company_id = company['id']
            response = self.fetch_person_from_db(name=name, current_company_id=current_company_id)
            if len(response) > 0:
                person = response[0]
                message = json.dumps(person, indent=2)
            else:
                message = f"There is no record for {name} from {company_name} in database."
        else:
            message = (
                f"There is no record for {company_name} in database.\n\n" +
                f"For a person to be successfully inserted into the database, first his/her current company should be inserted."
            )
        return state, message

    def _handle_insert_company(self, tool_call: dict, state: AgentState) -> tuple[AgentState, str]:
        company_name = tool_call['args']['name']
        response = self.fetch_company_by_name(company_name=company_name)
        if len(response) > 0:
            company = response[0]
            message = f"Company {company_name} already exists in database with id: {company['id']}"
        else:
            idx = self.insert_company_to_db(input_dict=tool_call['args'])
            message = f"{company_name} successfully inserted into database {Table.COMPANIES} table with id {idx}"
        return state, message

    def _handle_insert_person(self, tool_call: dict, state: AgentState) -> tuple[AgentState, str]:
        name = tool_call['args']['name']
        current_company = tool_call['args']['current_company']

        response = self.fetch_company_by_name(company_name=current_company)

        if len(response) > 0:
            company = response[0]
            current_company_id = company['id']

            response = self.fetch_person_from_db(name=name, current_company_id=current_company_id)
            if len(response) > 0:
                person = response[0]
                message = f"{name} from {current_company} already exist in the database with id: {person['id']}."
            else:
                idx = self.insert_person_to_db(input_dict=tool_call['args'], current_company_id=current_company_id)
                message = f"{name} from {current_company} successfully inserted into database {Table.PERSONS} table with id {idx}"
        else:
            state, out_dict = self.research_company(company_name=tool_call['args']['current_company'], state=state)
            current_company_id = self.insert_company_to_db(input_dict=out_dict['content'])
            idx = self.insert_person_to_db(input_dict=tool_call['args'], current_company_id=current_company_id)
            message = f"{tool_call['args']['name']} successfully inserted into database {Table.PERSONS} table with id {idx}"
        return state, message

    def _handle_update_company(self, tool_call: dict, state: AgentState) -> tuple[AgentState, str]:
        idx = self.update_company_in_db(input_dict=tool_call['args'])
        message = f"{tool_call['args']['name']} in database {Table.COMPANIES} table with id {idx} is successfully updated."
        return state, message

    def _handle_update_person(self, tool_call: dict, state: AgentState) -> tuple[AgentState, str]:
        name = tool_call['args']['name']
        new_company = tool_call['args']['current_company']

        response = self.fetch_company_by_name(company_name=new_company)
        if len(response) > 0:
            company = response[0]
            new_company_id = company['id']
            idx = self.update_person_in_db(input_dict=tool_call['args'], new_company_id=new_company_id)
            message = f"{name} in database {Table.PERSONS} table with id {idx} is successfully updated."
        else:
            state, out_dict = self.research_company(company_name=new_company, state=state)
            new_company_id = self.insert_company_to_db(input_dict=out_dict['content'])
            idx = self.update_person_in_db(input_dict=tool_call['args'], new_company_id=new_company_id)
            message = f"{name} in database {Table.PERSONS} table with id {idx} is successfully updated."
        return state, message

    def _handle_list_persons(self, _tool_call: dict, state: AgentState) -> tuple[AgentState, str]:
        response = self.list_all_names(table_name=Table.PERSONS)
        return state, json.dumps(response, indent=2)

    def _handle_list_companies(self, _tool_call: dict, state: AgentState) -> tuple[AgentState, str]:
        response = self.list_all_names(table_name=Table.COMPANIES)
        return state, json.dumps(response, indent=2)

    def _handle_list_persons_from_company(self, tool_call: dict, state: AgentState) -> tuple[AgentState, str]:
        response = self.list_persons_from_company_id(company_id=tool_call['args']['company_id'])
        return state, json.dumps(response, indent=2)
