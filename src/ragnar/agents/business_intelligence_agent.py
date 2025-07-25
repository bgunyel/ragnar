import asyncio
from uuid import uuid4
from typing import Any, Literal
import json

from langchain.chat_models import init_chat_model
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.callbacks import get_usage_metadata_callback

from business_researcher import BusinessResearcher, SearchType
from .state import AgentState
from .configuration import Configuration
from .enums import Node
from .tools import ResearchPerson, ResearchCompany

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
    def __init__(self, llm_config: dict[str, Any], web_search_api_key: str):
        self.memory_saver = MemorySaver()
        self.models = list({*[v['model'] for k, v in llm_config.items()]})
        self.business_researcher = BusinessResearcher(llm_config = llm_config, web_search_api_key = web_search_api_key)

        model_params = llm_config['reasoning_model']
        base_llm = init_chat_model(
            model=model_params['model'],
            model_provider=model_params['model_provider'],
            api_key=model_params['api_key'],
            **model_params['model_args']
        )
        self.model_name = model_params['model']
        self.structured_llm = base_llm.bind_tools(tools = [ResearchPerson, ResearchCompany])

        self.graph = self.build_graph()


    def run(self, query: str):
        in_state = AgentState(
            messages = [SystemMessage(content=AGENT_INSTRUCTIONS), HumanMessage(content=query)],
            token_usage = {m: {'input_tokens': 0, 'output_tokens': 0} for m in self.models},
        )

        config = {"configurable": {"thread_id": '1'}}

        out_state = self.graph.invoke(in_state, config)
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
            match tool_call['name']:
                case 'ResearchPerson':
                    input_dict = {
                        "name": tool_call['args']['name'],
                        "company": tool_call['args']['company'],
                        'search_type': SearchType.PERSON
                    }
                case 'ResearchCompany':
                    input_dict = {
                        "name": tool_call['args']['company_name'],
                        'search_type': SearchType.COMPANY
                    }
                case _:
                    raise RuntimeError('Unknown tool call')

            out_dict = self.run_event_loop(input_dict=input_dict)
            state.messages.append(
                ToolMessage(
                    content=json.dumps(out_dict['content'], indent=2),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
            ))

            for m in self.models:
                state.token_usage[m]['input_tokens'] += out_dict['token_usage'][m]['input_tokens']
                state.token_usage[m]['output_tokens'] += out_dict['token_usage'][m]['output_tokens']

        return state

    def run_event_loop(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        event_loop = asyncio.new_event_loop()
        out_dict = event_loop.run_until_complete(self.business_researcher.run(input_dict=input_dict, config=CONFIG))
        event_loop.close()
        return out_dict

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
