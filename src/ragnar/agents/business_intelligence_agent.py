import asyncio
from uuid import uuid4
from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

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
You are a smart and helpful business intelligence assistant.
"""

def should_continue(state: AgentState) -> Literal['continue', 'end']:
    messages = state["messages"]
    # If the last message is not a tool call, then we finish
    if not state.messages[-1].tool_calls:
        return "end"
    # default to continue
    return "continue"

class BusinessIntelligenceAgent:
    def __init__(self, llm_config: dict[str, Any], web_search_api_key: str):
        self.memory_saver = MemorySaver()
        self.models = list({llm_config['language_model']['model'], llm_config['reasoning_model']['model']})
        self.business_researcher = BusinessResearcher(llm_config = llm_config, web_search_api_key = web_search_api_key)

        model_params = llm_config['reasoning_model']
        base_llm = init_chat_model(
            model=model_params['model'],
            model_provider=model_params['model_provider'],
            api_key=model_params['api_key'],
            **model_params['model_args']
        )
        self.structured_llm = base_llm.bind_tools(tools = [ResearchPerson, ResearchCompany])

        self.graph = self.build_graph()


    def run(self):
        in_state = AgentState(
            messages = [],
            token_usage = {m: {'input_tokens': 0, 'output_tokens': 0} for m in self.models},
        )

        out_state = self.graph.ainvoke(in_state)
        dummy = -32

    def llm_call(self, state: AgentState) -> AgentState:
        return state

    def tools_call(self, state: AgentState) -> AgentState:
        return state

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
