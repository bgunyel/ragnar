from abc import ABC
from typing import Any, Literal
from pydantic import BaseModel

from ai_common import calculate_token_cost, get_llm
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph

from .configuration import Configuration
from .enums import Node
from .state import AgentState, DeepAgentState


def _should_continue(state: AgentState) -> Literal['continue', 'end']:
    # If the last message is not a tool call, then we finish
    if len(state.messages[-1].tool_calls) == 0:
        return "end"
    else:
        return "continue"


class BaseAgent(ABC):
    """
    Abstract base class for LLM-based agents with tool calling capabilities.

    This class provides a common framework for agents that can interact with LLMs,
    manage conversation state, and execute tools through a dispatcher pattern.

    Tool Handler Pattern:
        Child classes should populate the `tool_handlers` dictionary in their __init__ method
        to map tool names to handler methods. This enables a clean dispatcher pattern instead
        of large match-case blocks.

        Example in child class __init__:
            self.tool_handlers = {
                'ToolName1': self._handle_tool_1,
                'ToolName2': self._handle_tool_2,
                # ... more mappings
            }

        Handler Method Signature:
            All tool handler methods must follow this exact signature:

            def _handle_tool_name(self, tool_call: dict, state: AgentState) -> tuple[AgentState, str]:
                # Extract arguments from tool_call['args']
                # Perform tool-specific logic
                # Return updated state and response message
                return state, message_content

            For handlers that don't use tool_call parameters, prefix with underscore:
            def _handle_simple_tool(self, _tool_call: dict, state: AgentState) -> tuple[AgentState, str]:
                # Logic that doesn't need tool_call arguments
                return state, message_content
    """

    def __init__(self,
                 llm_config: dict[str, Any],
                 tools: list, agent_instructions: str,
                 runnable_config: RunnableConfig,
                 is_deep_agent: bool = False):
        self._memory_saver = MemorySaver()
        self._models = list({*[v['model'] for k, v in llm_config.items()]})
        self._message_memory = []
        self._llm_config = llm_config
        self._is_deep_agent = is_deep_agent
        self._runnable_config = runnable_config

        model_params = llm_config['reasoning_model']
        base_llm = get_llm(model_name=model_params['model'],
                           model_provider=model_params['model_provider'],
                           api_key=model_params['api_key'],
                           model_args=model_params['model_args'])


        self._model_name = model_params['model']
        self._structured_llm = base_llm.bind_tools(tools=tools)
        self._graph = self._build_graph()
        self._message_memory.append(SystemMessage(content=agent_instructions))
        self._tool_handlers = dict()

    def get_model_names(self) -> list[str]:
        return self._models

    async def run(self, query: str) -> dict[str, Any]:
        self._message_memory.append(HumanMessage(content=query))

        if self._is_deep_agent:
            in_state = DeepAgentState(
                messages=self._message_memory,
                token_usage={m: {'input_tokens': 0, 'output_tokens': 0} for m in self._models},
                todos=[]
            )
        else:
            in_state = AgentState(
                messages=self._message_memory,
                token_usage={m: {'input_tokens': 0, 'output_tokens': 0} for m in self._models},
            )

        out_state = await self._graph.ainvoke(in_state, self._runnable_config)
        self._message_memory = out_state['messages']
        cost_list, total_cost = calculate_token_cost(llm_config=self._llm_config, token_usage=out_state['token_usage'])

        out_dict = {
            'content': out_state['messages'][-1].content,
            'token_usage': out_state['token_usage'],
            'cost_list': cost_list,
            'total_cost': total_cost,
        }

        return out_dict

    def _llm_call(self, state: BaseModel) -> BaseModel:
        with get_usage_metadata_callback() as cb:
            response = self._structured_llm.invoke(state.messages)
            state.token_usage[self._model_name]['input_tokens'] += cb.usage_metadata[self._model_name]['input_tokens']
            state.token_usage[self._model_name]['output_tokens'] += cb.usage_metadata[self._model_name]['output_tokens']
            state.messages.extend([response])
        return state

    def _tools_call(self, state: BaseModel) -> BaseModel:
        for tool_call in state.messages[-1].tool_calls:
            handler = self._tool_handlers.get(tool_call['name'])
            if handler:
                state, tool_message_content = handler(tool_call, state)
            else:
                tool_message_content = f"Unknown tool call: {tool_call['name']}"

            state.messages.append(ToolMessage(
                content=tool_message_content,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            ))
        return state

    def _update_token_usage(self, state: AgentState, token_usage: dict[str, Any]) -> AgentState:
        for m in self._models:
            state.token_usage[m]['input_tokens'] += token_usage[m]['input_tokens']
            state.token_usage[m]['output_tokens'] += token_usage[m]['output_tokens']
        return state

    def _build_graph(self):

        if self._is_deep_agent:
            workflow = StateGraph(DeepAgentState, config_schema=Configuration)
        else:
            workflow = StateGraph(AgentState, config_schema=Configuration)

        ## Nodes
        workflow.add_node(node=Node.LLM_CALL, action=self._llm_call)
        workflow.add_node(node=Node.TOOLS_CALL, action=self._tools_call)

        ## Edges
        workflow.add_edge(start_key=START, end_key=Node.LLM_CALL)
        workflow.add_edge(start_key=Node.TOOLS_CALL, end_key=Node.LLM_CALL)
        workflow.add_conditional_edges(
            source=Node.LLM_CALL,
            path=_should_continue,
            path_map={
                "continue": Node.TOOLS_CALL,
                "end": END,
            },
        )

        ## Compile graph
        compiled_graph = workflow.compile(checkpointer=self._memory_saver)
        return compiled_graph
