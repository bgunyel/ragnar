from typing import Literal
from langchain_core.runnables import RunnableConfig

from ragnar.backend.state import GraphState
from ragnar.backend.models_config import Configuration
from ragnar.backend.enums import Node


def are_documents_relevant(state: GraphState, config: RunnableConfig) -> Literal['relevant', 'not relevant', 'max_iter']:
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state: The current graph state
        config (RunnableConfig): The configuration of the run

    Returns:
        str: Binary decision for next node to call
    """

    configurable = Configuration.from_runnable_config(config)

    if len(state.good_documents) >= configurable.number_of_retrieved_documents:
        return 'relevant'
    elif state.retrieval_iteration >= configurable.max_retrieval_iterations:
        return 'max_iter'
    else:
        return 'not relevant'


def is_answer_grounded(state: GraphState, config: RunnableConfig) -> Literal['grounded', 'not grounded', 'max_iter']:
    configurable = Configuration.from_runnable_config(config)

    if state.answer_grounded == 'yes':
        return 'grounded'
    elif state.generation_iteration >= configurable.max_generation_iterations:
        return 'max_iter'
    elif state.answer_grounded == 'no':
        return 'not grounded'
    else:
        raise RuntimeError(
            (f'Unknown state from hallucination grader --> '
            f'state.answer_grounded: {state.answer_grounded}')
        )


def is_answer_useful(state: GraphState, config: RunnableConfig) -> Literal['useful', 'not useful', 'max_iter']:
    configurable = Configuration.from_runnable_config(config)

    if state.answer_useful == 'yes':
        return 'useful'
    elif state.retrieval_iteration >= configurable.max_retrieval_iterations:
        return 'max_iter'
    elif state.answer_useful == 'no':
        return 'not useful'
    else:
        raise RuntimeError(
            (f'Unknown state from answer grader --> '
            f'state.answer_useful: {state.answer_useful}')
        )

def reset_generation(state: GraphState) -> GraphState:
    state.generation = 'I could not find the answer [reset generation]'
    state.steps.append(Node.RESET.value)
    return state
