from typing import TypedDict, Literal
from langgraph.graph import END

from ragnar.backend.enums import Node, StateField


def increment_iteration(state: TypedDict) -> TypedDict:
    state[StateField.ITERATION.value] += 1
    return state


def are_documents_relevant(state: TypedDict, max_iteration: int, number_of_documents: int) -> Literal['relevant', 'not relevant', 'max_iter']:
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state
        max_iteration (int): The maximum number of retrieval iterations
        number_of_documents (int): The number of documents expected from the retriever

    Returns:
        str: Binary decision for next node to call
    """

    if state[StateField.ITERATION.value] > max_iteration:
        return 'max_iter'
    elif len(state[StateField.GOOD_DOCUMENTS.value]) >= number_of_documents:
        return 'relevant'
    else:
        return 'not relevant'


def is_answer_grounded(state: TypedDict) -> Literal['grounded', 'not grounded']:

    if state[StateField.ANSWER_GROUNDED.value] == 'yes':
        return 'grounded'
    elif state[StateField.ANSWER_GROUNDED.value] == 'no':
        return 'not grounded'
    else:
        raise RuntimeError(
            (f'Unknown state from hallucination grader --> '
            f'state[{StateField.ANSWER_GROUNDED.value}]: {state[StateField.ANSWER_GROUNDED.value]}')
        )


def is_answer_useful(state: TypedDict) -> Literal['useful', 'not useful']:

    if state[StateField.ANSWER_USEFUL.value] == 'yes':
        return 'useful'
    elif state[StateField.ANSWER_USEFUL.value] == 'no':
        return 'not useful'
    else:
        raise RuntimeError(
            (f'Unknown state from answer grader --> '
            f'state[{StateField.ANSWER_USEFUL.value}]: {state[StateField.ANSWER_USEFUL.value]}')
        )
