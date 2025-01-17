from typing import TypedDict
from langgraph.graph import END

from ragnar.backend.enums import Node, StateField


def increment_iteration(state: TypedDict) -> TypedDict:
    state[StateField.ITERATION.value] += 1
    return state


def are_documents_relevant(state: TypedDict, max_iteration: int, number_of_documents: int) -> str:
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state
        max_iteration (int): The maximum number of retrieval iterations
        number_of_documents (int): The number of documents expected from the retriever

    Returns:
        str: Binary decision for next node to call
    """

    if (state[StateField.ITERATION.value] > max_iteration) or (len(state[StateField.GOOD_DOCUMENTS.value]) >= number_of_documents):
        out = Node.ANSWER_GENERATOR.value
    else:
        out = Node.REWRITE_QUESTION.value

    return out


def is_answer_grounded(state: TypedDict) -> str:

    if state[StateField.ANSWER_GROUNDED.value] == 'yes':
        out = Node.ANSWER_GRADER.value
    elif state[StateField.ANSWER_GROUNDED.value] == 'no':
        out = Node.ANSWER_GENERATOR.value
    else:
        raise RuntimeError(
            (f'Unknown state from hallucination grader --> '
            f'state[{StateField.ANSWER_GROUNDED.value}]: {state[StateField.ANSWER_GROUNDED.value]}')
        )

    return out


def is_answer_useful(state: TypedDict) -> str:

    if state[StateField.ANSWER_USEFUL.value] == 'yes':
        out = END
    elif state[StateField.ANSWER_USEFUL.value] == 'no':
        out = Node.REWRITE_QUESTION.value
    else:
        raise RuntimeError(
            (f'Unknown state from answer grader --> '
            f'state[{StateField.ANSWER_USEFUL.value}]: {state[StateField.ANSWER_USEFUL.value]}')
        )

    return out
