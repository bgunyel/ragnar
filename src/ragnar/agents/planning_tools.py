from pydantic import BaseModel, Field
from .state import DeepAgentState, ToDo

class WriteTodos(BaseModel):
    """Manage a structured task list for coordinating multi-step workflows.

    This tool creates and updates the agent's TODO list, enabling systematic tracking
    of progress through complex operations. Each list contains task objects with
    content descriptions, status indicators, and unique identifiers.

    Usage Guidelines:
        - Apply to multi-step or complex tasks requiring coordination
        - Create when user provides multiple tasks or explicitly requests tracking
        - Skip for single, trivial actions unless specifically requested

    Task List Structure:
        - Maintain a single list with multiple todo objects
        - Each object contains: content (str), status (str), and id (str)
        - Content should be clear and actionable
        - Status values: "pending", "in_progress", or "completed"

    Operational Best Practices:
        - Limit to one "in_progress" task at any time
        - Mark tasks "completed" immediately upon full completion
        - Always provide the complete updated list with each change
        - Remove irrelevant items to maintain focus

    Status Management:
        - Update task status or content by calling this tool again
        - Reflect progress in real-time; avoid batching status changes
        - If blocked, keep task "in_progress" and add new task describing obstacle

    Args:
        todos: List of task items containing 'content' and 'status' fields

    Returns:
        None. Updates the agent's internal state with the modified TODO list.
"""
    todos: list[ToDo] = Field(description="List of Todo items with content and status.")


class ReadTodos(BaseModel):
    """Retrieve the current TODO list from the agent's state.
    This tool fetches and returns the agent's TODO list, enabling it to review tasks, maintain focus on workflow objectives, and monitor progress through multi-step operations.

    Returns:
        The current TODO list from the agent state.
"""


def write_todos(todos: list[ToDo], state: DeepAgentState) -> tuple[DeepAgentState, str]:
    state.todos = todos
    message = f"Updated TODO List: \n\n {todos}"
    return state, message
