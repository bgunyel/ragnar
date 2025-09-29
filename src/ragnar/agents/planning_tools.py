import json
from pydantic import BaseModel, Field
from .state import DeepAgentState, ToDo


class WriteTodos(BaseModel):
    """
    Manage a structured task list for coordinating multi-step workflows.

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
    """
    Retrieve the current TODO list from the agent's state.
    This tool fetches and returns the agent's TODO list, enabling it to review tasks,
    maintain focus on workflow objectives, and monitor progress through multi-step operations.

    Args: None.

    Returns:
        The current TODO list from the agent state.
    """


PLANNING_INSTRUCTIONS = """
# TODO MANAGEMENT                                                                                              
Based upon the user's request:                                                                                 
1. Use the WriteTodos tool to create TODO at the start of a user request, per the tool description.           
2. After you accomplish a TODO, use the ReadTodos to read the TODOs in order to remind yourself of the plan.  
3. Reflect on what you've done and the TODO.                                                                   
4. Mark your task as completed, and proceed to the next TODO.                                                   
5. Continue this process until you have completed all TODOs.                                                   

IMPORTANT: Always create an action plan of TODOs and act following the above guidelines for ANY user request.                                                                                                  
IMPORTANT: Aim to batch research tasks into a *single TODO* in order to minimize the number of TODOs you have to keep track of.     
"""


##
# NOTE:
#   * The following functions below do not need to obey the signatures described above for the tools.
#   * The following functions below are the workhorse functions.
#   * The handler methods of the agent class (i.e. _handle_tool_name) must obey the signatures above.
#   * The handler methods should internally call the below workhorse functions.
##

def write_todos(todos: list[ToDo], state: DeepAgentState) -> tuple[DeepAgentState, str]:
    state.todos = [ToDo(**x) for x in todos]
    message = f"Updated TODO List: \n\n {json.dumps(todos, indent=2)}"
    return state, message

def read_todos(state: DeepAgentState) -> str:

    if len(state.todos) == 0:
        message = "Current TODO List is empty."
    else:
        message = "Current TODO List: \n\n"
        for i, item in enumerate(state.todos):
            message += f"{i+1}. {item.content} ({item.status})\n"

    return message

def handle_write_todos(tool_call: dict, state: DeepAgentState) -> tuple[DeepAgentState, str]:
    state, message = write_todos(todos=tool_call['args']['todos'], state=state)
    return state, message

def handle_read_todos(_tool_call: dict, state: DeepAgentState) -> tuple[DeepAgentState, str]:
    message = read_todos(state=state)
    return state, message