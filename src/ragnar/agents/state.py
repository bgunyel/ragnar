from typing import Literal
from pydantic import BaseModel, Field

class AgentState(BaseModel):
    messages: list
    token_usage: dict

class ToDo(BaseModel):
    content: str
    status: Literal["pending", "in_progress", "completed"]

class DeepAgentState(AgentState):
    todos: list[ToDo] = Field(description="List of Todo items for task planning and progress tracking")
