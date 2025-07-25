from pydantic import BaseModel

class AgentState(BaseModel):
    messages: list
    token_usage: dict
