from typing import ClassVar
from pydantic import BaseModel, ConfigDict

class Node(BaseModel):
    model_config = ConfigDict(frozen=True)
    # Class attributes
    LLM_CALL: ClassVar[str] = 'llm_call'
    TOOLS_CALL: ClassVar[str] = 'tools_call'

class Table(BaseModel):
    model_config = ConfigDict(frozen=True)
    # Class attributes
    COMPANIES: ClassVar[str] = 'companies'
    PERSONS: ClassVar[str] = 'persons'
    USERS: ClassVar[str] = 'users'
