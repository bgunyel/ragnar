import os
from dataclasses import dataclass, field, fields
from typing import Any, Optional, TypeAlias, Literal

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from typing_extensions import Annotated
from dataclasses import dataclass


TavilySearchCategory: TypeAlias = Literal['news', 'general']

@dataclass(kw_only=True)
class ConfigurationBase:

    @classmethod
    def from_runnable_config(
            cls, config: Optional[RunnableConfig] = None
    ) -> "ConfigurationBase":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})

"""
class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query for web search.")
"""

class Queries(BaseModel):
    queries: list[str] = Field(description="List of search queries.")
