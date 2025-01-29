import os
from dataclasses import dataclass, field, fields
from typing import Any, Optional, TypeAlias, Literal

from langchain_core.runnables import RunnableConfig


TavilySearchCategory: TypeAlias = Literal['news', 'general']

@dataclass(kw_only=True)
class ConfigurationBase:
    """
        * Modified from: https://github.com/langchain-ai/research-rabbit/blob/main/src/research_rabbit/configuration.py
    """

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
