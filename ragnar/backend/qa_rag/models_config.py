import os
from dataclasses import dataclass, field, fields
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from typing_extensions import Annotated
from dataclasses import dataclass


"""
    * Modified from: https://github.com/langchain-ai/research-rabbit/blob/main/src/research_rabbit/configuration.py
"""
@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the RAG models."""
    number_of_retrieved_documents: int = 3
    max_retrieval_iterations: int = 2
    max_generation_iterations: int = 2

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
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

