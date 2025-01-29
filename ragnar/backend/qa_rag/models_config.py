from ragnar.backend.base import ConfigurationBase


class Configuration(ConfigurationBase):
    """The configurable fields for the RAG models."""
    number_of_retrieved_documents: int = 3
    max_retrieval_iterations: int = 2
    max_generation_iterations: int = 2
