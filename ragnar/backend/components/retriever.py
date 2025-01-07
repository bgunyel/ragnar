from typing import TypedDict
from langchain_core.vectorstores import VectorStoreRetriever

class Retriever:
    def __init__(self, retriever: VectorStoreRetriever):
        self.retriever = retriever

    def run(self, state: TypedDict) -> TypedDict:
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """

        # state['documents'] = self.retriever.invoke(state["question"])
        state['documents'] = self.retriever.invoke(state["question"])
        state["steps"].append("retrieve_documents")
        return state
