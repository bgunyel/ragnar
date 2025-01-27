from langchain_core.vectorstores import VectorStoreRetriever
from ragnar.backend.enums import Node
from ragnar.backend.state import GraphState

class Retriever:
    def __init__(self, retriever: VectorStoreRetriever):
        self.retriever = retriever

    def run(self, state: GraphState) -> GraphState:
        """
        Retrieve documents

        Args:
            state: The current graph state

        Returns:
            state: New key added to state, documents, that contains retrieved documents
        """

        state.documents = self.retriever.invoke(state.query)
        state.steps.append(Node.RETRIEVE.value)
        state.retrieval_iteration += 1
        state.generation_iteration = 0
        return state
