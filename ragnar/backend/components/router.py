from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser

from ragnar.config import settings
from ragnar.backend.state import GraphState
from ragnar.backend.enums import Node


prompt = PromptTemplate(
    template="""You are an expert at routing a user question to a vectorstore or other data sources.

    Question: {question} \n
    
    The vectorstore contains documents related to the following topics:
    * Syria
    * Syrian civil war
    * Fall of the Assad regime
    * President of Syria
    
    Use the vectorstore for questions on these topics. For all else, use your internal knowledge, if possible.
    
    Return JSON with single key, datasource, that is 'internal' or 'vectorstore' depending on the question.
    """,
    input_variables=['question'],
)

def get_router(model_name: str):
    llm = ChatOllama(model=model_name, format="json", temperature=0, base_url=settings.OLLAMA_URL)
    router = prompt | llm | JsonOutputParser()
    return router


class Router:
    def __init__(self, model_name: str):
        self.router = get_router(model_name=model_name)

    def run(self, state: GraphState) -> GraphState:
        """
        Routes the query according to question

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updated graph state
        """

        out = self.router.invoke({'question': state.question})
        state.datasource = out['datasource']
        state.steps.append(Node.ROUTER.value)
        return state
