from typing import TypedDict

from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser

from ragnar.config import settings
from ragnar.backend.models.enums import States


prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}
    Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "documents"],
)


def get_hallucination_grader(model_name: str):
    llm = ChatOllama(model=model_name, format="json", temperature=0, base_url=settings.OLLAMA_URL)
    hallucination_grader = prompt | llm | JsonOutputParser()
    return hallucination_grader


class HallucinationGrader:
    def __init__(self, model_name: str):
        self.hallucination_grader = get_hallucination_grader(model_name=model_name)

    def run(self, state: TypedDict) -> TypedDict:
        """
         Determines whether the generated answer is grounded in / supported by the set documents.

         Args:
             state (dict): The current graph state

         Returns:
             state (dict):
         """

        score = self.hallucination_grader.invoke({"documents": state["documents"], "generation": state["generation"]})
        state[States.ANSWER_GROUNDED] = score["score"]
        return state
