from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser

from ragnar.config import settings
from ragnar.backend.state import GraphState
from ragnar.backend.enums import Node


prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
    Here is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question}
    Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "question"],
)


def get_answer_grader(model_name: str):
    llm = ChatOllama(model=model_name, format="json", temperature=0, base_url=settings.OLLAMA_URL)
    answer_grader = prompt | llm | JsonOutputParser()
    return answer_grader


class AnswerGrader:
    def __init__(self, model_name: str):
        self.answer_grader = get_answer_grader(model_name)

    def run(self, state: GraphState) -> GraphState:
        """
        Determines whether the generated answer is useful to resolve a question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updated graph state
        """
        score = self.answer_grader.invoke({"question": state.question, "generation": state.generation})
        state.answer_useful = score["score"]
        state.steps.append(Node.ANSWER_GRADER.value)
        return state
