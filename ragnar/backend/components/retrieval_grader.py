from typing import TypedDict

from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser

from ragnar.config import settings
from ragnar.backend.enums import Grades

prompt = PromptTemplate(
    template="""You are a teacher grading a quiz. You will be given: 
    1/ a QUESTION
    2/ A FACT provided by the student

    You are grading RELEVANCE RECALL:
    A score of 1 means that ANY of the statements in the FACT are relevant to the QUESTION. 
    A score of 0 means that NONE of the statements in the FACT are relevant to the QUESTION. 
    1 is the highest (best) score. 0 is the lowest score you can give. 

    Explain your reasoning in a step-by-step manner. Ensure your reasoning and conclusion are correct. 

    Avoid simply stating the correct answer at the outset.

    Question: {question} \n
    Fact: \n\n {documents} \n\n

    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
    """,
    input_variables=["question", "documents"],
)

def get_retrieval_grader(model_name: str):
    llm = ChatOllama(model=model_name, format="json", temperature=0, base_url=settings.OLLAMA_URL)
    retrieval_grader = prompt | llm | JsonOutputParser()
    return retrieval_grader


class RetrievalGrader:
    def __init__(self, model_name: str):
        self.retrieval_grader = get_retrieval_grader(model_name=model_name)

    def run(self, state: TypedDict) -> TypedDict:
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        state["steps"].append("grade_document_retrieval")

        if len(state["documents"]) > 0:
            state['documents_grade'] = Grades.GOOD.value

            for d in state['documents']:
                score = self.retrieval_grader.invoke({"question": state['question'], "documents": d.page_content})
                grade = score["score"]
                if grade == "yes":
                    state['good_documents'].append(d)
                else:
                    state['documents_grade'] = Grades.BAD.value
                    continue
        else:
            state['documents_grade'] = Grades.BAD.value

        return state
