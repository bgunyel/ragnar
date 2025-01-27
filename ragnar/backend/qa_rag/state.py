from langchain_core.documents import Document
from pydantic import BaseModel, PrivateAttr, Field
from typing_extensions import Annotated


class GraphState(BaseModel):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        documents: list of documents
        generation: LLM generation
        documents_grade: grade of documents

    """
    question: str
    query: str
    documents: list[Document]
    generation: str
    documents_grade: str
    steps: list[str]
    retrieval_iteration: int
    generation_iteration: int
    good_documents: list[Document]
    datasource: str
    answer_useful: str
    answer_grounded: str

    """
    def __init__(self, question: str, **kwargs):
        # self._question = question  # stores the read-only question
        # object.__setattr__(self, "_question", question)

        kwargs = {
            'question': question,
            'query': question,
            'documents': [],
            'good_documents': [],
            'generation': '',
            'documents_grade': '',
            "steps": [],
            "retrieval_iteration": 0,
            "generation_iteration": 0,
            'datasource': '',
            'answer_useful': '',
            'answer_grounded': ''
        }

        super().__init__(**kwargs)
        
    """

    """
    @property
    def question(self) -> str:
        return self._question
    
    def __setattr__(self, name, value) -> None:
        if name == "question":
            raise AttributeError("question is read-only")
        super().__setattr__(name, value)
    """