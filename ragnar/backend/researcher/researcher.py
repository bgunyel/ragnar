from typing import Literal
from uuid import uuid4
from pprint import pprint

from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.graph import START, END, StateGraph

from ragnar.config import settings
from ragnar.backend.researcher.components.planner import Planner


class Researcher:

    """
    Reproduction of https://github.com/langchain-ai/report-mAIstro
    """

    def __init__(self):
        self.planner = Planner(model_name=settings.MODEL)


    def get_response(self, question: str, verbose: bool = False) -> str:
        out = 'Hello, I am a researcher!'
        return out


    def build_graph(self):
        pass

