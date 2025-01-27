from langchain_core.documents import Document
from pydantic import BaseModel, PrivateAttr, Field
from typing_extensions import Annotated

"""
    Classes taken (and some are modified) from https://github.com/langchain-ai/report-mAIstro/report_masitro.py
"""

class Section(BaseModel):
    name: str = Field(description="Name for this section of the report.")
    description: str = Field(description="Brief overview of the main topics and concepts to be covered in this section.")
    research: bool = Field(description="Whether to perform web research for this section of the report.")
    content: str = Field(description="The content of the section.")

class Sections(BaseModel):
    sections: list[Section] = Field(description="Sections of the report.")


class ReportState(BaseModel):
    """
    Represents the state of our graph.

    Attributes:
        topic: research topic
        steps: steps followed during graph run

    """
    topic: str
    steps: list[str]
