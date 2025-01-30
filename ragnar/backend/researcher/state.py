from pydantic import BaseModel


class ReportState(BaseModel):
    """
    Represents the state of our research report.

    Attributes:
        topic: research topic
        steps: steps followed during graph run

    """
    topic: str
    steps: list[str]


class SummaryState(BaseModel):
    """
    Represents the state of our research summary.

    Attributes:
        topic: research topic
        search_queries: list of search queries
        source_str: String of formatted source content from web search
        content: Content generated from sources
        steps: steps followed during graph run

    """
    topic: str
    search_queries: list[str]
    source_str: str
    content: str
    summary_exists: bool = False
    steps: list[str]
