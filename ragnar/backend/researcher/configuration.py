from dataclasses import dataclass

from ragnar.backend.base import ConfigurationBase
from ragnar.backend.base import TavilySearchCategory


DEFAULT_REPORT_STRUCTURE = """The report structure should focus on breaking-down the user-provided topic:

1. Introduction (no research needed)
   - Brief overview of the topic area

2. Main Body Sections:
   - Each section should focus on a sub-topic of the user-provided topic
   - Include any key concepts and definitions
   - Provide real-world examples or case studies where applicable

3. Conclusion
   - Aim for 1 structural element (either a list of table) that distills the main body sections 
   - Provide a concise summary of the report"""


@dataclass(kw_only=True)
class Configuration(ConfigurationBase):
    """The configurable fields for the chatbot."""
    report_structure: str = DEFAULT_REPORT_STRUCTURE
    number_of_queries: int = 2
    search_category: TavilySearchCategory = "general"
    number_of_days_back: int = None
