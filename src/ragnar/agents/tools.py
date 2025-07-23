from pydantic import BaseModel, Field

class ResearchPerson(BaseModel):
    """Research a specific person within a company using web search and AI analysis."""
    name: str = Field(
        description="The full name of the person to research. Should be as specific as possible to ensure accurate search results.",
    )
    company: str = Field(
        description="The name of the company where the person works or is associated with. This helps narrow the search scope and improve result relevance.",
    )

class ResearchCompany(BaseModel):
    """Research a company using comprehensive web search and AI analysis."""
    company_name: str = Field(
        description="The name of the company to research. Should be the official company name or commonly recognized brand name to ensure accurate and comprehensive search results.",
    )
