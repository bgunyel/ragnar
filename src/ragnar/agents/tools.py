from pydantic import BaseModel, Field
from business_researcher import CompanySchema, PersonSchema

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

class InsertCompanyToDataBase(CompanySchema):
    """Insert a company to the database."""

class InsertPersonToDataBase(PersonSchema):
    """Insert a person to the database."""

class UpdateCompanyInDatabase(CompanySchema):
    """Update a company in the database."""
    id: int = Field(description="The id of the company in the database.")

class FetchCompanyFromDataBase(BaseModel):
    """Fetch details of a company from database."""
    company_name: str = Field(
        description="The name of the company to fetch. Should be the official company name or commonly recognized brand name to ensure accurate results.",
    )

class FetchPersonFromDataBase(BaseModel):
    """Fetch details of a specific person within a company from the database."""
    name: str = Field(
        description="The full name of the person to fetch. Should be as specific as possible to ensure accurate results.",
    )
    company: str = Field(
        description="The name of the company where the person works or is associated with. This helps narrow the search scope and improve result relevance.",
    )

class ListAllPersonNamesFromDataBase(BaseModel):
    """List names of all persons in the database."""

class ListAllCompanyNamesFromDataBase(BaseModel):
    """List names of all companies in the database."""

class ListPersonsFromCompanyId(BaseModel):
    """List names of all persons in a specific company from the database."""
    company_id: int = Field("The id of the company whose employees we want to list.")
