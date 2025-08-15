from typing import ClassVar
from pydantic import BaseModel, ConfigDict

class Node(BaseModel):
    model_config = ConfigDict(frozen=True)
    # Class attributes
    LLM_CALL: ClassVar[str] = 'llm_call'
    TOOLS_CALL: ClassVar[str] = 'tools_call'

class Table(BaseModel):
    model_config = ConfigDict(frozen=True)
    # Class attributes
    COMPANIES: ClassVar[str] = 'companies'
    PERSONS: ClassVar[str] = 'persons'
    USERS: ClassVar[str] = 'users'

# TODO: This class will be carried to ai_common
class ColumnsBase(BaseModel):
    model_config = ConfigDict(frozen=True)
    # Class attributes
    ID: ClassVar[str] = 'id'
    NAME: ClassVar[str] = 'name'
    CREATED_AT: ClassVar[str] = 'created_at'
    UPDATED_AT: ClassVar[str] = 'updated_at'
    CREATED_BY_ID: ClassVar[str] = 'created_by_id'
    UPDATED_BY_ID: ClassVar[str] = 'updated_by_id'

class CompaniesColumns(ColumnsBase):
    model_config = ConfigDict(frozen=True)
    # Class attributes (in Alphabetical Order)
    ADDRESS: ClassVar[str] = 'address'
    ALTERNATIVE_NAMES: ClassVar[str] = 'alternative_names'
    CEO: ClassVar[str] = 'ceo'
    COMPANY_SUMMARY: ClassVar[str] = 'company_summary'
    CRUNCHBASE_PROFILE: ClassVar[str] = 'crunchbase_profile'
    DISTINGUISHING_FEATURES: ClassVar[str] = 'distinguishing_features'
    IS_VERIFIED: ClassVar[str] = 'is_verified'
    KEY_EXECUTIVES: ClassVar[str] = 'key_executions'
    LATEST_FUNDING_ROUND: ClassVar[str] = 'latest_funding_round'
    LATEST_FUNDING_ROUND_DATE: ClassVar[str] = 'latest_funding_round_date'
    LATEST_FUNDING_ROUND_AMOUNT_MM_USD: ClassVar[str] = 'latest_funding_round_amount_mm_usd'
    LINKEDIN_PROFILE: ClassVar[str] = 'linkedin_profile'
    MAIN_PRODUCTS: ClassVar[str] = 'main_products'
    ORG_CHART_SUMMARY: ClassVar[str] = 'org_chart_summary'
    SERVICES: ClassVar[str] = 'services'
    SIMILAR_COMPANIES: ClassVar[str] = 'similar_companies'
    TOTAL_FUNDING_MM_USD: ClassVar[str] = 'total_funding_mm_usd'
    WEBSITE: ClassVar[str] = 'website'
    YEAR_FOUNDED: ClassVar[str] = 'year_founded'

class PersonsColumns(ColumnsBase):
    model_config = ConfigDict(frozen=True)
    # Class attributes (in Alphabetical Order)
    COMPANIES: ClassVar[str] = 'companies'
    CURRENT_COMPANY_ID: ClassVar[str] = 'current_company_id'
    CURRENT_LOCATION: ClassVar[str] = 'current_location'
    LINKEDIN_PROFILE: ClassVar[str] = 'linkedin_profile'
    ROLE: ClassVar[str] = 'role'
    WORK_EMAIL: ClassVar[str] = 'work_email'
    YEARS_EXPERIENCE: ClassVar[str] = 'years_experience'
