from pydantic import Field
from ai_common import CfgBase


class Configuration(CfgBase):
    """The configurable fields for the workflow"""
    name: str = Field(description="Name of the agent.")  # (0, inf)
