from pydantic import Field
from ai_common import CfgBase


class Configuration(CfgBase):
    """The configurable fields for the workflow"""
    max_iterations: int = Field(gt=0)  # (0, inf)
