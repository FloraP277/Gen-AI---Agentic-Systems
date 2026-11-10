"""The structured mine safety report both the agent and the workflow must produce."""
import json
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError


class MineReport(BaseModel):
    mine_id: str = Field(pattern=r"^\d{7}$")
    mine_name: str
    total_violations: int = Field(ge=0, description="Violations in the last 8 quarters")
    total_ss_violations: int = Field(ge=0, description="S&S violations in the last 8 quarters")
    accidents: int = Field(ge=0, description="Accidents in the last 5 years")
    injuries: int = Field(ge=0, description="Injuries in the last 5 years")
    risk_level: Literal["low", "medium", "high"]
    summary: str
    violation_trend: str
    accident_patterns: str
    relevant_regulations: list[str] = Field(description="Section numbers like 56.14100")
    recommendations: list[str]


SCHEMA_TEXT = json.dumps(MineReport.model_json_schema(), indent=1)

REPORT_INSTRUCTIONS = f"""Write a mine safety report as JSON matching this schema exactly, with no other text:
{SCHEMA_TEXT}
Use the last 8 quarters for violation totals and the last 5 years for accidents and injuries, copying the
totals exactly from the tool results. Only cite regulation sections that a search_regulations result returned.
Base risk_level on S&S violations, serious accidents and trends; explain why in the summary."""


def parse_report(text: str) -> tuple[MineReport | None, str | None]:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None, "no JSON object found"
    try:
        return MineReport.model_validate_json(m.group(0)), None
    except ValidationError as e:
        return None, f"schema error: {e.errors()[0]['loc']} {e.errors()[0]['msg']}"
