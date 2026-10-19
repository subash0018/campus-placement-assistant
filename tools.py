"""Deterministic project-specific placement eligibility tool."""
import json
from pathlib import Path
from typing import Any
from langchain_core.tools import tool

BASE_DIR = Path(__file__).resolve().parent
TOOL_DATA_PATH = BASE_DIR / "data" / "tool_data.json"
SUPPORTED_DEPARTMENTS = {"CSE", "IT", "ECE", "MECH"}

def _load_companies() -> list[dict[str, Any]]:
    if not TOOL_DATA_PATH.is_file():
        raise FileNotFoundError(f"Missing tool data: {TOOL_DATA_PATH}")
    with TOOL_DATA_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("tool_data.json must contain a JSON list.")
    return data

@tool
def match_placement_companies(cgpa: float, active_backlogs: int, department: str) -> dict:
    """Match a student's CGPA, active backlog count, and department against every company criterion.

    Args:
        cgpa: Student CGPA on a 0-to-10 scale.
        active_backlogs: Number of current active backlogs; must be a non-negative integer.
        department: Department code; accepted values are CSE, IT, ECE, and MECH.

    Returns:
        A dictionary with eligible_companies and ineligible_companies.
    """
    if isinstance(cgpa, bool) or not isinstance(cgpa, (int, float)) or not 0 <= cgpa <= 10:
        raise ValueError("CGPA must be between 0 and 10.")
    if isinstance(active_backlogs, bool) or not isinstance(active_backlogs, int) or active_backlogs < 0:
        raise ValueError("Active backlogs must be a non-negative whole number.")
    normalized_department = str(department).strip().upper()
    if normalized_department not in SUPPORTED_DEPARTMENTS:
        raise ValueError("Department must be one of: CSE, IT, ECE, MECH.")

    eligible = []
    ineligible = []
    for company in _load_companies():
        name = company["company"]
        reasons = []
        if cgpa < float(company["min_cgpa"]):
            reasons.append(
                f"CGPA {cgpa:g} is below the minimum {float(company['min_cgpa']):g}."
            )
        if active_backlogs > int(company["max_backlogs"]):
            reasons.append(
                f"Active backlogs ({active_backlogs}) exceed the maximum {int(company['max_backlogs'])}."
            )
        permitted = [str(item).upper() for item in company["departments"]]
        if normalized_department not in permitted:
            reasons.append(
                f"Department {normalized_department} is not permitted; allowed departments: {', '.join(permitted)}."
            )
        if reasons:
            ineligible.append({"company": name, "reasons": reasons})
        else:
            eligible.append(name)

    eligible.sort()
    ineligible.sort(key=lambda item: item["company"])
    return {
        "eligible_companies": eligible,
        "ineligible_companies": ineligible,
        "notice": "Eligibility matching only; it does not guarantee an interview or employment."
    }
