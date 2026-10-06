"""Recognize explicit container-deposit lines without inventing GPC products."""
import re

from display_mapping import DISPLAY_MAPPING_VERSION


DEPOSIT_PATTERN = re.compile(
    r"\b(?:beverage|container|bottle|can)\b.*\bdeposits?\s*$", re.IGNORECASE
)


def receipt_charge_result(query, include_candidates=False):
    if not DEPOSIT_PATTERN.search(query.normalized_text):
        return None
    result = {
        "id": None, "code": None, "title": "Container deposit", "full_title": None,
        "level_2_category": "Deposits & Fees", "category": "Deposits & Fees",
        "level_3_category": "Container Deposits", "subcategory": "Container Deposits",
        "display_label": "Container deposit", "description": "Container deposit charge",
        "description_fallback": False, "selection_source": "receipt_rule", "selection_fallback": False,
        "input_text": query.input_text, "normalized_text": query.normalized_text,
        "normalization": {"version": query.version, "expansions": query.expansions},
        "definition": None, "active": None, "confidence": None,
        "status": "non_product", "needs_review": False,
        "display_mapping_version": DISPLAY_MAPPING_VERSION,
        "display_mapping_source": "receipt_rule", "identity_rule": "container_deposit",
        "log_candidates": [],
    }
    if include_candidates:
        result["candidates"] = []
    return result
