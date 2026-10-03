import json
import math
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json_file(path: str | Path) -> Any:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    with candidate.open(encoding="utf-8") as file:
        return json.load(file)


def _first_match(patterns: list[str], text: str) -> re.Match[str] | None:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match
    return None


def extract_parameters(query: str, config: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    text = query.strip()

    price_match = _first_match(
        [
            r"(?:\$|usd\s*)(\d+(?:\.\d{1,2})?)\b",
            r"\b(?:paid|cost(?:s|ing)?|price(?:\s+was)?)\s*(\d+(?:\.\d{1,2})?)\b",
        ],
        text,
    )
    if price_match:
        price = float(price_match.group(1))
        if math.isfinite(price):
            params["purchase_price"] = price

    lowered = text.lower()
    if "yesterday" in lowered:
        params["days_since_delivery"] = 1
    elif "last week" in lowered:
        params["days_since_delivery"] = 7
    else:
        days_match = _first_match(
            [
                r"\b(?:delivered\s*)?(\d+)\s*days?\s*ago\b",
                r"\b(\d+)\s*days?\s*(?:since|from)\s+delivery\b",
                r"\bpast\s+(\d+)\s*days?\b",
            ],
            text,
        )
        if days_match:
            params["days_since_delivery"] = int(days_match.group(1))

    if re.search(r"\b(unopened|sealed|new[ -]in[ -]box|not\s+opened)\b", lowered):
        params["opened"] = False
    elif re.search(r"\b(opened|used|open[ -]box)\b", lowered):
        params["opened"] = True

    for category, synonyms in config["category_synonyms"].items():
        if any(
            re.search(rf"\b{re.escape(term)}s?\b", lowered)
            for term in [category, *synonyms]
        ):
            params["category"] = category
            break

    return params


def missing_refund_parameters(params: dict[str, Any]) -> list[str]:
    required = ("purchase_price", "days_since_delivery", "opened", "category")
    return [name for name in required if name not in params]
