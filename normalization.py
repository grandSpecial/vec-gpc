import re
from dataclasses import dataclass


NORMALIZATION_VERSION = "receipt-normalization-v1"

PHRASE_EXPANSIONS = {
    "GRND BEEF": "ground beef",
    "GROUND BEEF": "ground beef",
    "BNLS CHKN": "boneless chicken",
    "BNLS CHICKEN": "boneless chicken",
    "CHKN BRST": "chicken breast",
    "CHICKEN BRST": "chicken breast",
    "WHT BRD": "white bread",
    "WHL WHT BRD": "whole wheat bread",
    "ORG APPL": "organic apple",
    "ORG BNNA": "organic banana",
    "2% MLK": "milk",
}

TOKEN_EXPANSIONS = {
    "APPL": "apple",
    "APL": "apple",
    "APPLE": "apple",
    "BNNA": "banana",
    "BANA": "banana",
    "BAN": "banana",
    "ORG": "organic",
    "GRND": "ground",
    "GND": "ground",
    "BEEF": "beef",
    "CHKN": "chicken",
    "CHK": "chicken",
    "CHICK": "chicken",
    "BRST": "breast",
    "BNLS": "boneless",
    "SKNLS": "skinless",
    "PRK": "pork",
    "TURKY": "turkey",
    "SAUS": "sausage",
    "BRD": "bread",
    "WHT": "white",
    "WHL": "whole",
    "TORT": "tortilla",
    "TORTILLA": "tortilla",
    "TORTILLAS": "tortilla",
    "BUN": "bread bun",
    "BUNS": "bread buns",
    "HOTDOG": "hot dog",
    "HOTDOGS": "hot dogs",
    "BGL": "bagel",
    "BAGL": "bagel",
    "BAGEL": "bagel",
    "BAGELS": "bagel",
    "SCNE": "bread scone",
    "SCONE": "bread scone",
    "SCONES": "bread scone",
    "THINS": "bread thins",
    "TOAST": "bread toast",
    "LOAF": "bread loaf",
    "BAGUETTES": "baguette",
    "BAGUETTE": "baguette",
    "CHDR": "cheddar",
    "CHED": "cheddar",
    "MLK": "milk",
    "2%": "",
    "1%": "",
    "0%": "",
    "YOG": "yogurt",
    "YOGT": "yogurt",
    "YOGURT": "yogurt",
    "EGG": "egg",
    "EGGS": "eggs",
    "CHS": "cheese",
    "CHEESE": "cheese",
    "MOZZ": "mozzarella",
    "CRM": "cream",
    "BUTTR": "butter",
    "FRZ": "frozen",
    "FZN": "frozen",
    "FROZ": "frozen",
    "FRZN": "frozen",
    "RFG": "refrigerated",
    "REFRIG": "refrigerated",
    "TOM": "tomato",
    "TOMS": "tomatoes",
    "POT": "potato",
    "POTS": "potatoes",
    "CAR": "carrot",
    "LETT": "lettuce",
    "SPIN": "spinach",
    "AVO": "avocado",
    "STRW": "strawberry",
    "STRAWB": "strawberry",
    "BLUB": "blueberry",
    "BLUEB": "blueberry",
    "COKE": "cola soft drink",
    "POP": "soft drink",
    "SODA": "soft drink",
    "WTR": "water",
    "JCE": "juice",
    "OJ": "orange juice",
    "COF": "coffee",
    "TEA": "tea",
}

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9%]+")
SPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class NormalizedQuery:
    input_text: str
    normalized_text: str
    expansions: list[dict[str, str]]
    version: str = NORMALIZATION_VERSION


def _normalize_spacing(text: str) -> str:
    return SPACE_PATTERN.sub(" ", text).strip()


def normalize_receipt_text(text: str) -> NormalizedQuery:
    raw_text = text or ""
    compact_input = _normalize_spacing(raw_text)
    upper_text = compact_input.upper()

    phrase_expansion = PHRASE_EXPANSIONS.get(upper_text)
    if phrase_expansion:
        return NormalizedQuery(
            input_text=raw_text,
            normalized_text=phrase_expansion,
            expansions=[{"from": compact_input, "to": phrase_expansion, "type": "phrase"}],
        )

    tokens = TOKEN_PATTERN.findall(upper_text)
    expanded_tokens = []
    expansions = []
    for token in tokens:
        expanded = TOKEN_EXPANSIONS.get(token, token.lower())
        if not expanded:
            expansions.append({"from": token, "to": "", "type": "drop"})
            continue
        expanded_tokens.append(expanded)
        if expanded != token.lower():
            expansions.append({"from": token, "to": expanded, "type": "token"})

    normalized_text = _normalize_spacing(" ".join(expanded_tokens) or compact_input.lower())
    return NormalizedQuery(
        input_text=raw_text,
        normalized_text=normalized_text,
        expansions=expansions,
    )
