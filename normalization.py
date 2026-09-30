import re
from dataclasses import dataclass


NORMALIZATION_VERSION = "receipt-normalization-v2"

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
    "2% MLK": "2% milk",
    "SANDWICH THINS": "thin sandwich bread",
}

TOKEN_EXPANSIONS = {
    "APPL": "apple",
    "APL": "apple",
    "APPLE": "apple",
    "BNNA": "banana",
    "BANA": "banana",
    "ORG": "organic",
    "GRND": "ground",
    "GND": "ground",
    "BEEF": "beef",
    "CHKN": "chicken",
    "CHK": "chicken",
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
    "TOAST": "bread toast",
    "LOAF": "bread loaf",
    "BAGUETTES": "baguette",
    "BAGUETTE": "baguette",
    "CHDR": "cheddar",
    "CHED": "cheddar",
    "MLK": "milk",
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
    "TOMS": "tomatoes",
    "POTS": "potatoes",
    "LETT": "lettuce",
    "SPIN": "spinach",
    "AVO": "avocado",
    "STRW": "strawberry",
    "STRAWB": "strawberry",
    "BLUB": "blueberry",
    "BLUEB": "blueberry",
    "COKE": "cola soft drink",
    "WTR": "water",
    "JCE": "juice",
    "OJ": "orange juice",
    "COF": "coffee",
    "TEA": "tea",
}

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:\.[0-9]+)?%?")
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

    expansions = []
    # Longest phrases first; work within longer receipt lines too (e.g. sizes).
    for phrase, replacement in sorted(PHRASE_EXPANSIONS.items(), key=lambda p: -len(p[0])):
        pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
        if re.search(pattern, upper_text):
            upper_text = re.sub(pattern, replacement, upper_text)
            expansions.append({"from": phrase, "to": replacement, "type": "phrase"})

    def expand_token(match):
        token = match.group()
        # Lowercase spans were already expanded as phrases.
        if token != token.upper():
            return token
        expanded = TOKEN_EXPANSIONS.get(token, token.lower())
        if expanded != token.lower():
            expansions.append({"from": token, "to": expanded, "type": "token"})
        return expanded

    normalized_text = _normalize_spacing(TOKEN_PATTERN.sub(expand_token, upper_text).lower())
    return NormalizedQuery(
        input_text=raw_text,
        normalized_text=normalized_text,
        expansions=expansions,
    )
