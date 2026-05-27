import re
from dataclasses import dataclass


DISPLAY_MAPPING_VERSION = "display-mapping-v1"

STATE_SUFFIX_PATTERN = re.compile(
    r"\s*\((Frozen|Shelf Stable|Perishable|Chilled|Refrigerated)\)\s*$",
    re.IGNORECASE,
)
PROCESSED_SUFFIX_PATTERN = re.compile(
    r"\s*-\s*(Prepared/Processed|Unprepared/Unprocessed|Natural/Extruded).*?$",
    re.IGNORECASE,
)

LEVEL2_CATEGORY_OVERRIDES = {
    "Bread/Bakery Products": "Bakery",
    "Cereal/Grain/Pulse Products": "Pantry",
    "Confectionery/Sugar Sweetening Products": "Snacks & Candy",
    "Beverages": "Beverages",
    "Fruits - Unprepared/Unprocessed (Fresh)": "Produce",
    "Vegetables (Non Leaf) - Unprepared/Unprocessed (Fresh)": "Produce",
    "Leaf Vegetables - Unprepared/Unprocessed (Fresh)": "Produce",
    "Meat/Poultry/Other Animals": "Meat & Poultry",
    "Milk/Butter/Cream/Yogurts/Cheese/Eggs/Substitutes": "Dairy & Eggs",
    "Prepared/Preserved Foods": "Prepared Foods",
    "Seasonings/Preservatives/Extracts": "Pantry",
    "Sports Equipment": "Sports Equipment",
    "Sports/Recreational Equipment": "Sports Equipment",
}

LEVEL3_SUBCATEGORY_OVERRIDES = {
    "Bread": "Bread",
    "Bread/Bakery Products Variety Packs": "Bakery",
    "Sweet Bakery Products": "Desserts & Pastries",
    "Savoury Bakery Products": "Savory Baked Goods",
    "Biscuits/Cookies": "Cookies & Crackers",
    "Cheese/Cheese Substitutes": "Cheese",
    "Milk/Milk Substitutes": "Milk",
    "Yogurt/Yogurt Substitutes": "Yogurt",
    "Eggs/Eggs Substitutes": "Eggs",
    "Beef - Prepared/Processed": "Beef",
    "Chicken - Prepared/Processed": "Chicken",
    "Pork - Prepared/Processed": "Pork",
    "Meat/Poultry/Other Animals Sausages - Prepared/Processed": "Sausages",
    "Pome Fruits": "Apples & Pears",
    "Bananas": "Bananas",
    "Citrus": "Citrus",
    "Berries/Small Fruit": "Berries",
    "Soft Drinks": "Soft Drinks",
    "Non Alcoholic Beverages - Ready to Drink": "Soft Drinks",
    "Drinks Flavoured - Ready to Drink": "Soft Drinks",
    "Beverages Variety Packs": "Beverage Packs",
    "Waters": "Water",
    "Coffee": "Coffee",
    "Tea": "Tea",
    "Juice - Drinks": "Juice",
    "Baseball/Softball": "Baseball Equipment",
}

TITLE_SUBCATEGORY_OVERRIDES = {
    "Apples": "Apples",
    "Bananas": "Bananas",
    "Tomatillos": "Tomatillos",
    "Bread": "Bread",
    "Breadfruits": "Fruit",
    "Cheese": "Cheese",
}


@dataclass(frozen=True)
class DisplayLabels:
    category: str
    subcategory: str
    display_label: str
    source: str
    version: str = DISPLAY_MAPPING_VERSION


def split_gpc_path(full_title: str | None) -> list[str]:
    if not full_title:
        return []
    if " > " in full_title:
        return [part.strip() for part in full_title.split(" > ") if part.strip()]

    # Current imported data stores paths as space-joined titles. For known GPC
    # roots this heuristic recovers enough structure for deterministic display.
    text = full_title.strip()
    roots = [
        "Food/Beverage",
        "Sports Equipment",
        "Sports/Recreational Equipment",
        "Clothing",
        "Home Appliances",
        "Healthcare",
    ]
    for root in roots:
        if text.startswith(root):
            remainder = text[len(root):].strip()
            return [root] + _split_known_remainder(remainder)
    return [text]


def _split_known_remainder(remainder: str) -> list[str]:
    known_level2 = sorted(LEVEL2_CATEGORY_OVERRIDES, key=len, reverse=True)
    for level2 in known_level2:
        if remainder.startswith(level2):
            rest = remainder[len(level2):].strip()
            return [level2] + _split_known_level3(rest)
    return [remainder] if remainder else []


def _split_known_level3(remainder: str) -> list[str]:
    known_level3 = sorted(LEVEL3_SUBCATEGORY_OVERRIDES, key=len, reverse=True)
    for level3 in known_level3:
        if remainder.startswith(level3):
            rest = remainder[len(level3):].strip()
            return [level3] + ([rest] if rest else [])
    return [remainder] if remainder else []


def clean_gpc_title(title: str | None) -> str:
    if not title:
        return "Unclassified"
    cleaned = STATE_SUFFIX_PATTERN.sub("", title).strip()
    cleaned = PROCESSED_SUFFIX_PATTERN.sub("", cleaned).strip()
    cleaned = cleaned.replace(" / ", "/")
    return cleaned or title


def display_labels_for_gpc(gpc_item) -> DisplayLabels:
    path = split_gpc_path(gpc_item.full_title)
    level2_title = path[1] if len(path) > 1 else None
    level3_title = path[2] if len(path) > 2 else None
    cleaned_title = clean_gpc_title(gpc_item.title)

    category = (
        LEVEL2_CATEGORY_OVERRIDES.get(level2_title or "")
        or getattr(gpc_item, "level_2_category", None)
        or level2_title
        or "Unclassified"
    )
    subcategory = (
        TITLE_SUBCATEGORY_OVERRIDES.get(cleaned_title)
        or LEVEL3_SUBCATEGORY_OVERRIDES.get(level3_title or "")
        or cleaned_title
    )

    if subcategory == category and cleaned_title != category:
        subcategory = cleaned_title

    return DisplayLabels(
        category=category,
        subcategory=subcategory,
        display_label=cleaned_title,
        source="rules",
    )
