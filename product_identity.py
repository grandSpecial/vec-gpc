"""Conservative product-family constraints for reviewed, explicit receipt names.

These rules preserve GPC bricks and storage variants. They deliberately abstain
on accessories, mixes, composite foods and special-use products; they are not a
general food/non-food classifier.
"""
import re

from taxonomy import BRICK_PATHS


IDENTITY_RULE_VERSION = "receipt-identity-v1"
FAMILIES = {
    "sweet_muffins": tuple(code for code, path in BRICK_PATHS.items()
                           if path[0] == "Food/Beverage" and path[-1].startswith("Cakes - Sweet (")),
    "pizza": tuple(code for code, path in BRICK_PATHS.items()
                  if path[0] == "Food/Beverage" and path[-1].startswith("Pies/Pastries/Pizzas/Quiches - Savoury (")),
    "sparkling_water": tuple(code for code, path in BRICK_PATHS.items()
                             if path[0] == "Food/Beverage" and path[-1].startswith("Packaged Water - ")),
    "snack_chips": (10000177,),
    "bread_crumbs": (10000157, 10000069, 10000158),
    "pepper_rings": (10000244,),
}

# Only an explicit product at the end of the name (optionally followed by
# quantities/storage) qualifies. "Pizza chips" and "muffin pans" do not match.
SUFFIX = r"(?:\s+(?:\d+(?:\.\d+)?|g|kg|ml|l|oz|lb|pack|pk|ct|count|frozen|fresh|perishable|shelf|stable))*\s*$"
SPECIAL_CONTEXT = re.compile(
    r"\b(?:cannabis|thc|cbd|hemp|marijuana|tobacco|nicotine|cat|dog|pet|kitten|puppy|"
    r"mix|mixes|flour|dough|batter|kit|kits|maker|makers|machine|oven|pan|pans|"
    r"tray|trays|tin|tins|mould|mold|cutter|stone|scented|scent|candle|toy)\b"
)


def explicit_product_family(text):
    text = re.sub(r"[^a-z0-9.,]+", " ", text.lower()).strip()
    if SPECIAL_CONTEXT.search(text):
        return None
    if re.search(r"\b(?:tortilla|kettle|sour cream) chips?" + SUFFIX, text) and not re.search(
        r"\b(?:frozen|perishable|refrigerated|chilled)\b", text
    ):
        return "snack_chips"
    if re.search(r"\b(?:bread crumbs?|breadcrumbs?)" + SUFFIX, text):
        return "bread_crumbs"
    # The reviewed receipt convention refers to preserved pepper rings. Fresh
    # or frozen wording abstains rather than silently changing product state.
    if re.search(r"\bhot pepper rings?" + SUFFIX, text) and not re.search(r"\b(?:fresh|frozen|cut)\b", text):
        return "pepper_rings"
    if re.search(r"\bmuffins?" + SUFFIX, text):
        # English and savoury muffins cannot safely be assumed to be sweet cakes.
        if not re.search(r"\b(?:english|savoury|savory|cheese|cheddar|egg|corn|cornbread)\b", text):
            return "sweet_muffins"
    if re.search(r"\bpizzas?" + SUFFIX, text):
        return "pizza"
    # A comma-separated flavour is common in parsed receipts. It is still a
    # water purchase; unknown trailing product nouns without a comma abstain.
    if re.search(r"\bsparkling water" + SUFFIX, text) or re.search(
        r"\bsparkling water\s*,\s*(?:orange|lemon|lime|grapefruit|berry|peach|unflavou?red)" + SUFFIX, text
    ):
        return "sparkling_water"
    return None
