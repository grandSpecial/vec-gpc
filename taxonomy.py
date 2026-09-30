"""Canonical product ancestry, without collapsing shared attribute occurrences."""
import json
import re
from functools import lru_cache
from pathlib import Path


def _words(text):
    return {w.rstrip("s") for w in re.findall(r"[a-z]+", text.lower())}


def load_taxonomy():
    with Path(__file__).with_name("GPC_v20240603.json").open() as handle:
        schema = json.load(handle)["Schema"]
    paths = {}
    aliases = {}

    def walk(nodes, ancestors=()):
        for node in nodes:
            path = ancestors + (node["Title"],)
            if node["Level"] == 4:
                paths[node["Code"]] = path
                identity = re.sub(r"\([^)]*\)", "", node["Title"])
                identity_words = _words(identity)
                for attribute in node.get("Childs") or []:
                    title = attribute["Title"]
                    if not title.lower().startswith("type of "):
                        continue
                    subject = _words(title[8:])
                    # Type of Bread describes bread; Type of Flour under bread
                    # describes an ingredient and must not retrieve bread for flour.
                    if not subject or not subject <= identity_words:
                        continue
                    for value in attribute.get("Childs") or []:
                        alias = value["Title"].lower()
                        if alias not in {"unidentified", "unclassified", "other", "yes", "no"}:
                            aliases.setdefault(alias, set()).add(node["Code"])
            elif node["Level"] < 4:
                walk(node.get("Childs") or [], path)

    walk(schema)
    return paths, aliases


BRICK_PATHS, PRODUCT_TYPE_ALIASES = load_taxonomy()
CROP_CODES = tuple(code for code, path in BRICK_PATHS.items() if path[0] == "Crops")
PET_CODES = tuple(code for code, path in BRICK_PATHS.items() if path[0].startswith("Pet "))


@lru_cache(maxsize=1024)
def product_type_matches(text):
    """Additional product candidates from exact taxonomy product-type phrases."""
    words = " " + " ".join(re.findall(r"[a-z0-9]+", text.lower())) + " "
    matches = {}
    for alias, codes in PRODUCT_TYPE_ALIASES.items():
        phrase = " " + " ".join(re.findall(r"[a-z0-9]+", alias)) + " "
        if phrase.strip() and phrase in words:
            for code in codes:
                matches.setdefault(code, []).append(alias)
    return matches


def product_type_codes(text):
    return set(product_type_matches(text))
