import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from sqlalchemy import select

from display_mapping import display_labels_for_gpc
from models import GPCLevel, Items
from normalization import NormalizedQuery, normalize_receipt_text


FINAL_GPC_LEVEL = 4
LOG_CANDIDATE_LIMIT = 5
SEARCH_CANDIDATE_LIMIT = 35
TAXONOMY_PATH = Path(__file__).with_name("GPC_v20240603.json")
RERANKER_VERSION = "gpc-reranker-v1"
CONFIDENCE_HIGH_THRESHOLD = 0.42
CONFIDENCE_LOW_THRESHOLD = 0.30
BAKERY_PRODUCT_TERMS = {
    "bagel",
    "baguette",
    "bread",
    "brioche",
    "bun",
    "buns",
    "loaf",
    "naan",
    "scone",
    "sourdough",
    "toast",
    "tortilla",
}


@dataclass(frozen=True)
class ClassificationCandidateRow:
    item_id: int
    gpc_item: GPCLevel
    similarity_score: float
    rerank_score: float
    matched_raw_title: str
    raw_gpc_code: int
    raw_gpc_level: int
    reasons: list[str]


def build_level4_ancestor_index():
    """Map every raw GPC code to the Level 4 product brick(s) it appears under."""
    if not TAXONOMY_PATH.exists():
        return {}

    with TAXONOMY_PATH.open("r") as taxonomy_file:
        taxonomy = json.load(taxonomy_file)

    index = {}

    def walk(nodes, current_level4_code=None):
        for node in nodes:
            level = node.get("Level")
            code = node.get("Code")
            next_level4_code = code if level == FINAL_GPC_LEVEL else current_level4_code

            if code is not None and next_level4_code is not None:
                index.setdefault(code, [])
                if next_level4_code not in index[code]:
                    index[code].append(next_level4_code)

            walk(node.get("Childs") or [], next_level4_code)

    walk(taxonomy.get("Schema") or [])
    return index


GPC_CODE_TO_LEVEL4_CODES = build_level4_ancestor_index()


def similarity_from_distance(distance):
    if distance is None:
        return 0.0
    return float(1 - distance)


def is_generic_attribute_candidate(gpc_item):
    title = (gpc_item.title or "").strip().upper()
    return (
        title.startswith("IF ")
        or title in {"UNCLASSIFIED", "UNIDENTIFIED", "OTHER"}
        or "UNCLASSIFIED" in title
        or "UNIDENTIFIED" in title
    )


def choose_level4_item(candidate_level4_items, text: str):
    if not candidate_level4_items:
        return None

    text_lower = text.lower()
    state_preferences = []
    if "frozen" in text_lower:
        state_preferences.append("frozen")
    if any(word in text_lower for word in ("fresh", "refrigerated", "perishable")):
        state_preferences.append("perishable")
    state_preferences.append("shelf stable")

    for preference in state_preferences:
        for item in candidate_level4_items:
            if preference in (item.title or "").lower():
                return item

    return candidate_level4_items[0]


def promote_to_level4(gpc_item, db, text: str):
    if gpc_item.level == FINAL_GPC_LEVEL:
        return gpc_item

    if is_generic_attribute_candidate(gpc_item):
        return None

    level4_codes = GPC_CODE_TO_LEVEL4_CODES.get(gpc_item.code, [])
    if level4_codes:
        level4_items = (
            db.query(GPCLevel)
            .filter(GPCLevel.level == FINAL_GPC_LEVEL, GPCLevel.code.in_(level4_codes))
            .all()
        )
        level4_items_by_code = {item.code: item for item in level4_items}
        ordered_level4_items = [
            level4_items_by_code[code]
            for code in level4_codes
            if code in level4_items_by_code
        ]
        chosen_item = choose_level4_item(ordered_level4_items, text)
        if chosen_item is not None:
            return chosen_item

    current_item = gpc_item
    while current_item and current_item.level > FINAL_GPC_LEVEL:
        current_item = db.query(GPCLevel).filter_by(id=current_item.parent_id).first()

    if current_item and current_item.level == FINAL_GPC_LEVEL:
        return current_item

    return None


def _query_terms(normalized_query: NormalizedQuery) -> set[str]:
    terms = set(normalized_query.normalized_text.lower().split())
    for expansion in normalized_query.expansions:
        terms.update(expansion["to"].lower().split())
    return {term for term in terms if len(term) > 2}


def rerank_candidate(
    raw_gpc_item,
    final_gpc_item,
    similarity_score: float,
    normalized_query: NormalizedQuery,
):
    terms = _query_terms(normalized_query)
    title_text = f"{raw_gpc_item.title or ''} {final_gpc_item.title or ''}".lower()
    final_title = (final_gpc_item.title or "").lower()
    path_text = (final_gpc_item.full_title or "").lower()
    definition_text = (final_gpc_item.definition or "").lower()
    combined_final_text = f"{final_gpc_item.title or ''} {final_gpc_item.full_title or ''}".lower()
    reasons = []
    score = similarity_score

    exact_hits = [term for term in terms if term in title_text]
    if exact_hits:
        score += 0.08 + min(len(exact_hits), 3) * 0.02
        reasons.append(f"title_term_match:{','.join(exact_hits[:3])}")

    final_title_prefix_hits = [
        term
        for term in terms
        if final_title.startswith(f"{term} ") or final_title.startswith(f"{term}(")
    ]
    if final_title_prefix_hits:
        score += 0.10
        reasons.append(f"final_title_prefix_match:{','.join(final_title_prefix_hits[:3])}")

    path_hits = [term for term in terms if term in path_text]
    if path_hits:
        score += 0.03
        reasons.append(f"path_term_match:{','.join(path_hits[:3])}")

    definition_hits = [term for term in terms if term in definition_text]
    if definition_hits:
        score += 0.03
        reasons.append(f"definition_term_match:{','.join(definition_hits[:3])}")

    if raw_gpc_item.level > FINAL_GPC_LEVEL:
        score += 0.04
        reasons.append("specific_attribute_promoted")

    bakery_term_hits = BAKERY_PRODUCT_TERMS & terms
    if bakery_term_hits:
        if "bread/bakery products" in combined_final_text or final_title.startswith("bread"):
            score += 0.12
            reasons.append(f"bakery_product_boost:{','.join(sorted(bakery_term_hits)[:3])}")
        elif "food/beverage" not in combined_final_text:
            score -= 0.12
            reasons.append(f"bakery_product_non_food_penalty:{','.join(sorted(bakery_term_hits)[:3])}")
        elif "food/beverage beverages" in combined_final_text:
            score -= 0.10
            reasons.append(f"bakery_product_beverage_penalty:{','.join(sorted(bakery_term_hits)[:3])}")

    if "alternative" in combined_final_text and not (
        {"alternative", "plant", "vegan", "vegetarian", "substitute"} & terms
    ):
        score -= 0.08
        reasons.append("alternative_penalty")

    if "by-products" in combined_final_text and not ({"byproduct", "byproducts"} & terms):
        score -= 0.10
        reasons.append("byproducts_penalty")

    if (
        "alcoholic" in combined_final_text
        and "non alcoholic" not in combined_final_text
        and "non-alcoholic" not in combined_final_text
        and not (
        {"alcohol", "alcoholic", "beer", "wine", "liquor", "vodka", "rum"} & terms
        )
    ):
        score -= 0.08
        reasons.append("alcohol_penalty")

    if not final_gpc_item.active:
        score -= 0.05
        reasons.append("inactive_penalty")

    return score, reasons


def confidence_from_ranked_candidates(ranked_candidates):
    if not ranked_candidates:
        return 0.0
    top_score = ranked_candidates[0].rerank_score
    runner_up = ranked_candidates[1].rerank_score if len(ranked_candidates) > 1 else 0.0
    margin = max(0.0, top_score - runner_up)
    confidence = min(0.99, max(0.0, top_score + margin))
    return round(confidence, 4)


def status_from_confidence(confidence: float):
    if confidence >= CONFIDENCE_HIGH_THRESHOLD:
        return "classified", False
    if confidence >= CONFIDENCE_LOW_THRESHOLD:
        return "classified", True
    return "uncertain", True


def candidate_to_debug(row: ClassificationCandidateRow):
    labels = display_labels_for_gpc(row.gpc_item)
    return {
        "gpc_id": row.gpc_item.id,
        "gpc_code": row.gpc_item.code,
        "gpc_title": row.gpc_item.title,
        "gpc_full_title": row.gpc_item.full_title,
        "category": labels.category,
        "subcategory": labels.subcategory,
        "similarity_score": row.similarity_score,
        "rerank_score": round(row.rerank_score, 4),
        "matched_raw_title": row.matched_raw_title,
        "raw_gpc_code": row.raw_gpc_code,
        "raw_gpc_level": row.raw_gpc_level,
        "reasons": row.reasons,
    }


class GPCClassifier:
    def __init__(
        self,
        db,
        create_description,
        create_vector,
        log_candidate_limit=LOG_CANDIDATE_LIMIT,
        search_candidate_limit=SEARCH_CANDIDATE_LIMIT,
    ):
        self.db = db
        self.create_description = create_description
        self.create_vector = create_vector
        self.log_candidate_limit = log_candidate_limit
        self.search_candidate_limit = search_candidate_limit

    def classify(self, text: str, include_candidates: bool = False):
        started_at = time.time()
        normalized_query = normalize_receipt_text(text)
        description_response = self.create_description(normalized_query.normalized_text)
        description = description_response.choices[0].message.content.strip()
        vector = self.create_vector(description)
        ranked_candidates = self._retrieve_and_rank(vector, normalized_query)

        if not ranked_candidates:
            raise HTTPException(status_code=404, detail="GPCLevel item not found")

        winning_candidate = ranked_candidates[0]
        gpc_item = winning_candidate.gpc_item
        display_labels = display_labels_for_gpc(gpc_item)
        confidence = confidence_from_ranked_candidates(ranked_candidates)
        status, needs_review = status_from_confidence(confidence)

        response = {
            "id": gpc_item.id,
            "code": gpc_item.code,
            "title": gpc_item.title,
            "full_title": gpc_item.full_title,
            "level_2_category": display_labels.category,
            "level_3_category": display_labels.subcategory,
            "category": display_labels.category,
            "subcategory": display_labels.subcategory,
            "display_label": display_labels.display_label,
            "description": description,
            "input_text": normalized_query.input_text,
            "normalized_text": normalized_query.normalized_text,
            "normalization": {
                "version": normalized_query.version,
                "expansions": normalized_query.expansions,
            },
            "definition": gpc_item.definition,
            "active": gpc_item.active,
            "confidence": confidence,
            "status": status,
            "needs_review": needs_review,
            "display_mapping_version": display_labels.version,
            "display_mapping_source": display_labels.source,
            "reranker_version": RERANKER_VERSION,
            "latency_ms": int((time.time() - started_at) * 1000),
            "log_candidates": ranked_candidates[: self.log_candidate_limit],
        }

        if include_candidates:
            response["candidates"] = [
                candidate_to_debug(candidate)
                for candidate in ranked_candidates[: self.log_candidate_limit]
            ]

        return response

    def _retrieve_and_rank(self, vector: np.ndarray, normalized_query: NormalizedQuery):
        distance_expr = Items.vector.cosine_distance(vector).label("distance")
        candidate_results = self.db.execute(
            select(Items, GPCLevel, distance_expr)
            .join(GPCLevel, GPCLevel.id == Items.id)
            .order_by(distance_expr)
            .limit(self.search_candidate_limit)
        ).all()

        if not candidate_results:
            raise HTTPException(status_code=404, detail="No matching item found")

        candidates_by_final_id = {}
        for item, raw_gpc_item, distance in candidate_results:
            final_gpc_item = promote_to_level4(
                raw_gpc_item,
                self.db,
                normalized_query.normalized_text,
            )
            if final_gpc_item is None:
                continue

            similarity_score = similarity_from_distance(distance)
            rerank_score, reasons = rerank_candidate(
                raw_gpc_item,
                final_gpc_item,
                similarity_score,
                normalized_query,
            )
            candidate = ClassificationCandidateRow(
                item_id=final_gpc_item.id,
                gpc_item=final_gpc_item,
                similarity_score=similarity_score,
                rerank_score=rerank_score,
                matched_raw_title=raw_gpc_item.title,
                raw_gpc_code=raw_gpc_item.code,
                raw_gpc_level=raw_gpc_item.level,
                reasons=reasons,
            )

            existing = candidates_by_final_id.get(final_gpc_item.id)
            if existing is None or candidate.rerank_score > existing.rerank_score:
                candidates_by_final_id[final_gpc_item.id] = candidate

        return sorted(
            candidates_by_final_id.values(),
            key=lambda candidate: candidate.rerank_score,
            reverse=True,
        )
