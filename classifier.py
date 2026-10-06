import logging
import re
import time
from dataclasses import dataclass, replace

import numpy as np
from fastapi import HTTPException
from sqlalchemy import select

from display_mapping import display_labels_for_gpc
from models import GPCLevel, Items
from normalization import NormalizedQuery, normalize_receipt_text
from taxonomy import CROP_CODES, PET_CODES, product_type_codes
from product_identity import FAMILIES, IDENTITY_RULE_VERSION, explicit_product_family
from receipt_charges import receipt_charge_result


FINAL_GPC_LEVEL = 4
LOG_CANDIDATE_LIMIT = 5
SEARCH_CANDIDATE_LIMIT = 35
RERANKER_VERSION = "gpc-reranker-v3"
logger = logging.getLogger(__name__)
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


def similarity_from_distance(distance):
    return float(1 - distance) if distance is not None else 0.0


def word_terms(text):
    def singular(term):
        if len(term) > 4 and term.endswith("ies"):
            return term[:-3] + "y"
        if len(term) > 3 and term.endswith("s") and not term.endswith(("ss", "us", "is")):
            return term[:-1]
        return term
    return {singular(term) for term in re.findall(r"[a-z0-9]+", text.lower())}


def _query_terms(normalized_query: NormalizedQuery) -> set[str]:
    return {term for term in word_terms(normalized_query.normalized_text) if len(term) > 2}


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

    exact_hits = sorted(terms & word_terms(title_text))
    if exact_hits:
        score += min(len(exact_hits), 3) * 0.02
        reasons.append(f"title_term_match:{','.join(exact_hits[:3])}")

    # Whole title evidence is useful; a prefix can just be a flavour/ingredient.
    identity_title = re.sub(r"\([^)]*\)", "", final_title).strip()
    if word_terms(identity_title) and word_terms(identity_title) <= terms:
        score += 0.08
        reasons.append("whole_title_match")

    path_hits = sorted(terms & word_terms(path_text))
    if path_hits:
        score += 0.03
        reasons.append(f"path_term_match:{','.join(path_hits[:3])}")

    definition_hits = sorted(terms & word_terms(definition_text))
    if definition_hits:
        score += 0.03
        reasons.append(f"definition_term_match:{','.join(definition_hits[:3])}")

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
    """Conservative ranking signal, not a calibrated probability of correctness."""
    if not ranked_candidates:
        return 0.0
    top = ranked_candidates[0]
    similarity = max(0.0, min(0.99, top.similarity_score))
    if len(ranked_candidates) == 1:
        return round(min(similarity, CONFIDENCE_HIGH_THRESHOLD - 0.01), 4)
    margin = max(0.0, top.rerank_score - ranked_candidates[1].rerank_score)
    # Close alternatives require review even if keyword bonuses made both high.
    if margin < 0.04:
        similarity = min(similarity, CONFIDENCE_HIGH_THRESHOLD - 0.01)
    return round(similarity, 4)


def status_from_confidence(confidence: float):
    if confidence >= CONFIDENCE_HIGH_THRESHOLD:
        return "classified", False
    if confidence >= CONFIDENCE_LOW_THRESHOLD:
        return "classified", True
    return "uncertain", True


def candidate_to_debug(row: ClassificationCandidateRow, ancestor_categories=None, receipt_text=""):
    labels = display_labels_for_gpc(row.gpc_item, ancestor_categories, receipt_text)
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
        select_candidate=None,
    ):
        self.db = db
        self.create_description = create_description
        self.create_vector = create_vector
        self.log_candidate_limit = log_candidate_limit
        self.search_candidate_limit = search_candidate_limit
        self.ancestor_categories = {}
        self.select_candidate = select_candidate

    def classify(self, text: str, include_candidates: bool = False):
        started_at = time.monotonic()
        normalized_query = normalize_receipt_text(text)
        if not normalized_query.normalized_text.strip():
            raise HTTPException(status_code=422, detail="text must not be empty")
        charge = receipt_charge_result(normalized_query, include_candidates)
        if charge is not None:
            charge.update(reranker_version=RERANKER_VERSION, identity_rule_version=IDENTITY_RULE_VERSION,
                          latency_ms=int((time.monotonic() - started_at) * 1000))
            return charge
        description = normalized_query.normalized_text
        description_fallback = False
        if self.create_description is not None:
            try:
                description_response = self.create_description(normalized_query.normalized_text)
                generated = description_response.choices[0].message.content
                if not generated or not generated.strip():
                    raise ValueError("Empty description")
                description = generated.strip()
            except Exception as exc:
                description_fallback = True
                logger.warning("description_fallback error_type=%s", type(exc).__name__)
        # A short optional expansion improves recall for unfamiliar product names.
        # The original ALWAYS remains in the embedding, and final selection sees
        # the original receipt rather than treating the expansion as ground truth.
        vector = self.create_vector(
            f"Product: {normalized_query.normalized_text}\nDescription: {description}"
        )
        ranked_candidates = self._retrieve_and_rank(vector, normalized_query)

        if not ranked_candidates:
            raise HTTPException(status_code=404, detail="GPCLevel item not found")

        selection_fallback = False
        selection_needs_review = False
        selection_source = "heuristic"
        if self.select_candidate is not None:
            try:
                selection = self.select_candidate(normalized_query, ranked_candidates)
                if type(selection.get("needs_review")) is not bool:
                    raise ValueError("Missing selection review flag")
                selected_code = selection.get("code")
                if type(selected_code) is not int:
                    raise ValueError("Invalid selected code")
                chosen = next((row for row in ranked_candidates if row.gpc_item.code == selected_code), None)
                if chosen is None:
                    raise ValueError("Selected code is not a retrieved candidate")
                ranked_candidates = [replace(chosen, reasons=chosen.reasons + ["model_selected"])] + [
                    row for row in ranked_candidates if row is not chosen
                ]
                selection_needs_review = selection["needs_review"]
                selection_source = "model"
                generated = selection.get("description")
                if isinstance(generated, str) and generated.strip():
                    description = generated.strip()[:1024]
            except Exception as exc:
                selection_fallback = True
                selection_needs_review = True
                logger.warning("selection_fallback error_type=%s", type(exc).__name__)

        winning_candidate = ranked_candidates[0]
        gpc_item = winning_candidate.gpc_item
        # One small lookup, not a per-candidate parent walk. Existing retail
        # overrides take precedence over the already curated ancestor labels.
        self.ancestor_categories = dict(self.db.execute(
            select(GPCLevel.title, GPCLevel.level_2_category)
            .where(GPCLevel.level == 2, GPCLevel.level_2_category.isnot(None))
        ).all())
        display_labels = display_labels_for_gpc(gpc_item, self.ancestor_categories, normalized_query.normalized_text)
        confidence = confidence_from_ranked_candidates(ranked_candidates)
        if selection_needs_review:
            confidence = min(confidence, CONFIDENCE_HIGH_THRESHOLD - 0.01)
        status, needs_review = status_from_confidence(confidence)
        needs_review = needs_review or selection_needs_review

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
            "description_fallback": description_fallback,
            "selection_source": selection_source,
            "selection_fallback": selection_fallback,
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
            "identity_rule_version": IDENTITY_RULE_VERSION,
            "identity_rule": explicit_product_family(normalized_query.normalized_text),
            "latency_ms": int((time.monotonic() - started_at) * 1000),
            "log_candidates": ranked_candidates[: self.log_candidate_limit],
        }

        if include_candidates:
            response["candidates"] = [
                candidate_to_debug(candidate, self.ancestor_categories, normalized_query.normalized_text)
                for candidate in ranked_candidates[: self.log_candidate_limit]
            ]

        return response

    def _retrieve_and_rank(self, vector: np.ndarray, normalized_query: NormalizedQuery):
        distance_expr = Items.vector.cosine_distance(vector).label("distance")
        statement = (
            select(Items, GPCLevel, distance_expr)
            .join(GPCLevel, GPCLevel.id == Items.id)
            .where(GPCLevel.level == FINAL_GPC_LEVEL, GPCLevel.active.is_(True), Items.vector.isnot(None))
        )
        terms = word_terms(normalized_query.normalized_text)
        family = explicit_product_family(normalized_query.normalized_text)
        if family:
            # Restrict retrieval itself, so a bad retrieval hint cannot crowd
            # out the known product family. Selection and fallback both inherit
            # the constraint. Explicit special-use/accessory names abstain.
            statement = statement.where(GPCLevel.code.in_(FAMILIES[family]))
        # Retail produce is not the crop used to grow it. Explicit growing inputs
        # retain access to agricultural categories; houseplants remain available.
        if not terms & {"seedling", "seedlings", "plant", "plants", "shrub", "shrubs", "tree", "trees", "growing"}:
            statement = statement.where(GPCLevel.code.notin_(CROP_CODES))
        # Keep an explicit animal audience when ingredients also resemble human food.
        if (terms & {"cat", "dog", "kitten", "puppy"}
                and terms & {"food", "chow", "kibble", "treat", "biscuit", "chew", "bone", "litter", "collar", "salmon", "chicken", "beef", "tuna", "turkey"}) and not (
            {"hot", "dog"} <= terms or {"corn", "dog"} <= terms
        ):
            statement = statement.where(GPCLevel.code.in_(PET_CODES))
        candidate_results = self.db.execute(
            statement.order_by(distance_expr, GPCLevel.code).limit(self.search_candidate_limit)
        ).all()
        # Recover product-type synonyms absent from terse brick titles (tortilla
        # is a Type of Bread). Keep all retrieved parents for model comparison.
        alias_codes = product_type_codes(normalized_query.normalized_text)
        if alias_codes:
            lexical_results = self.db.execute(
                statement.where(GPCLevel.code.in_(alias_codes))
                .order_by(distance_expr, GPCLevel.code).limit(15)
            ).all()
            seen = {row[1].id for row in candidate_results}
            candidate_results.extend(row for row in lexical_results if row[1].id not in seen)

        if not candidate_results:
            raise HTTPException(status_code=404, detail="No matching item found")

        candidates_by_final_id = {}
        for item, raw_gpc_item, distance in candidate_results:
            final_gpc_item = raw_gpc_item

            similarity_score = similarity_from_distance(distance)
            rerank_score, reasons = rerank_candidate(
                raw_gpc_item,
                final_gpc_item,
                similarity_score,
                normalized_query,
            )
            if family:
                reasons.append(f"explicit_product_family:{family}")
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
            key=lambda candidate: (candidate.rerank_score, -candidate.gpc_item.code),
            reverse=True,
        )
