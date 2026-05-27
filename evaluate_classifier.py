#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

from classifier import GPCClassifier
from main import create_description, create_vector
from models import SessionLocal


DEFAULT_GOLD_SET = Path(__file__).with_name("evaluation") / "gold_receipt_queries.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate vec-gpc classification quality.")
    parser.add_argument("--gold-set", type=Path, default=DEFAULT_GOLD_SET)
    parser.add_argument("--show-candidates", action="store_true")
    return parser.parse_args()


def read_gold_set(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def matches(expected: str | None, actual: str | None):
    if not expected:
        return None
    return (expected or "").strip().casefold() == (actual or "").strip().casefold()


def main():
    args = parse_args()
    rows = read_gold_set(args.gold_set)
    db = SessionLocal()
    totals = {
        "category": 0,
        "subcategory": 0,
        "gpc_code": 0,
        "category_possible": 0,
        "subcategory_possible": 0,
        "gpc_code_possible": 0,
    }

    try:
        classifier = GPCClassifier(db, create_description, create_vector)
        for row in rows:
            result = classifier.classify(row["query"], include_candidates=args.show_candidates)
            category_match = matches(row.get("expected_category"), result.get("category"))
            subcategory_match = matches(row.get("expected_subcategory"), result.get("subcategory"))
            gpc_match = matches(row.get("expected_gpc_code"), str(result.get("code")))

            if category_match is not None:
                totals["category_possible"] += 1
                totals["category"] += int(category_match)
            if subcategory_match is not None:
                totals["subcategory_possible"] += 1
                totals["subcategory"] += int(subcategory_match)
            if gpc_match is not None:
                totals["gpc_code_possible"] += 1
                totals["gpc_code"] += int(gpc_match)

            print(
                f"{row['query']}: "
                f"{result['category']} / {result['subcategory']} "
                f"({result['code']} {result['title']}) "
                f"confidence={result['confidence']} status={result['status']}"
            )
            print(f"  normalized={result['normalized_text']} description={result['description']}")
            if args.show_candidates:
                for candidate in result.get("candidates", [])[:3]:
                    print(
                        "  candidate "
                        f"{candidate['gpc_code']} {candidate['gpc_title']} "
                        f"{candidate['category']} / {candidate['subcategory']} "
                        f"score={candidate['rerank_score']}"
                    )

        print("\nSummary")
        for metric in ("category", "subcategory", "gpc_code"):
            possible = totals[f"{metric}_possible"]
            correct = totals[metric]
            if possible:
                print(f"{metric}: {correct}/{possible} = {correct / possible:.1%}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
