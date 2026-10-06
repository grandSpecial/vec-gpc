"""Run in Google Colab after installing requests and gspread.

Paste this file into a cell, or upload it and run `%run evaluate_sheet_colab.py`.
Add VEC_GPC_API_KEY to Colab Secrets and enable notebook access first.
Products must already be in the worksheet (for example, imported from a CSV).
"""

import json
import re
import time
from urllib.parse import urlsplit, urlunsplit

import requests


SPREADSHEET_ID = "1ugpL5CnQPssWqU2vEoEZXJN8ZbI46YnAwFZBMJUtsO4"
WORKSHEET_ID = 262133907
# Paste the public tunnel URL ending in /search, or leave blank to enter it
# when the script starts. Hosted Colab cannot reach your Mac at 127.0.0.1.
API_URL = ""
PRODUCT_COLUMN = 1  # One-based column number; row 1 contains headers.
INCLUDE_CANDIDATES = True  # API returns its diagnostic shortlist, not every candidate.

# Keep test-N compatible with earlier single-column category runs.
# category/subcategory are aliases of level_2_category/level_3_category.
RESULT_FIELDS = (
    "level_2_category",
    "level_3_category",
    "display_label",
    "description",
    "code",
    "title",
    "full_title",
    "definition",
    "confidence",
    "needs_review",
    "status",
    "description_fallback",
    "selection_source",
    "selection_fallback",
    "normalized_text",
    "display_mapping_version",
    "display_mapping_source",
    "reranker_version",
    "identity_rule_version",
    "identity_rule",
    "latency_ms",
    "candidates",
)
REVIEW_FIELDS = (
    "product_correct", "category_correct", "expected_category",
    "expected_gpc_code", "issue_type", "notes", "error",
)


def check_connection(session):
    target = urlsplit(API_URL)
    if target.scheme not in {"http", "https"} or not target.hostname or target.path != "/search":
        raise ValueError("Set API_URL to your reachable API URL ending in /search.")
    ping_url = urlunsplit((target.scheme, target.netloc, "/ping", "", ""))
    try:
        response = session.get(ping_url, timeout=(10, 20))
    except (requests.ConnectionError, requests.Timeout) as error:
        raise RuntimeError(
            "Cannot reach the API. In hosted Colab, 127.0.0.1 refers to Colab, "
            "not your Mac. Use the current HTTPS tunnel URL and keep both "
            "uvicorn and the tunnel running. No sheet results were written."
        ) from error
    if response.status_code in (401, 403):
        raise PermissionError(
            "API authentication failed. Colab's VEC_GPC_API_KEY must match "
            "API_AUTH_TOKEN in the local server's environment."
        )
    response.raise_for_status()
    if response.json() != {"ok": True}:
        raise ValueError("Unexpected /ping response; check that the URL points to vec-gpc.")
    print("API connection and authentication verified.")


def output_layout(rows):
    """Append beyond all populated cells, including unlabeled notes columns."""
    last_used_column = max(
        (i + 1 for row in rows for i, value in enumerate(row) if value.strip()),
        default=0,
    )
    test_numbers = []
    for header in rows[0]:
        match = re.match(r"^test-(\d+)(?:$|\s)", header.strip(), re.IGNORECASE)
        if match:
            test_numbers.append(int(match.group(1)))
    test_header = f"test-{max(test_numbers, default=0) + 1}"
    headers = [test_header] + [
        f"{test_header} {field}" for field in RESULT_FIELDS[1:] + REVIEW_FIELDS
    ]
    return last_used_column + 1, test_header, headers


def classify(session, product):
    """Retry transient HTTP/connection failures, but stop on invalid credentials."""
    for attempt in range(3):
        try:
            response = session.post(
                API_URL,
                params={"text": product, "include_candidates": str(INCLUDE_CANDIDATES).lower()},
                data="",
                timeout=(10, 60),
            )
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 2:
                raise
        else:
            if response.status_code in (401, 403):
                raise PermissionError(
                    f"API authorization failed (HTTP {response.status_code})."
                )
            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt == 2:
                    response.raise_for_status()
            else:
                response.raise_for_status()
                result = response.json()
                if not isinstance(result, dict):
                    raise ValueError("Expected a JSON response object.")
                if "level_2_category" not in result:
                    raise ValueError("Response is missing level_2_category.")
                return result
        time.sleep(2 ** (attempt + 1))


def result_cells(result):
    # Preserve numeric values, booleans (including False), and empty nulls.
    cells = []
    for field in RESULT_FIELDS:
        value = result.get(field)
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        cells.append("" if value is None else value)
    return cells


def evaluate_sheet(sheet, session):
    from gspread.utils import rowcol_to_a1

    if PRODUCT_COLUMN < 1:
        raise ValueError("PRODUCT_COLUMN must be at least 1.")
    check_connection(session)
    rows = sheet.get_all_values()
    if len(rows) < 2 or not any(
        len(row) >= PRODUCT_COLUMN and row[PRODUCT_COLUMN - 1].strip()
        for row in rows[1:]
    ):
        raise ValueError("No product rows found.")

    output_column, test_header, headers = output_layout(rows)
    results = [headers]
    completed = failed = 0

    for row_number, row in enumerate(rows[1:], start=2):
        product = row[PRODUCT_COLUMN - 1].strip() if len(row) >= PRODUCT_COLUMN else ""
        cells = [""] * len(headers)
        if product:
            try:
                result = classify(session, product)
                cells[:len(RESULT_FIELDS)] = result_cells(result)
                completed += 1
                print(
                    f"Row {row_number}: {product} → "
                    f"{result.get('level_2_category')} / {result.get('level_3_category')}"
                )
            except PermissionError:
                # No sheet writes have occurred; fix the secret before rerunning.
                raise
            except (requests.RequestException, ValueError) as error:
                # Do not put errors in the category or shift subsequent rows.
                cells[-1] = f"ERROR: {type(error).__name__}"
                if isinstance(error, requests.HTTPError) and error.response is not None:
                    cells[-1] += f" (HTTP {error.response.status_code})"
                failed += 1
                print(f"Row {row_number}: {product} → {cells[-1]}")
        results.append(cells)

    # Refuse to overwrite cells populated while the API requests were running.
    # Avoid concurrent runs/edits in the destination columns: this is not a lock.
    end_column = output_column + len(headers) - 1
    if end_column > sheet.col_count:
        sheet.add_cols(end_column - sheet.col_count)
    start = rowcol_to_a1(1, output_column)
    end = rowcol_to_a1(len(results), end_column)
    output_range = f"{start}:{end}"
    if any(value.strip() for row in sheet.get(output_range) for value in row):
        raise RuntimeError("Destination cells changed during this run; rerun to append safely.")
    sheet.update(range_name=output_range, values=results, value_input_option="RAW")
    print(
        f"\nSaved {test_header} to {output_range}. "
        f"Successful: {completed}; failed: {failed}."
    )
    print("Review product_correct and category_correct separately (TRUE/FALSE/UNKNOWN).")
    print("Use issue_type: product_identity, display_mapping, insufficient_context, or non_product.")


def main():
    global API_URL
    # Colab-specific imports stay here so helpers can also be checked locally.
    from google.colab import auth, userdata
    import google.auth
    import gspread

    if not API_URL.strip():
        API_URL = input("API URL reachable from Colab (ending in /search): ").strip()
    api_token = userdata.get("VEC_GPC_API_KEY")
    if not api_token or not api_token.strip():
        raise ValueError("Set VEC_GPC_API_KEY in Colab Secrets and allow notebook access.")
    auth.authenticate_user()
    credentials, _ = google.auth.default()
    gc = gspread.authorize(credentials)
    spreadsheet = gc.open_by_key(SPREADSHEET_ID)
    sheet = spreadsheet.get_worksheet_by_id(WORKSHEET_ID)

    with requests.Session() as session:
        session.headers.update({"accept": "application/json", "Authorization": f"Bearer {api_token}"})
        evaluate_sheet(sheet, session)


if __name__ == "__main__":
    main()
