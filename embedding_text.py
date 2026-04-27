def build_gpc_embedding_text(title: str, full_title: str, definition: str | None):
    parts = []

    if title:
        parts.append(f"Item: {title}.")

    if full_title:
        parts.append(f"Classification path: {full_title}.")

    if definition:
        parts.append(f"Definition: {definition}")

    return " ".join(part.strip() for part in parts if part).strip()
