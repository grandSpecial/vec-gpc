import hashlib

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_VERSION = "gpc-brick-v2"


def build_gpc_embedding_text(title: str, full_title: str, definition: str | None):
    parts = []

    if title:
        parts.append(f"Item: {title}.")

    if full_title:
        parts.append(f"Classification path: {full_title}.")

    if definition:
        parts.append(f"Definition: {definition}")

    return " ".join(part.strip() for part in parts if part).strip()


def embedding_revision(text: str) -> str:
    """Invalidate embeddings when content, format version, or model changes."""
    digest = hashlib.sha256(f"{EMBEDDING_MODEL}\n{text}".encode()).hexdigest()
    return f"{EMBEDDING_VERSION}:{digest}"
