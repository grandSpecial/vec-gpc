"""Audit or refresh active product-brick embeddings; dry run unless --apply.

Only vectors and refresh markers are updated. GPC IDs, codes and category labels
are never rewritten. Each batch commits its vectors and markers atomically.
"""
import argparse
from openai import OpenAI
from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert

from embedding_text import build_gpc_embedding_text, embedding_revision, EMBEDDING_MODEL
from models import GPCLevel, Items, EmbeddingRefreshState, SessionLocal
from taxonomy import BRICK_PATHS


def pending_embeddings(db):
    rows = db.execute(
        select(GPCLevel, EmbeddingRefreshState.embedding_version, Items.vector.isnot(None))
        .outerjoin(EmbeddingRefreshState, EmbeddingRefreshState.gpc_item_id == GPCLevel.id)
        .outerjoin(Items, Items.id == GPCLevel.id)
        .where(GPCLevel.level == 4, GPCLevel.active.is_(True))
        .order_by(GPCLevel.id)
    ).all()
    pending = []
    for item, version, has_vector in rows:
        path = " > ".join(BRICK_PATHS.get(item.code, (item.full_title,)))
        content = build_gpc_embedding_text(item.title, path, item.definition)
        revision = embedding_revision(content)
        if version != revision or not has_vector:
            pending.append((item.id, content, revision))
    return pending


def save_batch(db, batch, embeddings):
    if len(batch) != len(embeddings):
        raise ValueError("Embedding response count does not match batch")
    for (item_id, _, revision), vector in zip(batch, embeddings):
        if len(vector) != 1536:
            raise ValueError("Unexpected embedding dimension")
        db.execute(insert(Items).values(id=item_id, vector=vector).on_conflict_do_update(
            index_elements=[Items.id], set_={"vector": vector}))
        db.execute(insert(EmbeddingRefreshState).values(
            gpc_item_id=item_id, embedding_version=revision,
        ).on_conflict_do_update(index_elements=[EmbeddingRefreshState.gpc_item_id],
                                set_={"embedding_version": revision, "updated_at": func.now()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 128:
        parser.error('--batch-size must be between 1 and 128')
    with SessionLocal() as db:
        pending = pending_embeddings(db)
    print(f"Active product bricks needing refresh: {len(pending)}")
    if not args.apply:
        print("Dry run. Use --apply to write vectors to the configured DATABASE_URL.")
        return
    client = OpenAI(timeout=30, max_retries=2)
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL,
                                            input=[row[1] for row in batch])
        embeddings = [r.embedding for r in sorted(response.data, key=lambda r: r.index)]
        with SessionLocal.begin() as db:
            save_batch(db, batch, embeddings)
        print(f"Refreshed {start + len(batch)}/{len(pending)}")


if __name__ == '__main__':
    main()
