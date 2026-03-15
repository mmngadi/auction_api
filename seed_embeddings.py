#!/usr/bin/env python3
"""
Populate the ``condition_embedding`` column for every auction_lots row
where it is currently NULL.

Idempotent — safe to re-run at any time.
Requires Ollama to be serving the nomic-embed-text model.
"""

import sys
import logging
import psycopg2
import ollama

from app.config import settings

BATCH_COMMIT_SIZE = 50

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("seed_embeddings")


def main() -> int:
    # ── connections ──────────────────────────────────────────
    logger.info("DB  → %s", settings.db.dsn)
    conn = psycopg2.connect(settings.db.dsn)

    logger.info("LLM → %s  model=%s", settings.ollama.base_url, settings.ollama.embed_model)
    client = ollama.Client(
        host=settings.ollama.base_url,
        timeout=settings.ollama.timeout,
    )

    try:
        cur = conn.cursor()

        # ── fetch rows that still need embeddings ────────────
        cur.execute(
            "SELECT id, condition FROM auction_lots "
            "WHERE condition_embedding IS NULL "
            "  AND condition IS NOT NULL "
            "  AND TRIM(condition) <> ''"
        )
        rows = cur.fetchall()
        total = len(rows)

        if total == 0:
            logger.info("✓ All rows already have embeddings — nothing to do.")
            return 0

        logger.info("Found %d rows to embed.", total)

        success = 0
        skipped = 0
        failed = 0

        # ── generate + write ─────────────────────────────────
        for i, (row_id, condition_text) in enumerate(rows, 1):
            if not condition_text or not condition_text.strip():
                skipped += 1
                continue

            try:
                resp = client.embeddings(
                    model=settings.ollama.embed_model,
                    prompt=condition_text,
                )
                cur.execute(
                    "UPDATE auction_lots "
                    "SET condition_embedding = %s WHERE id = %s",
                    (resp["embedding"], row_id),
                )
                success += 1
            except Exception as exc:
                logger.warning("Row %d failed: %s", row_id, exc)
                failed += 1

            # Periodic commit + progress log
            if i % BATCH_COMMIT_SIZE == 0 or i == total:
                conn.commit()
                pct = i / total * 100
                logger.info("Progress: %d / %d  (%.0f%%)", i, total, pct)

        conn.commit()
        logger.info(
            "✓ Done — %d embedded · %d skipped (empty) · %d failed",
            success, skipped, failed,
        )
        return 0 if failed == 0 else 1

    except Exception:
        logger.exception("Seeding failed")
        conn.rollback()
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())