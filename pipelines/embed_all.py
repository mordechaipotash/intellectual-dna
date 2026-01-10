#!/usr/bin/env python3
"""
Embed ALL User Messages - Background Script

Runs until all user messages are embedded.
Progress logged to embed_progress.log

Usage:
    python -m pipelines embed --all              # Embed all remaining
    python -m pipelines embed --all --checkpoint 1000  # Log every 1000
"""

import sys
import time
import logging
from pathlib import Path

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PARQUET_PATH, EMBEDDINGS_DB
from pipelines.utils.id_utils import EFFECTIVE_ID_SQL
from pipelines.embed_messages import run_embedding_pipeline, get_stats
import duckdb

# Setup logging
LOG_FILE = Path(__file__).parent.parent / "data" / "embed_progress.log"
LOG_FILE.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def get_total_to_embed() -> int:
    """Get count of user messages that need embedding."""
    con = duckdb.connect()
    result = con.execute(f'''
        SELECT COUNT(DISTINCT {EFFECTIVE_ID_SQL})
        FROM read_parquet("{PARQUET_PATH}")
        WHERE role = 'user'
        AND content IS NOT NULL
        AND char_count > 20
    ''').fetchone()
    return result[0]


def get_current_count() -> int:
    """Get count of already embedded messages."""
    try:
        con = duckdb.connect(str(EMBEDDINGS_DB))
        return con.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0]
    except:
        return 0


def embed_all(checkpoint_every: int = 500):
    """
    Embed all remaining user messages.

    Args:
        checkpoint_every: Log progress every N messages
    """
    total_target = get_total_to_embed()
    start_count = get_current_count()
    remaining = total_target - start_count

    log.info("=" * 60)
    log.info("EMBEDDING ALL USER MESSAGES")
    log.info("=" * 60)
    log.info(f"Total target: {total_target:,}")
    log.info(f"Already embedded: {start_count:,}")
    log.info(f"Remaining: {remaining:,}")
    log.info(f"Estimated time: {remaining / 15 / 60:.0f} minutes")
    log.info("=" * 60)

    if remaining <= 0:
        log.info("All messages already embedded!")
        return

    start_time = time.time()
    last_checkpoint = start_count
    batches_per_run = 20  # 1000 messages per run

    while True:
        current = get_current_count()
        remaining = total_target - current

        if remaining <= 0:
            break

        log.info(f"Starting batch... ({current:,}/{total_target:,} complete)")

        try:
            run_embedding_pipeline(batch_count=batches_per_run)
        except KeyboardInterrupt:
            log.info("Interrupted by user")
            break
        except Exception as e:
            log.error(f"Error in batch: {e}")
            time.sleep(5)
            continue

        new_count = get_current_count()
        embedded_this_session = new_count - start_count
        elapsed = time.time() - start_time
        rate = embedded_this_session / elapsed if elapsed > 0 else 0

        if new_count - last_checkpoint >= checkpoint_every:
            eta_seconds = (total_target - new_count) / rate if rate > 0 else 0
            log.info("-" * 40)
            log.info(f"CHECKPOINT: {new_count:,}/{total_target:,} ({new_count/total_target*100:.1f}%)")
            log.info(f"Session: +{embedded_this_session:,} @ {rate:.1f} msg/s")
            log.info(f"ETA: {eta_seconds/60:.0f} minutes")
            log.info("-" * 40)
            last_checkpoint = new_count

    elapsed = time.time() - start_time
    final_count = get_current_count()
    embedded_total = final_count - start_count

    log.info("=" * 60)
    log.info("EMBEDDING COMPLETE")
    log.info("=" * 60)
    log.info(f"Total embedded this session: {embedded_total:,}")
    log.info(f"Final count: {final_count:,}/{total_target:,}")
    log.info(f"Total time: {elapsed/60:.1f} minutes")
    log.info(f"Average rate: {embedded_total/elapsed:.1f} msg/s" if elapsed > 0 else "N/A")
    log.info("=" * 60)


if __name__ == "__main__":
    checkpoint = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    embed_all(checkpoint_every=checkpoint)
