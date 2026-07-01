"""Admin script: index a curated corpus folder into the persistent Qdrant DB.

Drop PDFs/TXTs into data/corpus/ (or --dir) and run this once so the served app
can auto-connect and answer queries without anyone uploading files.

    python -m scripts.seed_corpus [--dir data/corpus] [--force]

Ingestion is additive: already-indexed sources are skipped unless --force is
passed, and re-running never duplicates content (deterministic point IDs).
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend import config  # noqa: E402
from backend.embeddings import get_dense_embeddings, get_sparse_embeddings  # noqa: E402
from backend.ingestion import load_and_split_from_paths  # noqa: E402
from backend.vectorstore import (  # noqa: E402
    check_qdrant_connection,
    get_indexed_sources,
    upsert_documents,
)

SUPPORTED_SUFFIXES = {".pdf", ".txt"}


def _gather_files(corpus_dir: Path) -> list[Path]:
    """Return sorted supported files directly under the corpus directory."""
    return sorted(
        p
        for p in corpus_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def seed(corpus_dir: Path, force: bool) -> int:
    ok, error = check_qdrant_connection()
    if not ok:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if not corpus_dir.is_dir():
        print(f"ERROR: corpus directory not found: {corpus_dir}", file=sys.stderr)
        return 1

    files = _gather_files(corpus_dir)
    if not files:
        print(f"No PDF/TXT files found in {corpus_dir}. Nothing to seed.")
        return 0

    already_indexed = set() if force else set(get_indexed_sources())
    to_index = [f for f in files if f.name not in already_indexed]
    skipped = [f for f in files if f.name in already_indexed]

    for f in skipped:
        print(f"skip (already indexed): {f.name}")

    if not to_index:
        print("Nothing new to index.")
        return 0

    print(f"Loading and splitting {len(to_index)} file(s)...")
    chunks = load_and_split_from_paths([str(f) for f in to_index])
    if not chunks:
        print("No chunks produced from the selected files.")
        return 0

    print("Loading embedding models (first run downloads them)...")
    dense = get_dense_embeddings()
    sparse = get_sparse_embeddings()

    print(f"Indexing {len(chunks)} chunks into '{config.QDRANT_COLLECTION}'...")
    upsert_documents(chunks, dense, sparse)

    print(
        f"Done. Added {len(to_index)} file(s), {len(chunks)} chunks. "
        f"Skipped {len(skipped)} already-indexed file(s)."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed the RAG corpus into Qdrant.")
    parser.add_argument(
        "--dir",
        default=str(config.CORPUS_DIR),
        help="Folder of PDF/TXT files to index (default: %(default)s).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-index files even if their source name is already in the DB.",
    )
    args = parser.parse_args()
    return seed(Path(args.dir), args.force)


if __name__ == "__main__":
    raise SystemExit(main())
