"""
Data ingestion script for CrossIndexRAG.

Run once (or whenever your data changes) to embed and store all documents:

    python ingest_data.py

Expected folder layout
----------------------
data/
  text_docs/    *.txt files  — each file = one text document
  image_docs/   *.jpg / *.png / *.jpeg files
  table_docs/   *.csv files  — each row = one table document
"""

import os
import sys
import csv
import glob

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from embedders.text_embedder  import TextEmbedder
from embedders.image_embedder import ImageEmbedder
from embedders.table_embedder import TableEmbedder
from vector_db.text_index     import TextIndex
from vector_db.image_index    import ImageIndex
from vector_db.table_index    import TableIndex
from utils.logger             import get_logger

logger = get_logger("ingest_data")

DATA_DIR       = os.path.join(CURRENT_DIR, "data")
TEXT_DOCS_DIR  = os.path.join(DATA_DIR, "text_docs")
IMAGE_DOCS_DIR = os.path.join(DATA_DIR, "image_docs")
TABLE_DOCS_DIR = os.path.join(DATA_DIR, "table_docs")


# ── Text ──────────────────────────────────────────────────────────────────────

def ingest_text(index: TextIndex, embedder: TextEmbedder) -> int:
    files = glob.glob(os.path.join(TEXT_DOCS_DIR, "*.txt"))
    if not files:
        logger.warning(f"No .txt files found in {TEXT_DOCS_DIR}")
        return 0
    count = 0
    for path in files:
        source_id = "text_" + os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            continue
        embedding = embedder.embed(content)
        index.add(
            source_id=source_id,
            embedding=embedding,
            content=content,
            metadata={"file": os.path.basename(path)},
        )
        logger.info(f"  [text]  {source_id}")
        count += 1
    return count


# ── Image ─────────────────────────────────────────────────────────────────────

def ingest_images(index: ImageIndex, embedder: ImageEmbedder) -> int:
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(IMAGE_DOCS_DIR, ext)))
    if not files:
        logger.warning(f"No image files found in {IMAGE_DOCS_DIR}")
        return 0
    count = 0
    for path in files:
        source_id = "image_" + os.path.splitext(os.path.basename(path))[0]
        try:
            embedding = embedder.embed(path)
        except Exception as e:
            logger.error(f"  [image] skipping {path}: {e}")
            continue
        index.add(
            source_id=source_id,
            embedding=embedding,
            image_path=path,
            metadata={"file": os.path.basename(path)},
        )
        logger.info(f"  [image] {source_id}")
        count += 1
    return count


# ── Table ─────────────────────────────────────────────────────────────────────

def ingest_tables(index: TableIndex, embedder: TableEmbedder) -> int:
    files = glob.glob(os.path.join(TABLE_DOCS_DIR, "*.csv"))
    if not files:
        logger.warning(f"No .csv files found in {TABLE_DOCS_DIR}")
        return 0
    count = 0
    for path in files:
        basename = os.path.splitext(os.path.basename(path))[0]
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                source_id = f"table_{basename}_{row_idx}"
                content   = " | ".join(f"{k}: {v}" for k, v in row.items())
                embedding = embedder.embed(row)
                index.add(
                    source_id=source_id,
                    embedding=embedding,
                    content=content,
                    metadata={"file": os.path.basename(path), "row": row_idx},
                )
                logger.info(f"  [table] {source_id}")
                count += 1
    return count


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting ingestion...")

    text_index    = TextIndex()
    image_index   = ImageIndex()
    table_index   = TableIndex()

    text_embedder  = TextEmbedder()
    image_embedder = ImageEmbedder()
    table_embedder = TableEmbedder()

    n_text  = ingest_text(text_index,    text_embedder)
    n_image = ingest_images(image_index, image_embedder)
    n_table = ingest_tables(table_index, table_embedder)

    logger.info("=" * 50)
    logger.info(f"Ingestion complete:")
    logger.info(f"  Text docs   : {n_text}")
    logger.info(f"  Image docs  : {n_image}")
    logger.info(f"  Table rows  : {n_table}")
    logger.info(f"  Total       : {n_text + n_image + n_table}")
    logger.info("=" * 50)
    logger.info("You can now run: python app/hybridRetrievalDemo.py")