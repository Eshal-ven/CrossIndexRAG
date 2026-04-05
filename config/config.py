"""
Central configuration for CrossIndexRAG.
Change values here; everything else reads from this file.
"""

# ── Embedder models ────────────────────────────────────────────────────────────
TEXT_EMBED_MODEL   = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
IMAGE_EMBED_MODEL  = "openai/clip-vit-base-patch32"
TABLE_EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

# ── Reranker model ─────────────────────────────────────────────────────────────
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── ChromaDB ───────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = "./chroma_store"
TEXT_COLLECTION    = "text_index"
IMAGE_COLLECTION   = "image_index"
TABLE_COLLECTION   = "table_index"

# ── Retrieval hyperparameters ──────────────────────────────────────────────────
TOP_K_PER_MODALITY  = 5      # candidates pulled per modality before fusion
CE_TOP_M            = 50     # max candidates sent to cross-encoder
FINAL_TOP_K         = 5      # results returned to the user
ALPHA               = 0.35   # weight of embedding score in final fusion (0=CE only, 1=embed only)

# ── Modality agreement ─────────────────────────────────────────────────────────
REQUIRE_MODALITIES  = 2      # minimum distinct modalities in top-K for "safe" flag

# ── Evaluation ─────────────────────────────────────────────────────────────────
NDCG_K = 10