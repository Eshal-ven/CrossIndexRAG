import os
import sys
import time
import logging
from typing import List, Dict, Any
from collections import Counter, defaultdict
import numpy as np


# ==========================================================
# FIX PYTHON PATH
# ==========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)


# ==========================================================
# IMPORT MODULES (Corrected for CrossIndexRAG folder)
# ==========================================================
try:
    from retrievers.text_retriever import TextRetriever
    from retrievers.image_retriever import ImageRetriever
    from retrievers.table_retriever import TableRetriever
except Exception as e:
    raise ImportError(f"Failed to import retriever classes. Error: {e}")

try:
    from vector_db.text_index import TextIndex
    from vector_db.image_index import ImageIndex
    from vector_db.table_index import TableIndex
except Exception as e:
    raise ImportError(f"Failed to import index classes. Error: {e}")

try:
    from embedders.text_embedder import TextEmbedder
    from embedders.image_embedder import ImageEmbedder
    from embedders.table_embedder import TableEmbedder
except Exception as e:
    raise ImportError(f"Failed to import embedders. Error: {e}")

try:
    from reranker.crossencoder_reranker import CrossEncoderReranker
except Exception as e:
    raise ImportError(f"Failed to import CrossEncoderReranker. Error: {e}")


# ==========================================================
# LOGGER
# ==========================================================
logger = logging.getLogger("hybrid_retrieval_demo")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(ch)



# ==========================================================
# UTILITY FUNCTIONS
# ==========================================================
def try_construct(cls, *args, **kwargs):
    """Try multiple constructor signatures."""
    try:
        return cls(*args, **kwargs)
    except TypeError:
        try:
            return cls()
        except Exception as e:
            raise RuntimeError(f"Cannot instantiate {cls.__name__}: {e}") from e


def normalize_retriever_output(raw_list: List[Any], modality: str) -> List[Dict[str, Any]]:
    normalized = []
    for item in raw_list or []:
        if item is None:
            continue
        if isinstance(item, dict):
            _id = item.get("id") or item.get("source_id")
            content = item.get("text") or item.get("content") or item.get("image_path", "")
            score = float(item.get("score", item.get("similarity", 0.0)))
            meta = item.get("metadata", {})
        else:
            _id = getattr(item, "source_id", None)
            content = getattr(item, "content", None) or getattr(item, "text", "") or getattr(item, "image_path", "")
            score = float(getattr(item, "score", 0.0))
            meta = getattr(item, "metadata", {})
        normalized.append({
            "id": str(_id),
            "text": str(content) if content else "",
            "score": score,
            "metadata": meta,
            "modality": modality
        })
    return normalized


def minmax_norm(arr: np.ndarray):
    if len(arr) == 0:
        return arr
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx - mn < 1e-9:
        return np.ones_like(arr)
    return (arr - mn) / (mx - mn)


def fuse_scores_array(embed_scores, ce_scores, alpha=0.35):
    embed_n = minmax_norm(embed_scores)
    ce_n = minmax_norm(ce_scores)
    return alpha * embed_n + (1 - alpha) * ce_n


def modality_agreement_topk(results, require_modalities=2):
    mods = [r.get("modality") for r in results]
    return len(set(mods)) >= require_modalities



# ==========================================================
# HYBRID APP
# ==========================================================
class HybridRetrievalApp:
    def __init__(self,
                 text_index_cls=TextIndex,
                 image_index_cls=ImageIndex,
                 table_index_cls=TableIndex,
                 text_retriever_cls=TextRetriever,
                 image_retriever_cls=ImageRetriever,
                 table_retriever_cls=TableRetriever,
                 text_embedder_cls=TextEmbedder,
                 image_embedder_cls=ImageEmbedder,
                 table_embedder_cls=TableEmbedder,
                 reranker_cls=CrossEncoderReranker,
                 top_k_per_mod=5,
                 alpha=0.35):

        self.top_k_per_mod = top_k_per_mod
        self.alpha = alpha

        logger.info("Initializing indexes...")
        self.text_index = try_construct(text_index_cls)
        self.image_index = try_construct(image_index_cls)
        self.table_index = try_construct(table_index_cls)

        logger.info("Initializing embedders...")
        self.text_embedder = try_construct(text_embedder_cls)
        self.image_embedder = try_construct(image_embedder_cls)
        self.table_embedder = try_construct(table_embedder_cls)

        logger.info("Initializing retrievers...")
        self.text_retriever = try_construct(text_retriever_cls, self.text_index, self.text_embedder)
        self.image_retriever = try_construct(image_retriever_cls, self.image_index, self.image_embedder)
        self.table_retriever = try_construct(table_retriever_cls, self.table_index, self.table_embedder)

        logger.info("Initializing reranker...")
        self.reranker = try_construct(reranker_cls)

        logger.info("HybridRetrievalApp initialized.")


    # ------------------------------
    # RETRIEVAL PIPELINE
    # ------------------------------
    def retrieve_candidates(self, query: str):
        results_by_mod = {}

        # Text
        try:
            raw_text = self.text_retriever.retrieve(query, top_k=self.top_k_per_mod)
        except:
            raw_text = self.text_retriever.search(query, top_k=self.top_k_per_mod)
        results_by_mod["text"] = normalize_retriever_output(raw_text, "text")

        # Image
        try:
            raw_img = self.image_retriever.retrieve(query, top_k=self.top_k_per_mod)
        except:
            raw_img = self.image_retriever.search(query, top_k=self.top_k_per_mod)
        results_by_mod["image"] = normalize_retriever_output(raw_img, "image")

        # Table
        try:
            raw_tbl = self.table_retriever.retrieve(query, top_k=self.top_k_per_mod)
        except:
            raw_tbl = self.table_retriever.search(query, top_k=self.top_k_per_mod)
        results_by_mod["table"] = normalize_retriever_output(raw_tbl, "table")

        return results_by_mod


    def merge_candidates(self, results_by_mod):
        merged = {}
        for mod, items in results_by_mod.items():
            for it in items:
                sid = it["id"]
                if sid not in merged or it["score"] > merged[sid]["score"]:
                    merged[sid] = dict(it)
        return list(merged.values())


    def rerank_with_cross_encoder(self, query, candidates, top_m=50):
        if not candidates:
            return []
        subset = candidates[:top_m]
        reranked = self.reranker.rerank(query, subset)

        for item in reranked:
            if "ce_score" not in item:
                item["ce_score"] = 0.0
        return reranked


    def final_fusion_and_sort(self, reranked, alpha):
        embed_scores = np.array([r.get("score", 0.0) for r in reranked])
        ce_scores = np.array([r.get("ce_score", 0.0) for r in reranked])

        fused = fuse_scores_array(embed_scores, ce_scores, alpha)

        for r, fs in zip(reranked, fused):
            r["final_score"] = float(fs)

        return sorted(reranked, key=lambda x: x["final_score"], reverse=True)


    def query(self, query: str, top_k=5, ce_top_m=50, require_modalities=2):
        start = time.time()

        results_by_mod = self.retrieve_candidates(query)
        merged = self.merge_candidates(results_by_mod)

        if not merged:
            return {"query": query, "results": [], "time": time.time() - start}

        reranked = self.rerank_with_cross_encoder(query, merged, ce_top_m)
        final_sorted = self.final_fusion_and_sort(reranked, self.alpha)

        topk = final_sorted[:top_k]
        safe = modality_agreement_topk(topk, require_modalities)

        return {
            "query": query,
            "results": topk,
            "safe": safe,
            "time": time.time() - start
        }


# ==========================================================
# PRETTY PRINT
# ==========================================================
def pretty_print_response(resp):
    print("\n============================================================")
    print(f"Query: {resp['query']}")
    print(f"Time: {resp['time']:.3f}s  | Safe: {resp['safe']}")
    print("------------------------------------------------------------")

    for i, r in enumerate(resp["results"], 1):
        print(f"\nRank {i} | Modality: {r['modality']} | ID: {r['id']}")
        txt = r.get("text", "")[:200].replace("\n", " ")
        print(f"Preview: {txt}")
        print(f"Scores -> embed: {r['score']:.4f}, ce: {r['ce_score']:.4f}, final: {r['final_score']:.4f}")
        if r.get("metadata"):
            print(f"Metadata: {r['metadata']}")
    print("============================================================\n")



# ==========================================================
# RUN
# ==========================================================
if __name__ == "__main__":
    logger.info("Starting HybridRetrievalApp...")

    try:
        app = HybridRetrievalApp()
    except Exception as e:
        logger.exception("Failed to initialize HybridRetrievalApp.")
        sys.exit(1)

    print("\nHybrid Retrieval System Ready (type 'exit' to quit)\n")
    while True:
        q = input("Query> ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue
        response = app.query(q)
        pretty_print_response(response)

