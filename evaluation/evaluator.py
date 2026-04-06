"""
Evaluator: runs a list of benchmark queries through the retrieval pipeline
and reports precision, recall, nDCG, MRR, and modality coverage.

Usage
-----
    from evaluation.evaluator import Evaluator

    benchmark = [
        {
            "query": "Who created Python?",
            "relevant_ids": ["text_001", "table_003"]
        },
        ...
    ]

    evaluator = Evaluator(app)          # pass your HybridRetrievalApp instance
    report    = evaluator.run(benchmark, k=5)
    evaluator.print_report(report)
"""
from typing import List, Dict, Any

import numpy as np

from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    mean_reciprocal_rank,
    modality_coverage,
)
from configs.config import NDCG_K
from utils.logger import get_logger

logger = get_logger("evaluator")


class Evaluator:
    def __init__(self, app):
        """
        Parameters
        ----------
        app : HybridRetrievalApp instance (or any object with a .query() method
              that returns {'results': [{'id': ..., 'modality': ...}, ...]})
        """
        self.app = app

    def run(self, benchmark: List[Dict[str, Any]], k: int = NDCG_K) -> Dict[str, Any]:
        """
        Run the full benchmark.

        Parameters
        ----------
        benchmark : list of {'query': str, 'relevant_ids': [str, ...]}
        k         : cutoff for metrics

        Returns
        -------
        dict with per-query rows and aggregate averages
        """
        rows = []
        all_retrieved   = []
        all_relevant    = []

        for item in benchmark:
            query        = item["query"]
            relevant_ids = item["relevant_ids"]

            try:
                response = self.app.query(query, top_k=k)
            except Exception as e:
                logger.error(f"Query failed: '{query}' — {e}")
                continue

            retrieved_ids = [r["id"] for r in response.get("results", [])]
            results       = response.get("results", [])

            p   = precision_at_k(retrieved_ids, relevant_ids, k)
            r   = recall_at_k(retrieved_ids, relevant_ids, k)
            n   = ndcg_at_k(retrieved_ids, relevant_ids, k)
            mc  = modality_coverage(results)
            safe = response.get("safe", False)

            rows.append({
                "query":            query,
                "precision@k":      round(p,  4),
                "recall@k":         round(r,  4),
                "ndcg@k":           round(n,  4),
                "modality_cov":     round(mc, 4),
                "safe":             safe,
                "retrieved_ids":    retrieved_ids,
                "relevant_ids":     relevant_ids,
            })

            all_retrieved.append(retrieved_ids)
            all_relevant.append(relevant_ids)

        mrr = mean_reciprocal_rank(all_retrieved, all_relevant)

        aggregates = {
            "num_queries":      len(rows),
            "k":                k,
            "avg_precision@k":  round(float(np.mean([r["precision@k"] for r in rows])), 4) if rows else 0.0,
            "avg_recall@k":     round(float(np.mean([r["recall@k"]     for r in rows])), 4) if rows else 0.0,
            "avg_ndcg@k":       round(float(np.mean([r["ndcg@k"]       for r in rows])), 4) if rows else 0.0,
            "avg_modality_cov": round(float(np.mean([r["modality_cov"] for r in rows])), 4) if rows else 0.0,
            "mrr":              round(mrr, 4),
            "safe_rate":        round(sum(r["safe"] for r in rows) / len(rows), 4) if rows else 0.0,
        }

        return {"per_query": rows, "aggregates": aggregates}

    @staticmethod
    def print_report(report: Dict[str, Any]) -> None:
        agg = report["aggregates"]
        print("\n" + "=" * 60)
        print(f"  BENCHMARK RESULTS  ({agg['num_queries']} queries, k={agg['k']})")
        print("=" * 60)
        print(f"  Precision@{agg['k']:<3}    {agg['avg_precision@k']:.4f}")
        print(f"  Recall@{agg['k']:<3}       {agg['avg_recall@k']:.4f}")
        print(f"  nDCG@{agg['k']:<3}         {agg['avg_ndcg@k']:.4f}")
        print(f"  MRR              {agg['mrr']:.4f}")
        print(f"  Modality cov.    {agg['avg_modality_cov']:.4f}")
        print(f"  Safe rate        {agg['safe_rate']:.4f}")
        print("=" * 60)

        print("\nPer-query breakdown:")
        for row in report["per_query"]:
            safe_flag = "✓" if row["safe"] else "✗"
            print(f"  [{safe_flag}] {row['query'][:50]:<50} "
                  f"P={row['precision@k']:.2f} R={row['recall@k']:.2f} "
                  f"nDCG={row['ndcg@k']:.2f} cov={row['modality_cov']:.2f}")
        print()