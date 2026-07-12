"""
Reranker (Cross-Encoder) using MLX for query-document relevance scoring.

Usage
-----
>>> from code_radar.reranker import Reranker
>>> reranker = Reranker()
>>> reranker.load()
>>> scores = reranker.rerank("query", ["doc1", "doc2"])
"""

from code_radar.reranker._helpers import RerankTimeoutError
from code_radar.reranker._reranker import Reranker

__all__ = ["Reranker", "RerankTimeoutError"]
