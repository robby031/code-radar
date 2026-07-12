"""
Embedding engine with MLX-accelerated inference on Apple Silicon.

Usage
-----
>>> from code_radar.engine import EmbeddingEngine
>>> engine = EmbeddingEngine()
>>> engine.load()
>>> vec = engine.embed_text("Hello, world!")
"""

from code_radar.engine._engine import EmbeddingEngine

__all__ = ["EmbeddingEngine"]
