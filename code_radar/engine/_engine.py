import gc
import re
import time
from typing import Any, Sequence, cast

import mlx.core as mx
from mlx_lm.utils import load

from code_radar.engine._helpers import (
    env_flag,
    env_int,
    is_oom_error,
    plan_batches,
    cosine_similarity,
)
from code_radar.envvars import has_env
from code_radar.logging import get_logger

log = get_logger(__name__)


class EmbeddingEngine:
    # Keep backward-compatible class-level access for static methods
    # moved to _helpers.py.
    cosine_similarity = staticmethod(cosine_similarity)
    def __init__(self, model_name: str | None = None):
        if model_name is None:
            from code_radar.config import resolve_model_id

            model_name = resolve_model_id()

        self.model_name = model_name
        self.model: Any = None
        self.tokenizer: Any = None
        self.max_length = 8192

        # Tuning knobs (can be overridden via env vars)
        self._batch_set_by_env = has_env("CODE_EMBED_BATCH_SIZE")
        self._max_batch_tokens_set_by_env = has_env("CODE_EMBED_MAX_BATCH_TOKENS")
        self._max_tokens_per_text_set_by_env = has_env("CODE_EMBED_MAX_TOKENS_PER_TEXT")
        self._cleanup_interval_set_by_env = has_env("CODE_EMBED_CLEANUP_INTERVAL")

        self.default_batch_size = env_int("CODE_EMBED_BATCH_SIZE", 24)
        self.max_batch_tokens = env_int("CODE_EMBED_MAX_BATCH_TOKENS", 24 * 2048)
        self.max_tokens_per_text = env_int("CODE_EMBED_MAX_TOKENS_PER_TEXT", 2048)
        self.cleanup_interval = env_int("CODE_EMBED_CLEANUP_INTERVAL", 16)

        # Auto memory policy can be disabled if users want full manual control.
        self.disable_auto_memory_policy = env_flag("CODE_EMBED_DISABLE_AUTO_POLICY")

    def load(self) -> None:
        log.info("Loading embedding model: %s ...", self.model_name)
        t0 = time.perf_counter()

        result = load(self.model_name)
        self.model = result[0]
        self.tokenizer = result[1]

        elapsed = time.perf_counter() - t0

        if hasattr(self.model, "args"):
            if hasattr(self.model.args, "max_position_embeddings"):
                self.max_length = self.model.args.max_position_embeddings
            elif hasattr(self.model.args, "max_seq_len"):
                self.max_length = self.model.args.max_seq_len

        self._apply_auto_memory_policy()

        log.info(
            (
                "Model loaded in %.2fs  |  model=%s  |  max_length=%d  "
                "|  batch=%d  |  max_batch_tokens=%d  |  max_tokens_per_text=%d"
            ),
            elapsed,
            self.model_name,
            self.max_length,
            self.default_batch_size,
            self.max_batch_tokens,
            self.max_tokens_per_text,
        )

    def _hidden_dim(self) -> int:
        model = self.model
        return int(getattr(getattr(model, "args", None), "hidden_size", 1024))

    def _infer_model_size_b(self) -> float | None:
        """Best-effort inference of model size in billions from model id/name."""
        lower = self.model_name.lower()
        m = re.search(r"(\d+(?:\.\d+)?)\s*b", lower)
        if m is None:
            return None

        try:
            return float(m.group(1))
        except ValueError:
            return None

    def _apply_auto_memory_policy(self) -> None:
        """Apply conservative memory policy for larger models unless env overrides it.

        Policy goals:
        - Menjaga agar penggunaan peak memory tetap dapat diprediksi.
        - Menerapkan cost token maksimum per batch.
        - Mengurangi risiko swap pada device memori terpadu.
        """
        if self.disable_auto_memory_policy:
            log.info("Auto memory policy disabled via CODE_EMBED_DISABLE_AUTO_POLICY")
            return

        inferred_b = self._infer_model_size_b()
        hidden_dim = self._hidden_dim()

        # Defaults (small model friendly)
        rec_batch = 24
        rec_max_batch_tokens = 24 * 2048  # 49152
        rec_cleanup_interval = 16

        reason = "small-model/default"

        if inferred_b is not None:
            if inferred_b > 6.0:
                rec_batch = 2
                rec_max_batch_tokens = 8192
                rec_cleanup_interval = 2
                reason = f"very-large-model({inferred_b:g}B)"
            elif inferred_b > 3.0:
                rec_batch = 4
                rec_max_batch_tokens = 8192
                rec_cleanup_interval = 4
                reason = f"large-model({inferred_b:g}B)"
            elif inferred_b > 1.0:
                rec_batch = 8
                rec_max_batch_tokens = 16384
                rec_cleanup_interval = 8
                reason = f"medium-model({inferred_b:g}B)"
            else:
                reason = f"small-model({inferred_b:g}B)"
        else:
            # Fallback by hidden size if model name doesn't contain B-size.
            if hidden_dim >= 3000:
                rec_batch = 2
                rec_max_batch_tokens = 8192
                rec_cleanup_interval = 2
                reason = f"hidden-dim({hidden_dim})"
            elif hidden_dim >= 2000:
                rec_batch = 4
                rec_max_batch_tokens = 8192
                rec_cleanup_interval = 4
                reason = f"hidden-dim({hidden_dim})"
            elif hidden_dim >= 1500:
                rec_batch = 8
                rec_max_batch_tokens = 16384
                rec_cleanup_interval = 8
                reason = f"hidden-dim({hidden_dim})"

        old_batch = self.default_batch_size
        old_max_batch_tokens = self.max_batch_tokens
        old_cleanup = self.cleanup_interval

        if not self._batch_set_by_env:
            self.default_batch_size = max(1, min(self.default_batch_size, rec_batch))
        if not self._max_batch_tokens_set_by_env:
            self.max_batch_tokens = max(1024, min(self.max_batch_tokens, rec_max_batch_tokens))
        if not self._cleanup_interval_set_by_env:
            self.cleanup_interval = max(1, min(self.cleanup_interval, rec_cleanup_interval))

        # keep max_tokens_per_text predictable for big models unless manually set
        if not self._max_tokens_per_text_set_by_env and (inferred_b is not None and inferred_b > 3.0):
            self.max_tokens_per_text = min(self.max_tokens_per_text, 2048)

        if (
            old_batch != self.default_batch_size
            or old_max_batch_tokens != self.max_batch_tokens
            or old_cleanup != self.cleanup_interval
        ):
            log.info(
                (
                    "Auto memory policy applied  |  reason=%s  |  hidden_dim=%d  "
                    "|  batch=%d  |  max_batch_tokens=%d  |  cleanup_interval=%d"
                ),
                reason,
                hidden_dim,
                self.default_batch_size,
                self.max_batch_tokens,
                self.cleanup_interval,
            )

    def _pad_id(self) -> int:
        tokenizer = self.tokenizer
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tokenizer, "eos_token_id", 0)
        if pad_id is None:
            pad_id = 0
        return int(pad_id)

    def _tokenize(self, text: str) -> list[int]:
        tokenizer = self.tokenizer
        encode_fn = getattr(tokenizer, "encode", None)
        if encode_fn is None:
            return list(tokenizer(text).input_ids[: self.max_tokens_per_text])
        return list(encode_fn(text)[: self.max_tokens_per_text])

    def _forward_batch(
        self, token_ids_batch: Sequence[Sequence[int]], pad_id: int
    ) -> list[list[float]]:
        model = self.model
        if model is None:
            raise RuntimeError("model is not loaded")

        inner_model = getattr(model, "model", model)

        max_len = max(len(ids) for ids in token_ids_batch)
        padded: list[list[int]] = []
        mask: list[list[float]] = []

        for ids in token_ids_batch:
            pad_len = max_len - len(ids)
            padded.append(list(ids) + [pad_id] * pad_len)
            mask.append([1.0] * len(ids) + [0.0] * pad_len)

        inputs = mx.array(padded)

        raw_hidden = cast(Any, inner_model)(inputs)
        if isinstance(raw_hidden, (list, tuple)):
            hidden = raw_hidden[0]
        elif hasattr(raw_hidden, "last_hidden_state"):
            hidden = raw_hidden.last_hidden_state
        else:
            hidden = raw_hidden

        hidden = hidden.astype(mx.float32)

        m = mx.array(mask, dtype=mx.float32)[:, :, None]
        sum_mask = mx.clip(m.sum(axis=1), 1e-9, None)
        emb = (hidden * m).sum(axis=1) / sum_mask

        norm = mx.sqrt(mx.sum(emb**2, axis=1, keepdims=True))
        norm = mx.clip(norm, 1e-9, None)
        emb = emb / norm

        mx.eval(emb)
        out = cast(list[list[float]], emb.tolist())

        del inputs, raw_hidden, hidden, m, sum_mask, norm, emb, padded, mask
        return out

    def embed_text(self, text: str) -> list[float]:
        if not text or not text.strip():
            log.warning("embed_text called with empty text")
            return []

        result = self.embed_texts([text], batch_size=1)[0]
        return result

    def embed_texts(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        if not texts:
            log.warning("embed_texts called with empty list")
            return []

        model = self.model
        tokenizer = self.tokenizer
        if model is None or tokenizer is None:
            raise RuntimeError("call load() before embed_texts()")

        effective_batch_size = max(1, batch_size or self.default_batch_size)
        pad_id = self._pad_id()

        total_texts = len(texts)
        t0 = time.perf_counter()

        # Pre-tokenize once
        tk0 = time.perf_counter()
        tokenized: list[list[int]] = [self._tokenize(t) for t in texts]
        tk_elapsed = time.perf_counter() - tk0

        hidden_dim = self._hidden_dim()
        # Default all results to zero vectors (covers empty-token texts)
        results: list[list[float]] = [[0.0] * hidden_dim for _ in texts]

        non_empty_items = [(idx, ids) for idx, ids in enumerate(tokenized) if ids]
        total_tokens = sum(len(ids) for _, ids in non_empty_items)

        if not non_empty_items:
            log.info("Embedding skipped: all texts are empty/blank")
            return results

        batch_queue = plan_batches(
            non_empty_items,
            max_items=effective_batch_size,
            max_batch_tokens=self.max_batch_tokens,
        )

        batch_count = 0
        while batch_queue:
            batch_items = batch_queue.popleft()
            ids_batch = [ids for _, ids in batch_items]

            try:
                batch_vectors = self._forward_batch(ids_batch, pad_id)
            except Exception as exc:
                if len(batch_items) <= 1:
                    raise RuntimeError(
                        f"Embedding failed for a single item (tokens={len(ids_batch[0])}): {exc}"
                    ) from exc

                # Adaptive split for better survivability (especially on memory pressure)
                mid = len(batch_items) // 2
                left = batch_items[:mid]
                right = batch_items[mid:]

                if is_oom_error(exc):
                    log.warning(
                        "Batch OOM/failure, splitting  |  size=%d -> %d + %d  |  error=%s",
                        len(batch_items),
                        len(left),
                        len(right),
                        exc,
                    )
                else:
                    log.warning(
                        "Batch failure, retry by split  |  size=%d -> %d + %d  |  error=%s",
                        len(batch_items),
                        len(left),
                        len(right),
                        exc,
                    )

                batch_queue.appendleft(right)
                batch_queue.appendleft(left)
                continue

            for (orig_idx, _), vec in zip(batch_items, batch_vectors):
                results[orig_idx] = vec

            batch_count += 1
            if self.cleanup_interval > 0 and batch_count % self.cleanup_interval == 0:
                gc.collect()
                mx.clear_cache()

        elapsed = time.perf_counter() - t0
        texts_per_sec = total_texts / elapsed if elapsed > 0 else 0.0
        toks_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0

        log.info(
            (
                "Embedding done  |  texts=%d  |  non_empty=%d  |  batches=%d  "
                "|  elapsed=%.2fs  |  %.1f texts/s  |  %.0f tok/s  "
                "|  tokenization=%.2fs"
            ),
            total_texts,
            len(non_empty_items),
            batch_count,
            elapsed,
            texts_per_sec,
            toks_per_sec,
            tk_elapsed,
        )

        # Final cleanup once
        gc.collect()
        mx.clear_cache()

        return results
