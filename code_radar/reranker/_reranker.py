import gc
import inspect
import time
from typing import Any

import mlx.core as mx
from mlx_lm.utils import load

from code_radar.envvars import get_env
from code_radar.logging import get_logger
from code_radar.reranker._helpers import RerankTimeoutError, env_int

log = get_logger(__name__)


class Reranker:
    """Cross-encoder reranker for query-document relevance scoring."""

    def __init__(self, model_name: str | None = None):
        if model_name is None:
            from code_radar.config import resolve_reranker_model_id

            model_name = resolve_reranker_model_id()

        self.model_name = model_name
        self._model: Any = None
        self._tokenizer: Any = None
        self._score_head: Any = None
        self.max_length: int = 8192
        self._hidden_dim: int = 1024
        self._load_error: Exception | None = None

        # Runtime safety & latency knobs
        self.max_doc_tokens: int = env_int("CODE_RERANKER_DOC_MAX_TOKENS", 256, minimum=32)
        self.max_doc_chars: int = env_int("CODE_RERANKER_DOC_MAX_CHARS", 1000, minimum=256)
        self.max_query_tokens: int = env_int("CODE_RERANKER_QUERY_MAX_TOKENS", 128, minimum=8)
        self.max_batch_size: int = env_int("CODE_RERANKER_MAX_BATCH_SIZE", 16, minimum=1)
        self.cleanup_interval: int = env_int("CODE_RERANKER_CLEANUP_INTERVAL", 8, minimum=1)

        self._cls_token_id: int | None = None
        self._sep_ids: list[int] = []

        # Fallback scoring mode for LLM-style rerankers without explicit score head
        self._use_chat_logits: bool = False
        self._yes_token_ids: list[int] = []
        self._no_token_ids: list[int] = []
        self._warned_no_valid_scoring: bool = False
        self._rerank_instruction: str = get_env(
            "CODE_RERANKER_INSTRUCTION",
            "Given a code search query, retrieve relevant passages that answer the query",
        ) or "Given a code search query, retrieve relevant passages that answer the query"

    # Public API
    def load(self) -> None:
        """Load the reranker model into memory."""
        log.info("Loading reranker model: %s ...", self.model_name)
        t0 = time.perf_counter()

        try:
            result = load(self.model_name)
        except Exception as exc:
            self._load_error = exc
            log.error("Failed to load reranker model  |  model=%s  |  error=%s", self.model_name, exc)
            raise RuntimeError(f"Reranker load failed: {exc}") from exc

        self._model = result[0]
        self._tokenizer = result[1]
        elapsed = time.perf_counter() - t0

        # Detect score head architecture
        if hasattr(self._model, "score"):
            self._score_head = self._model.score
            log.info("Reranker has built-in score head (model.score)")
        elif hasattr(self._model, "classifier"):
            self._score_head = self._model.classifier
            log.info("Reranker has built-in classifier head (model.classifier)")

        # Extract model dimensions
        args = getattr(self._model, "args", None)
        if args is not None:
            if hasattr(args, "max_position_embeddings"):
                self.max_length = int(args.max_position_embeddings)
            elif hasattr(args, "max_seq_len"):
                self.max_length = int(args.max_seq_len)
            self._hidden_dim = int(getattr(args, "hidden_size", 1024))

        tok = self._tokenizer
        self._cls_token_id = int(tok.cls_token_id) if getattr(tok, "cls_token_id", None) is not None else None
        self._sep_ids = self._tokenize("\n", max_tokens=4, add_special_tokens=False)

        # LLM-logit fallback mode for reranker checkpoints that expose no score head.
        self._use_chat_logits = False
        self._yes_token_ids = []
        self._no_token_ids = []
        if self._score_head is None and hasattr(tok, "apply_chat_template"):
            self._yes_token_ids = self._collect_single_token_ids(["yes", " yes", "Yes", " Yes"])
            self._no_token_ids = self._collect_single_token_ids(["no", " no", "No", " No"])
            if self._yes_token_ids and self._no_token_ids:
                self._use_chat_logits = True

        scoring_mode = (
            "score_head"
            if self._score_head is not None
            else ("chat_yes_no_logits" if self._use_chat_logits else "disabled")
        )
        if scoring_mode == "disabled":
            log.warning(
                "Reranker loaded without score_head and without chat-logit fallback. "
                "Rerank will fallback to neutral scores."
            )

        log.info(
            "Reranker loaded in %.2fs  |  model=%s  |  max_length=%d  |  hidden_dim=%d  |  has_score_head=%s  |  scoring_mode=%s  |  doc_max_tokens=%d  |  doc_max_chars=%d",
            elapsed,
            self.model_name,
            self.max_length,
            self._hidden_dim,
            self._score_head is not None,
            scoring_mode,
            self.max_doc_tokens,
            self.max_doc_chars,
        )

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._score_head = None
        gc.collect()
        mx.clear_cache()
        log.info("Reranker unloaded  |  model=%s", self.model_name)

    def rerank(
        self,
        query: str,
        documents: list[str],
        batch_size: int = 4,
        timeout_sec: float | None = None,
    ) -> list[float]:
        """Score query-document pairs using the cross-encoder.

        Args:
            query: Search query.
            documents: Document candidates.
            batch_size: Batch size for inference.
            timeout_sec: Optional hard latency budget. If exceeded,
                raises :class:`RerankTimeoutError`.
        """
        if not documents:
            return []

        if self._model is None:
            raise RuntimeError("Reranker not loaded — call .load() first")

        effective_batch = max(1, min(int(batch_size), self.max_batch_size))
        start = time.perf_counter()
        deadline = (start + timeout_sec) if timeout_sec and timeout_sec > 0 else None

        all_scores: list[float] = []
        n_batches = 0

        for i in range(0, len(documents), effective_batch):
            if deadline is not None and time.perf_counter() >= deadline:
                elapsed = time.perf_counter() - start
                raise RerankTimeoutError(
                    f"rerank timeout after {elapsed:.3f}s "
                    f"(budget={timeout_sec:.3f}s, processed={len(all_scores)}/{len(documents)})"
                )

            batch = documents[i : i + effective_batch]
            batch_scores = self._score_batch(query, batch)
            all_scores.extend(batch_scores)
            n_batches += 1

            if n_batches % self.cleanup_interval == 0:
                gc.collect()
                mx.clear_cache()

        elapsed = time.perf_counter() - start
        log.info(
            "Reranked %d docs in %.2fs  |  query=%s  |  batches=%d  |  batch_size=%d",
            len(documents),
            elapsed,
            query[:80],
            n_batches,
            effective_batch,
        )

        return all_scores

    # Internal
    def _tokenize(
        self,
        text: str,
        *,
        max_tokens: int | None = None,
        add_special_tokens: bool = False,
    ) -> list[int]:
        tokenizer = self._tokenizer
        if tokenizer is None:
            return []

        token_ids: Any
        encode_fn = getattr(tokenizer, "encode", None)

        if encode_fn is not None:
            try:
                sig = inspect.signature(encode_fn)
                if "add_special_tokens" in sig.parameters:
                    token_ids = encode_fn(text, add_special_tokens=add_special_tokens)
                else:
                    token_ids = encode_fn(text)
            except (TypeError, ValueError):
                token_ids = encode_fn(text)
        else:
            try:
                tokenized = tokenizer(text, add_special_tokens=add_special_tokens)
            except TypeError:
                tokenized = tokenizer(text)
            token_ids = getattr(tokenized, "input_ids", tokenized)

        if isinstance(token_ids, dict):
            token_ids = token_ids.get("input_ids", [])

        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]

        ids = [int(t) for t in token_ids]

        if max_tokens is not None:
            ids = ids[: max(1, max_tokens)]

        return ids

    def _collect_single_token_ids(self, pieces: list[str]) -> list[int]:
        ids: set[int] = set()
        for piece in pieces:
            token_ids = self._tokenize(piece, add_special_tokens=False)
            if len(token_ids) == 1:
                ids.add(int(token_ids[0]))
        return sorted(ids)

    def _decode_ids(self, token_ids: list[int], fallback_text: str) -> str:
        tokenizer = self._tokenizer
        if tokenizer is None or not token_ids:
            return fallback_text

        decode_fn = getattr(tokenizer, "decode", None)
        if decode_fn is None:
            return fallback_text

        try:
            out = decode_fn(token_ids)
            return str(out) if out else fallback_text
        except Exception:
            return fallback_text

    def _truncate_query_text(self, query: str) -> str:
        q_ids = self._tokenize(query, add_special_tokens=False)
        if not q_ids:
            return query
        q_ids = q_ids[: self.max_query_tokens]
        return self._decode_ids(q_ids, query)

    def _truncate_doc_text(self, doc_text: str) -> tuple[str, bool]:
        char_truncated = len(doc_text) > self.max_doc_chars
        source = doc_text[: self.max_doc_chars]

        raw_doc_ids = self._tokenize(source, add_special_tokens=False)
        token_truncated = len(raw_doc_ids) > self.max_doc_tokens
        doc_ids = raw_doc_ids[: self.max_doc_tokens]

        text = self._decode_ids(doc_ids, source)
        return text, (char_truncated or token_truncated)

    def _build_chat_prompt_ids(self, query: str, doc_text: str) -> tuple[list[int], bool]:
        tokenizer = self._tokenizer
        if tokenizer is None:
            return [], False

        truncated_query = self._truncate_query_text(query)
        truncated_doc, doc_was_truncated = self._truncate_doc_text(doc_text)

        messages = [
            {"role": "system", "content": self._rerank_instruction},
            {"role": "query", "content": truncated_query},
            {"role": "document", "content": truncated_doc},
        ]

        apply_template = getattr(tokenizer, "apply_chat_template", None)
        if apply_template is None:
            return [], doc_was_truncated

        try:
            prompt_ids = apply_template(messages, tokenize=True)
        except TypeError:
            prompt_ids = apply_template(messages)

        if prompt_ids and isinstance(prompt_ids[0], list):
            prompt_ids = prompt_ids[0]

        ids = [int(t) for t in prompt_ids]
        if not ids:
            return [], doc_was_truncated

        max_len = max(1, self.max_length)
        budget_truncated = len(ids) > max_len
        if budget_truncated:
            # Keep tail so assistant answer position remains intact.
            ids = ids[-max_len:]

        return ids, (doc_was_truncated or budget_truncated)

    def _get_pad_id(self) -> int:
        tokenizer = self._tokenizer
        if tokenizer is None:
            return 0

        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tokenizer, "eos_token_id", 0)
        if pad_id is None:
            pad_id = 0

        return int(pad_id)

    def _build_pair_ids(self, query_ids: list[int], doc_text: str) -> tuple[list[int], bool]:
        char_truncated = len(doc_text) > self.max_doc_chars
        doc_source = doc_text[: self.max_doc_chars]

        # Token truncation is measured explicitly for observability.
        raw_doc_ids = self._tokenize(doc_source, add_special_tokens=False)
        token_truncated = len(raw_doc_ids) > self.max_doc_tokens
        doc_ids = raw_doc_ids[: self.max_doc_tokens]

        sep_ids = self._sep_ids
        if not sep_ids:
            eos_id = getattr(self._tokenizer, "eos_token_id", None)
            sep_ids = [int(eos_id)] if eos_id is not None else []

        doc_budget = max(0, self.max_length - len(query_ids) - len(sep_ids))
        budget_truncated = len(doc_ids) > doc_budget
        if budget_truncated:
            doc_ids = doc_ids[:doc_budget]

        pair_ids = query_ids + sep_ids + doc_ids
        was_truncated = char_truncated or token_truncated or budget_truncated
        return pair_ids, was_truncated

    def _pool_index_for_pair(self, pair_ids: list[int]) -> int:
        # Cross-encoder pooling should not use mean pooling.
        # Prefer CLS when present at position 0, otherwise last token.
        if self._cls_token_id is not None and pair_ids and pair_ids[0] == self._cls_token_id:
            return 0
        return max(0, len(pair_ids) - 1)

    def _score_from_head(self, token_vec: Any) -> Any:
        score_head = self._score_head
        if score_head is None:
            return mx.array(0.0, dtype=mx.float32)

        attempts = (
            token_vec,
            mx.expand_dims(token_vec, axis=0),
            mx.expand_dims(mx.expand_dims(token_vec, axis=0), axis=1),
        )

        raw_score: Any = None
        for candidate in attempts:
            try:
                raw_score = score_head(candidate)
                break
            except Exception:
                continue

        if raw_score is None:
            return mx.array(0.0, dtype=mx.float32)

        while getattr(raw_score, "ndim", 0) > 0:
            raw_score = raw_score[0]

        if hasattr(raw_score, "astype"):
            return raw_score.astype(mx.float32)
        return mx.array(raw_score, dtype=mx.float32)

    def _materialize_scores(self, score_values: list[Any]) -> list[float]:
        if not score_values:
            return []

        # FIX: Evaluate MLX scores once per batch before crossing into Python.
        # This avoids repeated scalar float() synchronizations that can stall the caller.
        scores_arr = mx.stack(score_values).astype(mx.float32)
        mx.eval(scores_arr)

        scores = []
        for value in scores_arr.tolist():
            out = float(value)
            scores.append(0.0 if out != out else out)
        return scores

    def _score_batch_with_chat_logits(self, query: str, documents: list[str]) -> list[float]:
        model = self._model
        tokenizer = self._tokenizer
        if model is None or tokenizer is None:
            return [0.0] * len(documents)

        if not self._yes_token_ids or not self._no_token_ids:
            return [0.0] * len(documents)

        token_ids_list: list[list[int]] = []
        batch_max_len = 0
        truncated_docs = 0

        for doc_text in documents:
            prompt_ids, was_truncated = self._build_chat_prompt_ids(query, doc_text)
            if was_truncated:
                truncated_docs += 1
            if not prompt_ids:
                # Keep shape stable; score will become neutral.
                prompt_ids = [self._get_pad_id()]

            token_ids_list.append(prompt_ids)
            batch_max_len = max(batch_max_len, len(prompt_ids))

        if batch_max_len == 0:
            return [0.0] * len(documents)

        pad_id = self._get_pad_id()
        padded = [ids + [pad_id] * (batch_max_len - len(ids)) for ids in token_ids_list]

        inputs = mx.array(padded)
        raw_logits: Any = model(inputs)

        if isinstance(raw_logits, (list, tuple)):
            logits = raw_logits[0]
        elif hasattr(raw_logits, "logits"):
            logits = raw_logits.logits
        else:
            logits = raw_logits

        logits = logits.astype(mx.float32)  # [batch, seq_len, vocab]

        yes_ids = mx.array(self._yes_token_ids)
        no_ids = mx.array(self._no_token_ids)
        score_values: list[Any] = []
        for i, ids in enumerate(token_ids_list):
            last_pos = max(0, len(ids) - 1)
            next_token_logits = logits[i, last_pos, :]

            # FIX: Keep yes/no logit reduction on MLX and materialize once per batch.
            yes_logit = mx.max(next_token_logits[yes_ids])
            no_logit = mx.max(next_token_logits[no_ids])
            score_values.append(yes_logit - no_logit)

        scores = self._materialize_scores(score_values)

        if truncated_docs > 0:
            log.debug(
                "Reranker truncation  |  docs_truncated=%d/%d  |  doc_max_tokens=%d  |  doc_max_chars=%d",
                truncated_docs,
                len(documents),
                self.max_doc_tokens,
                self.max_doc_chars,
            )

        del inputs, raw_logits, logits, padded, token_ids_list, yes_ids, no_ids, score_values
        return scores

    def _score_batch(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []

        model = self._model
        tokenizer = self._tokenizer
        if model is None or tokenizer is None:
            return [0.0] * len(documents)

        if self._score_head is None:
            if self._use_chat_logits:
                return self._score_batch_with_chat_logits(query, documents)

            if not self._warned_no_valid_scoring:
                log.warning(
                    "Reranker has no valid scoring mode (no score head / no chat-logit fallback). "
                    "Returning neutral scores to keep retrieval order."
                )
                self._warned_no_valid_scoring = True
            return [0.0] * len(documents)

        query_ids = self._tokenize(query, max_tokens=self.max_query_tokens, add_special_tokens=False)
        if not query_ids:
            query_ids = self._tokenize(query, max_tokens=self.max_query_tokens, add_special_tokens=True)
        if not query_ids:
            return [0.0] * len(documents)

        token_ids_list: list[list[int]] = []
        pool_indices: list[int] = []
        batch_max_len = 0
        truncated_docs = 0

        for doc_text in documents:
            pair_ids, was_truncated = self._build_pair_ids(query_ids, doc_text)
            if was_truncated:
                truncated_docs += 1
            if not pair_ids:
                pair_ids = query_ids[:]

            token_ids_list.append(pair_ids)
            pool_indices.append(self._pool_index_for_pair(pair_ids))
            batch_max_len = max(batch_max_len, len(pair_ids))

        if batch_max_len == 0:
            return [0.0] * len(documents)

        pad_id = self._get_pad_id()
        padded = [ids + [pad_id] * (batch_max_len - len(ids)) for ids in token_ids_list]

        inputs = mx.array(padded)
        inner_model = getattr(model, "model", model)
        raw_hidden: Any = inner_model(inputs)

        if isinstance(raw_hidden, (list, tuple)):
            hidden = raw_hidden[0]
        elif hasattr(raw_hidden, "last_hidden_state"):
            hidden = raw_hidden.last_hidden_state
        else:
            hidden = raw_hidden

        hidden = hidden.astype(mx.float32)

        score_values: list[Any] = []
        for i, pool_idx in enumerate(pool_indices):
            token_vec = hidden[i, pool_idx, :]  # [hidden_dim]
            score_values.append(self._score_from_head(token_vec))

        scores = self._materialize_scores(score_values)

        if truncated_docs > 0:
            log.debug(
                "Reranker truncation  |  docs_truncated=%d/%d  |  doc_max_tokens=%d  |  doc_max_chars=%d",
                truncated_docs,
                len(documents),
                self.max_doc_tokens,
                self.max_doc_chars,
            )

        del inputs, raw_hidden, hidden, padded, token_ids_list, pool_indices, score_values
        return scores

    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded() else "not-loaded"
        return f"Reranker(model={self.model_name}, status={status})"
