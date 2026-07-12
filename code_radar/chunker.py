import ast
import hashlib
from pathlib import Path
from typing import Any

from code_radar.logging import get_logger

log = get_logger(__name__)


def estimate_tokens(text: str) -> int:
    # Fast heuristic for code/text (roughly 1 token ~ 4 chars)
    return max(1, len(text) // 4)


def count_tokens(tokenizer: Any, text: str) -> int:
    # Kept for tests / utilities that need exact tokenizer count.
    return len(tokenizer.encode(text))


def _token_count_fast(_tokenizer: Any, text: str) -> int:
    # Chunking should stay cheap; exact tokenization happens again in embedding stage.
    return estimate_tokens(text)


def chunk_file(filepath: Path, root: Path, tokenizer: Any, content: str | None = None) -> list[dict[str, Any]]:
    rel_path = str(filepath.relative_to(root))

    if content is None:
        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            log.warning("Failed to read file  |  path=%s  |  error=%s", rel_path, exc)
            return []

    if not content.strip():
        log.debug("Skipping empty file  |  path=%s", rel_path)
        return []

    lang = filepath.suffix.lstrip(".").lower()

    if estimate_tokens(content) <= 1000:
        return [
            {
                "id": hashlib.md5(f"{rel_path}:file".encode()).hexdigest(),
                "filepath": rel_path,
                "content": content,
                "chunk_type": "file",
                "start_line": 1,
                "end_line": len(content.splitlines()),
                "metadata": {
                    "language": lang,
                    "token_count": _token_count_fast(tokenizer, content),
                },
            }
        ]

    if lang == "py":
        chunks = _chunk_python(content, rel_path, tokenizer)
    elif lang == "go":
        chunks = _chunk_go(content, rel_path, tokenizer)
    else:
        chunks = _chunk_fallback(content, rel_path, tokenizer, lang)

    log.debug(
        "Chunked  |  file=%s  |  lang=%s  |  chunks=%d  |  lines=%d",
        rel_path,
        lang,
        len(chunks),
        len(content.splitlines()),
    )
    return chunks


# PYTHON PARSER (AST)
def _chunk_python(content: str, rel_path: str, tokenizer: Any) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return _chunk_fallback(content, rel_path, tokenizer, "py")

    lines = content.splitlines(keepends=True)
    chunks: list[dict[str, Any]] = []
    cursor = 1

    def _process_children(node: ast.AST, parent_name: str | None = None) -> None:
        nonlocal cursor

        funcs = [
            n
            for n in ast.iter_child_nodes(node)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        classes = [n for n in ast.iter_child_nodes(node) if isinstance(n, ast.ClassDef)]

        all_nodes = sorted(funcs + classes, key=lambda x: x.lineno)
        parent_end = getattr(node, "end_lineno", None) or len(lines)

        for n in all_nodes:
            if n.lineno > cursor:
                _push_block(lines, cursor, n.lineno - 1, rel_path, tokenizer, "py", chunks)

            is_class = isinstance(n, ast.ClassDef)
            node_type = "class" if is_class else "method" if parent_name else "function"

            n_end = n.end_lineno or n.lineno
            chunk_content = "".join(lines[n.lineno - 1 : n_end])

            prefix = f"# Class: {parent_name}\n" if parent_name and not is_class else ""
            final_content = prefix + chunk_content

            chunks.append(
                {
                    "id": hashlib.md5(
                        f"{rel_path}:{node_type}:{parent_name or ''}:{n.name}:{n.lineno}".encode()
                    ).hexdigest(),
                    "filepath": rel_path,
                    "content": final_content,
                    "chunk_type": node_type,
                    "start_line": n.lineno,
                    "end_line": n_end,
                    "metadata": {
                        "language": "py",
                        "token_count": _token_count_fast(tokenizer, final_content),
                        "name": n.name,
                        "parent": parent_name,
                    },
                }
            )

            if is_class:
                cursor = n.lineno + 1
                _process_children(n, parent_name=n.name)

            cursor = n_end + 1

        if cursor <= parent_end:
            _push_block(lines, cursor, parent_end, rel_path, tokenizer, "py", chunks)
        cursor = parent_end + 1

    _process_children(tree)
    return chunks


# GO PARSER (TREE-SITTER)
def _chunk_go(content: str, rel_path: str, tokenizer: Any) -> list[dict[str, Any]]:
    try:
        from tree_sitter_language_pack import get_parser

        parser = get_parser("go")
        tree = parser.parse(content.encode("utf-8"))
    except ImportError:
        log.warning("tree-sitter not installed, falling back to line chunking for Go")
        return _chunk_fallback(content, rel_path, tokenizer, "go")
    except Exception:
        return _chunk_fallback(content, rel_path, tokenizer, "go")

    lines = content.splitlines(keepends=True)
    chunks: list[dict[str, Any]] = []
    cursor = 1

    query_str = """
    (function_declaration) @func
    (method_declaration) @method
    (type_declaration) @type
    """

    try:
        query = parser.language.query(query_str)  # type: ignore[union-attr]
        captures = query.captures(tree.root_node)  # type: ignore[union-attr]

        nodes = sorted([node for node, _ in captures], key=lambda n: n.start_point[0])

        for node in nodes:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1

            if start_line > cursor:
                _push_block(lines, cursor, start_line - 1, rel_path, tokenizer, "go", chunks)

            node_type = node.type
            chunk_content = "".join(lines[start_line - 1 : end_line])

            name_node = node.child_by_field_name("name")
            name = name_node.text.decode("utf-8") if name_node else "anonymous"

            chunks.append(
                {
                    "id": hashlib.md5(f"{rel_path}:{node_type}:{name}:{start_line}".encode()).hexdigest(),
                    "filepath": rel_path,
                    "content": chunk_content,
                    "chunk_type": node_type,
                    "start_line": start_line,
                    "end_line": end_line,
                    "metadata": {
                        "language": "go",
                        "token_count": _token_count_fast(tokenizer, chunk_content),
                        "name": name,
                    },
                }
            )
            cursor = end_line + 1

    except Exception as e:
        log.error("Tree-sitter query error for Go  |  error=%s", e)
        return _chunk_fallback(content, rel_path, tokenizer, "go")

    if cursor <= len(lines):
        _push_block(lines, cursor, len(lines), rel_path, tokenizer, "go", chunks)

    return chunks


# HELPERS and FALLBACK
def _push_block(
    lines: list[str],
    start: int,
    end: int,
    rel_path: str,
    tokenizer: Any,
    lang: str,
    chunks: list[dict[str, Any]],
) -> None:
    block = "".join(lines[start - 1 : end])
    if not block.strip():
        return

    chunks.append(
        {
            "id": hashlib.md5(f"{rel_path}:block:{start}".encode()).hexdigest(),
            "filepath": rel_path,
            "content": block,
            "chunk_type": "code_block",
            "start_line": start,
            "end_line": end,
            "metadata": {
                "language": lang,
                "token_count": _token_count_fast(tokenizer, block),
            },
        }
    )


def _chunk_fallback(content: str, rel_path: str, tokenizer: Any, lang: str) -> list[dict[str, Any]]:
    lines = content.splitlines(keepends=True)
    chunks: list[dict[str, Any]] = []

    buf: list[str] = []
    buf_chars = 0
    start = 1
    max_chars = 4000  # ~1000 tokens

    for i, line in enumerate(lines, 1):
        line_chars = len(line)

        if buf and (buf_chars + line_chars > max_chars):
            block = "".join(buf)
            chunks.append(
                {
                    "id": hashlib.md5(f"{rel_path}:block:{start}".encode()).hexdigest(),
                    "filepath": rel_path,
                    "content": block,
                    "chunk_type": "code_block",
                    "start_line": start,
                    "end_line": i - 1,
                    "metadata": {
                        "language": lang,
                        "token_count": _token_count_fast(tokenizer, block),
                    },
                }
            )
            buf, buf_chars = [], 0
            start = i

        buf.append(line)
        buf_chars += line_chars

    if buf:
        block = "".join(buf)
        chunks.append(
            {
                "id": hashlib.md5(f"{rel_path}:block:{start}".encode()).hexdigest(),
                "filepath": rel_path,
                "content": block,
                "chunk_type": "code_block",
                "start_line": start,
                "end_line": len(lines),
                "metadata": {
                    "language": lang,
                    "token_count": _token_count_fast(tokenizer, block),
                },
            }
        )

    return chunks
