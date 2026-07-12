from pathlib import Path
from typing import Iterator

import pathspec

from code_radar.constants import SKIP_DIRS, TEXT_EXTENSIONS

class CodeReader:
    def __init__(self, root_path: str):
        self.root = Path(root_path).resolve()
        self.spec = self._load_gitignore()

    def _load_gitignore(self) -> pathspec.PathSpec:
        gitignore_path = self.root / ".gitignore"
        if gitignore_path.is_file():
            with open(gitignore_path, "r", encoding="utf-8", errors="ignore") as f:
                return pathspec.PathSpec.from_lines("gitwildmatch", f)
        return pathspec.PathSpec.from_lines("gitwildmatch", [])

    def files(self) -> Iterator[Path]:
        for dirpath, dirnames, filenames in self.root.walk():

            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

            for filename in filenames:
                if filename.startswith("."):
                    continue

                filepath = dirpath / filename
                rel_path = filepath.relative_to(self.root)

                if self.spec.match_file(rel_path.as_posix()):
                    continue

                if _is_binary(filepath):
                    continue

                yield filepath


def _is_binary(path: Path) -> bool:
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return False

    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
        return b"\0" in chunk
    except OSError:
        return True
