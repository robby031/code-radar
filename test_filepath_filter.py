import sys
import types
from pathlib import Path


server_pkg = types.ModuleType("code_radar.server")
server_pkg.__path__ = [str(Path(__file__).parent / "code_radar" / "server")]
state = types.ModuleType("code_radar.server.state")
state._store = None
state._root = None
state.env_int = lambda _name, default, minimum=1: max(minimum, default)

sys.modules["code_radar.server"] = server_pkg
sys.modules["code_radar.server.state"] = state

reranker = types.ModuleType("code_radar.reranker")
reranker.RerankTimeoutError = TimeoutError
reranker.Reranker = object
sys.modules["code_radar.reranker"] = reranker

from code_radar.server import search_utils as utils


class FakeStore:
    def __init__(self, filepaths: list[str]) -> None:
        self.filepaths = filepaths

    def get_all_metadatas(self):
        return {
            "metadatas": [
                {"filepath": filepath}
                for filepath in self.filepaths
            ]
        }


def test_filepath_filter_prefixed_path_does_not_fall_back_to_basename() -> None:
    old_store = state._store
    old_root = state._root
    try:
        state._root = "/repo/gateway"
        state._store = FakeStore(
            [
                "src/core/integration/otomax/datasource_product.go",
                "src/core/integration/host/datasource_product.go",
                "src/core/integration/swpulsa/datasource_product.go",
            ]
        )

        resolved = utils.resolve_filepath_filters(
            "gateway/src/core/integration/swpulsa/datasource_product.go"
        )

        assert resolved == ["src/core/integration/swpulsa/datasource_product.go"]
    finally:
        state._store = old_store
        state._root = old_root


def test_filepath_filter_basename_only_matches_same_basename() -> None:
    old_store = state._store
    old_root = state._root
    try:
        state._root = "/repo/gateway"
        state._store = FakeStore(
            [
                "src/core/integration/host/host.go",
                "src/core/integration/swpulsa/datasource_product.go",
            ]
        )

        resolved = utils.resolve_filepath_filters("host.go")

        assert resolved == ["src/core/integration/host/host.go"]
    finally:
        state._store = old_store
        state._root = old_root


def test_filepath_filter_directory_prefix_matches_children() -> None:
    old_store = state._store
    old_root = state._root
    try:
        state._root = "/repo/gateway"
        state._store = FakeStore(
            [
                "src/core/integration/host/host.go",
                "src/core/integration/swpulsa/datasource_product.go",
                "src/controllers/helpers/transaction/daemon.go",
            ]
        )

        resolved = utils.resolve_filepath_filters("src/core/integration")

        assert resolved == [
            "src/core/integration/host/host.go",
            "src/core/integration/swpulsa/datasource_product.go",
        ]
    finally:
        state._store = old_store
        state._root = old_root
