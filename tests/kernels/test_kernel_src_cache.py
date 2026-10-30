"""Kernel sources build from a copy named by their content, so a JIT key built from include paths survives a reinstall
of the same sources under another prefix and changes with the sources."""

from __future__ import annotations

import pathlib

from freetoken.kernel.utils import KERNEL_PATH, content_addressed


def _tree(root: pathlib.Path, body: str) -> pathlib.Path:
    (root / "include").mkdir(parents=True)
    (root / "include" / "k.cuh").write_text(body)
    (root / "jit").mkdir()
    (root / "jit" / "k.cu").write_text('#include "k.cuh"\n')
    return root


def test_the_same_sources_under_two_prefixes_share_one_copy(tmp_path):
    a = content_addressed(_tree(tmp_path / "pkg-a" / "csrc", "int k;\n"), tmp_path / "cache")
    b = content_addressed(_tree(tmp_path / "pkg-b" / "csrc", "int k;\n"), tmp_path / "cache")
    assert a == b and a.parent == tmp_path / "cache"
    assert (a / "include" / "k.cuh").read_text() == "int k;\n"
    assert (a / "jit" / "k.cu").read_text() == '#include "k.cuh"\n'
    assert [p.name for p in (tmp_path / "cache").iterdir()] == [a.name]


def test_changed_sources_get_their_own_copy(tmp_path):
    a = content_addressed(_tree(tmp_path / "pkg-a" / "csrc", "int k;\n"), tmp_path / "cache")
    b = content_addressed(_tree(tmp_path / "pkg-b" / "csrc", "int k2;\n"), tmp_path / "cache")
    assert a != b
    assert (b / "include" / "k.cuh").read_text() == "int k2;\n"


def test_an_unwritable_cache_builds_from_the_sources_in_place(tmp_path):
    src = _tree(tmp_path / "pkg" / "csrc", "int k;\n")
    blocker = tmp_path / "file"
    blocker.write_text("")
    assert content_addressed(src, blocker / "cache") == src


def test_the_kernel_sources_resolve_to_a_complete_tree():
    assert (KERNEL_PATH / "include").is_dir() and (KERNEL_PATH / "jit").is_dir()
