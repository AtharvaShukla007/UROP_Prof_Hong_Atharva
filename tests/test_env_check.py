"""CPU-only checks for src/urop/env_check.py -- pure version-parsing logic,
no network, no actual torch/vllm install needed.
"""
from __future__ import annotations

from urop.env_check import public_version, torch_versions_compatible


def test_public_version_strips_local_cuda_suffix():
    assert public_version("2.4.0+cu121") == "2.4.0"


def test_public_version_none_for_missing_or_unparseable():
    assert public_version(None) is None
    assert public_version("") is None
    assert public_version("not-a-version") is None


def test_torch_versions_compatible_true_when_installed_has_local_suffix_but_release_matches():
    """The real bug this fixes: Kaggle reports installed torch WITH a local
    CUDA-build suffix (e.g. "2.4.0+cu121"), while a package's declared pin
    (e.g. vLLM's `torch==2.7.1`) never carries one. Literal string equality
    between these would never match even when the release is identical.
    """
    assert torch_versions_compatible("2.4.0+cu121", "2.4.0")


def test_torch_versions_compatible_false_for_genuinely_different_releases():
    assert not torch_versions_compatible("2.4.0+cu121", "2.7.1")


def test_torch_versions_compatible_false_when_installed_is_missing():
    assert not torch_versions_compatible(None, "2.7.1")


def test_torch_versions_compatible_false_when_required_is_unparseable():
    assert not torch_versions_compatible("2.4.0+cu121", "not-a-version")
