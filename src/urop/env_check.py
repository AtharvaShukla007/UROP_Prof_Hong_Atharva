"""Environment/dependency compatibility checks -- pure logic, testable
without a GPU, network access, or any actual package install.
"""
from __future__ import annotations

from packaging.version import InvalidVersion, Version


def public_version(raw: str | None) -> str | None:
    """The "public" release segment of a version string (release/pre/post/
    dev, per PEP 440), stripping any local version label -- e.g. torch's own
    "+cu121" CUDA-build suffix. Returns None for a missing or unparseable
    string; never guesses.
    """
    if not raw:
        return None
    try:
        return str(Version(raw).public)
    except InvalidVersion:
        return None


def torch_versions_compatible(installed: str | None, required: str | None) -> bool:
    """True iff `installed` and `required` name the same PUBLIC torch
    release, ignoring a local build suffix like "+cu121".

    Kaggle reports its preinstalled torch with a local CUDA-build suffix
    (e.g. "2.4.0+cu121"), while a package's declared pin (e.g. vLLM's
    `torch==2.7.1`) does not carry one -- literal string equality between
    the two would never match even when the underlying release is actually
    identical. This compares only the PEP 440 public segment.

    A True result is only ONE necessary compatibility check. It does NOT by
    itself prove a server will actually start on a given GPU/driver/
    attention-backend combination -- that stays unverified until the server
    process is actually observed to come up (see docs/kaggle_quickstart.md).
    """
    a, b = public_version(installed), public_version(required)
    return a is not None and a == b
