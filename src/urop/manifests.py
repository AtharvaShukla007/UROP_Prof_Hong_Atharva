"""Provenance manifests: what produced a run's outputs, from what inputs."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


@dataclass
class RunManifest:
    run_id: str
    stage: str
    created_at: str
    config: dict[str, Any]
    config_hash: str
    input_files: dict[str, str]
    backend_label: str
    python_version: str = field(default_factory=lambda: sys.version)
    platform_info: str = field(default_factory=platform.platform)
    stop_reason: str | None = None
    notes: str = ""

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.__dict__, indent=indent, sort_keys=True)


def build_run_manifest(
    *,
    run_id: str,
    stage: str,
    config: dict[str, Any],
    input_files: list[str | Path],
    backend_label: str,
) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        stage=stage,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        config=config,
        config_hash=sha256_json(config),
        input_files={str(p): sha256_file(p) for p in input_files},
        backend_label=backend_label,
    )


def save_manifest(manifest: RunManifest, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(manifest.to_json(), encoding="utf-8")


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
