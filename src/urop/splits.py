"""Deterministic, task-grouped train/test splits.

All trials (run_ids) of one task stay on the same side of the split, so no
task leaks across train/test even when it has multiple trials (handoff §5:
"Deterministic split by task identity BEFORE codebook induction. All trials
of one task stay together.").
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from urop.data import CleanedRecord


def task_key(record: CleanedRecord) -> str:
    return f"{record.domain}:{record.instance_id}"


@dataclass
class SplitManifest:
    seed: int
    train_frac: float
    config_hash: str
    train_task_keys: list[str]
    test_task_keys: list[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _config_hash(seed: int, train_frac: float, task_keys: list[str]) -> str:
    payload = json.dumps({"seed": seed, "train_frac": train_frac, "task_keys": sorted(task_keys)}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_split(records: Iterable[CleanedRecord], *, seed: int = 42, train_frac: float = 0.8) -> SplitManifest:
    """Deterministic split, grouped by task_key.

    Determinism holds in two senses: (1) same seed + same task set gives the
    same split regardless of input record order, because the task-key set is
    deduplicated and sorted before shuffling; (2) the recorded config_hash
    lets a caller detect when the underlying task set changed.
    """
    if not 0 < train_frac < 1:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")

    keys = sorted({task_key(r) for r in records})
    if not keys:
        raise ValueError("no records to split")

    rng = random.Random(seed)
    shuffled = list(keys)
    rng.shuffle(shuffled)

    if len(shuffled) == 1:
        n_train = 1
    else:
        n_train = round(len(shuffled) * train_frac)
        n_train = max(1, min(n_train, len(shuffled) - 1))

    train_keys = sorted(shuffled[:n_train])
    test_keys = sorted(shuffled[n_train:])

    return SplitManifest(
        seed=seed,
        train_frac=train_frac,
        config_hash=_config_hash(seed, train_frac, keys),
        train_task_keys=train_keys,
        test_task_keys=test_keys,
    )


def verify_no_overlap(manifest: SplitManifest) -> None:
    overlap = set(manifest.train_task_keys) & set(manifest.test_task_keys)
    if overlap:
        raise AssertionError(f"train/test task overlap: {sorted(overlap)[:5]}")


def assign_split(records: Iterable[CleanedRecord], manifest: SplitManifest) -> dict[str, list[CleanedRecord]]:
    train_set = set(manifest.train_task_keys)
    test_set = set(manifest.test_task_keys)
    out: dict[str, list[CleanedRecord]] = {"train": [], "test": [], "unassigned": []}
    for r in records:
        k = task_key(r)
        if k in train_set:
            out["train"].append(r)
        elif k in test_set:
            out["test"].append(r)
        else:
            out["unassigned"].append(r)
    return out


def save_manifest(manifest: SplitManifest, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(manifest.to_json(), encoding="utf-8")


def load_manifest(path: str | Path) -> SplitManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return SplitManifest(**data)
