#!/usr/bin/env python3
"""Fetch and verify the public SpreadsheetBench data used by DEGS."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence


SOURCE_URL = "https://github.com/Qwen-Applications/Trace2Skill.git"
SOURCE_COMMIT = "3d0b52a140f002a512930252b613c49048f7d5ac"
VERIFIED_RELATIVE = Path(
    "data/spreadsheetbench_verified/spreadsheetbench_verified_400"
)
FULL_RELATIVE = Path("data/all_data_912_v0.1")
VERIFIED_DATASET_SHA256 = (
    "bcecaa89a005bd4e3bbe98da150a86e8062c27f262e575d5e47bd9861b3525e7"
)
VERIFIED_TREE_SHA256 = (
    "24d0574150633eda8953db512436a8ee0e355c861a5c427f2137fca7a6cc1e18"
)
FULL_DATASET_SHA256 = (
    "e5137ecbec4273d91344a0c8feb2aff2d4a93d5881ac40e490250dfd8db227de"
)
FULL_TREE_SHA256 = (
    "c8a376cad23b9ca9b17f013d154390fb2d373a543488d6ff6053e57c789e532f"
)
VERIFIED_TRAIN_IDS_SHA256 = (
    "4e7e54c59d34e671fb02861636f636fe46f50271a40df055cb4a349561cbef93"
)
VERIFIED_DEVELOPMENT_IDS_SHA256 = (
    "eb68c9b69307708ebf7ac04c86c82efddafebc85c5b428ba65d24ee5478f2d3d"
)
FULL_IDS_SHA256 = (
    "2062dfa283b12f92860d138a78478bb2ccd7459c42d8a03958271f3b89dad717"
)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _dataset_ids(root: Path) -> list[str]:
    dataset = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    if type(dataset) is not list or any(type(row) is not dict for row in dataset):
        raise ValueError(f"dataset rows differ: {root}")
    identifiers = [str(row.get("id", "")) for row in dataset]
    if any(not item for item in identifiers) or len(set(identifiers)) != len(
        identifiers
    ):
        raise ValueError(f"dataset task identities differ: {root}")
    return identifiers


def _tree_sha256(root: Path) -> str:
    if not root.is_dir():
        raise ValueError(f"dataset root differs: {root}")
    files = []
    for path in root.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            files.append(path)
    digest = hashlib.sha256()
    for path in sorted(files):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_file_sha256(path)))
    return digest.hexdigest()


def verify_checkout(checkout: Path) -> dict[str, Any]:
    root = checkout.expanduser().resolve()
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode or completed.stdout.strip() != SOURCE_COMMIT:
        raise ValueError("Trace2Skill checkout is not at the pinned commit")
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=no"],
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode or status.stdout.strip():
        raise ValueError("tracked files in the Trace2Skill checkout are modified")
    verified = root / VERIFIED_RELATIVE
    full = root / FULL_RELATIVE
    verified_ids = _dataset_ids(verified)
    full_ids = _dataset_ids(full)
    observed = {
        "source_url": SOURCE_URL,
        "source_commit": SOURCE_COMMIT,
        "verified_root": str(verified),
        "full_root": str(full),
        "verified_dataset_sha256": _file_sha256(verified / "dataset.json"),
        "verified_tree_sha256": _tree_sha256(verified),
        "verified_task_count": len(verified_ids),
        "verified_train_range": [0, 200],
        "verified_development_range": [200, 400],
        "verified_train_ids_sha256": _canonical_sha256(verified_ids[:200]),
        "verified_development_ids_sha256": _canonical_sha256(verified_ids[200:400]),
        "full_dataset_sha256": _file_sha256(full / "dataset.json"),
        "full_tree_sha256": _tree_sha256(full),
        "full_task_count": len(full_ids),
        "full_ids_sha256": _canonical_sha256(full_ids),
    }
    expected = {
        **observed,
        "verified_dataset_sha256": VERIFIED_DATASET_SHA256,
        "verified_tree_sha256": VERIFIED_TREE_SHA256,
        "verified_task_count": 400,
        "verified_train_ids_sha256": VERIFIED_TRAIN_IDS_SHA256,
        "verified_development_ids_sha256": VERIFIED_DEVELOPMENT_IDS_SHA256,
        "full_dataset_sha256": FULL_DATASET_SHA256,
        "full_tree_sha256": FULL_TREE_SHA256,
        "full_task_count": 912,
        "full_ids_sha256": FULL_IDS_SHA256,
    }
    if observed != expected:
        differences = {
            key: {"expected": expected[key], "observed": observed[key]}
            for key in expected
            if observed.get(key) != expected[key]
        }
        raise ValueError(f"SpreadsheetBench identity differs: {differences}")
    return observed


def fetch(checkout: Path) -> None:
    target = checkout.expanduser().absolute()
    if target.exists():
        if not (target / ".git").exists():
            raise FileExistsError("checkout target exists and is not a Git checkout")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--no-checkout", SOURCE_URL, str(target)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(target), "checkout", "--detach", SOURCE_COMMIT],
        check=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="do not clone; verify an existing checkout",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.verify_only:
        fetch(args.checkout)
    result = verify_checkout(args.checkout)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
