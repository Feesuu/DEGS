#!/usr/bin/env python3
"""Recreate and verify the fixed seed-42 Skill2Bench train/test split."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Pinned Skill-Entropy-RL skill2_bench directory.",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        required=True,
        help="Expanded pinned Trace2Skill_Skill2Bench baseline runtime.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(ROOT / "src"))
    baseline_root = args.baseline_root.expanduser().resolve()
    if not (baseline_root / "skill2bench/dataset.py").is_file():
        raise ValueError("Skill2Bench baseline root differs")
    sys.path.insert(0, str(baseline_root))
    from skill2bench.dataset import prepare_splits
    from degs_skill2bench.dataset import load_split

    manifest = prepare_splits(
        args.source_dir,
        args.output_dir,
        seed=42,
        train_size=100,
    )
    load_split(args.output_dir / "train_100.jsonl", split="train")
    load_split(args.output_dir / "test_200.jsonl", split="test")
    print(
        f"verified Skill2Bench split: train={manifest['train_size']} "
        f"test={manifest['test_size']} seed={manifest['seed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
