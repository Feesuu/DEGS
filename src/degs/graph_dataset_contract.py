from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class GraphDatasetContract:
    """Dataset-owned bounds for the shared incremental graph algorithm."""

    identity: str
    source_split: str
    train_count: int
    batch_size: int
    allow_final_partial_batch: bool = False

    def __post_init__(self) -> None:
        if not self.identity or not self.source_split:
            raise ValueError("graph dataset identity differs")
        if self.train_count <= 0 or self.batch_size <= 0:
            raise ValueError("graph dataset bounds differ")
        if not self.allow_final_partial_batch and self.train_count % self.batch_size:
            raise ValueError("graph dataset requires a full final batch")

    @property
    def batch_count(self) -> int:
        return math.ceil(self.train_count / self.batch_size)

    def batch_indices(self, batch_index: int) -> tuple[int, ...]:
        if not 0 <= batch_index < self.batch_count:
            raise ValueError("graph batch index differs")
        start = batch_index * self.batch_size
        stop = min(start + self.batch_size, self.train_count)
        indices = tuple(range(start, stop))
        if len(indices) != self.batch_size and not self.allow_final_partial_batch:
            raise ValueError("graph final batch differs")
        return indices

    def next_batch_indices(self, processed_count: int) -> tuple[int, ...]:
        if not 0 <= processed_count <= self.train_count:
            raise ValueError("processed graph count differs")
        if processed_count == self.train_count:
            return ()
        if processed_count % self.batch_size:
            raise ValueError("processed graph count is not a batch boundary")
        return self.batch_indices(processed_count // self.batch_size)

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "source_split": self.source_split,
            "train_count": self.train_count,
            "batch_size": self.batch_size,
            "allow_final_partial_batch": self.allow_final_partial_batch,
        }


SPREADSHEETBENCH_GRAPH_CONTRACT = GraphDatasetContract(
    identity="spreadsheetbench-train-0-200-v1",
    source_split="train[0,200)",
    train_count=200,
    batch_size=8,
)


__all__ = ["GraphDatasetContract", "SPREADSHEETBENCH_GRAPH_CONTRACT"]
