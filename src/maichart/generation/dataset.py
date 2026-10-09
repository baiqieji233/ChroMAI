"""Training dataset adapter for conditional generation."""

from __future__ import annotations

import random
from typing import Any

from torch.utils.data import Dataset

from maichart.generation.conditions import condition_from_targets
from maichart.training.dataset_v25 import MaichartV25Dataset


class GeneratorDataset(Dataset):
    """Wrap the existing V2.5 cache with conditions and random frame crops."""

    def __init__(
        self,
        manifest_path: str,
        *,
        cache_dir: str = "cache",
        segment_frames: int = 512,
        random_crop: bool = True,
        include_unusable: bool = False,
    ) -> None:
        self.base = MaichartV25Dataset(
            manifest_path,
            cache_dir=cache_dir,
            include_unusable=include_unusable,
        )
        self.segment_frames = int(segment_frames)
        self.random_crop = bool(random_crop)
        if self.segment_frames <= 0:
            raise ValueError("segment_frames must be positive.")

    @property
    def input_dim(self) -> int:
        return self.base.input_dim

    @property
    def num_note_types(self) -> int:
        return self.base.num_note_types

    @property
    def note_type_vocab(self) -> tuple[str, ...]:
        return self.base.note_type_vocab

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base[index]
        x = sample["x"]
        y = sample["y"]
        level = sample.get("meta", {}).get("level")
        # Derive controls from the whole difficulty before taking a random
        # crop. Otherwise the same song receives a different style/radar
        # condition on every crop and the target leaks local statistics into
        # what is supposed to be a song-level control.
        condition = condition_from_targets(y, self.note_type_vocab, level=_as_float(level))
        if x.size(0) > self.segment_frames:
            start = random.randint(0, x.size(0) - self.segment_frames) if self.random_crop else 0
            end = start + self.segment_frames
            x = x[start:end]
            y = {key: value[start:end] for key, value in y.items()}
        return {
            "x": x,
            "y": y,
            "condition": condition.to_vector(),
            "meta": sample.get("meta", {}),
        }


def _as_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
