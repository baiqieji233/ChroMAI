"""Batch collation for conditional generator samples."""

from __future__ import annotations

from typing import Any

import torch


def collate_generator(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("collate_generator requires at least one sample.")
    lengths = [int(sample["x"].size(0)) for sample in samples]
    batch_size = len(samples)
    max_len = max(lengths)
    input_dim = int(samples[0]["x"].size(1))
    x = samples[0]["x"].new_zeros((batch_size, max_len, input_dim))
    padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)
    loss_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    y = {
        "note_presence": samples[0]["y"]["note_presence"].new_zeros((batch_size, max_len, 1)),
        "buttons": samples[0]["y"]["buttons"].new_zeros((batch_size, max_len, 8)),
        "note_type": samples[0]["y"]["note_type"].new_zeros((batch_size, max_len)),
        "density": samples[0]["y"]["density"].new_zeros((batch_size, max_len, 1)),
    }
    condition = samples[0]["condition"].new_zeros((batch_size, samples[0]["condition"].numel()))
    for index, sample in enumerate(samples):
        length = lengths[index]
        x[index, :length] = sample["x"]
        padding_mask[index, :length] = False
        loss_mask[index, :length] = True
        condition[index] = sample["condition"]
        for key in y:
            y[key][index, :length] = sample["y"][key]
    return {
        "x": x,
        "condition": condition,
        "padding_mask": padding_mask,
        "loss_mask": loss_mask,
        "y": y,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "meta": [sample.get("meta", {}) for sample in samples],
    }
