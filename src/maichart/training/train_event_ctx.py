"""Train the cross-window family (``event_ctx_v1``).

This is ``train_events`` with a different dataset and a different
``target_schema``.  Everything else -- the optimizer, the validation loop, the
checkpoint format, the contract comparison, the acceptance integration -- is the
same code, so the two families stay comparable and a fix in the training loop
reaches both.

What changes:

===================  ==========================  ==============================
                     ``event_v1``                ``event_ctx_v1``
===================  ==========================  ==============================
window               isolated                    prefixed with the preceding
                                                 ``context_beats`` of notes
audio slice          exactly the window           window + ``audio_margin_beats``
                                                 on both sides
stride               ``segment_beats``            ``segment_beats`` - ``overlap_beats``
vocabulary           shared specials              shared + ``<ctx>`` / ``<ctx_end>``
loss                 every window token           only the window, not the lead-in
===================  ==========================  ==============================

Usage is identical to ``train_events``, with four extra flags::

    python -m maichart.training.train_event_ctx --manifest ... --checkpoint-dir ... \
        --context-beats 8 --audio-margin-beats 4 --context-tokens 192 --overlap-beats 0

Checkpoints from this family carry ``target_schema = event_ctx_v1`` and a
different vocabulary signature, so they can never be mixed with ``event_v1``
weights in either direction.

Touch oversampling takes priority in mixed Touch/Slide windows, so its multiplier
also changes how often those mixed windows appear. Fine-rhythm oversampling is
versioned separately; multiplier 1 records fine windows for audit without
changing their sampling distribution.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from maichart.generation.event_ctx import FAMILY_LABEL, TARGET_SCHEMA
from maichart.training import train_events
from maichart.training.event_ctx_dataset import EventCtxDataset


def family_dataset_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "context_beats": int(args.context_beats),
        "audio_margin_beats": int(args.audio_margin_beats),
        "context_tokens": int(args.context_tokens),
        "overlap_beats": int(args.overlap_beats),
        "context_dropout": float(args.context_dropout),
        "slide_oversample": int(getattr(args, "slide_oversample", 1)),
        "touch_oversample": int(getattr(args, "touch_oversample", 1)),
        "fine_rhythm_oversample": getattr(args, "fine_rhythm_oversample", 1),
        "fine_rhythm_ratio_threshold": float(getattr(args, "fine_rhythm_ratio_threshold", 0.25)),
    }


def configure() -> None:
    """Install the family into the shared trainer (idempotent)."""

    train_events.DATASET_CLASS = EventCtxDataset
    train_events.TARGET_SCHEMA = TARGET_SCHEMA
    train_events.FAMILY_LABEL = FAMILY_LABEL
    train_events.dataset_kwargs = family_dataset_kwargs


def main(argv: list[str] | None = None) -> int:
    configure()
    return train_events.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
