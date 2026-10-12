"""Cross-window context: let each window see the chart around it.

The problem this family solves
------------------------------
``event_v1`` trains one window of ``segment_beats`` in isolation.  A window
therefore cannot know what came before it or what the music does next, so the
seam between two generated windows is arbitrary: the model repeats a phrase it
just wrote, or restarts a pattern mid-bar.  Human charts are continuous -- the
author writes measure 5 *because* of measure 4.

What is added here
------------------
1. **Lead-in context (notes).**  Each training window is prefixed with the notes
   of the preceding ``context_beats`` beats, wrapped in ``<ctx>`` / ``<ctx_end>``
   delimiters.  The window is still the only thing the loss is computed on, so
   the model learns "given what I just wrote, continue it".  At generation the
   prefix is the model's *own* previously generated notes, which makes the
   window seam explicit instead of implicit.
2. **Neighbourhood audio.**  The audio slice fed to the encoder is extended by
   ``audio_margin_beats`` on both sides.  The music of the next few beats is
   known at generation time (the whole song is available), so looking ahead is
   legitimate -- only the *notes* are causal.  This is how a window can "notice
   what the next section will be".
3. **Rolling overlap (optional).**  ``overlap_beats`` shortens the window stride
   so consecutive windows share material.  The earlier window keeps the shared
   beats (its output already conditioned the next window's prefix), so merging
   stays conflict-free.

Why this is a separate family rather than a flag on ``event_v1``: the vocabulary
gains two control tokens, so the alphabet -- and therefore every checkpoint --
is different.  Two families can never load each other's weights, which is the
behaviour we want.

Prior art (the same ideas appear in the literature):

* Transformer-XL's segment-level recurrence carries the previous segment's
  hidden state into the current one (see the Transformer-XL write-up at
  https://research.google/blog/transformer-xl-unleashing-the-potential-of-attention-models/).
* Dance Dance Convolution predicts, per short audio window, whether a step
  occurs, with a receptive field wider than the window itself, then selects
  arrows conditioned on the surrounding chart context
  (https://ar5iv.labs.arxiv.org/html/1703.06891).
* Long-form generation systems generate small chunks conditioned on the
  previous chunk and stitch them together.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Iterable, Sequence

TARGET_SCHEMA = "event_ctx_v1"
FAMILY_LABEL = "event_ctx_v1（跨窗口上下文）"

# Control tokens.  They live after the shared EVENT_SPECIALS so the event_v1
# alphabet is untouched.
CTX_OPEN = "<ctx>"
CTX_CLOSE = "<ctx_end>"
CTX_SPECIALS: tuple[str, ...] = (CTX_OPEN, CTX_CLOSE)

# Defaults: eight beats of lead-in (half a window at segment_beats=16) and four
# beats of audio on each side.  A window plus margins stays inside the encoder's
# 512-frame budget for typical tempos, so no pooling is triggered.
DEFAULT_CONTEXT_BEATS = 8
DEFAULT_AUDIO_MARGIN_BEATS = 4
DEFAULT_CONTEXT_TOKENS = 192
# Fresh training uses a larger history cap; legacy generation fallbacks above
# remain stable. Saved checkpoint layouts always take precedence.
FRESH_CONTEXT_TOKENS = 768
# PerceiverS observed that a fixed, long context makes the model emit repetitive
# segments; dropping the lead-in at random is the cheap form of their
# "progressively increasing context" and stops the model from simply copying it.
DEFAULT_CONTEXT_DROPOUT = 0.15


def context_start_beat(start: Fraction, context_beats: int) -> Fraction:
    """First beat of the lead-in block for a window starting at ``start``."""

    if context_beats <= 0:
        return Fraction(start)
    return max(Fraction(0), Fraction(start) - int(context_beats))


def lead_in_events(points: Iterable[Any], start: Fraction, context_beats: int) -> list[tuple[Fraction, str]]:
    """Points inside the lead-in block, positioned relative to the block start.

    Positions are relative to the *block* rather than to the window because a
    position token cannot be negative.  The same rule is applied at generation
    time, so the token layout always matches training.
    """

    block_start = context_start_beat(start, context_beats)
    selected = [
        (Fraction(point.beat), str(point.note_text))
        for point in points
        if block_start <= Fraction(point.beat) < Fraction(start)
    ]
    selected.sort(key=lambda item: item[0])
    # Positions must be strictly increasing and start at 0 for the block.
    return [(beat - block_start, text) for beat, text in selected]


def truncate_lead_in(events: Sequence[tuple[Fraction, str]], context_tokens: int) -> list[tuple[Fraction, str]]:
    """Keep the newest lead-in events that fit in ``context_tokens``.

    Dropping from the left keeps the beats nearest the window, which are the
    ones that actually determine what comes next.
    """

    if context_tokens <= 0:
        return []
    kept: list[tuple[Fraction, str]] = []
    used = 0
    for beat, text in reversed(events):
        cost = 2 + len("".join(text.split()))  # position token + characters + <event_end>
        if used + cost > context_tokens:
            break
        kept.append((beat, text))
        used += cost
    kept.reverse()
    return kept


def prefix_token_ids(vocabulary: Any, events: Sequence[tuple[Fraction, str]], context_tokens: int) -> list[int]:
    """Token ids of a lead-in block: ``<ctx>`` events ``<ctx_end>``.

    Raises ``ValueError`` when an event falls outside the checkpoint vocabulary,
    so the caller can fall back to an empty lead-in instead of emitting a
    sequence the model has never seen.
    """

    from maichart.generation.event_model import EventSequence

    if not events:
        return [vocabulary.token_to_id[CTX_OPEN], vocabulary.token_to_id[CTX_CLOSE]]
    kept = truncate_lead_in(events, context_tokens)
    if not kept:
        return [vocabulary.token_to_id[CTX_OPEN], vocabulary.token_to_id[CTX_CLOSE]]
    encoded = vocabulary.encode(EventSequence(beats=[beat for beat, _ in kept],
                                              note_texts=[text for _, text in kept]))
    body = encoded[1:-1]  # drop <bos> ... <eos>
    return ([vocabulary.token_to_id[CTX_OPEN]] + body
            + [vocabulary.token_to_id[CTX_CLOSE]])


def loss_mask_after(prefix_tokens: int, total: int) -> list[bool]:
    """``False`` over the given lead-in, ``True`` over the window itself.

    The composite sequence is ``[<bos>] + prefix + window + [<eos>]`` and the
    decoder targets are that sequence shifted by one, so target ``j`` predicts
    sequence token ``j+1``.  The first window token sits at index
    ``len(prefix) + 1``, which means it is predicted by target ``len(prefix)``:
    exactly the first position that should count.  Everything before it is the
    given context, and training on it would teach the model to reconstruct its
    own input.
    """

    return [index >= prefix_tokens for index in range(total)]
