"""Generation must never be driven by a layout the checkpoint was not trained with.

Two regressions are pinned here:

* the numbering convention.  A checkpoint trained before the position origin was
  unified numbers its window from 0, with an alphabet that stops at
  ``segment_beats``; asking it for the block offset left the grammar with no
  legal position token and it emitted <eos> after a couple of notes;
* the window layout.  Generation used whatever the caller supplied (an empty UI
  field became 0), while the layout is part of the data contract.
"""

from fractions import Fraction

import pytest

from maichart.generation.event_cli import _vocabulary_position_limit
from maichart.generation.event_ctx_cli import _with_defaults
from maichart.generation.event_model import EventVocabulary

SEGMENT = 8
CONTEXT = 8


def _vocabulary(segment_beats: int) -> EventVocabulary:
    """A vocabulary whose position alphabet stops below ``segment_beats``."""

    from maichart.generation.event_model import EventSequence, char_token, position_token
    from maichart.generation.event_model import EVENT_SPECIALS, NOTE_CHARACTERS

    values = list(EVENT_SPECIALS) + [char_token(char) for char in NOTE_CHARACTERS]
    positions = [Fraction(numerator, denominator)
                 for denominator in (1, 2, 4, 8)
                 for numerator in range(segment_beats * denominator)]
    return EventVocabulary(values + [position_token(beat) for beat in sorted(set(positions))])


def test_position_limit_reflects_the_training_convention():
    old = _vocabulary(SEGMENT)
    # The pre-unification alphabet tops out just under segment_beats, so the
    # block offset (8) is not expressible.
    assert _vocabulary_position_limit(old) < SEGMENT
    assert _vocabulary_position_limit(old) < Fraction(CONTEXT)

    # A vocabulary built by the current dataset covers block + window.
    from maichart.generation.event_model import EventSequence, char_token, position_token
    from maichart.generation.event_model import EVENT_SPECIALS, NOTE_CHARACTERS

    values = list(EVENT_SPECIALS) + [char_token(char) for char in NOTE_CHARACTERS]
    positions = [Fraction(numerator, denominator)
                 for denominator in (1, 2, 4, 8)
                 for numerator in range((SEGMENT + CONTEXT) * denominator)]
    new = EventVocabulary(values + [position_token(beat) for beat in sorted(set(positions))])
    assert _vocabulary_position_limit(new) >= Fraction(CONTEXT)


def test_layout_comes_from_the_checkpoint_when_not_given():
    contract = {"family_params": {"context_beats": 8, "audio_margin_beats": 4,
                                  "context_tokens": 256, "overlap_beats": 0}}
    filled = _with_defaults(["--checkpoint", "x"], contract)
    assert "--context-tokens" in filled and filled[filled.index("--context-tokens") + 1] == "256"
    assert filled[filled.index("--context-beats") + 1] == "8"

    # Without a contract the family defaults apply.
    plain = _with_defaults(["--checkpoint", "x"])
    assert plain[plain.index("--context-tokens") + 1] == "192"

    # An explicit flag is preserved (and compared by _check_layout).
    explicit = _with_defaults(["--context-beats", "4"], contract)
    assert explicit[explicit.index("--context-beats") + 1] == "4"


def test_layout_mismatch_is_reported(tmp_path, capsys):
    from maichart.generation.event_ctx_cli import _check_layout

    _check_layout(["--context-beats", "0"],
                  {"family_params": {"context_beats": 8}})
    captured = capsys.readouterr()
    assert "与训练时不一致" in captured.err
    _check_layout(["--context-beats", "8"], {"family_params": {"context_beats": 8}})
    assert capsys.readouterr().err == ""
