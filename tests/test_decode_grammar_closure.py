"""The decode grammar must make button-colliding cells unreachable.

`is_complete` already refused to close such a cell; the hole was that
`can_continue` accepted every *prefix* of a catalog atom, so the decoder could
type a part that repeats a button the cell already strikes and then be stuck
(no character can extend it, no close is legal).  The engine's escape hatch turns
that dead end into an output cell with a `duplicate_button_head` warning -- the
intended "pass through but warn" behaviour -- but the dead end itself is
avoidable, and it wasted window budget whenever it was hit.
"""

from __future__ import annotations

import pytest

from maichart.generation.event_cli import _NoteGrammar

ATOMS = [
    # plain buttons
    "1", "2", "3", "4", "5", "6", "7", "8",
    # holds (incl. modifier order variants)
    "4h[8:1]", "5h[8:1]", "5h[16:1]", "4h[4:1]x", "8h[8:1]",
    # slides and compound slides
    "1-5[8:1]", "1-5[8:1]*-3[8:1]", "5-1[8:1]*-8[8:1]",
    "8b-4[8:1]", "8b-4[8:1]b", "8-5[8:1]", "2-8[8:1]",
    "7x-4[8:1]", "1-4b[8:1]",
    # compact chords
    "47", "46", "12", "36",
    # touches
    "A", "A4", "A5", "A4f", "B8", "B1", "E2", "E8",
]

# Every entry above must be a legal cell on its own.
assert all(atom.strip() for atom in ATOMS)


def grammar(atoms: list[str] | None = None) -> _NoteGrammar:
    built = _NoteGrammar.from_atoms(atoms or ATOMS)
    assert built is not None
    return built


@pytest.mark.parametrize(
    "text",
    [
        "1/1",                                  # same tap twice
        "1/1-5[8:1]",                           # tap + slide from the same button
        "1/1-5[8:1]*-3[8:1]",                   # tap + compound slide, same head
        "5/5h[8:1]",                            # tap + hold on the same button
        "4/4h[4:1]x",
        "4h[8:1]/4h[8:1]",                      # identical hold twice
        "4h[8:1]/5h[8:1]/47",                   # compact chord reusing 4
        "8b-4[8:1]/8b-4[8:1]b",                 # identical slide twice
        "A4/A4",                                # identical touch twice
        "36/E8/B8/B1/E2/36",                    # second compact chord repeats 3 and 6
    ],
)
def test_colliding_parts_are_not_offered(text):
    assert grammar().can_continue(text) is False, text
    assert grammar().is_complete(text) is False, text


@pytest.mark.parametrize(
    "text",
    [
        "1/5",                                  # legal chord, distinct heads
        "2/8-5[8:1]",                           # tap + slide from another button
        "6/7x-4[8:1]",                          # tap 6 + slide from 7
        "1-4b[8:1]/8-5[8:1]",                   # two slides, distinct heads
        "36/E8/B8/B1/E2",                       # compact chord + four touches
        "A/A4",                                 # distinct touch heads
        "1", "1-5[8:1]", "4h[8:1]", "47",       # single parts
        "8b-4[8:1]b", "4h[4:1]x",
    ],
)
def test_legal_cells_still_pass(text):
    assert grammar().can_continue(text) is True, text
    assert grammar().is_complete(text) is True, text


def test_the_dead_end_is_refused_at_the_first_colliding_character():
    """`1/1-5[8:1]*-3[8:1]` must stop being typed right after the second `1`."""

    grammar_ = grammar()
    text = "1/1-5[8:1]*-3[8:1]"
    accepted = [text[:length] for length in range(1, len(text) + 1)
                if grammar_.can_continue(text[:length])]
    assert accepted == ["1", "1/"], accepted
    assert "1/1" not in accepted


def test_the_eight_button_chord_can_be_closed_but_not_extended():
    """With every button struck, `/` is refused; closing the chord is legal."""

    digits = [str(index) for index in range(1, 9)]
    grammar_ = grammar(digits)
    assert grammar_.can_continue("1/2/3/4/5/6/7/8") is True
    assert grammar_.is_complete("1/2/3/4/5/6/7/8") is True
    assert grammar_.can_continue("1/2/3/4/5/6/7/8/") is False
    # A touch is a different button, so with touch atoms the slash stays legal.
    assert grammar_ is not None
    assert grammar().can_continue("A/A") is True


def test_longer_spelling_with_a_different_head_is_still_offered():
    """`A` collides with an existing `A`, but `A4` is a different button.

    The character stays available, otherwise the legal longer spelling `A/A4`
    could never be reached; `4/47` has no such escape and is refused.
    """

    grammar_ = grammar()
    assert grammar_.can_continue("A/A") is True
    assert grammar_.can_continue("A/A4") is True
    assert grammar_.is_complete("A/A4") is True
    assert grammar_.can_continue("4/4") is False
    assert grammar_.can_continue("4/47") is False


def test_grammar_without_atoms_is_not_built():
    assert _NoteGrammar.from_atoms([]) is None
    assert _NoteGrammar.from_atoms(None) is None


def test_same_head_slide_branch_cap_is_hard_but_configurable():
    double = "1-4[8:1]*-6[8:1]"
    triple = "1-4[8:1]*-6[8:1]*-8[8:1]"
    capped = _NoteGrammar.from_atoms(["1", double, triple], max_star_slides=2)
    assert capped is not None
    assert triple not in capped.atoms
    assert capped.can_continue("1-4[8:1]")
    assert capped.can_continue(double)
    assert not capped.can_continue(double + "*")
    assert not capped.is_complete(triple)

    only_over_cap = _NoteGrammar.from_atoms([triple], max_star_slides=2)
    assert only_over_cap is not None
    assert not only_over_cap.can_continue(triple)
    assert not only_over_cap.is_complete(triple)

    diagnostic = _NoteGrammar.from_atoms([triple], max_star_slides=3)
    assert diagnostic is not None
    assert triple in diagnostic.atoms
    assert diagnostic.can_continue(triple)
    assert diagnostic.is_complete(triple)


def test_same_head_cap_does_not_change_slash_chords():
    grammar_ = _NoteGrammar.from_atoms(["1", "2", "3", "1/2"])
    assert grammar_ is not None
    assert grammar_.can_continue("1/2")
    assert grammar_.is_complete("1/2")


def test_head_closures_cover_every_prefix():
    """The closures are consistent: each atom's own heads are reachable from it."""

    grammar_ = grammar()
    for atom in ATOMS:
        heads = grammar_.heads[atom]
        assert heads, atom
        assert heads <= grammar_.head_closure[atom], atom
        assert grammar_.can_continue(atom) is True, atom
        assert grammar_.is_complete(atom) is True, atom
