"""Conservative proof that a changed prefix preserves an authored configuration."""
from __future__ import annotations

from maichart.event_timing import EventPoint
from .event_set_codec import _atoms, rotate_atom
from .stream_data import NOTE


def rotated_text(text, shift):
    """Use only codec-proved atom rotations; retain duplicate atoms."""
    return '/'.join(sorted(rotate_atom(atom, shift) for atom in _atoms(text)))


def corresponding_shift(authored, actual, *, minimum_changed_notes=3):
    """Return an exact C8 correspondence for complete, aligned event sequences.

    Exact identity may have any length. A changed rotation requires several
    actual NOTE observations, including two distinct authored positions.
    This never infers a transform from the target being supervised.
    """
    if len(authored) != len(actual):
        return None
    if any(a.kind != b.kind or a.end != b.end or a.start != b.start
           for a, b in zip(authored, actual)):
        return None
    if all(a.kind != NOTE or _atoms(a.text) == _atoms(b.text)
           for a, b in zip(authored, actual)):
        return 0
    notes = [(a.text, b.text) for a, b in zip(authored, actual) if a.kind == NOTE]
    if len(notes) < minimum_changed_notes or len({a for a, _ in notes}) < 2:
        return None
    possible = []
    for shift in range(1, 8):
        try:
            if all(rotated_text(a, shift) == '/'.join(_atoms(b)) for a, b in notes):
                possible.append(shift)
        except ValueError:
            continue
    return possible[0] if len(possible) == 1 else None


def rotation_boundary_clear(authored_prefix, first_changed_second, guard):
    """Refuse a changed C8 frame while an old action can affect its boundary.

    EventGuard's native interaction cutoff includes neighboring head timing;
    an earlier Hold/Slide may remain live much longer than a fixed history
    window, so inspect the complete authored prefix only for rare candidates.
    """
    cutoff = guard.interaction_start(float(first_changed_second))
    for step in authored_prefix:
        if step.kind != NOTE:
            continue
        try:
            _start, end, _sustained, errors = guard._lifetime(EventPoint(step.end,step.text))
        except (ValueError, TypeError):
            return False
        if errors or end >= cutoff:
            return False
    return True
