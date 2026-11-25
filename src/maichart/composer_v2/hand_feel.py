"""Interface reserved for external, read-only phrase diagnostics."""
from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol, Sequence

from maichart.event_timing import EventPoint
from maichart.tempo_clock import TempoClock


@dataclass(frozen=True)
class HandFeelHint:
    beat: Fraction
    reason: str
    alternatives: tuple[str, ...]


class HandFeelAnalyzer(Protocol):
    def analyze(self, points: Sequence[EventPoint], clock: TempoClock,
                start: Fraction, stop: Fraction) -> Sequence[HandFeelHint]:
        """Analyze a range in the context of the complete chart."""
        ...
