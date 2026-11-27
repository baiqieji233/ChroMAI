"""Read-only hand assignment hints at authored beat positions."""

from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol, Sequence

from maichart.event_timing import EventPoint, EventTempo
from maichart.tempo_clock import TempoClock


@dataclass(frozen=True)
class HandFeelHint:
    beat: Fraction
    reason: str
    alternatives: tuple[str, ...]


class HandFeelAnalyzer(Protocol):
    def analyze(
        self, points: Sequence[EventPoint], clock: TempoClock, start: Fraction, stop: Fraction
    ) -> Sequence[HandFeelHint]:
        """Analyze a range in the context of the complete chart."""
        ...


class PhraseHandAnalyzer:
    """Summarize competing full paths at notes in a selected beat range."""

    def analyze(
        self, points: Sequence[EventPoint], clock: TempoClock, start: Fraction, stop: Fraction
    ) -> Sequence[HandFeelHint]:
        from .hand_flow import recognize_hands

        tempo = [EventTempo(beat, bpm) for beat, bpm in zip(clock.beats, clock.bpms)]
        end = clock.at(stop)
        chart_end = max(end, clock.at(points[-1].beat)) if points else end
        result = recognize_hands(points, tempo, clock, clock.at(start), end, chart_end)
        beats = {round(clock.at(p.beat), 7): p.beat for p in points if start <= p.beat <= stop}
        hints = []
        for frames in zip(*(path["frames"] for path in result["solutions"])):
            beat = beats.get(frames[0]["second"])
            if beat is None:
                continue
            alternatives = []
            for frame in frames:
                hands = []
                for side, word in (("left", "左手"), ("right", "右手")):
                    hits = [c for c in frame[side] if c["phase"] in ("pulse", "start")]
                    if hits:
                        hands.append(word + " " + ", ".join(str(c["lane"] or c["pad"]) for c in hits))
                summary = " / ".join(hands)
                if summary and summary not in alternatives:
                    alternatives.append(summary)
            if len(alternatives) > 1:
                hints.append(HandFeelHint(beat, "这组有几种代价接近的分配", tuple(alternatives)))
        return hints
