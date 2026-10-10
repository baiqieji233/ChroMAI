"""Exact-event controls and diagnostic-only chart quality assessment.

No fixed-frame labels, candidate selection, or gameplay verification are used.
Event inputs only need ``beat`` and ``note_text`` attributes. Tempo inputs need
``beat`` and ``bpm`` attributes (mappings and two-item tuples also work).
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from maichart.generation.conditions import GenerationCondition
from maichart.notes import (
    BASIC_COMPACT_CHORD_RE,
    ParsedNote,
    ParsedTimingPointNotes,
    parse_timing_point_notes,
)
from maichart.timing import RawTimingPoint, TICKS_PER_BEAT

NOTE_TYPES = ("tap", "hold", "slide", "touch", "touch_hold")
PEAK_WINDOW_SEC = 1.0
DENSITY_REFERENCE_NPS = 16.0
PEAK_REFERENCE_NPS = 32.0
JACK_WINDOW_SEC = 0.5


@dataclass(slots=True)
class EventGenerationCondition(GenerationCondition):
    """A GenerationCondition with provenance, not extra model coordinates."""

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = GenerationCondition.to_dict(self)
        result["metadata"] = dict(self.metadata)
        return result


@dataclass(slots=True)
class EventNoteParse:
    """Lossless parser output. Any fatal error makes controls unavailable."""

    points: list[ParsedTimingPointNotes] = field(default_factory=list)
    unsupported: list[dict[str, Any]] = field(default_factory=list)
    fatal_errors: list[dict[str, Any]] = field(default_factory=list)
    # Non-fatal observations, e.g. an accepted negative delay: the chart is
    # usable, but the caller should know the clock starts before the audio.
    warnings: list[dict[str, Any]] = field(default_factory=list)

    @property
    def notes(self) -> list[ParsedNote]:
        return [note for point in self.points for note in point.notes]

    @property
    def syntax_valid(self) -> bool:
        return not self.fatal_errors


class EventNoteError(ValueError):
    """Raised when a condition would otherwise hide invalid event syntax."""

    def __init__(self, parsed: EventNoteParse):
        self.unsupported = parsed.unsupported
        self.fatal_errors = parsed.fatal_errors
        super().__init__("Invalid event chart: " + "; ".join(
            f"{issue['code']}: {issue['message']}" for issue in parsed.fatal_errors
        ))


def _value(item: Any, key: str, default: Any = None) -> Any:
    return item.get(key, default) if isinstance(item, Mapping) else getattr(item, key, default)


def _fraction(value: Any) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(str(value))


class _TempoMap:
    def __init__(self, changes: Iterable[Any], first: float = 0.0):
        self.first = float(first)
        if not math.isfinite(self.first):
            raise ValueError("first must be finite.")
        entries = []
        for item in changes:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                beat, bpm = item
            else:
                beat, bpm = _value(item, "beat"), _value(item, "bpm")
            beat, bpm = _fraction(beat), float(bpm)
            if beat < 0 or not math.isfinite(bpm) or bpm <= 0:
                raise ValueError("Tempo beats must be nonnegative and BPM finite and positive.")
            entries.append((beat, bpm))
        entries.sort()
        if not entries or entries[0][0] != 0:
            raise ValueError("tempo_changes must explicitly include a BPM at beat 0.")
        if len({beat for beat, _ in entries}) != len(entries):
            raise ValueError("Duplicate tempo-change beats are ambiguous.")
        self.beats = [beat for beat, _ in entries]
        self.bpms = [bpm for _, bpm in entries]
        self.seconds = [self.first]
        for index in range(1, len(entries)):
            self.seconds.append(self.seconds[-1] + float(
                self.beats[index] - self.beats[index - 1]
            ) * 60.0 / self.bpms[index - 1])

    def bpm(self, beat: Fraction) -> float:
        return self.bpms[bisect_right(self.beats, beat) - 1]

    def time(self, beat: Fraction) -> float:
        index = bisect_right(self.beats, beat) - 1
        return self.seconds[index] + float(beat - self.beats[index]) * 60.0 / self.bpms[index]


def parse_event_notes(
    points: Iterable[Any], tempo_changes: Iterable[Any], first: float = 0.0,
) -> EventNoteParse:
    """Parse complete raw tokens without quantization or silent repair.

    Returns parsed timing points plus explicit ``unsupported`` and
    ``fatal_errors`` lists. Slash chords, compact chords, and star branches
    are delegated intact to the existing note parser. ``tick=0`` is a parser
    placeholder, never a quantized event coordinate. Duplicates are checked
    across *all* input points at the same exact beat.
    """
    result = EventNoteParse()
    try:
        tempo = _TempoMap(tempo_changes, first)
    except (ValueError, TypeError, OverflowError) as exc:
        result.fatal_errors.append({"code": "invalid_tempo_map", "message": str(exc)})
        return result
    heads: dict[Fraction, Counter[str]] = defaultdict(Counter)
    for index, event in enumerate(points):
        try:
            beat = _fraction(_value(event, "beat"))
            text = _value(event, "note_text")
            if beat < 0 or not isinstance(text, str):
                raise ValueError("Event beat must be nonnegative and note_text must be a string.")
        except (ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
            result.fatal_errors.append({
                "code": "invalid_event", "point_index": index, "message": str(exc),
            })
            continue
        raw_point = RawTimingPoint(
            index=index, raw=text, note_text=text, beat=beat, tick=0,
            time_sec=tempo.time(beat), division=4, bpm=tempo.bpm(beat),
        )
        try:
            parsed = parse_timing_point_notes(raw_point)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
            issue = {"code": "note_parse_error", "point_index": index,
                     "beat": str(beat), "raw": text, "message": str(exc)}
            result.unsupported.append(dict(issue))
            result.fatal_errors.append(issue)
            continue
        result.points.append(parsed)
        for unknown in parsed.unknown_tokens:
            issue = {"code": "unsupported_token", "point_index": index,
                     "beat": str(beat), "raw": unknown.raw, "message": unknown.reason}
            result.unsupported.append(dict(issue))
            result.fatal_errors.append(issue)
            # The raw parser rejects compact duplicates before emitting heads.
            compact = BASIC_COMPACT_CHORD_RE.fullmatch("".join(unknown.raw.split()))
            if compact:
                for button, count in Counter(compact.group("positions")).items():
                    if count > 1:
                        result.fatal_errors.append({
                            "code": "duplicate_button_head", "beat": str(beat),
                            "button": int(button), "count": count,
                            "message": f"Repeated button {button} in compact chord {unknown.raw!r}.",
                        })
        for note in parsed.notes:
            # Touch notes carry a string position ("A4", "B7", "C"); taps and
            # slide heads carry 1..8.  Both must be counted, otherwise repeated
            # touch buttons (e.g. "A4/A4") never register as duplicates.
            heads[beat][str(note.position)] += 1
            durations = [segment.get("duration") for segment in note.slide_segments]
            if not durations and note.duration is not None:
                durations = [note.duration]
            for duration in durations:
                if duration is None:
                    continue
                values = list(duration.values or [])
                # `[delay##duration]`: a NEGATIVE DELAY is legitimate.  A chart
                # needs blank space before its body, and when the audio has no
                # lead-in the charter writes a negative delay so the chart moves
                # earlier as a whole -- equivalent to prepending silent audio
                # (see EventChart.time_offset / audio_segment_tensor).  Only the
                # duration part has to be positive.
                delay = values[0] if duration.kind == "timing_pair" and values else None
                duration_values = values[1:] if duration.kind == "timing_pair" else values
                if delay is not None and delay < 0:
                    result.warnings.append({
                        "code": "negative_delay", "point_index": index,
                        "beat": str(beat), "raw": note.raw, "delay": delay,
                        "message": "Negative delay accepted: the chart starts before the audio; "
                                   "the missing region is treated as silent audio.",
                    })
                # Two idioms are legal in real maimai charts but used to reject the
                # whole chart here:
                #   * `[1:0]` (zero grid count) on a hold -- a placeholder, 158 uses
                #     in this corpus, essentially all on holds and touch holds;
                #   * a negative duration expression, which is how a charter shifts
                #     a note (or the whole chart, when it leads) earlier in time.
                # Neither changes where the note's head sits, which is all the
                # training token stream encodes, so they are reported instead of
                # fatal.  Only non-finite values still break the chart.
                if duration.beats is not None and math.isfinite(float(duration.beats)) and duration.beats <= 0:
                    result.warnings.append({
                        "code": "zero_or_negative_grid_duration", "point_index": index,
                        "beat": str(beat), "raw": note.raw, "beats": str(duration.beats),
                        "message": "Zero/negative grid duration accepted as a placeholder or time shift.",
                    })
                if (duration.seconds is not None and math.isfinite(duration.seconds)
                        and duration.seconds <= 0):
                    result.warnings.append({
                        "code": "zero_or_negative_seconds_duration", "point_index": index,
                        "beat": str(beat), "raw": note.raw, "seconds": duration.seconds,
                        "message": "Zero/negative seconds duration accepted as a time shift.",
                    })
                if (duration.beats is not None and not math.isfinite(float(duration.beats))
                        or duration.seconds is not None and not math.isfinite(duration.seconds)
                        or any(not math.isfinite(value) for value in duration_values)):
                    result.fatal_errors.append({
                        "code": "invalid_duration", "point_index": index,
                        "beat": str(beat), "raw": note.raw,
                        "message": "Note duration must be finite.",
                    })
            # Grid durations cross BPM boundaries; explicit seconds do not.
            for segment in note.slide_segments:
                duration = segment.get("duration")
                if duration is not None and duration.kind == "grid_fraction":
                    segment["duration_sec"] = tempo.time(beat + duration.beats) - tempo.time(beat)
            if note.slide_segments:
                note.duration_sec = max(segment["duration_sec"] for segment in note.slide_segments)
            elif note.duration is not None and note.duration.kind == "grid_fraction":
                note.duration_sec = tempo.time(beat + note.duration.beats) - tempo.time(beat)
    for beat, buttons in sorted(heads.items()):
        for button, count in sorted(buttons.items()):
            if count > 1:
                label = f"Button {button}" if button.isdigit() else f"Touch {button}"
                result.fatal_errors.append({
                    "code": "duplicate_button_head", "beat": str(beat),
                    "button": button, "count": count,
                    "message": f"{label} has {count} heads at beat {beat}.",
                })
    return result


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def _entropy(counts: Iterable[int], categories: int) -> float:
    values = list(counts)
    total = sum(values)
    return (-sum((count / total) * math.log2(count / total) for count in values if count)
            / math.log2(categories)) if total else 0.0


def _peak_nps(groups: list[tuple[Fraction, list[ParsedNote]]]) -> float:
    """Peak occupied-event rate, so one chord cannot become a fake spike."""
    times = sorted(float(notes[0].time_sec) for _, notes in groups if notes)
    left, maximum = 0, 0
    for right, time in enumerate(times):
        # Half-open one-second windows, so regularly spaced boundary notes
        # are not counted twice. Chords contribute all of their heads.
        while time - times[left] >= PEAK_WINDOW_SEC - 1e-9:
            left += 1
        maximum = max(maximum, right - left + 1)
    return maximum / PEAK_WINDOW_SEC


def _signature(note: ParsedNote) -> tuple[Any, ...]:
    return (note.note_type, str(note.position), note.is_break,
            bool(note.modifiers.get("ex")), bool(note.modifiers.get("firework")),
            note.duration.raw if note.duration else "", tuple(
                (segment.get("path_text", segment.get("pattern")),
                 tuple(segment.get("path_args", [])),
                 segment["duration"].raw if segment.get("duration") else "")
                for segment in note.slide_segments))


def _motif_stats(groups: list[tuple[Fraction, list[ParsedNote]]]) -> dict[str, Any]:
    bars: dict[int, list[tuple[Any, ...]]] = defaultdict(list)
    for beat, notes in groups:
        bar = int(beat // 4)
        signatures = tuple(sorted(_signature(note) for note in notes))
        bars[bar].append((str(beat - 4 * bar), signatures))
    # Constant single-position patterns must not masquerade as motif reuse.
    motifs = [tuple(events) for events in bars.values()
              if len(events) >= 2 and len({event[1] for event in events}) >= 2]
    counts = Counter(motifs)
    reused = sum(count - 1 for count in counts.values())
    return {"window_beats": 4, "eligible_windows": len(motifs),
            "unique_motifs": len(counts), "repeated_windows": reused,
            "nonconstant_only": True, "reuse_ratio": reused / len(motifs) if motifs else 0.0}


def _statistics(parsed: EventNoteParse, duration: float) -> dict[str, Any]:
    notes = sorted(parsed.notes, key=lambda note: note.beat)
    grouped: dict[Fraction, list[ParsedNote]] = defaultdict(list)
    for note in notes:
        grouped[note.beat].append(note)
    groups = sorted(grouped.items())
    types = Counter(note.note_type for note in notes)
    buttons = Counter(str(note.position) for note in notes if isinstance(note.position, int))
    touches = Counter(str(note.position) for note in notes if not isinstance(note.position, int))
    signatures = [tuple(sorted(_signature(note) for note in group)) for _, group in groups]
    run, max_run, previous = 0, 0, None
    for signature in signatures:
        run = run + 1 if signature == previous else 1
        max_run, previous = max(max_run, run), signature
    chords = sum(len(group) >= 2 for _, group in groups)
    jack_pairs, comparable, directions = 0, 0, []
    for (_, before), (_, after) in zip(groups, groups[1:]):
        old = {note.position for note in before if isinstance(note.position, int)}
        new = {note.position for note in after if isinstance(note.position, int)}
        if old and new:
            comparable += len(new)
            gap = float(after[0].time_sec) - float(before[0].time_sec)
            if gap <= JACK_WINDOW_SEC + 1e-9:
                jack_pairs += len(old & new)
        if len(old) == len(new) == 1:
            delta = (next(iter(new)) - next(iter(old))) % 8
            if delta not in (0, 4):
                directions.append(1 if delta < 4 else -1)
    slides = [note for note in notes if note.note_type == "slide"]
    paths, durations, tricky, path_steps = Counter(), [], 0, []
    for note in slides:
        complex_note = len(note.slide_segments) > 1
        for segment in note.slide_segments:
            paths[str(segment.get("trajectory", {}).get("raw", note.raw))] += 1
            seconds = float(segment.get("duration_sec") or 0.0)
            durations.append(seconds)
            parts = segment.get("path_parts") or [segment]
            start = note.position
            distance = 0
            for part in parts:
                end = part.get("end_position")
                if isinstance(start, int) and isinstance(end, int):
                    delta = (end - start) % 8
                    distance += min(delta, 8 - delta)
                start = end
            path_steps.append(distance)
            complex_note |= (len(parts) > 1 or seconds < 0.25 or
                             str(segment.get("pattern")) in {"pp", "qq", "s", "z", "V", "w"})
        tricky += int(complex_note)
    slide_sequences = sum(0 < after.beat - before.beat <= 2
                          for before, after in zip(slides, slides[1:]))
    offgrid = [note for note in notes if (note.beat * TICKS_PER_BEAT).denominator != 1]
    return {
        "counts": {"events": len(parsed.points), "occupied_events": len(groups),
                   "notes": len(notes), "button_heads": sum(buttons.values()),
                   "touch_heads": sum(touches.values()), "break_notes": sum(note.is_break for note in notes),
                   "chords": chords, "slide_branches": sum(paths.values())},
        "types": {name: types[name] for name in NOTE_TYPES},
        "type_ratios": {name: types[name] / len(notes) if notes else 0.0 for name in NOTE_TYPES},
        "buttons": {str(index): buttons[str(index)] for index in range(1, 9)},
        "touch_positions": dict(sorted(touches.items())),
        "entropy": {"buttons": _entropy(buttons.values(), 8), "types": _entropy(types.values(), 5)},
        "duration_sec": duration, "note_density_nps": len(notes) / duration if duration > 0 else 0.0,
        "peak_nps": _peak_nps(groups), "peak_window_sec": PEAK_WINDOW_SEC,
        "chord_ratio": chords / len(groups) if groups else 0.0,
        "jack": {"repeated_heads": jack_pairs, "comparable_heads": comparable,
                 "window_sec": JACK_WINDOW_SEC, "ratio": jack_pairs / comparable if comparable else 0.0},
        "direction": {"transitions": len(directions), "clockwise": directions.count(1),
                      "counterclockwise": directions.count(-1),
                      "consistency": abs(sum(directions)) / len(directions) if directions else 0.0},
        "motif": _motif_stats(groups), "max_repeated_run": max_run,
        "offgrid_fractions": {"tick_reference": TICKS_PER_BEAT,
                              "note_count": len(offgrid),
                              "ratio": len(offgrid) / len(notes) if notes else 0.0,
                              "beats": sorted({str(note.beat) for note in offgrid}),
                              "denominators": dict(sorted(Counter(str(note.beat.denominator) for note in offgrid).items()))},
        "slides": {"paths": dict(sorted(paths.items())), "branch_durations_sec": durations,
                   "mean_duration_sec": sum(durations) / len(durations) if durations else 0.0,
                   "mean_path_steps": sum(path_steps) / len(path_steps) if path_steps else 0.0,
                   "tricky_ratio": tricky / len(slides) if slides else 0.0,
                   "sequence_ratio": slide_sequences / (len(slides) - 1) if len(slides) > 1 else 0.0},
    }


def _audio_proxy(parsed: EventNoteParse, audio: Any, *, first: float | None = None) -> tuple[float, dict[str, Any]]:
    metadata = {"kind": "unavailable", "available": False, "true_melody_measurement": False,
                "reason": "No audio supplied; melody_following is explicitly 0."}
    if audio is None:
        return 0.0, metadata
    # Accept precomputed AudioFeatureSet or JSON features, not raw waveforms.
    offset = float(_value(audio, "first", 0.0) if first is None else first)
    onset_times = []
    for onset in _value(audio, "onsets", []) or []:
        value = onset if isinstance(onset, (int, float)) else _value(onset, "time_sec")
        try:
            time = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(time):
            onset_times.append(time)
    if not onset_times:
        # EventAudio stores normalized onset strength in its feature matrix
        # rather than as a separate onset list. Keep the dependency structural
        # so this sidecar does not import the optional audio implementation.
        times_data = _value(audio, "times", [])
        features = _value(audio, "features", [])
        try:
            feature_times = list(times_data)
            rows = list(features)
            onset_index = 80 + 12
            strengths = [float(row[onset_index]) for row in rows]
            if len(strengths) == len(feature_times) and strengths:
                finite = [value for value in strengths if math.isfinite(value)]
                if finite:
                    mean = sum(finite) / len(finite)
                    variance = sum((value - mean) ** 2 for value in finite) / len(finite)
                    threshold = mean + 0.5 * math.sqrt(variance)
                    onset_times = [float(time) for index, time in enumerate(feature_times)
                                   if math.isfinite(float(time)) and strengths[index] >= threshold]
                    metadata["feature_source"] = "EventAudio.features[onset]"
        except (TypeError, ValueError, IndexError, OverflowError):
            onset_times = []
    onset_times.sort()
    times = sorted({float(note.time_sec) + offset for note in parsed.notes})
    tolerance = 0.08
    matched = 0
    for time in times:
        index = bisect_right(onset_times, time)
        candidates = onset_times[max(0, index - 1):index + 1]
        matched += int(any(abs(time - candidate) <= tolerance for candidate in candidates))
    if not onset_times:
        metadata["reason"] = "Audio features contain no usable onsets; melody_following is 0."
        return 0.0, metadata
    metadata.update(kind="onset_alignment_proxy", available=True, reason=
                    "Coarse rhythm/onset alignment only; not pitch or true melody following.",
                    tolerance_sec=tolerance, first=offset, matched_events=matched,
                    occupied_events=len(times))
    return matched / len(times) if times else 0.0, metadata


def _condition(stats: dict[str, Any], *, level: float, melody: float = 0.0,
               melody_metadata: dict[str, Any] | None = None) -> EventGenerationCondition:
    ratios, slides = stats["type_ratios"], stats["slides"]
    return EventGenerationCondition(
        level=level,
        radar={"note_density": _clip(stats["note_density_nps"] / DENSITY_REFERENCE_NPS),
               "peak": _clip(stats["peak_nps"] / PEAK_REFERENCE_NPS),
               "sweep": _clip(ratios["slide"] * slides["mean_path_steps"] / 4),
               "slide_tricky": ratios["slide"] * slides["tricky_ratio"],
               "slide_sequence": ratios["slide"] * slides["sequence_ratio"],
               "jack": stats["jack"]["ratio"], "chord": stats["chord_ratio"]},
        style={"tap_bias": ratios["tap"], "slide_bias": ratios["slide"],
               "hold_bias": ratios["hold"] + ratios["touch_hold"],
               "touch_bias": ratios["touch"] + ratios["touch_hold"],
               "chord_bias": stats["chord_ratio"], "melody_following": melody,
               "pattern_reuse": stats["motif"]["reuse_ratio"],
               "directionality": stats["direction"]["consistency"]},
        metadata={"source": "exact_events", "computed_in_memory": True,
                  "persisted_to_training_zip": False, "statistics": stats,
                  "normalization": {"note_density_reference_nps": DENSITY_REFERENCE_NPS,
                                    "peak_reference_nps": PEAK_REFERENCE_NPS},
                  "melody_following": melody_metadata or _audio_proxy(EventNoteParse(), None)[1],
                  "duration_semantics": "Grid durations integrate the tempo map from the head; explicit seconds stay seconds. Slide wait/gameplay semantics are not verified."},
    )


def condition_from_events(
    points: Iterable[Any], tempo_changes: Iterable[Any], *, level: float,
    duration_sec: float, audio: Any = None, audio_first: float | None = None,
) -> GenerationCondition:
    """Compute training controls dynamically from exact events.

    ``duration_sec`` is the chart/song interval length, not last-note time.
    ``audio`` optionally accepts precomputed onset features; its optional
    ``first`` field aligns chart-relative beats with absolute audio seconds.
    ``audio_first`` explicitly overrides that field when the feature object
    has no chart-clock metadata (as with cached ``EventAudio``).
    The returned GenerationCondition subclass carries JSON-safe ``metadata``.
    Invalid tokens, duplicate heads, or durations raise EventNoteError.
    """
    duration = float(duration_sec)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration_sec must be finite and positive.")
    parsed = parse_event_notes(points, tempo_changes)
    if not parsed.syntax_valid:
        raise EventNoteError(parsed)
    melody, metadata = _audio_proxy(parsed, audio, first=audio_first)
    return _condition(_statistics(parsed, duration), level=level,
                      melody=melody, melody_metadata=metadata)


def assess_event_chart(
    points: Iterable[Any], tempo_changes: Iterable[Any], *, first: float = 0.0,
    duration_sec: float | None = None,
    target_condition: GenerationCondition | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return JSON diagnostics, never a claim of verified playability.

    Invalid syntax is explicit and actual controls are withheld, not derived
    from a silently truncated chart. Only empty charts or repeated all-tap
    single-position charts are labeled degenerate. Other charts need review.
    """
    changes = list(tempo_changes)
    parsed = parse_event_notes(points, changes, first)
    duration_source = "provided"
    if duration_sec is None:
        duration_source = "inferred_extent_with_one_beat_after_last_head"
        duration = 0.0
        explicit_end = bool(parsed.points and parsed.points[-1].note_text == "")
        if parsed.points:
            tempo = _TempoMap(changes, first)
            duration = max(float(point.time_sec) - first for point in parsed.points)
            if not explicit_end:
                for note in parsed.notes:
                    duration = max(duration, float(note.time_sec) - first + (
                        note.duration_sec if note.duration_sec is not None else 60 / tempo.bpm(note.beat)))
            else:
                duration_source = "explicit_empty_end_point"
    else:
        duration = float(duration_sec)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration_sec must be finite and positive.")
    stats = _statistics(parsed, duration)
    reasons = []
    count = stats["counts"]["notes"]
    if not count:
        reasons.append("empty_chart")
    elif count >= 2 and stats["types"]["tap"] == count and len({
        note.position for note in parsed.notes
    }) == 1:
        reasons.append("all_taps_single_position")
    requested = (target_condition if isinstance(target_condition, GenerationCondition)
                 else GenerationCondition.from_dict(dict(target_condition)) if target_condition is not None else None)
    actual = _condition(stats, level=requested.level if requested else 13.0) if parsed.syntax_valid else None
    deltas = None
    if actual is not None and requested is not None:
        deltas = {section: {key: getattr(actual, section)[key] - getattr(requested, section)[key]
                            for key in getattr(actual, section)} for section in ("radar", "style")}
        # This diagnostic has neither audio nor a difficulty estimator.
        deltas["style"]["melody_following"] = None
    return {"schema": "maichart-event-quality-v1", "syntax_valid": parsed.syntax_valid,
            "quality_status": "degenerate" if parsed.syntax_valid and reasons else "needs_review",
            "playable_verified": False, "degenerate_reasons": reasons,
            "unsupported": parsed.unsupported, "fatal_errors": parsed.fatal_errors,
            "statistics_partial": not parsed.syntax_valid,
            **stats, "duration_source": duration_source,
            "control": {"actual": actual.to_dict() if actual else None,
                        "requested": requested.to_dict() if requested else None,
                        "actual_minus_requested": deltas, "level_estimated": False,
                        "unmeasured": ["level", "melody_following"]},
            "limitations": ["Syntax and descriptive statistics are not gameplay verification.",
                            "Off-grid rational events are retained, not rounded or rejected.",
                            "No audio: melody_following is 0, not a true melody measurement."],
            }
