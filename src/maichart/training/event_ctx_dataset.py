"""Dataset for the cross-window family: windows that know their neighbourhood.

``EventCtxDataset`` is ``EventDataset`` plus three changes:

* every window carries a **lead-in** block of the notes immediately before it,
  delimited by ``<ctx>`` / ``<ctx_end>`` and excluded from the loss;
* the audio slice is extended by ``audio_margin_beats`` on both sides, so the
  encoder sees the music just before and just after the window;
* the window stride can be shortened (``overlap_beats``), which turns the
  windows into a rolling, overlapping sweep of the song.

The event_v1 dataset is untouched: this class only adds a vocabulary extension
and its own window list, and it subclasses ``EventDataset`` so manifest
handling, chart filtering, the audio cache and the condition vector are shared.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from fractions import Fraction
from typing import Any, Sequence

import torch

from maichart.event_timing import seconds_at_beat
from maichart.generation.event_audio import (
    _normalize_tempo_map,
    _time_to_beat,
    audio_segment_tensor,
    segment_frame_times,
)
from maichart.generation.event_ctx import (
    CTX_SPECIALS,
    DEFAULT_AUDIO_MARGIN_BEATS,
    DEFAULT_CONTEXT_BEATS,
    DEFAULT_CONTEXT_DROPOUT,
    DEFAULT_CONTEXT_TOKENS,
    TARGET_SCHEMA,
    context_start_beat,
    lead_in_events,
    loss_mask_after,
    prefix_token_ids,
)
from maichart.generation.event_model import EventSequence
from maichart.generation.event_quality import parse_event_notes
from maichart.training.event_dataset import EventDataset
from maichart.training.event_input_policy import CONTEXT_BUDGET_POLICIES

OVERSAMPLE_POLICY = "independent_note_types_touch_priority_fine_rhythm_v2"
# A window whose shared fine-note ratio reaches this
# share is copied fine_rhythm_oversample times.  It targets the passages where the
# model currently falls back to quarter/eighth notes: 12th-16th-24th alternations,
# dotted sixteenths and dense drum/piano figuration.
FINE_RHYTHM_RATIO_THRESHOLD = 0.25
FINE_RHYTHM_STATS_VERSION = "fine_note_ratio_quarter_plus_1_128_v1"


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer at least 1.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer at least 1.") from exc
    if not math.isfinite(number) or not number.is_integer() or number < 1:
        raise ValueError(f"{name} must be an integer at least 1.")
    return int(number)


class EventCtxDataset(EventDataset):
    EXTRA_SPECIALS = CTX_SPECIALS

    def __init__(
        self,
        manifest_path,
        *,
        context_beats: int = DEFAULT_CONTEXT_BEATS,
        audio_margin_beats: int = DEFAULT_AUDIO_MARGIN_BEATS,
        context_tokens: int = DEFAULT_CONTEXT_TOKENS,
        context_budget_policy: str = "bounded_v2",
        overlap_beats: int = 0,
        context_dropout: float = DEFAULT_CONTEXT_DROPOUT,
        touch_oversample: int = 1,
        **kwargs: Any,
    ) -> None:
        if context_beats < 0 or audio_margin_beats < 0 or overlap_beats < 0:
            raise ValueError("context_beats, audio_margin_beats and overlap_beats must not be negative.")
        if context_tokens <= 0:
            raise ValueError("context_tokens must be positive (it is the lead-in budget).")
        if context_budget_policy not in CONTEXT_BUDGET_POLICIES:
            raise ValueError(f"Invalid context_budget_policy: {context_budget_policy!r}")
        self.context_budget_policy = context_budget_policy
        if not 0.0 <= float(context_dropout) <= 1.0:
            raise ValueError("context_dropout must be within [0, 1].")
        # The sampling knobs are family-level settings consumed here; they must
        # not reach ``EventDataset.__init__``, which has no notion of them.
        self.slide_oversample = int(kwargs.pop("slide_oversample", 1))
        if self.slide_oversample < 1:
            raise ValueError("slide_oversample must be at least 1.")
        if int(touch_oversample) < 1:
            raise ValueError("touch_oversample must be at least 1 (1 disables oversampling).")
        self.touch_oversample = int(touch_oversample)
        self.fine_rhythm_oversample = _positive_integer(
            kwargs.pop("fine_rhythm_oversample", 1), "fine_rhythm_oversample")
        try:
            self.fine_rhythm_ratio_threshold = float(
                kwargs.pop("fine_rhythm_ratio_threshold", FINE_RHYTHM_RATIO_THRESHOLD))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("fine_rhythm_ratio_threshold must be finite and in (0, 1].") from exc
        if not math.isfinite(self.fine_rhythm_ratio_threshold) or not (
            0.0 < self.fine_rhythm_ratio_threshold <= 1.0
        ):
            raise ValueError("fine_rhythm_ratio_threshold must be finite and in (0, 1].")
        self.sampling_windows = Counter()
        self.sampling_segments = Counter()
        self.context_beats = int(context_beats)
        self.audio_margin_beats = int(audio_margin_beats)
        self.context_tokens = int(context_tokens)
        self.overlap_beats = int(overlap_beats)
        # PerceiverS reports that a fixed, long context makes the model produce
        # repetitive segments; dropping the lead-in at random keeps it from
        # relying on copying.  The block geometry (and therefore the window's
        # positions) does not change, so the model still knows where it sits.
        self.context_dropout = float(context_dropout)
        self._context_rng = random.Random(int(kwargs.get("condition_seed", 0)) + 7919)
        # Per-song normalized tempo map; recomputing it for every window was pure
        # repetition (76k windows over ~2k songs).
        self._tempo_cache: dict[int, Any] = {}
        super().__init__(manifest_path, **kwargs)
        # `super()` built the plain event_v1 window list; replace it with the
        # context-aware one.  Building it twice is a few seconds and keeps the
        # parent untouched.
        self.segments = []
        self.segment_quality_weights = []
        self.segment_quality_token_weights = []
        self.segment_quality_loss_starts = []
        self.skipped = [item for item in self.skipped if item.get("stage") != "segment"]
        self._build_context_segments()
        if self.quality_annotations is not None and self.segments:
            self._filter_quality_excluded_segments()
        self._print_context_summary()
        if not self.segments:
            raise ValueError(
                f"No usable context segments in {self.manifest_path}. "
                f"See dataset audit reasons: {self.skipped[:5]}"
            )

    @property
    def stride_beats(self) -> int:
        return max(1, self.segment_beats - self.overlap_beats)

    def vocabulary_segment_beats(self) -> int:
        """The composite spans the lead-in block plus the window."""

        return self.segment_beats + self.context_beats

    def _build_context_segments(self) -> None:
        """Windows as ``(sample, start, end, tokens, lead_in_tokens, prefix_len, offset)``.

        Position origin (the "temporal confusion" fix): every token of the
        composite -- lead-in *and* window -- is positioned relative to the start
        of the lead-in block, so positions increase monotonically across the
        whole sequence instead of restarting at the window boundary.  Two
        origins in one sequence is exactly the failure Transformer-XL had to
        solve with relative positional encodings.

        The offset is how many beats the block starts before the window.  It
        stays the same even when the lead-in is empty or truncated, so the model
        can always tell where the window sits in the block, and generation can
        reproduce the layout exactly.
        """

        for index, ref in enumerate(self.samples):
            # Parse once per chart, not once per duplicate/window. A note's
            # actual type determines sampling; letters in raw Simai text or
            # the terminating E must not masquerade as touch notes.
            note_types = defaultdict(set)
            for note in parse_event_notes(ref.chart.points, ref.chart.tempo_changes).notes:
                note_types[note.beat].add(note.note_type)
            count = max(1, math.ceil(ref.chart.end_beat / self.stride_beats))
            for step in range(count):
                start = Fraction(step * self.stride_beats)
                if start >= ref.chart.end_beat and step:
                    break
                end = min(start + self.segment_beats, ref.chart.end_beat)
                block_start = context_start_beat(start, self.context_beats)
                offset = start - block_start
                window_points = [point for point in ref.chart.points if start <= point.beat < end]
                window = EventSequence(
                    beats=[point.beat - block_start for point in window_points],
                    note_texts=[point.note_text for point in window_points],
                )
                events = lead_in_events(ref.chart.points, start, self.context_beats)
                if self.quality_annotations is not None and events:
                    events = [event for event in events
                              if self._quality_event_weight(index, block_start + event[0]) > 0.0]
                if events and self.context_dropout > 0 and self._context_rng.random() < self.context_dropout:
                    events = []  # training with no lead-in: generation must handle it too
                try:
                    window_tokens = self.vocabulary.encode(window)
                    prefix = prefix_token_ids(self.vocabulary, events, self.context_tokens)
                except ValueError as exc:
                    self.skipped.append({
                        "song_id": ref.song_id, "stage": "segment", "start_beat": str(start),
                        "reason": "vocabulary_mismatch", "message": str(exc),
                    })
                    continue
                # New runs use the same fixed history cap as generation. Never
                # choose a larger/smaller context after inspecting future target
                # length. Over-budget composites are explicitly discarded.
                # Exact resume of an older experiment keeps its recorded policy.
                if self.context_budget_policy == "legacy_dynamic_v1":
                    budget = int(self.max_tokens) - len(window_tokens) - 2
                    if budget != self.context_tokens:
                        try:
                            prefix = prefix_token_ids(self.vocabulary, events, budget)
                        except ValueError:
                            prefix = prefix_token_ids(self.vocabulary, events, self.context_tokens)
                total = len(window_tokens) + len(prefix)  # composite length minus one
                if total > self.max_tokens:
                    self.skipped.append({
                        "song_id": ref.song_id, "stage": "segment", "start_beat": str(start),
                        "reason": "segment_token_limit",
                        "window_tokens": len(window_tokens), "prefix_tokens": len(prefix),
                        "context_budget_policy": self.context_budget_policy,
                        "message": (f"Window {len(window_tokens)} + context {len(prefix)} = {total} tokens "
                                    f"exceeds max_tokens={self.max_tokens}; increase max_tokens, or reduce "
                                    f"segment_beats/context_beats/context_tokens."),
                    })
                    continue
                kinds = {kind for point in window_points for kind in note_types[point.beat]}
                has_slide = "slide" in kinds
                has_touch = bool(kinds & {"touch", "touch_hold"})
                category = ("mixed" if has_slide and has_touch else "touch_only" if has_touch
                            else "slide_only" if has_slide else "plain")
                # Touch takes priority in mixed windows. With touch=1, raising
                # slide must not quietly oversample touch via those windows.
                # The multipliers are never added or multiplied together.
                copies = self.touch_oversample if has_touch else self.slide_oversample if has_slide else 1
                fine_ratio = self._fine_note_ratio(window_points, start, end)
                fine_multiplier = (self.fine_rhythm_oversample
                                   if fine_ratio >= self.fine_rhythm_ratio_threshold else 1)
                copies *= fine_multiplier
                if fine_ratio >= self.fine_rhythm_ratio_threshold:
                    category = category + "+fine"
                self.sampling_windows[category] += 1
                self.sampling_segments[category] += copies
                for _ in range(copies):
                    self.segments.append((index, start, end, window_tokens, prefix, len(prefix), offset))
                    quality_weight, token_weights = self._quality_weights_for_window(
                        index, start, end, window_points,
                        target_length=len(prefix) + len(window_tokens) - 1,
                        prefix_len=len(prefix),
                    )
                    self.segment_quality_weights.append(quality_weight)
                    self.segment_quality_token_weights.append(token_weights)
                    self.segment_quality_loss_starts.append(len(prefix))

    def _fine_note_ratio(self, window_points, start: Fraction, end: Fraction) -> float:
        """Share of adjacent onsets inside the shared fine-rhythm boundary.

        Delegates to ``music_plan_targets.fine_note_ratio`` so the windows that
        get oversampled and the value the planner is asked to predict are
        computed by one implementation; two copies would eventually disagree and
        the mismatch would look like a modelling failure.
        """

        from maichart.training.music_plan_targets import fine_note_ratio

        return fine_note_ratio(point.beat for point in window_points)

    def _print_context_summary(self) -> None:
        with_lead_in = sum(1 for item in self.segments if item[4] and item[5] > 2)
        lead_in_tokens = sum(item[5] for item in self.segments)
        average = lead_in_tokens / max(1, len(self.segments))
        maximum = max((item[5] for item in self.segments), default=0)
        over_budget = sum(item.get("reason") == "segment_token_limit" for item in self.skipped)
        print(
            f"event_ctx_dataset segments={len(self.segments)} songs={len(self.samples)} "
            f"context_beats={self.context_beats} audio_margin_beats={self.audio_margin_beats} "
            f"overlap_beats={self.overlap_beats} stride={self.stride_beats} "
            f"segments_with_lead_in={with_lead_in} avg_prefix_tokens={average:.1f} "
            f"max_prefix_tokens={maximum} context_tokens={self.context_tokens} "
            f"context_budget_policy={self.context_budget_policy} dropped_token_windows={over_budget} "
            f"slide_oversample={self.slide_oversample} touch_oversample={self.touch_oversample} "
            f"fine_rhythm_oversample={self.fine_rhythm_oversample} "
            f"fine_rhythm_stats_version={self.family_params().get('fine_rhythm_stats_version', 'disabled')} "
            "mixed_windows_use_touch_multiplier=true "
            f"oversample_policy={OVERSAMPLE_POLICY} windows={dict(self.sampling_windows)} "
            f"weighted_windows={dict(self.sampling_segments)}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.segments)

    def _window_audio(self, ref, start: Fraction, end: Fraction, audio):
        """Sample the window plus its audio margins."""

        margin = Fraction(self.audio_margin_beats)
        begin_beat = max(Fraction(0), start - margin)
        end_beat = end + margin
        # `audio_first` folds in a leading negative delay, so a chart that needs
        # blank space before its body maps onto silence in front of the audio
        # instead of being pushed onto the first frame.
        anchor = ref.chart.audio_first
        begin_sec = seconds_at_beat(begin_beat, ref.chart.tempo_changes, anchor)
        end_sec = max(begin_sec, seconds_at_beat(end_beat, ref.chart.tempo_changes, anchor))
        if audio.duration_sec:
            end_sec = min(end_sec, float(audio.duration_sec))
        # The normalized tempo map depends only on the song, and this runs once
        # per window (76k times per epoch), so it is cached per chart identity.
        key = id(ref.chart)
        tempo_entries = self._tempo_cache.get(key)
        if tempo_entries is None:
            tempo_entries = _normalize_tempo_map(ref.chart.tempo_changes,
                                                 ref.chart.tempo_changes[0].bpm)
            self._tempo_cache[key] = tempo_entries
        anchor_beat = _time_to_beat(begin_sec - anchor, tempo_entries)
        return audio_segment_tensor(
            audio, begin_sec, end_sec, song_duration=audio.duration_sec,
            bpm=ref.chart.tempo_changes[0].bpm, start_beat=anchor_beat, first=anchor,
            tempo_changes=ref.chart.tempo_changes,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_index, start, end, window_tokens, prefix, prefix_len, offset = self.segments[index]
        ref = self.samples[sample_index]
        audio = self._load_audio(ref)
        x = self._window_audio(ref, start, end, audio)
        # event_ctx intentionally includes audio margins for continuity. The
        # music-structure planner must pool only the target window, otherwise
        # the lead-in/tail changes the authored local workload target.
        from maichart.event_timing import seconds_at_beat
        margin = Fraction(self.audio_margin_beats)
        block_start = max(Fraction(0), start - margin)
        block_end = end + margin
        begin_sec = float(seconds_at_beat(block_start, ref.chart.tempo_changes, ref.chart.audio_first))
        finish_sec = max(begin_sec, float(seconds_at_beat(block_end, ref.chart.tempo_changes, ref.chart.audio_first)))
        core_begin = float(seconds_at_beat(start, ref.chart.tempo_changes, ref.chart.audio_first))
        core_end = float(seconds_at_beat(end, ref.chart.tempo_changes, ref.chart.audio_first))
        from maichart.generation.event_audio import FEATURE_RATE_HZ
        sample_times = begin_sec + torch.arange(x.size(0), dtype=torch.float32) / FEATURE_RATE_HZ
        plan_audio_mask = (sample_times < core_begin) | (sample_times >= core_end)
        if sample_index not in self.conditions:
            self._ensure_condition(sample_index, ref, audio)
        # Composite target: <bos> <ctx> lead-in <ctx_end> window <eos>
        ids = [self.vocabulary.bos_id] + list(prefix) + list(window_tokens[1:])
        tokens = torch.tensor(ids, dtype=torch.long)
        targets = tokens[1:]
        mask = torch.tensor(loss_mask_after(prefix_len, targets.numel()), dtype=torch.bool)
        condition, condition_null = self._condition_with_null_flag(sample_index)
        item = {
            "audio": x, "global_audio": self.global_audio[sample_index],
            "condition": self._window_condition(sample_index, ref, audio, start, condition),
            "condition_null": torch.tensor(condition_null, dtype=torch.bool),
            "plan_audio_mask": plan_audio_mask,
            "plan_target": self._quality_window_plan(sample_index, ref, start, end),
            "quality_sample_weight": torch.tensor(self.segment_quality_weights[index], dtype=torch.float32),
            "quality_plan_weight": torch.tensor(self._quality_plan_weight(
                sample_index, start, end, self.segment_quality_weights[index]), dtype=torch.float32),
            "quality_token_weights": self.quality_token_vector(index, targets.numel()),
            "decoder_input": tokens[:-1], "target": targets, "loss_mask": mask,
            "meta": {"song_id": ref.song_id, "start_beat": str(start), "end_beat": str(end),
                     "lead_in_tokens": prefix_len, "context_beats": self.context_beats,
                     "beat_offset_denominator": int(offset.denominator),
                     "beat_offset_numerator": int(offset.numerator)},
        }
        if self.onset_profile != "none":
            frame_end_sec = finish_sec
            if audio.duration_sec:
                frame_end_sec = min(frame_end_sec, float(audio.duration_sec))
            sample_times = segment_frame_times(begin_sec, frame_end_sec)
            if sample_times.size != x.size(0):
                raise RuntimeError("onset frame-time grid does not match audio segment frames")
            item["onset_target"], item["onset_weight"], item["onset_valid"] = self._onset_labels(
                sample_index, ref, sample_times, core_begin, core_end, float(audio.duration_sec))
        self._attach_action_state(item, sample_index, ref, start)
        return item

    def audit(self) -> dict[str, Any]:
        payload = super().audit()
        payload.update({
            "target_schema": TARGET_SCHEMA,
            "family": TARGET_SCHEMA,
            "onset_profile": self.onset_profile,
            "context_beats": self.context_beats,
            "audio_margin_beats": self.audio_margin_beats,
            "context_tokens": self.context_tokens,
            "context_budget_policy": self.context_budget_policy,
            "max_prefix_tokens": max((item[5] for item in self.segments), default=0),
            "overlap_beats": self.overlap_beats,
            "context_dropout": self.context_dropout,
            "stride_beats": self.stride_beats,
            "segments_with_lead_in": sum(1 for item in self.segments if item[5] > 2),
            "slide_oversample": self.slide_oversample,
            "touch_oversample": self.touch_oversample,
            "fine_rhythm_oversample": self.fine_rhythm_oversample,
            "fine_rhythm_ratio_threshold": self.fine_rhythm_ratio_threshold,
            "fine_rhythm_stats_version": (FINE_RHYTHM_STATS_VERSION
                                           if self.fine_rhythm_oversample > 1 else None),
            "oversample_policy": OVERSAMPLE_POLICY,
            "sampling_windows": dict(self.sampling_windows),
            "sampling_segments": dict(self.sampling_segments),
        })
        return payload

    def family_params(self) -> dict[str, Any]:
        """Values that must match for a resume to be meaningful."""

        params = {
            "context_beats": self.context_beats,
            "audio_margin_beats": self.audio_margin_beats,
            "context_tokens": self.context_tokens,
            "overlap_beats": self.overlap_beats,
            "context_dropout": self.context_dropout,
            # Part of the contract: a checkpoint resampled with a different
            # slide share is not the same distribution, and resuming across the
            # boundary would silently mix two training regimes.
            "slide_oversample": self.slide_oversample,
            "touch_oversample": self.touch_oversample,
            "fine_rhythm_oversample": self.fine_rhythm_oversample,
            "fine_rhythm_ratio_threshold": self.fine_rhythm_ratio_threshold,
            "oversample_policy": OVERSAMPLE_POLICY,
        }
        if self.fine_rhythm_oversample > 1:
            params["fine_rhythm_stats_version"] = FINE_RHYTHM_STATS_VERSION
        if self.onset_profile != "none":
            params["onset_profile"] = self.onset_profile
        return params
