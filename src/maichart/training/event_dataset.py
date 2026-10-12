"""Exact-event dataset independent of the legacy fixed-frame cache."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, OrderedDict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from maichart.event_timing import parse_event_chart, seconds_at_beat
from maichart.generation.event_audio import (
    GLOBAL_MEMORY_FRAMES,
    _normalize_tempo_map,
    _time_to_beat,
    audio_cache_dir,
    audio_segment_tensor,
    compress_event_audio,
    event_audio_cache_path,
    extract_event_audio,
    load_event_audio,
    segment_frame_times,
)
from maichart.generation.event_model import EventSequence, EventVocabulary, NOTE_CHARACTERS
from maichart.generation.event_quality import condition_from_events, parse_event_notes
from maichart.generation.music_plan import is_music_structure, validate_profile
from maichart.training.onset_targets import (
    build_onset_targets, onset_times, quality_onset_params, validate_onset_profile,
)
from maichart.training.chart_sanitization import (
    DEFAULT_CHART_FILTER_POLICY,
    sanitize_chart,
    validate_chart_filter_policy,
)


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


# Decoded features run ~3 MB per song (a 95-column float32 frame per 20 ms), so
# the resident cache is capped by bytes rather than by entry count.  4 GiB holds
# roughly 1300 songs, which covers the current corpus without approaching the
# machine's RAM ceiling.
DEFAULT_AUDIO_CACHE_BYTES = 4 * 1024 ** 3
LEGACY_CONDITION_TARGET_VERSION = "legacy_unshifted_v1"
CURRENT_CONDITION_TARGET_VERSION = "chart_audio_clock_v2"
CONDITION_TARGET_VERSIONS = {LEGACY_CONDITION_TARGET_VERSION, CURRENT_CONDITION_TARGET_VERSION}


@dataclass
class EventSample:
    song_id: str
    audio_path: Path
    chart: Any
    level: float
    source_path: Path


def numeric_level(value: Any) -> float:
    """Chart level as a number.

    ``level_raw`` is occasionally the literal string ``"None"`` (or empty, or a
    placeholder) when a chart has no level field.  That used to surface as
    "could not convert string to float: 'None'" buried in the dataset audit; it
    now raises a named error the caller maps to a clear skip reason.
    """

    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan", "-"}:
        raise ValueError(f"Missing chart level ({value!r}); cannot apply the level filter.")
    try:
        return float(text[:-1]) + 0.7 if text.endswith("+") else float(text)
    except ValueError as exc:
        raise ValueError(f"Chart level {value!r} is not numeric; cannot apply the level filter.") from exc


def inspect_chart(chart: Any) -> None:
    """Reject unsupported documents instead of dropping parts of a target."""
    if not chart.points:
        raise ValueError("Empty event chart.")
    for point in chart.points:
        compact = "".join(point.note_text.split())
        if any(char not in NOTE_CHARACTERS for char in compact):
            raise ValueError(f"Unsupported syntax at beat {point.beat}: {compact[:100]}")
    # The public quality helper also checks duplicate heads and unknown text.
    from maichart.generation.event_quality import assess_event_chart
    report = assess_event_chart(chart.points, chart.tempo_changes, first=chart.first)
    if not report.get("syntax_valid", False):
        errors = report.get("fatal_errors") or report.get("unsupported") or []
        if errors:
            summary = "; ".join(
                f"{item.get('code', 'invalid')}@{item.get('beat', '?')}: {item.get('message', '')}"
                for item in errors[:3]
            )
            suffix = " ..." if len(errors) > 3 else ""
            raise ValueError(f"Invalid event document: {summary}{suffix}")
        raise ValueError("Invalid event document: parser reported fatal syntax errors.")


class EventDataset(Dataset):
    # Family-specific control tokens appended to the vocabulary.  event_v1 adds
    # none, which keeps its alphabet (and every existing checkpoint) identical.
    EXTRA_SPECIALS: tuple[str, ...] = ()

    def __init__(
        self, manifest_path: str | Path, *, cache_dir: str | Path = "cache/events",        segment_beats: int = 16, vocabulary: EventVocabulary | None = None,
        max_tokens: int = 2048, sample_rate: int = 22050, hop_length: int = 441,
        eager_audio: bool = True, audio_cache_bytes: int | None = DEFAULT_AUDIO_CACHE_BYTES,
        global_frames: int = GLOBAL_MEMORY_FRAMES,
        condition_mask_prob: float = 0.5, condition_default: float = 0.5,
        condition_target_version: str = LEGACY_CONDITION_TARGET_VERSION,
        condition_seed: int = 0,
        slide_oversample: int = 1,
        min_difficulty_index: int = 4, max_difficulty_index: int = 6,
        min_level: float = 12.0, max_level: float = 15.0,
        log_audio_loads: bool = False,
        audio_cache_label: str = "dataset",
        quality_annotations: str | Path | None = None,
        planner_profile: str = "legacy_v1",
        onset_profile: str = "none",
        chart_filter_policy: str | None = None,
    ) -> None:
        if segment_beats <= 0 or max_tokens < 4:
            raise ValueError("segment_beats must be positive and max_tokens at least 4.")
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest_filter_policy = (self.manifest.get("preprocessing") or {}).get("chart_filter_policy")
        self.chart_filter_policy = validate_chart_filter_policy(
            chart_filter_policy if chart_filter_policy is not None else
            manifest_filter_policy or DEFAULT_CHART_FILTER_POLICY
        )
        self.chart_filter_audit: list[dict[str, Any]] = []
        validate_profile(planner_profile)
        self.planner_profile = planner_profile
        self.onset_profile = validate_onset_profile(onset_profile)
        self._onset_times_by_sample: dict[int, list[float]] = {}
        self._music_plan_targets: dict[int, Any] = {}
        self.quality_annotations_path = str(Path(quality_annotations).resolve()) if quality_annotations else ""
        self.quality_annotations = None
        self.quality_annotation_fingerprint = ""
        self.quality_records_by_sample: dict[int, dict[str, Any]] = {}
        self.quality_records_by_source: dict[tuple[str, int], dict[str, Any]] = {}
        self.segment_quality_weights: list[float] = []
        # Each segment stores either None (uniform chart weight) or a compact
        # list of (token_offset, token_length, weight) event spans. Do not retain
        # one Python float per token across a 100k-window full corpus.
        self.segment_quality_token_weights: list[list[tuple[int, int, float]] | None] = []
        self.segment_quality_loss_starts: list[int] = []
        if quality_annotations:
            from maichart.training.quality_annotations import annotations_fingerprint, load_quality_annotations

            self.quality_annotations = load_quality_annotations(quality_annotations)
            self.quality_annotation_fingerprint = annotations_fingerprint(self.quality_annotations)
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.segment_beats = int(segment_beats)
        self.max_tokens = int(max_tokens)
        self.sample_rate, self.hop_length = int(sample_rate), int(hop_length)
        if int(global_frames) <= 0:
            raise ValueError("global_frames must be positive.")
        # Full-song memory resolution.  It must match the model's
        # global_audio_tokens, so it is carried on the dataset instead of a
        # hard-coded 64.
        self.global_frames = int(global_frames)
        self.samples: list[EventSample] = []
        self._sample_index_by_id: dict[int, int] = {}
        self.skipped: list[dict[str, Any]] = []
        # Decoded audio features are ~3 MB per song, so an unbounded cache grows
        # linearly with the corpus (a 3000-song manifest would hold ~9 GB).  The
        # cache is an LRU bounded by bytes; `None` restores the old unbounded
        # behaviour and 0 disables caching entirely.
        self.audio_cache_bytes = None if audio_cache_bytes is None else max(0, int(audio_cache_bytes))
        # Target-derived controls are masked with this probability (see
        # `_masked_condition`). Default 0.5 balances masked/unmasked controls.
        # At 1.0 only `level` survives. Set 0.0 to reproduce the old
        # behaviour, where the condition leaked the answer's statistics.
        self.condition_mask_prob = max(0.0, min(1.0, float(condition_mask_prob)))
        self.condition_default = float(condition_default)
        if condition_target_version not in CONDITION_TARGET_VERSIONS:
            raise ValueError(f"Unknown condition_target_version {condition_target_version!r}.")
        self.condition_target_version = str(condition_target_version)
        self.min_difficulty_index = int(min_difficulty_index)
        self.max_difficulty_index = int(max_difficulty_index)
        self.min_level = float(min_level)
        self.max_level = float(max_level)
        if not 1 <= self.min_difficulty_index <= self.max_difficulty_index <= 6:
            raise ValueError("difficulty index range must be within 1..6.")
        if self.min_level > self.max_level:
            raise ValueError("min_level must not exceed max_level.")
        self._condition_rng = random.Random(int(condition_seed))
        self.audio: "OrderedDict[str, Any]" = OrderedDict()
        # Set when relative song paths had to be measured from a base other than
        # the manifest's own folder, so the run can say so once.
        self._path_base_note: dict[str, Any] = {}
        # Lazily discovered fallback base for a manifest that was moved after it
        # was written.
        self._extra_base: Path | None = None
        self._extra_base_checked = False
        self._audio_bytes = 0
        self.audio_cache_evictions = 0
        self.audio_cache_loads = 0
        # One line per song floods the 300-line WebUI log during the first epoch
        # (~1600 lines in minutes) and pushes the epoch summaries out of view.
        # Default: an aggregate line every `audio_load_log_every` loads, which
        # still shows the loading curve; `log_audio_loads=True` restores the
        # per-song detail for debugging a single song.
        self.log_audio_loads = bool(log_audio_loads)
        # Training and validation own separate EventDataset instances and
        # therefore separate LRU caches/load counters. Include the role in
        # aggregate lines so the validation counter restarting at 1 is clear.
        self.audio_cache_label = str(audio_cache_label or "dataset")
        self.audio_load_log_every = 25
        # Resolved cache manifest per source path, so a spilled song can be
        # reloaded from its cache file without re-reading and re-hashing the
        # source audio on every miss.
        self._audio_cache_paths: dict[str, Path | None] = {}
        self.conditions: dict[int, torch.Tensor] = {}
        self.global_audio: dict[int, torch.Tensor] = {}
        self._intensity_factors: dict[int, dict[Fraction, float]] = {}
        self._collect_samples(eager_audio=eager_audio)
        self._sample_index_by_id = {id(ref): index for index, ref in enumerate(self.samples)}
        if self.quality_annotations is not None:
            self._resolve_quality_records()
        self.vocabulary = vocabulary or EventVocabulary.from_event_charts(
            [ref.chart for ref in self.samples], segment_beats=self.vocabulary_segment_beats(),
            extra_specials=self.EXTRA_SPECIALS,
        )
        self.segments: list[tuple[int, Fraction, Fraction, list[int]]] = []
        for index, ref in enumerate(self.samples):
            count = max(1, math.ceil(ref.chart.end_beat / self.segment_beats))
            for segment in range(count):
                start = Fraction(segment * self.segment_beats)
                end = min(start + self.segment_beats, ref.chart.end_beat)
                selected = [point for point in ref.chart.points if start <= point.beat < end]
                sequence = EventSequence(
                    beats=[point.beat - start for point in selected],
                    note_texts=[point.note_text for point in selected],
                )
                try:
                    tokens = self.vocabulary.encode(sequence)
                    if len(tokens) > self.max_tokens:
                        raise ValueError(f"Segment has {len(tokens)} tokens, limit={self.max_tokens}; increase max_tokens or reduce segment_beats.")
                    self.segments.append((index, start, end, tokens))
                    sample_weight, token_weights = self._quality_weights_for_window(
                        index, start, end, selected, target_length=len(tokens) - 1,
                    )
                    self.segment_quality_weights.append(sample_weight)
                    self.segment_quality_token_weights.append(token_weights)
                    self.segment_quality_loss_starts.append(0)
                except ValueError as exc:
                    self.skipped.append({"song_id": ref.song_id, "stage": "segment", "start_beat": str(start),
                                         "reason": ("vocabulary_mismatch" if "checkpoint vocabulary" in str(exc)
                                                    else "segment_token_limit"), "message": str(exc)})
        if self.quality_annotations is not None and self.segments:
            self._filter_quality_excluded_segments()
        self._print_collection_summary()
        if not self.segments:
            raise ValueError(f"No usable event segments in {self.manifest_path}. See dataset audit reasons: {self.skipped[:5]}")

    def _candidate_bases(self) -> list[Path]:
        """Directories a relative song path may be measured from.

        A manifest's paths are normally relative to its own folder, but a derived
        manifest (a gold split written into a subfolder) can end up with paths
        measured from somewhere else -- most often the dataset root recorded in
        ``source_root``.  Trying the plausible bases turns a confusing "all songs
        missing" failure into a working run; the chosen base is reported once so a
        genuinely wrong manifest is still visible.
        """

        bases: list[Path] = [self.manifest_path.parent]
        declared = self.manifest.get("source_root")
        if declared and not str(declared).startswith("<"):
            root = Path(str(declared))
            bases.append(root if root.is_absolute() else (self.manifest_path.parent / root))
        bases.append(self.manifest_path.parent.parent)
        unique: list[Path] = []
        for base in bases:
            resolved = base.resolve()
            if resolved not in unique:
                unique.append(resolved)
        return unique

    def _discover_extra_base(self, value: str) -> Path | None:
        """Find the folder a moved manifest's relative paths are measured from.

        A derived manifest (a gold split) can be copied somewhere else after it
        was written; its relative paths then point nowhere and every song is
        reported as missing.  The data is usually one level away, so the
        immediate subfolders of the manifest's own folder are probed once and the
        working base is remembered for the rest of the manifest.
        """

        if self._extra_base_checked:
            return self._extra_base
        self._extra_base_checked = True
        relative = Path(value)
        try:
            children = [child for child in self.manifest_path.parent.iterdir() if child.is_dir()]
        except OSError:
            return None
        for child in children:
            if (child / relative).exists():
                self._extra_base = child
                return child
        return None

    def _resolve_song_path(self, value: str) -> Path:
        """Resolve one manifest path, accepting any base that actually exists."""

        path = Path(str(value))
        if path.is_absolute():
            return path
        bases = self._candidate_bases()
        if self._extra_base is not None:
            bases = [self._extra_base, *bases]
        for base in bases:
            candidate = base / path
            if candidate.exists():
                if base != bases[0] or self._extra_base is not None:
                    self._path_base_note["count"] = int(self._path_base_note.get("count", 0)) + 1
                    self._path_base_note["base"] = str(base)
                    self._path_base_note.setdefault("example", str(candidate))
                return candidate
        discovered = self._discover_extra_base(value)
        if discovered is not None:
            candidate = discovered / path
            if candidate.exists():
                self._path_base_note["count"] = int(self._path_base_note.get("count", 0)) + 1
                self._path_base_note["base"] = str(discovered)
                self._path_base_note.setdefault("example", str(candidate))
                return candidate
        return bases[0] / path

    def _collect_samples(self, *, eager_audio: bool) -> None:
        for song in self.manifest.get("songs", []):
            song_id = str(song.get("song_id", "unknown"))
            for difficulty in song.get("difficulties", []):
                index = difficulty.get("difficulty_index", difficulty.get("index"))
                if index is None:
                    continue
                try:
                    difficulty_index = int(index)
                    if difficulty_index < self.min_difficulty_index or difficulty_index > self.max_difficulty_index:
                        continue
                    level = numeric_level(difficulty.get("level_raw", difficulty.get("level")))
                    if level < self.min_level or level > self.max_level:
                        continue
                    source = self._resolve_song_path(song["maidata_path"])
                    audio_path = self._resolve_song_path(song["audio_path"])
                    if not source.is_file() or not audio_path.is_file():
                        raise ValueError(
                            "Source maidata/audio is missing; event training requires original files. "
                            f"Tried bases: {[str(base) for base in self._candidate_bases()]}"
                        )
                    chart = parse_event_chart(source, difficulty_index)
                    sanitized = sanitize_chart(chart, self.chart_filter_policy)
                    chart = sanitized.chart
                    if sanitized.removed:
                        audit = {
                            "song_id": song_id,
                            "difficulty_index": difficulty_index,
                            "source_path": str(source),
                            **sanitized.to_dict(),
                        }
                        self.chart_filter_audit.append(audit)
                        preview = ", ".join(
                            f"{item['beat']}:{item['atom']}" for item in audit["removed_preview"]
                        )
                        print(
                            f"[chart_filter] song={song_id} difficulty={difficulty_index} "
                            f"removed={audit['removed_count']} sample=[{preview}]",
                            flush=True,
                        )
                    inspect_chart(chart)
                    ref = EventSample(song_id, audio_path, chart, level, source)
                    if eager_audio:
                        self._load_audio(ref)
                    self.samples.append(ref)
                except Exception as exc:
                    message = str(exc)
                    self.skipped.append({"song_id": song_id, "difficulty_index": index, "stage": "document",
                                         "reason": self._skip_reason(message), "message": message})

    def _resolve_quality_records(self) -> None:
        """Attach annotations only when the exact difficulty content still matches."""
        from maichart.training.quality_annotations import resolve_annotation

        records = list((self.quality_annotations or {}).get("records", []))
        for sample_index, ref in enumerate(self.samples):
            slot = int(getattr(ref.chart, "difficulty", 0))
            source = str(ref.source_path.resolve())
            for record in records:
                if int(record.get("difficulty_index", -1)) != slot:
                    continue
                if record.get("song_id") and str(record["song_id"]) != ref.song_id:
                    continue
                try:
                    resolved = resolve_annotation(record, ref.source_path, ref.chart, slot, ref.audio_path)
                except (OSError, TypeError, ValueError):
                    resolved = None
                if resolved is None:
                    continue
                self.quality_records_by_sample[sample_index] = resolved
                self.quality_records_by_source[(source, slot)] = resolved
                break

    def quality_record(self, ref: EventSample) -> dict[str, Any] | None:
        """Content-validated annotation lookup used by the preference builder."""
        key = (str(Path(ref.source_path).resolve()), int(getattr(ref.chart, "difficulty", 0)))
        return self.quality_records_by_source.get(key)

    def _quality_event_weight(self, sample_index: int, beat: Fraction) -> float:
        record = self.quality_records_by_sample.get(int(sample_index))
        if record is None:
            return 1.0
        label = str(record.get("label", "normal"))
        regions = list(record.get("regions", []))
        if label == "excluded" or (label == "negative" and not regions):
            return 0.0
        base = 1.0 if label == "negative" and regions else float(record.get("weight", 1.0))
        if base <= 0.0:
            return 0.0
        ref = self.samples[int(sample_index)]
        when = float(seconds_at_beat(Fraction(beat), ref.chart.tempo_changes, ref.chart.audio_first))
        factor = 1.0
        for region in record.get("regions", []):
            if float(region["start_sec"]) <= when < float(region["end_sec"]):
                factor *= float(region.get("weight", 1.0))
        return max(0.0, min(4.0, base * factor))

    def _quality_onset_params(self, sample_index: int) -> tuple[float, list[dict[str, Any]]]:
        return quality_onset_params(self.quality_records_by_sample.get(int(sample_index)))

    def _onset_labels(self, sample_index: int, ref: EventSample, sample_times,
                      core_start_sec: float, core_end_sec: float, audio_duration_sec: float):
        if sample_index not in self._onset_times_by_sample:
            self._onset_times_by_sample[sample_index] = onset_times(ref.chart)
        base_weight, regions = self._quality_onset_params(sample_index)
        return build_onset_targets(
            ref.chart, sample_times,
            core_start_sec=core_start_sec,
            core_end_sec=core_end_sec,
            audio_duration_sec=audio_duration_sec,
            quality_base_weight=base_weight,
            quality_regions=regions,
            onset_seconds=self._onset_times_by_sample[sample_index],
        )

    def _quality_weights_for_window(self, sample_index: int, start: Fraction, end: Fraction,
                                    points: list[Any], *, target_length: int,
                                    prefix_len: int = 0) -> tuple[float, list[float]]:
        """Weight target tokens by the annotated chart or the event's time region.

        Region weights multiply the chart-level quality weight; a zero-weight
        region suppresses only events whose onsets fall inside that interval.
        Cross-window context remains excluded by its loss mask.
        """
        record = self.quality_records_by_sample.get(int(sample_index))
        if record is None:
            return 1.0, None
        label = str(record.get("label", "normal"))
        base = (0.0 if label == "excluded" or (label == "negative" and not record.get("regions"))
                else 1.0 if label == "negative" else float(record.get("weight", 1.0)))
        base = max(0.0, min(4.0, base))
        cursor = int(prefix_len)
        event_weights: list[float] = []
        event_spans: list[tuple[int, int, float]] = []
        for point in points:
            text = "".join(str(point.note_text).split())
            event_weight = self._quality_event_weight(sample_index, Fraction(point.beat))
            event_weights.append(event_weight)
            size = len(text) + 2  # AT position + CHAR tokens + <event_end>
            event_spans.append((cursor, size, event_weight))
            cursor += size
        window_weight = (sum(event_weights) / len(event_weights)) if event_weights else base
        # No span list for the usual uniform case; a compact event-span list is
        # needed only when a local rating differs from the chart-level weight.
        # In that case all events are listed so unaffected events keep the
        # chart-level weight rather than inheriting the window mean.
        varying = any(abs(value - base) > 1e-9 for value in event_weights)
        return max(0.0, min(4.0, window_weight)), (event_spans if varying else None)

    def _filter_quality_excluded_segments(self) -> None:
        """Remove wholly zero-weight windows while retaining them for preference pairing."""
        self.preference_segments = list(self.segments)
        self.preference_segment_quality_weights = list(self.segment_quality_weights)
        self.preference_segment_quality_token_weights = list(self.segment_quality_token_weights)
        self.preference_segment_quality_loss_starts = list(self.segment_quality_loss_starts)
        keep = [i for i, weight in enumerate(self.segment_quality_weights) if weight > 0.0]
        self.segments = [self.segments[i] for i in keep]
        self.segment_quality_weights = [self.segment_quality_weights[i] for i in keep]
        self.segment_quality_token_weights = [self.segment_quality_token_weights[i] for i in keep]
        self.segment_quality_loss_starts = [self.segment_quality_loss_starts[i] for i in keep]

    def quality_normalizers(self) -> dict[str, float]:
        """Fixed dataset-level scales prevent batch-local weight cancellation."""
        weighted_tokens = token_count = weighted_windows = window_count = 0.0
        for index, (weight, token_weights, loss_start) in enumerate(zip(
            self.segment_quality_weights, self.segment_quality_token_weights, self.segment_quality_loss_starts,
        )):
            segment = self.segments[index]
            length = len(segment[3]) - 1 if len(segment) == 4 else len(segment[3]) + len(segment[4]) - 1
            start = max(0, int(loss_start))
            valid_length = max(0, length - start)
            weighted = float(weight) * valid_length
            for offset, span_length, span_weight in token_weights or ():
                left = max(start, int(offset))
                right = min(length, int(offset) + int(span_length))
                if right > left:
                    weighted += (float(span_weight) - float(weight)) * (right - left)
            weighted_tokens += weighted
            token_count += valid_length
            sample_index, begin, end = segment[:3]
            weighted_windows += self._quality_plan_weight(sample_index, begin, end, weight)
            window_count += 1
        return {
            "token": weighted_tokens / max(token_count, 1.0),
            "planner": weighted_windows / max(window_count, 1.0),
            "positive_windows": float(sum(weight > 0 for weight in self.segment_quality_weights)),
        }

    @staticmethod
    def _skip_reason(message: str) -> str:
        lowered = message.lower()
        if "chart level" in lowered:
            return "level_missing"
        if "unsupported syntax" in lowered or "unsupported token" in lowered:
            return "unsupported_syntax"
        if "invalid event document" in lowered or "duplicate" in lowered:
            return "invalid_chart"
        if "missing" in lowered:
            return "missing_source"
        if "malformed" in lowered or "parse" in lowered:
            return "parse_error"
        return "other"

    def _print_collection_summary(self) -> None:
        document_skips = [item for item in self.skipped if item.get("stage") == "document"]
        segment_skips = [item for item in self.skipped if item.get("stage") == "segment"]
        if self._path_base_note:
            print(f"event_dataset_path_base manifest_folder={self.manifest_path.parent} "
                  f"used_base={self._path_base_note.get('base')} "
                  f"example={self._path_base_note.get('example')} "
                  f"songs={self._path_base_note.get('count')} "
                  f"(this manifest's relative paths are not measured from its own folder)", flush=True)
        if not document_skips and not segment_skips:
            return
        counts = Counter(item.get("reason", "other") for item in self.skipped)
        print(
            f"event_dataset_summary manifest={self.manifest_path} accepted_documents={len(self.samples)} "
            f"skipped_documents={len(document_skips)} accepted_segments={len(self.segments)} "
            f"skipped_segments={len(segment_skips)} reasons={dict(sorted(counts.items()))}",
            flush=True,
        )
        for item in self.skipped[:10]:
            location = f"{item.get('song_id', '?')}:{item.get('difficulty_index', item.get('start_beat', '?'))}"
            print(f"event_dataset_skip={location} reason={item.get('reason', 'other')} message={item.get('message', '')}", flush=True)

    def audio_cache_summary(self) -> str:
        """One-line report of the resident audio cache, for epoch/time logging."""

        limit = "unbounded" if self.audio_cache_bytes is None else f"{self.audio_cache_bytes / 1024 ** 3:.2f} GiB"
        return (f"audio_cache songs={len(self.audio)} resident={self._audio_bytes / 1024 ** 3:.2f} GiB "
                f"limit={limit} spills={self.audio_cache_evictions} loads={self.audio_cache_loads}")

    @staticmethod
    def _audio_size(audio: Any) -> int:
        """Bytes held by one cached EventAudio (features dominate the times array)."""

        total = 0
        for name in ("features", "times"):
            array = getattr(audio, name, None)
            nbytes = getattr(array, "nbytes", None)
            if nbytes:
                total += int(nbytes)
        return total

    def _read_audio(self, ref: EventSample):
        """Load one song's features, preferring its cache file over re-analysis."""

        key = str(ref.audio_path)
        if key not in self._audio_cache_paths:
            try:
                self._audio_cache_paths[key] = event_audio_cache_path(
                    ref.audio_path, audio_cache_dir(self.cache_dir),
                    sample_rate=self.sample_rate, hop_length=self.hop_length,
                )
            except OSError:
                self._audio_cache_paths[key] = None
        cache_path = self._audio_cache_paths[key]
        if cache_path is not None and cache_path.exists():
            try:
                return load_event_audio(cache_path, sample_rate=self.sample_rate, hop_length=self.hop_length)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                pass  # corrupt or partial cache: fall through to a full extraction
        return extract_event_audio(ref.audio_path, cache_dir=audio_cache_dir(self.cache_dir),
                                   sample_rate=self.sample_rate, hop_length=self.hop_length)

    def _load_audio(self, ref: EventSample):
        key = str(ref.audio_path)
        cached = self.audio.get(key)
        if cached is not None:
            self.audio.move_to_end(key)
            return cached
        audio = self._read_audio(ref)
        self.audio_cache_loads += 1
        size = self._audio_size(audio)
        if self.audio_cache_bytes is not None and self.audio_cache_bytes > 0:
            while self.audio and self._audio_bytes + size > self.audio_cache_bytes:
                _, evicted = self.audio.popitem(last=False)
                self._audio_bytes -= self._audio_size(evicted)
                self.audio_cache_evictions += 1
        if self.audio_cache_bytes == 0:
            self._note_audio_load(ref)
            return audio  # caching disabled: decode and hand back without retaining
        self.audio[key] = audio
        self._audio_bytes += size
        self._note_audio_load(ref)
        return audio

    def _note_audio_load(self, ref: EventSample) -> None:
        """Report one audio load, aggregated unless per-song detail was asked for."""

        if self.log_audio_loads:
            print(f"event_audio_ready={ref.song_id} dataset={self.audio_cache_label}", flush=True)
            return
        if self.audio_cache_loads == 1 or self.audio_cache_loads % self.audio_load_log_every == 0:
            resident = self._audio_bytes / 1024 ** 3
            print(f"event_audio_loads={self.audio_cache_loads} last={ref.song_id} "
                  f"dataset={self.audio_cache_label} "
                  f"resident={resident:.2f}GiB spills={self.audio_cache_evictions}", flush=True)

    def __len__(self) -> int:
        return len(self.segments)

    def vocabulary_segment_beats(self) -> int:
        """Beat span the position alphabet must cover.

        event_v1 needs exactly one window.  A family whose sequence spans more
        than one window (the cross-window block plus its window) overrides this,
        which is how positions stay continuous instead of restarting.
        """

        return self.segment_beats

    def _ensure_condition(self, sample_index: int, ref: EventSample, audio: Any) -> None:
        """Cache one song's condition vector and full-song memory.

        Shared with the cross-window family, which builds its own windows but
        needs exactly the same vectors.
        """

        if sample_index in self.conditions:
            return
        condition = condition_from_events(
            ref.chart.points, ref.chart.tempo_changes, level=ref.level,
            duration_sec=audio.duration_sec, audio=audio,
            **({"audio_first": ref.chart.audio_first}
               if self.condition_target_version == CURRENT_CONDITION_TARGET_VERSION else {}),
        )
        self.conditions[sample_index] = condition.to_vector()
        # The full-song memory starts at audio time 0, which is not necessarily
        # chart beat 0 when `first` is nonzero.  The compression helper derives
        # that anchor through the same exact tempo integration as the segment
        # path.  The frame count comes from the dataset so it always matches the
        # model's global_audio_tokens.
        self.global_audio[sample_index] = compress_event_audio(
            audio,
            song_duration=audio.duration_sec,
            bpm=ref.chart.tempo_changes[0].bpm,
            first=ref.chart.audio_first,
            tempo_changes=ref.chart.tempo_changes,
            frames=self.global_frames,
        )

    def _masked_condition(self, sample_index: int) -> torch.Tensor:
        """Condition vector for one training item, with the answers masked out.

        Every control except ``level`` is computed *from the chart being
        predicted*: chord ratio, slide ratios, density, reverb of style.  A model
        with enough capacity will happily read the answer off the condition and
        ignore the audio, and then at generation time -- where no chart exists
        yet and the controls default to something the corpus never contains -- it
        produces one arbitrary style and holds it for hundreds of beats.

        ``condition_mask_prob`` replaces those target-derived controls with the
        value generation will actually supply, so the model has to learn the
        chart from the audio while the masked input stays in distribution.  With
        the probability at 1.0 only ``level`` remains, which is the honest
        setting: everything else must come from the music.
        """

        vector = self.conditions[sample_index]
        if self.condition_mask_prob <= 0.0:
            return vector
        if self.condition_mask_prob >= 1.0 or self._condition_rng.random() < self.condition_mask_prob:
            masked = vector.clone()
            # Keep level (index 0); replace the rest with the inference default.
            masked[1:] = self.condition_default
            return masked
        return vector

    @staticmethod
    def _window_plan(ref: EventSample, start: Fraction, end: Fraction,
                     points_override: list[Any] | None = None) -> torch.Tensor:
        points = (list(points_override) if points_override is not None else
                  [point for point in ref.chart.points if start <= point.beat < end])
        span = max(float(end - start), 1.0)
        # Planner targets must use the same unit as generation quality: parsed
        # playable note heads, rather than raw timing points.  A chord or a
        # multi-branch slide is one timing point but several playable heads;
        # using point count here taught the planner a systematically different
        # density from the acceptance metrics used after generation.
        parsed = parse_event_notes(points, ref.chart.tempo_changes, first=getattr(ref.chart, "first", 0.0))
        notes = parsed.notes if parsed.syntax_valid else []
        note_count = len(notes)
        count = note_count
        texts = [str(point.note_text) for point in points]
        # Chord ratio is the fraction of occupied timing points containing at
        # least two parsed heads.  The parser also catches compact chords that
        # have no slash in their source text.
        heads_per_point = Counter()
        for point in parsed.points:
            heads_per_point[point.beat] += len(point.notes)
        occupied = max(len(points), 1)
        chord = sum(value > 1 for value in heads_per_point.values()) / occupied
        slide_count = sum(note.note_type == "slide" for note in notes)
        slide = slide_count / max(note_count, 1)
        last = max((float(point.beat - start) / span for point in points), default=0.0)
        # Log-scaled proxies preserve distinctions in dense windows without the
        # old hard saturation at 32 event points.  The scales are the same
        # references used by quality reporting (16 notes/sec).
        count_proxy = min(1.0, math.log1p(note_count) / math.log1p(64.0))
        # The sixth value is a coarse configuration class used as an auxiliary
        # anti-collapse target: 0 plain-single, 1 plain-chord, 2 slide, 3 touch.
        # It is deliberately low-bandwidth; the event decoder still decides the
        # exact hand pattern.
        from maichart.event_timing import seconds_at_beat
        duration_sec = max(
            float(seconds_at_beat(end, ref.chart.tempo_changes, ref.chart.audio_first)
            - seconds_at_beat(start, ref.chart.tempo_changes, ref.chart.audio_first)),
            1e-6,
        )
        absolute_density = note_count / duration_sec
        config = 0.0
        touch_count = sum(note.note_type in {"touch", "touch_hold"} for note in notes)
        if touch_count:
            config = 3.0
        elif slide_count:
            config = 2.0
        elif chord > 0.0:
            config = 1.0
        return torch.tensor([count_proxy, min(1.0, absolute_density / 16.0),
                             chord, slide, last, config / 3.0], dtype=torch.float32)

    def _quality_window_plan(self, sample_index: int, ref: EventSample,
                             start: Fraction, end: Fraction) -> torch.Tensor:
        # Ignoring an author's bad passage is not a request for a sparse chart.
        # Keep the real target; partially annotated windows are masked out of
        # planner supervision because its target describes the entire window.
        if is_music_structure(self.planner_profile):
            from maichart.training.music_plan_targets import MusicPlanTargets

            if sample_index not in self._music_plan_targets:
                self._music_plan_targets[sample_index] = MusicPlanTargets(
                    ref.chart, profile=self.planner_profile)
            return self._music_plan_targets[sample_index].window(start, end)
        return self._window_plan(ref, start, end)

    def _quality_plan_weight(self, sample_index: int, start: Fraction,
                             end: Fraction, default: float) -> float:
        record = self.quality_records_by_sample.get(int(sample_index))
        if record is None:
            return float(default)
        ref = self.samples[int(sample_index)]
        begin = float(seconds_at_beat(start, ref.chart.tempo_changes, ref.chart.audio_first))
        finish = float(seconds_at_beat(end, ref.chart.tempo_changes, ref.chart.audio_first))
        for region in record.get("regions", []):
            if (begin < float(region["end_sec"]) and finish > float(region["start_sec"])
                    and float(region.get("weight", 1.0)) != 1.0):
                return 0.0
        return float(default)

    def quality_token_vector(self, index: int, target_length: int) -> torch.Tensor:
        weight = float(self.segment_quality_weights[index])
        result = torch.full((int(target_length),), weight, dtype=torch.float32)
        for offset, span_length, span_weight in self.segment_quality_token_weights[index] or ():
            low = max(0, int(offset))
            high = min(result.numel(), int(offset) + int(span_length))
            if high > low:
                result[low:high] = float(span_weight)
        return result

    def _condition_with_null_flag(self, sample_index: int) -> tuple[torch.Tensor, bool]:
        vector = self.conditions[sample_index]
        if self.condition_mask_prob <= 0.0:
            return vector, False
        masked = self.condition_mask_prob >= 1.0 or self._condition_rng.random() < self.condition_mask_prob
        if masked:
            result = vector.clone()
            result[1:] = self.condition_default
            return result, True
        return vector, False

    def _window_condition(self, sample_index: int, ref: EventSample, audio: Any,
                          start: Fraction, condition: torch.Tensor | None = None) -> torch.Tensor:
        condition = (self._masked_condition(sample_index) if condition is None else condition).clone()
        if is_music_structure(self.planner_profile):
            # A planner learns authored local workload from music. Multiplying
            # the global density (including numeric null values) by RMS creates
            # a conflicting instruction and exposes a fake condition state.
            return condition
        factors = self._intensity_factors.get(sample_index)
        if factors is None:
            from maichart.generation.intensity import window_factors
            windows = [(item[1], item[2]) for item in self.segments if item[0] == sample_index]
            values = window_factors(audio, windows, tempo_changes=ref.chart.tempo_changes,
                                    first=ref.chart.audio_first, span=0.5, song_duration=audio.duration_sec)
            factors = {window[0]: factor for window, factor in zip(windows, values)}
            self._intensity_factors[sample_index] = factors
        factor = factors.get(start, 1.0)
        condition[1:3] = torch.clamp(condition[1:3] * float(factor), 0.0, 1.0)
        return condition


    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_index, start, end, token_ids = self.segments[index]
        ref = self.samples[sample_index]
        audio = self._load_audio(ref)
        from maichart.event_timing import seconds_at_beat
        # No clamp to audio time 0: a chart may start before the audio does
        # (negative `first` or a leading negative delay), and that region must be
        # sampled as silence instead of collapsing onto the first frame.
        begin_sec = seconds_at_beat(start, ref.chart.tempo_changes, ref.chart.audio_first)
        end_sec = max(begin_sec, seconds_at_beat(end, ref.chart.tempo_changes, ref.chart.audio_first))
        tempo_entries = _normalize_tempo_map(ref.chart.tempo_changes, ref.chart.tempo_changes[0].bpm)
        # A segment can begin before audio time 0 when `first` is positive.
        # audio_segment_tensor samples the clipped time, so derive the chart
        # beat for that actual sample instead of passing the unclipped window
        # start and triggering a false anchor mismatch.
        segment_start_beat = _time_to_beat(begin_sec - ref.chart.audio_first, tempo_entries)
        x = audio_segment_tensor(
            audio, begin_sec, end_sec, song_duration=audio.duration_sec,
            bpm=ref.chart.tempo_changes[0].bpm, start_beat=segment_start_beat,
            first=ref.chart.audio_first, tempo_changes=ref.chart.tempo_changes,
        )
        if sample_index not in self.conditions:
            self._ensure_condition(sample_index, ref, audio)
        tokens = torch.tensor(token_ids, dtype=torch.long)
        condition, condition_null = self._condition_with_null_flag(sample_index)
        quality_weight = float(self.segment_quality_weights[index])
        quality_tokens = self.quality_token_vector(index, len(token_ids) - 1)
        item = {"audio": x, "global_audio": self.global_audio[sample_index],
                "condition": self._window_condition(sample_index, ref, audio, start, condition),
                "condition_null": torch.tensor(condition_null, dtype=torch.bool),
                "plan_target": self._quality_window_plan(sample_index, ref, start, end),
                "quality_sample_weight": torch.tensor(quality_weight, dtype=torch.float32),
                "quality_plan_weight": torch.tensor(
                    self._quality_plan_weight(sample_index, start, end, quality_weight), dtype=torch.float32),
                "quality_token_weights": quality_tokens,
                "decoder_input": tokens[:-1], "target": tokens[1:],
                "meta": {"song_id": ref.song_id, "start_beat": str(start), "end_beat": str(end)}}
        if self.onset_profile != "none":
            sample_times = segment_frame_times(begin_sec, end_sec)
            if sample_times.size != x.size(0):
                raise RuntimeError("onset frame-time grid does not match audio segment frames")
            item["onset_target"], item["onset_weight"], item["onset_valid"] = self._onset_labels(
                sample_index, ref, sample_times, float(begin_sec), float(end_sec), float(audio.duration_sec))
        return item

    @property
    def signature(self) -> str:
        sources = [(str(ref.source_path), hashlib.sha256(ref.source_path.read_bytes()).hexdigest(), str(ref.audio_path), ref.audio_path.stat().st_size, ref.audio_path.stat().st_mtime_ns) for ref in self.samples]
        payload = {
            "sources": sources,
            "difficulty_filter": {
                "min_index": self.min_difficulty_index,
                "max_index": self.max_difficulty_index,
                "min_level": self.min_level,
                "max_level": self.max_level,
            },
            "segment_beats": self.segment_beats,
            "max_tokens": self.max_tokens,
            "condition_mask_prob": self.condition_mask_prob,
            "condition_default": self.condition_default,
        }
        if self.condition_target_version != LEGACY_CONDITION_TARGET_VERSION:
            payload["condition_target_version"] = self.condition_target_version
        if self.quality_annotations is not None:
            payload["quality_annotations_fingerprint"] = self.quality_annotation_fingerprint
        if self.planner_profile != "legacy_v1":
            payload["planner_profile"] = self.planner_profile
        if self.onset_profile != "none":
            payload["onset_profile"] = self.onset_profile
        # Preserve legacy resume signatures while recording changed targets.
        if self.chart_filter_policy != "legacy_strict_v1":
            payload["chart_filter_policy"] = self.chart_filter_policy
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def audit(self) -> dict[str, Any]:
        return {"schema": "maichart-event-dataset-audit-v1", "manifest": str(self.manifest_path),
                "samples": len(self.samples), "segments": len(self), "vocab_size": len(self.vocabulary),
                "signature": self.signature, "skipped": self.skipped,
                "target_schema": "event_v1", "segment_beats": self.segment_beats,
                "onset_profile": self.onset_profile,
                "chart_filter_policy": self.chart_filter_policy,
                "chart_filter_audit": self.chart_filter_audit,
                "filtered_note_count": sum(item["removed_count"] for item in self.chart_filter_audit),
                "style_in_dataset": False, "frame_division_used": False,
                "difficulty_filter": {"min_index": self.min_difficulty_index,
                                      "max_index": self.max_difficulty_index,
                                      "min_level": self.min_level, "max_level": self.max_level},
                "condition_target_version": self.condition_target_version}


def collate_events(batch: list[dict[str, Any]], *, pad_id: int = 0) -> dict[str, Any]:
    max_audio = max(item["audio"].size(0) for item in batch)
    max_tokens = max(item["target"].size(0) for item in batch)
    audio = torch.zeros(len(batch), max_audio, batch[0]["audio"].size(1))
    audio_mask = torch.ones(len(batch), max_audio, dtype=torch.bool)
    decoder = torch.full((len(batch), max_tokens), pad_id, dtype=torch.long)
    target = torch.full_like(decoder, pad_id)
    token_mask = torch.ones_like(decoder, dtype=torch.bool)
    plan_audio_mask = torch.ones((len(batch), max_audio), dtype=torch.bool)
    # Positions the loss should ignore beyond padding.  The cross-window family
    # uses this to exclude its context block: the model must *see* the lead-in
    # notes (so they stay in decoder_input) but must not be trained to predict
    # them, because at generation time they are given, not produced.
    loss_mask = torch.zeros_like(decoder, dtype=torch.bool)
    quality_token_weights = torch.ones((len(batch), max_tokens), dtype=torch.float32)
    for index, item in enumerate(batch):
        a_len, t_len = item["audio"].size(0), item["target"].size(0)
        audio[index, :a_len] = item["audio"]
        audio_mask[index, :a_len] = False
        item_plan_mask = item.get("plan_audio_mask")
        if item_plan_mask is None:
            plan_audio_mask[index, :a_len] = False
        else:
            plan_audio_mask[index, :min(a_len, item_plan_mask.numel())] = item_plan_mask[:a_len]
        decoder[index, :t_len] = item["decoder_input"]
        target[index, :t_len] = item["target"]
        token_mask[index, :t_len] = False
        item_mask = item.get("loss_mask")
        if item_mask is None:
            loss_mask[index, :t_len] = True
        else:
            loss_mask[index, :t_len] = item_mask[:t_len]
        if item.get("quality_token_weights") is not None:
            item_weights = item["quality_token_weights"]
            quality_token_weights[index, :min(t_len, item_weights.numel())] = item_weights[:t_len]
    result = {"audio": audio, "audio_padding_mask": audio_mask, "decoder_input": decoder, "target": target,
            "decoder_padding_mask": token_mask, "loss_mask": loss_mask,
            "plan_audio_mask": plan_audio_mask,
            "quality_token_weights": quality_token_weights,
            "global_audio": torch.stack([item["global_audio"] for item in batch]),
             "condition_null": torch.stack([item["condition_null"] for item in batch]),
            "condition": torch.stack([item["condition"] for item in batch]),
            "quality_sample_weights": torch.stack([item.get("quality_sample_weight", torch.tensor(1.0)).float() for item in batch]),
            "quality_plan_weights": torch.stack([item.get("quality_plan_weight", item.get("quality_sample_weight", torch.tensor(1.0))).float() for item in batch]),
             "plan_target": torch.stack([item["plan_target"] for item in batch])}
    if any("onset_target" in item for item in batch):
        for key in ("onset_target", "onset_weight", "onset_valid"):
            values = torch.zeros((len(batch), max_audio), dtype=torch.float32)
            for index, item in enumerate(batch):
                if key not in item:
                    continue
                source = item[key].to(dtype=torch.float32).flatten()
                if source.numel() != item["audio"].size(0):
                    raise ValueError(f"{key} length must match the unpadded audio frames")
                values[index, :source.numel()] = source
            result[key] = values
    return result
