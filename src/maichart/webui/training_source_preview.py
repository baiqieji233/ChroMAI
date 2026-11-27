"""Read-only quality preview for a prospective Composer v2 training run."""
from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, replace
from functools import cache, lru_cache
import hashlib
import json
import os
import threading
from pathlib import Path
from tempfile import TemporaryDirectory

from maichart.composer_v2.difficulty import (
    DIFFICULTY_SLOTS, FITTED_SOURCES, LEGACY_DIFFICULTY_SLOTS, metadata_exclusion, resolve_selection,
)
from maichart.composer_v2.quality_data import TrainingQuality, quality_excludes_chart, stream_region_weight_fn
from maichart.portable_paths import material_path
from maichart.training.dataset_catalog import _source_asset, manifest_dataset_name
from maichart.training.quality_annotations import merge_annotation_records, resolve_auto_annotations, split_annotation_source


_SLOT_NAMES = {4: "EXPERT", 5: "MASTER", 6: "Re:MASTER", 7: "UTAGE"}
_MEMORY_AUDIO_DIGESTS_LOCK = threading.Lock()
_MEMORY_AUDIO_DIGESTS: OrderedDict[tuple[str, int, int], str] = OrderedDict()
_MEMORY_AUDIO_DIGESTS_MAX = 16384


class _CachedReadOnlyDigests:
    """Stat-keyed in-memory audio digest cache wrapping read-only SourceDigests (zero disk writes)."""

    def __init__(self, cache_dir: Path):
        from maichart.form.cache_support import SourceDigests

        self._inner = SourceDigests(cache_dir, read_only=True)

    def digest(self, path) -> str:
        target = Path(path)
        stat = target.stat()
        key = (os.path.normcase(str(target.resolve())), stat.st_size, stat.st_mtime_ns)
        with _MEMORY_AUDIO_DIGESTS_LOCK:
            cached = _MEMORY_AUDIO_DIGESTS.get(key)
            if cached is not None:
                _MEMORY_AUDIO_DIGESTS.move_to_end(key)
                return cached
        value = self._inner.digest(target)
        with _MEMORY_AUDIO_DIGESTS_LOCK:
            _MEMORY_AUDIO_DIGESTS[key] = value
            _MEMORY_AUDIO_DIGESTS.move_to_end(key)
            while len(_MEMORY_AUDIO_DIGESTS) > _MEMORY_AUDIO_DIGESTS_MAX:
                _MEMORY_AUDIO_DIGESTS.popitem(last=False)
        return value

    def close(self) -> None:
        self._inner.close()


def _describe_included_chart(chart, record, supervision, fitted_score=None):
    rec = record or {}
    zero_unmarked = bool(chart.reviewed_only or rec.get("unmarked_weight") == 0)
    pos_regions = [(s, e, wt) for s, e, wt in supervision.regions if max(wt) > 0]
    raw_regions = rec.get("regions") or ()
    if zero_unmarked:
        eff_weight = round(max((max(wt) for _, _, wt in pos_regions), default=0.0), 4)
        masked_spans = sum(
            1 for r in raw_regions
            if r.get("label") == "problem"
            or max(float(r.get("timing_weight", 1.0)), float(r.get("component_weight", 1.0))) <= 0
        )
        approved_spans = len(pos_regions)
        is_full = rec.get("candidate_chart_review") == "learn_all" and masked_spans == 0
        if is_full:
            weight_summary = f"{eff_weight:g}× 整谱入训"
        elif masked_spans > 0:
            weight_summary = f"{eff_weight:g}×（屏蔽 {masked_spans} 段无理）"
        else:
            weight_summary = f"{eff_weight:g}×（仅学 {approved_spans} 段，其余 0×）"
    else:
        base_w = max(supervision.base)
        eff_weight = round(base_w, 4)
        zero_spans = sum(1 for _, _, wt in supervision.regions if max(wt) <= 0)
        down_spans = sum(1 for _, _, wt in supervision.regions if 0 < min(wt) < base_w - 1e-6)
        boost_spans = sum(1 for _, _, wt in supervision.regions if max(wt) > base_w + 1e-6)
        masked_spans = zero_spans
        approved_spans = sum(1 for _, _, wt in supervision.regions if max(wt) > 0)
        is_full = zero_spans == 0 and down_spans == 0
        notes = []
        if zero_spans:
            notes.append(f"屏蔽 {zero_spans} 段(0×)")
        if down_spans:
            notes.append(f"降权 {down_spans} 段")
        if boost_spans:
            notes.append(f"加权 {boost_spans} 段")
        if not notes:
            if eff_weight > 1.0 + 1e-6:
                weight_summary = f"{eff_weight:g}× 加权整谱"
            elif eff_weight < 1.0 - 1e-6:
                weight_summary = f"{eff_weight:g}× 降权整谱"
            else:
                weight_summary = "1× 标准整谱"
        else:
            weight_summary = f"{eff_weight:g}×（{'，'.join(notes)}）"

    if eff_weight > 1.0 + 1e-6:
        tier = "boosted"
    elif eff_weight < 1.0 - 1e-6:
        tier = "downweighted"
    elif is_full:
        tier = "standard_full"
    else:
        tier = "partial_or_masked"

    row = dict(
        chart.describe(),
        slot_name=_SLOT_NAMES.get(chart.slot, f"槽 {chart.slot}"),
        effective_weight=eff_weight,
        weight_tier=tier,
        weight_summary=weight_summary,
        label=str(rec.get("label") or "normal"),
        reason=str(rec.get("reason") or ""),
        auto_screened=bool(rec.get("auto_screened")),
        reviewed_only=bool(chart.reviewed_only),
        masked_span_count=int(masked_spans),
        approved_span_count=int(approved_spans),
    )
    if fitted_score is not None:
        row["fitted_level"] = round(float(fitted_score), 2)
    if chart.metadata.get("learning_role") == "mechanics":
        row.update(learning_role="mechanics", effective_weight=0., weight_summary="仅练动作 · 应用本谱的标注范围")
    return row


@dataclass
class Chart:
    manifest: Path
    name: str
    title: str
    level: str
    slot: int
    path: Path
    audio: Path | None
    candidate: bool
    reviewed_only: bool
    metadata: dict

    @property
    def key(self):
        return self.path, self.slot

    def describe(self):
        return dict(manifest=str(self.manifest), source=self.name, path=str(self.path),
                    title=self.title, slot=self.slot, level=self.level)


def _charts(manifest, payload, resolve_asset):
    default_name = manifest_dataset_name(manifest, payload)
    for song in payload.get("songs", ()):
        if not song.get("maidata_path"):
            continue
        source = song.get("source_dataset")
        name = source.get("name") if isinstance(source, dict) else source
        owner = Path(song["source_manifest"]) if song.get("source_manifest") else manifest
        path = resolve_asset(song["maidata_path"], manifest)
        audio = resolve_asset(song["audio_path"], manifest) if song.get("audio_path") else None
        candidate = bool(payload.get("candidate_pool") or song.get("candidate_pool"))
        reviewed_only = candidate or bool(payload.get("reviewed_training") or song.get("reviewed_only"))
        for difficulty in song.get("difficulties", ()):
            try:
                slot = int(difficulty.get("difficulty_index", difficulty.get("index")))
            except (TypeError, ValueError):
                continue
            level = difficulty.get("level_raw", difficulty.get("level"))
            if "learning_role" in song and "learning_role" not in difficulty:
                difficulty = dict(difficulty, learning_role=song["learning_role"])
            yield Chart(owner, name or default_name, str(song.get("title") or path.parent.name),
                        str(level or ""), slot, path, audio, candidate, reviewed_only, difficulty)


def _candidate_scan_exclusions(manifest, payload, read_charts, resolve_asset):
    if not payload.get("candidate_pool"):
        return
    path = manifest.with_name("pool.json")
    if not path.is_file():
        return
    metadata = json.loads(path.read_text(encoding="utf-8-sig"))
    if not metadata.get("source_manifest"):
        return
    original = Path(metadata["source_manifest"]).resolve()
    if not original.is_file():
        return
    # An initial scan can drop duplicate charts before creating the pool
    # manifest. Recover their identity from the original source for comparison.
    if not metadata.get("excluded"):
        return
    originals = {chart.key: chart for chart in read_charts(original)}
    name = manifest_dataset_name(manifest, payload)
    for entry in metadata.get("excluded", ()):
        if not entry.get("source_path") or entry.get("slot") is None:
            continue
        source = resolve_asset(entry["source_path"], original)
        slot = int(entry["slot"])
        chart = originals.get((source, slot))
        if chart is None:
            chart = Chart(manifest, name, source.parent.name, "", slot, source, None, True, True, {})
        yield replace(chart, manifest=manifest, name=name, candidate=True, reviewed_only=True), str(entry.get("reason") or "scan_excluded")


@dataclass
class Supervision:
    base: tuple[float, float]
    regions: list[tuple[float, float, tuple[float, float]]]

    def spans(self):
        """Zero-weight portions, including gaps between explicit approvals."""
        end = 0.
        for start, stop, weights in self.regions:
            if start > end and not all(self.base):
                yield end, start, self.base
            if not all(weights):
                yield start, stop, weights
            end = stop
        if not all(self.base):
            yield end, None, self.base

    def at(self, second):
        return next((weight for start, end, weight in self.regions if start <= second < end), self.base)


def _supervision(record, sequence, legacy):
    record = record or {}
    if quality_excludes_chart(record):
        return Supervision((0., 0.), [])
    unmarked = float(record.get("unmarked_weight", 1.))
    outer = float(record.get("weight", 1.))
    weight_fn = stream_region_weight_fn(unmarked, legacy)
    regions = []
    for region in record.get("regions", ()):
        if sequence == "window_v1":
            weight = float(region.get("weight", 1.))
            weights = (weight, weight)
        else:
            weights = weight_fn(region)
        regions.append((float(region["start_sec"]), float(region["end_sec"]),
                        (weights[0] * outer, weights[1] * outer)))
    return Supervision((unmarked * outer, unmarked * outer), regions)


def _restored_spans(requested, actual, contribution=None):
    regions = requested.regions + actual.regions + (contribution.regions if contribution else [])
    boundaries = sorted({0., *(value for start, end, _ in regions for value in (start, end))})
    result = []
    for start, end in zip(boundaries, boundaries[1:] + [None]):
        old, new = requested.at(start), actual.at(start)
        supplied = contribution.at(start) if contribution else new
        parts = [name for name, before, after, provided in zip(("timing", "configuration"), old, new, supplied)
                 if before == 0 and after > 0 and provided > 0]
        if not parts:
            continue
        if result and result[-1]["end_sec"] == start and result[-1]["learns"] == parts:
            result[-1]["end_sec"] = end
        else:
            result.append(dict(start_sec=start, end_sec=end, learns=parts))
    return result


def _quality_record(quality, chart, sequence, legacy):
    """Detect a restrictive record whose current chart/audio no longer matches."""
    try:
        manual, auto = quality.records_for(chart.path, chart.slot, chart.audio)
    except (OSError, ValueError, UnicodeError):
        manual = auto = None
    key = chart.slot, str(chart.path)
    stale = any(
        record is None and any(_supervision(row, sequence, legacy).spans()
                               for row in index.get(key, ()))
        for record, index in ((manual, quality.manual_index), (auto, quality.auto_index))
    )
    return manual, auto, stale


def _same_chart_index(relevant, actual, digest):
    """Check audio bytes only for paths with a matching authored difficulty."""
    from maichart.training.candidate_pool import _content_identity

    resolved = {}

    def identity(chart):
        if chart.key in resolved:
            return resolved[chart.key]
        try:
            content = _content_identity(chart.path, chart.slot)
            key = (chart.slot, content) if content else None
        except (OSError, ValueError, StopIteration, UnicodeError):
            key = None
        resolved[chart.key] = key
        return key

    keys = {key for chart in relevant if (key := identity(chart)) is not None}
    result = defaultdict(list)
    if keys:
        for chart, supervision in actual:
            key = identity(chart)
            if key not in keys or chart.audio is None:
                continue
            try:
                result[key, digest(chart.audio)].append((chart, supervision))
            except (OSError, ValueError):
                continue
    return result, identity


def _quality_rows(manifest, data, sources, options, digest, composition, *, read_charts, resolve_path, resolve_asset):
    from maichart.training.event_dataset import numeric_level
    from maichart.training.candidate_pool import _read, _valid_regions, candidate_record

    quality = TrainingQuality(manifest, data, enabled=options["quality"], use_auto=options["use_auto"],
                              audio_digest=digest, resolve_path=resolve_path)
    expected_auto = options["saved_identity"].get("auto_annotations_sha256")
    if expected_auto and quality.auto_fingerprint != expected_auto:
        raise ValueError("父 checkpoint 的正式体检标注已变化，请恢复原标注或使用新目录微调")
    difficulty = options["difficulty"]
    sequence = options["sequence"]
    legacy = options["legacy"]
    excluded, context, restored, uncertain = [], [], [], []
    included, held_out = [], []
    held_out_by_reason: Counter[str] = Counter()
    actual, requested = [], []
    source_charts = {}
    noted = set()
    retained = 0
    charts = read_charts(manifest)
    has_reviewed = data.get("candidate_pool") or data.get("reviewed_training") or any(chart.reviewed_only for chart in charts)
    if has_reviewed and (not options["quality"] or sequence != "continuous_events_v1"):
        raise ValueError("候选库和示范库需要启用质量标注，并使用 Composer v2 连续事件训练")

    @cache
    def parseable(path, slot):
        return _parseable(path, slot, options["dialect"])

    def _eligibility_reason(chart):
        if not 4 <= chart.slot <= 6:
            return "non_training_slot"
        raw = chart.metadata.get("level_raw", chart.metadata.get("level"))
        reason = metadata_exclusion(
            chart.slot, raw, difficulty.effective_slots, None, difficulty.min_declared_level
        )
        if reason is not None:
            return reason
        if difficulty.source not in FITTED_SOURCES:
            try:
                lvl = numeric_level(raw)
            except ValueError:
                return "invalid_declared_level"
            if not difficulty.min_level <= lvl <= difficulty.max_level:
                return "declared_level_out_of_range"
        return None

    def eligibility(chart):
        return _eligibility_reason(chart) is None

    def mask(chart, record, reason=None):
        excluded.append(dict(chart.describe(), reason=reason or "quality_excluded", label=(record or {}).get("label")))

    def warn(chart, reason):
        key = chart.manifest, chart.key, reason
        if key not in noted:
            uncertain.append(dict(chart.describe(), reason=reason))
            noted.add(key)

    for cand_chart in (options.get("pre_split_charts") or charts):
        elig_reason = _eligibility_reason(cand_chart)
        if elig_reason is not None:
            held_out_by_reason[elig_reason] += 1
            if cand_chart.slot >= 4:
                held_out.append(dict(
                    cand_chart.describe(),
                    slot_name=_SLOT_NAMES.get(cand_chart.slot, f"槽 {cand_chart.slot}"),
                    reason=elig_reason,
                ))
        elif cand_chart.key in options["holdout_keys"]:
            held_out_by_reason["validation_holdout"] += 1
            held_out.append(dict(
                cand_chart.describe(),
                slot_name=_SLOT_NAMES.get(cand_chart.slot, f"槽 {cand_chart.slot}"),
                reason="validation_holdout",
            ))

    remaining_keys = {chart.key for chart in charts if eligibility(chart)}
    uncertain_limit = False
    for chart in charts:
        if not eligibility(chart):
            continue
        if options["limit"] and retained >= options["limit"] and not uncertain_limit:
            break
        if not chart.path.is_file() or chart.audio is None or not chart.audio.is_file():
            warn(chart, "missing_asset")
            actual.append((chart, None))
            continue
        manual, auto, stale = _quality_record(quality, chart, sequence, legacy)
        if stale:
            warn(chart, "stale_quality_annotation")
        record = merge_annotation_records(manual, auto) if auto is not None else manual
        if chart.reviewed_only and (record or {}).get("unmarked_weight") != 0:
            warn(chart, "missing_review")
            continue
        if quality_excludes_chart(record):
            reason = "manual_exclusion" if quality_excludes_chart(manual) else "automatic_exclusion"
            mask(chart, record, reason)
            requested.append((chart, record, reason, True))
            continue
        fitted_score = None
        if options.get("scorer") is not None:
            try:
                score = _fitted_score(options["scorer"], chart.path, chart.slot)
            except (OSError, RuntimeError, ValueError):
                warn(chart, "missing_score")
                actual.append((chart, None))
                uncertain_limit = uncertain_limit or bool(options["limit"])
                continue
            if not difficulty.min_level <= score <= difficulty.max_level:
                held_out_by_reason["fitted_level_out_of_range"] += 1
                held_out.append(dict(
                    chart.describe(),
                    slot_name=_SLOT_NAMES.get(chart.slot, f"槽 {chart.slot}"),
                    reason="fitted_level_out_of_range",
                    fitted_level=round(float(score), 2),
                ))
                continue
            fitted_score = score
        if options["limit"] and not parseable(chart.path, chart.slot):
            warn(chart, "parse_failed")
            continue
        if uncertain_limit:
            actual.append((chart, None))
            continue
        retained += 1
        supervision = _supervision(record, sequence, legacy)
        actual.append((chart, supervision))
        included.append(_describe_included_chart(chart, record, supervision, fitted_score))
        spans = list(supervision.spans())
        if spans:
            requested.append((chart, record, "source_restriction", True))
            context.extend(dict(chart.describe(), start_sec=start, end_sec=end, timing_weight=weights[0],
                                configuration_weight=weights[1]) for start, end, weights in spans)

    # Compare independent source intent with the resulting selection. Even a
    # dropped candidate can be supplied again by the original ordinary corpus.
    for source_manifest, payload in sources:
        if len(sources) == 1 and not payload.get("candidate_pool") and options["quality"]:
            continue
        scope = options.get('auto_quality_manifests')
        source_auto = (options["use_auto"] or (not options["quality"] and options["policy"] != "off"))
        source_auto = source_auto and (scope is None or str(source_manifest) in scope)
        source_quality = TrainingQuality(
            source_manifest, payload, enabled=True,
            use_auto=source_auto,
            audio_digest=digest, resolve_path=resolve_path,
        )
        pool_metadata = _read(source_manifest.with_name("pool.json")) if payload.get("candidate_pool") else {}
        is_candidate_pool = bool(payload.get("candidate_pool"))
        for chart in read_charts(source_manifest):
            if not eligibility(chart):
                continue
            if chart.key in options["holdout_keys"]:
                continue
            source_charts[chart.manifest, chart.key] = chart
            if is_candidate_pool:
                key = chart.slot, str(chart.path)
                records = source_quality.manual_index.get(key, ())
                approved = [row for row in records if _valid_regions(row)]
                assets = dict(maidata_path=str(chart.path), audio_path=str(chart.audio))
                intended = (candidate_record(source_manifest, assets, chart.metadata, approved, _metadata=pool_metadata)
                            if approved else None)
                auto, stale = None, False
                manual = intended
                if intended is None:
                    if any(quality_excludes_chart(row) for row in records):
                        manual, _, stale = _quality_record(source_quality, chart, sequence, legacy)
                    intended = manual if quality_excludes_chart(manual) else {"label": "excluded"}
            else:
                manual, auto, stale = _quality_record(source_quality, chart, sequence, legacy)
                intended = merge_annotation_records(manual, auto) if auto is not None else manual
            if stale:
                warn(chart, "stale_quality_annotation")
            if chart.reviewed_only and not quality_excludes_chart(intended):
                intended = dict(intended or {}, unmarked_weight=0.)
                if not any(max(stream_region_weight_fn(0., legacy)(span)) > 0 for span in intended.get("regions", ())):
                    intended = {"label": "excluded"}
            if quality_excludes_chart(intended) or any(_supervision(intended, sequence, legacy).spans()):
                if quality_excludes_chart(manual):
                    reason = "manual_exclusion"
                elif quality_excludes_chart(auto):
                    reason = "automatic_exclusion"
                else:
                    reason = "candidate_not_approved" if chart.reviewed_only else "source_restriction"
                requested.append((chart, intended, reason, False))
        for chart, reason in _candidate_scan_exclusions(source_manifest, payload, read_charts, resolve_asset):
            if eligibility(chart) and chart.key not in options["holdout_keys"]:
                requested.append((chart, {"label": "excluded"}, reason, False))

    copies, content_identity = (_same_chart_index([chart for chart, *_ in requested], actual, digest)
                                if len(sources) > 1 else ({}, None))
    actual_by_key = {chart.key: (chart, supervision) for chart, supervision in actual}
    excluded_keys = {(row["manifest"], row["path"], row["slot"]) for row in excluded}
    restored_by_chart = defaultdict(list)
    contributors = composition.get("preview_contributors", {})
    annotation_owners = composition.get("preview_annotation_owners", {})
    handled = set()
    for chart, record, reason, effective in requested:
        original = _supervision(record, sequence, legacy)
        source_key = chart.manifest, chart.key, original.base, tuple(original.regions)
        if source_key in handled:
            continue
        handled.add(source_key)
        matches = []
        if chart.key in actual_by_key:
            matches.append(actual_by_key[chart.key])
        if content_identity is not None and chart.audio and (content := content_identity(chart)):
            try:
                matches.extend((other, supervision) for other, supervision in copies.get((content, digest(chart.audio)), ())
                               if other.key != chart.key)
            except (OSError, ValueError):
                warn(chart, "identity_unverified")
        suppliers = []
        unknown = False
        for other, supervision in matches:
            if other.metadata.get('learning_role') == 'mechanics':
                continue  # An action-only example cannot restore note/timing supervision.
            other_key = str(other.path), other.slot
            contributions = contributors.get(other_key, ())
            if not effective and len(contributions) > 1:
                providers = []
                for source, path, regions in contributions:
                    provider = source_charts.get((Path(source), (path, other.slot)))
                    if provider is not None and provider.manifest != chart.manifest:
                        approval = Supervision((0., 0.), [(r["start_sec"], r["end_sec"], (r["eff_timing"], r["eff_component"]))
                                                           for r in regions])
                        providers.append((provider, approval))
            else:
                provider = other
                if not effective and other.key == chart.key and not chart.reviewed_only and options["quality"]:
                    # Ordinary sources can share a single physical chart. A
                    # merged sidecar may retain the other source's decision
                    # even when its duplicate row was collapsed.
                    for channel in ("manual", "auto"):
                        owner = annotation_owners.get(other_key, {}).get(channel)
                        candidate = source_charts.get((Path(owner), other.key)) if owner else None
                        if candidate is not None and candidate.manifest != chart.manifest:
                            provider = candidate
                            break
                providers = [(provider, None)] if provider.manifest != chart.manifest or not options["quality"] else []
            if not providers:
                continue
            if supervision is None:
                unknown = True
                continue
            for provider, approval in providers:
                spans = _restored_spans(original, supervision, approval)
                if spans and other.metadata.get("event_usable_for_training") is not True and not parseable(other.path, other.slot):
                    warn(other, "parse_failed")
                    unknown = True
                    continue
                suppliers.extend(dict(provider.describe(), **span, same_file=chart.key == provider.key)
                                 for span in spans)
        if suppliers:
            restored.append(dict(chart.describe(), reason="other_source" if options["quality"] else "quality_disabled",
                                 original_reason=reason, suppliers=suppliers))
            restored_by_chart[(str(chart.manifest), str(chart.path), chart.slot)].extend(suppliers)
        elif unknown:
            warn(chart, "possible_restoration")
        elif not matches and quality_excludes_chart(record) and (chart.reviewed_only or chart.key in remaining_keys):
            key = str(chart.manifest), str(chart.path), chart.slot
            if key not in excluded_keys and chart.key not in options["holdout_keys"]:
                mask(chart, record, reason)
                excluded_keys.add(key)
    for row in excluded:
        row["also_restored"] = (row["manifest"], row["path"], row["slot"]) in restored_by_chart
    for row in context:
        row["also_restored"] = any(
            _overlap(row, supplier) and (
                row["timing_weight"] == 0 and "timing" in supplier["learns"]
                or row["configuration_weight"] == 0 and "configuration" in supplier["learns"]
            ) for supplier in restored_by_chart.get((row["manifest"], row["path"], row["slot"]), ())
        )
    by_slot = Counter()
    by_tier = Counter()
    source_stats = defaultdict(lambda: dict(
        included_charts=0,
        weight_sum=0.0,
        boosted_charts=0,
        standard_charts=0,
        masked_charts=0,
        downweighted_charts=0,
        excluded_charts=0,
        held_out_charts=0,
    ))
    total_weight_sum = 0.0
    for item in included:
        mechanics = item.get("learning_role") == "mechanics"
        w = 0. if mechanics else float(item["effective_weight"])
        tier = item["weight_tier"]
        src_name = item["source"]
        by_slot[str(item["slot"])] += 1
        by_tier[tier] += 1
        total_weight_sum += w
        st = source_stats[src_name]
        st["mechanics_charts"] = st.get("mechanics_charts",0) + int(mechanics)
        st["included_charts"] += 1
        st["weight_sum"] += w
        if tier == "boosted":
            st["boosted_charts"] += 1
        elif tier == "standard_full":
            st["standard_charts"] += 1
        elif tier == "partial_or_masked":
            st["masked_charts"] += 1
        elif tier == "downweighted":
            st["downweighted_charts"] += 1
    for item in excluded:
        if not item.get("also_restored"):
            source_stats[item["source"]]["excluded_charts"] += 1
    for item in held_out:
        source_stats[item["source"]]["held_out_charts"] += 1

    by_source = []
    for src_name, st in sorted(source_stats.items(), key=lambda kv: (-kv[1]["weight_sum"], -kv[1]["included_charts"], kv[0])):
        inc = st["included_charts"]
        w_sum = round(st["weight_sum"], 3)
        by_source.append(dict(
            source=src_name,
            included_charts=inc,
            weight_sum=w_sum,
            mechanics_charts=st.get("mechanics_charts",0),
            weight_share=round(st["weight_sum"] / total_weight_sum, 4) if total_weight_sum > 0 else 0.0,
            avg_weight=round(st["weight_sum"] / inc, 3) if inc > 0 else 0.0,
            boosted_charts=st["boosted_charts"],
            standard_charts=st["standard_charts"],
            masked_charts=st["masked_charts"],
            downweighted_charts=st["downweighted_charts"],
            excluded_charts=st["excluded_charts"],
            held_out_charts=st["held_out_charts"],
        ))

    inclusion_summary = dict(
        total_included_charts=len(included),
        total_weight_sum=round(total_weight_sum, 3),
        difficulty_slots=list(difficulty.effective_slots),
        slot_names=[_SLOT_NAMES.get(s, f"槽 {s}") for s in difficulty.effective_slots],
        difficulty_source=difficulty.source,
        level_range=[difficulty.min_level, difficulty.max_level],
        min_declared_level=difficulty.min_declared_level,
        by_slot=dict(sorted(by_slot.items())),
        by_weight_tier=dict(
            boosted=by_tier["boosted"],
            standard_full=by_tier["standard_full"],
            partial_or_masked=by_tier["partial_or_masked"],
            downweighted=by_tier["downweighted"],
        ),
        by_source=by_source,
        held_out_by_reason=dict(sorted(held_out_by_reason.items())),
    )
    return dict(
        included=included,
        held_out=held_out,
        inclusion_summary=inclusion_summary,
        excluded=excluded,
        context=context,
        restored=restored,
        uncertain=uncertain,
        retained_charts=retained,
        limit_uncertain=uncertain_limit,
    )


def _overlap(left, right):
    return ((left["end_sec"] is None or right["start_sec"] < left["end_sec"])
            and (right["end_sec"] is None or left["start_sec"] < right["end_sec"]))


def _parseable(path, slot, dialect):
    from maichart.composer_v2 import chart_dialect
    from maichart.event_timing import parse_event_chart
    from maichart.training.event_dataset import inspect_chart

    try:
        chart = parse_event_chart(path, slot, strict_notes=False)
        if dialect in chart_dialect.NORMALIZED_POLICIES:
            chart, _ = chart_dialect.normalize_training_chart(chart, dialect)
        inspect_chart(chart)
    except (OSError, ValueError, KeyError, UnicodeError):
        return False
    return True


@lru_cache(maxsize=8)
def _chart_bytes(path, modified_ns, size):
    content = path.read_bytes()
    return hashlib.sha256(content).hexdigest(), content


def _fitted_score(scorer, path, slot):
    stat = path.stat()
    checksum, content = _chart_bytes(path, stat.st_mtime_ns, stat.st_size)
    return scorer.score(checksum, slot, path, content)


def _manifest_content(data):
    paths = {"manifest", "raw_root", "raw_roots", "source_root", "cache_dir", "quality_annotations",
             "auto_quality_annotations"}
    comparable = {key: value for key, value in data.items() if key not in paths}
    comparable["summary"] = {key: value for key, value in data.get("summary", {}).items() if key != "supervision"}
    return comparable


def _preview_skip_reasons(result, selection):
    codes = {"stale_quality_annotation", "quality_audio_changed"}
    existing = {(row["path"], row["slot"]) for row in result["uncertain"]
                if row["reason"] in codes}
    for warning in selection.get("warnings", ()):
        key = warning.get("source_path"), warning.get("difficulty_index")
        if warning.get("reason") in codes and key not in existing:
            result["uncertain"].append(dict(
                manifest=warning.get("source_manifest", ""), source=warning.get("source_manifest", ""),
                path=warning["source_path"], title=Path(warning["source_path"]).parent.name,
                slot=warning.get("difficulty_index"), reason=warning["reason"],
            ))
            existing.add(key)


def inspect_training_sources(payload, runtime):
    from maichart.composer_v2 import chart_dialect
    from maichart.composer_v2.train import resolve_zero_unmarked_policy
    from .composer_api import _path
    from .composer_v2_api import _choice, _integer, inspection

    settings = payload["training_source_preview"]
    if not isinstance(settings, dict):
        raise ValueError("开训前预览设置须为对象")
    mode = _choice(settings, "mode", "fresh", {"fresh", "finetune", "resume"})
    resume = mode == "resume"
    saved, parent = {}, {}
    checked = {}
    sequence = "continuous_events_v1"
    if mode != "fresh":
        checked = inspection(_path(settings.get("init_checkpoint"), "训练检查点", exists=True))
        if not checked.get("valid") or not checked.get("can_resume" if resume else "can_init"):
            raise ValueError("检查点不支持所选训练方式：" + str(checked.get("errors") or checked.get("error")))
        saved, parent = checked.get("args") or {}, checked.get("training_contract") or {}
        sequence = checked.get("sequence_policy", "window_v1")

    quality = settings.get("quality_enabled", saved.get("quality_enabled", True) if resume else True)
    if not isinstance(quality, bool):
        raise ValueError("质量加权必须是布尔值")
    policy = _choice(settings, "auto_quality_policy", "auto", {"auto", "on", "off"})
    difficulty = resolve_selection(settings.get("difficulty_source", "auto"), settings.get("min_level"),
                                   settings.get("max_level"), min_declared_level=settings.get("min_declared_level"),
                                   difficulty_slots=settings.get("difficulty_slots"), parent=parent, resume=resume)
    limit = _integer(settings, "limit_charts", saved.get("limit_charts", 0) if resume else 0)
    legacy = resolve_zero_unmarked_policy(parent, resume=resume) == "legacy_weights_v1"
    dialect = parent.get("chart_dialect", chart_dialect.LEGACY_STRICT_POLICY) if resume else chart_dialect.POLICY
    cache_dir = _path(settings.get("cache_dir") or "cache/chart_composer_v2", "训练缓存目录")
    digest = _CachedReadOnlyDigests(cache_dir)
    try:
        with TemporaryDirectory(prefix="maichart_training_preview_") as folder:
            return _inspect_selection(payload, runtime, Path(folder), settings, options=dict(
                quality=quality, policy=policy, difficulty=difficulty, sequence=sequence, legacy=legacy,
                limit=limit, mode=mode, parent=parent, saved=saved, cache_dir=cache_dir, dialect=dialect,
                saved_identity=((checked.get("data_contract") or {}).get("train") or {}) if resume else {},
            ), digest=digest)
    finally:
        digest.close()


def _inspect_selection(payload, runtime, folder, settings, options, digest):
    from maichart.composer.evaluation import assert_unseen, file_hash, split_manifest
    from maichart.composer_v2.difficulty import locate_scorer, make_scorer, scorer_path_key
    from maichart.composer_v2.train import resolve_auto_quality_policy
    from .composer_api import _path
    from .composer_v2_api import (_auto_quality_scope_summary, _choice, _float, _integer,
                                  _manifest_paths, _prepare_scoped_resume, _requested_auto_quality_scope)

    mode = options["mode"]
    resumed = mode == "resume"
    request = {**payload, "cache_dir": str(options["cache_dir"])}
    _requested_auto_quality_scope(request, policy=options['policy'])
    frozen = _prepare_scoped_resume(request, options['saved'], policy=options['policy']) if resumed else None
    if frozen is None:
        prepared_payload, selection = runtime._prepare_training_sources(
            request, preview_root=folder, preview_audio_digest=digest.digest)
    else:
        prepared_payload, selection = frozen
    if not prepared_payload.get("manifest"):
        raise ValueError("请先选择训练清单")
    source_manifest = runtime._resolve_path(prepared_payload["manifest"])
    pending = runtime._manifest_pending_status(source_manifest)
    if pending is not None:
        raise ValueError(pending["message"])
    data = runtime._read_training_manifest(source_manifest, prepared_payload.get("raw_root"))
    known_manifests = {source_manifest: data}
    resolved_paths = {}

    def read_manifest(path):
        if path not in known_manifests:
            known_manifests[path] = runtime._read_training_manifest(path)
        return known_manifests[path]

    def resolve_path(value):
        if value not in resolved_paths:
            path = material_path(value).resolve()
            resolved_paths[value] = path
            resolved_paths.setdefault(str(path), path)
        return resolved_paths[value]

    @cache
    def resolve_asset(value, owner):
        raw = material_path(value)
        if raw.is_absolute():
            return resolve_path(value)
        path = _source_asset(raw, owner, read_manifest(owner))
        resolved_paths.setdefault(str(path), path)
        return path

    @cache
    def read_charts(path):
        return list(_charts(path, read_manifest(path), resolve_asset))
    published = Path((selection or {}).get("published_manifest", source_manifest))
    composition = selection or {}
    selection = {key: value for key, value in composition.items() if key not in {
        "songs", "warnings", "quality_annotations", "auto_quality_annotations", "raw_root", "raw_roots",
        "source_root", "cache_dir", "published_manifest", "preview_contributors", "preview_annotation_owners",
    }} or None
    if selection is not None:
        selection["manifest"] = str(published)
        selection["preview_only"] = published != source_manifest

    # A direct formal audit on an existing composition takes precedence over
    # inherited sidecars. Only use it if it still binds to that composition.
    if published != source_manifest and published.is_file() and not resumed and 'auto_quality_manifests' not in data:
        live_data = read_manifest(published)
        live_auto, auto = resolve_auto_annotations(published, live_data)
        if (live_auto is not None and live_auto.name == "auto_quality_annotations.json"
                and _manifest_content(live_data) == _manifest_content(data)):
            from maichart.training.quality_annotations import save_auto_annotations
            saved_metadata = dict(auto.get("audit_metadata") or {})
            saved_metadata["manifest_sha256"] = file_hash(source_manifest)
            save_auto_annotations(source_manifest, auto["records"], audit_metadata=saved_metadata)

    effective = source_manifest
    effective_data = data
    difficulty = options["difficulty"]
    validation_source = _choice(settings, "validation_source", "manual" if settings.get("validation_manifest") else "auto",
                                {"auto", "off", "manual"})
    if resumed:
        saved_manifest = options["saved"].get("manifest")
        if frozen is not None and saved_manifest:
            saved_manifest = material_path(saved_manifest)
        effective = _path(saved_manifest, "父 checkpoint 训练清单", exists=True)
        effective_data = read_manifest(effective)
        split = effective_data.get("split_metadata", {})
        accepted = {effective}
        if split.get("source_manifest"):
            accepted.add(resolve_path(split["source_manifest"]) if frozen is not None else runtime._resolve_path(split["source_manifest"]))
        if published not in accepted:
            raise ValueError("续训请检查父 checkpoint 的训练清单或其原始划分来源")
        expected = options["saved_identity"].get("manifest_sha256")
        if published != effective and split.get("source_sha256") and split["source_sha256"] != file_hash(published):
            raise ValueError("原始划分来源已变化，请核对来源或使用新目录微调")
        if expected and expected != file_hash(effective):
            raise ValueError("父 checkpoint 的训练清单已变化，请核对来源或使用新目录微调")
        validation_source = "inherited"
    elif validation_source == "auto":
        fraction = _float(settings, "holdout_fraction", .05, minimum=0)
        if fraction and data.get("songs"):
            eligibility = (difficulty.split_exclusion
                           if difficulty.min_declared_level or difficulty.difficulty_slots != LEGACY_DIFFICULTY_SLOTS else None)
            split = split_manifest(source_manifest, folder / "split", fraction=fraction,
                                   seed=_integer(settings, "seed", 0), chart_eligibility=eligibility,
                                   audio_digest=digest.digest)
            effective = Path(split["splits"]["train"]["path"])
            effective_data = read_manifest(effective)
        else:
            validation_source = "off"
    elif validation_source == "manual":
        validation = _path(settings.get("validation_manifest"), "手动验证清单", exists=True)
        assert_unseen(source_manifest, validation, audio_digest=digest.digest)

    use_auto = resolve_auto_quality_policy(
        options["policy"], effective, options["parent"], resume=resumed,
        quality_enabled=options["quality"], sequence_policy=options["sequence"],
    ) == "auto_v1"
    options["use_auto"] = use_auto
    options['auto_quality_manifests'] = _manifest_paths(effective_data.get('auto_quality_manifests'))
    options["pre_split_charts"] = read_charts(source_manifest)
    options["holdout_keys"] = ({chart.key for chart in read_charts(source_manifest)} - {chart.key for chart in read_charts(effective)}
                               if source_manifest != effective else set())
    notes = []
    if difficulty.source in FITTED_SOURCES:
        key = scorer_path_key(difficulty.source)
        root = locate_scorer(difficulty.source, options["cache_dir"], settings.get(key), options["saved"].get(key) if resumed else None)
        options["scorer"] = make_scorer(difficulty.source, root, options["cache_dir"], read_only=True)
        notes.append("估分沿用已有缓存；未缓存的谱面列入待确认。")
    if resumed:
        notes.append("续训按父检查点清单和策略预览；启动时仍会校验完整续训合同。")
    if options["limit"]:
        notes.append(f"按清单顺序考虑前 {options['limit']} 张可用谱；实际解析或音频准备失败可能改变入选边界。")
    if validation_source == "auto":
        notes.append("已按当前种子和比例模拟自动验证划分，验证及终测留出的谱面不计入训练。")
    if options["sequence"] == "window_v1":
        notes.append("旧窗口训练使用片段总权重，采音与配置共用该权重。")
    if not data.get("songs"):
        notes.append("本次合并没有可供训练的谱面。")
    if not options["quality"]:
        notes.append("质量标注已关闭，人工排除和局部屏蔽不会生效。")
    notes.append("预览显示质量标注控制的监督范围。谱面与音频准备、词表和最终训练窗口由开训时确认。")
    manifests = list(dict.fromkeys(runtime._resolve_path(value) for value in prepared_payload.get("manifests") or [published]))
    original_sources = [runtime._resolve_path(path) for path in data.get("merged_from", ())]
    if len(manifests) == 1 and original_sources:
        manifests = original_sources
    sources = [(path, read_manifest(path)) for path in manifests if path.is_file()]
    if not sources:
        sources = [(effective, effective_data)]
    source_paths = manifests
    if resumed:
        origin = split_annotation_source(effective, effective_data)
        origin_data = read_manifest(origin) if origin is not None and origin.is_file() else effective_data
        original_sources = [resolve_path(path) for path in origin_data.get("merged_from", ())]
        source_paths = original_sources or ([origin] if origin is not None else [effective])
        sources = [(path, read_manifest(path)) for path in original_sources if path.is_file()]
        sources = sources or [(origin or effective, origin_data)]
        notes.append("来源对照读取当前原库，续训采用父检查点的训练清单。")
    missing = [path for path in source_paths if not path.is_file()]
    if missing:
        notes.append("部分原始来源清单无法读取，无法完整核对重新纳入。")
    preview = _quality_rows(effective, effective_data, sources, options, digest.digest,
                            composition, read_charts=read_charts,
                            resolve_path=resolve_path, resolve_asset=resolve_asset)
    _preview_skip_reasons(preview, composition)
    preview["uncertain"].extend(dict(manifest=str(path), source=path.parent.name, path=str(path),
                                      title=path.parent.name, reason="missing_source") for path in missing)
    if preview["limit_uncertain"]:
        notes.append("有谱面尚未估分，之后的入选顺序和来源对照待确认。")
    preview.update(quality_enabled=options["quality"], auto_quality_enabled=use_auto,
                   auto_quality_scope=_auto_quality_scope_summary(effective,effective_data,use_auto),
                   mode=mode, notes=notes, held_out_charts=len(options["holdout_keys"]))
    display_manifest = effective if resumed else published
    return {**prepared_payload, "manifest": str(display_manifest)}, selection, data, preview
