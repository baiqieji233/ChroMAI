"""Quality-weighted and corrected-chart preference training for event models.

This module deliberately stays beside the existing event datasets.  A corrected
chart is parsed only when a negative annotation has a ``corrected_path``; the
audio, window geometry, condition and context all come from the existing
dataset item.  Thus a preference pair never needs a second full dataset.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from maichart.event_timing import parse_event_chart, seconds_at_beat
from maichart.generation.event_model import EventSequence
from maichart.generation.sustain_state import action_state_kwargs
from maichart.training.event_dataset import collate_events, inspect_chart, sanitize_chart


@dataclass
class PreferencePair:
    """One corrected (preferred) and original (rejected) shared-context pair."""

    chosen: dict[str, Any]
    rejected: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)
    reference: tuple[float, float] | None = None

    @property
    def key(self) -> str:
        payload = {
            "meta": self.meta,
            "chosen": self.chosen.get("target", torch.empty(0, dtype=torch.long)).detach().cpu().tolist(),
            "rejected": self.rejected.get("target", torch.empty(0, dtype=torch.long)).detach().cpu().tolist(),
        }
        profile = self.meta.get("action_state_profile", "none")
        action_state_present = any("action_state" in item for item in (self.chosen, self.rejected))
        if profile != "none" or action_state_present:
            payload["action_state_profile"] = profile
            payload["action_inputs"] = {
                role: {
                    name: (example[name].detach().cpu().tolist()
                           if isinstance(example.get(name), Tensor) else None)
                    for name in ("action_state", "action_state_padding_mask")
                }
                for role, example in (("chosen", self.chosen), ("rejected", self.rejected))
            }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _records(annotations: Any) -> list[dict[str, Any]]:
    if annotations is None:
        return []
    if isinstance(annotations, (str, Path)):
        data = json.loads(Path(annotations).read_text(encoding="utf-8"))
        annotations = data.get("records", data) if isinstance(data, Mapping) else data
    if isinstance(annotations, Mapping):
        annotations = annotations.get("records", [])
    return [dict(item) for item in annotations if isinstance(item, Mapping)]


def _resolve(value: Any, base: Path) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _same_path(a: Any, b: Any) -> bool:
    try:
        return Path(str(a)).resolve() == Path(str(b)).resolve()
    except (TypeError, ValueError, OSError):
        return str(a) == str(b)


def _sample_record(dataset: Any, ref: Any, records: list[dict[str, Any]]) -> dict[str, Any] | None:
    resolver = getattr(dataset, "quality_record", None)
    if callable(resolver):
        # The dataset resolver is content-aware (chart/audio hashes).  Once a
        # dataset exposes it, a miss means the stored annotation is stale or
        # mismatched; never resurrect it by falling back to path matching.
        try:
            resolved = resolver(ref)
        except (TypeError, ValueError, OSError):
            resolved = None
        if isinstance(resolved, Mapping) and resolved.get("label") == "negative" and resolved.get("corrected_path"):
            return dict(resolved)
        return None
    source = Path(ref.source_path).resolve()
    difficulty = int(getattr(ref.chart, "difficulty", getattr(ref, "difficulty_index", 0)))
    for record in records:
        if record.get("label") != "negative" or not record.get("corrected_path"):
            continue
        if record.get("difficulty_index") is not None and int(record["difficulty_index"]) != difficulty:
            continue
        record_source = record.get("source_path")
        if record_source:
            candidate = Path(str(record_source))
            if not candidate.is_absolute():
                candidate = source.parent / candidate
            if _same_path(candidate, source):
                return record
        # chart_id is optional in hand-created/test records; source and slot are
        # the safe fallback when the catalog generated it.
        if record.get("chart_id") and getattr(ref, "song_id", None) == record.get("song_id"):
            return record
    return None


def _tempo_signature(chart: Any) -> list[tuple[Fraction, float]]:
    return [(Fraction(change.beat), float(change.bpm)) for change in getattr(chart, "tempo_changes", [])]


def _validate_corrected(original: Any, corrected: Any, *, tolerance: float = 1e-3) -> str | None:
    if abs(float(original.audio_first) - float(corrected.audio_first)) > tolerance:
        return "corrected chart changes first/audio_first"
    left, right = _tempo_signature(original), _tempo_signature(corrected)
    if len(left) != len(right) or any(a != b and (a[0] != b[0] or abs(a[1] - b[1]) > tolerance)
                                      for a, b in zip(left, right)):
        return "corrected chart changes tempo map"
    # A corrected note layout may move the last note slightly, but a different
    # song or a truncated document should never silently become a pair.
    original_end = float(seconds_at_beat(original.end_beat, original.tempo_changes, original.audio_first))
    corrected_end = float(seconds_at_beat(corrected.end_beat, corrected.tempo_changes, corrected.audio_first))
    if abs(original_end - corrected_end) > max(0.25, tolerance * 100):
        return "corrected chart end differs too much"
    return None


def _region_hit(chart: Any, start: Fraction, end: Fraction, regions: list[Mapping[str, Any]]) -> bool:
    if not regions:
        return True
    begin = float(seconds_at_beat(start, chart.tempo_changes, chart.audio_first))
    finish = float(seconds_at_beat(end, chart.tempo_changes, chart.audio_first))
    return any(begin < float(region.get("end_sec", 0)) and finish > float(region.get("start_sec", 0))
               for region in regions)


def _window_info(dataset: Any, index: int) -> tuple[int, Fraction, Fraction, list[int], list[int] | None, int]:
    entry = dataset.segments[index]
    sample_index, start, end, tokens = entry[:4]
    prefix = None
    prefix_len = 0
    if len(entry) >= 7:  # EventCtxDataset: (..., window_tokens, prefix, prefix_len, offset)
        prefix = list(entry[4])
        prefix_len = int(entry[5])
    return int(sample_index), Fraction(start), Fraction(end), list(tokens), prefix, prefix_len


def _synthetic_segment(dataset: Any, sample_index: int, start: Fraction, end: Fraction) -> tuple[Any, ...]:
    """Recreate one candidate tuple when quality filtering removed its segment."""
    ref = dataset.samples[sample_index]
    if hasattr(dataset, "context_beats"):
        from maichart.generation.event_ctx import context_start_beat, lead_in_events, prefix_token_ids
        block_start = context_start_beat(start, int(dataset.context_beats))
        points = [point for point in ref.chart.points if start <= point.beat < end]
        tokens = dataset.vocabulary.encode(EventSequence(
            [point.beat - block_start for point in points], [point.note_text for point in points]))
        events = lead_in_events(ref.chart.points, start, int(dataset.context_beats))
        event_weight = getattr(dataset, "_quality_event_weight", None)
        if callable(event_weight):
            events = [event for event in events if event_weight(sample_index, block_start + event[0]) > 0.0]
        policy = getattr(dataset, "context_budget_policy", "legacy_dynamic_v1")
        budget = int(dataset.context_tokens)
        if policy == "legacy_dynamic_v1":
            budget = max(0, min(budget, int(dataset.max_tokens) - len(tokens) - 2))
        prefix = prefix_token_ids(dataset.vocabulary, events, budget)
        if (policy == "bounded_v2"
                and len(tokens) + len(prefix) > int(dataset.max_tokens)):
            raise ValueError(
                f"fixed context preference window has {len(tokens) + len(prefix)} tokens; "
                f"max_tokens={int(dataset.max_tokens)}"
            )
        return (sample_index, start, end, tokens, prefix, len(prefix), start - block_start)
    points = [point for point in ref.chart.points if start <= point.beat < end]
    tokens = dataset.vocabulary.encode(EventSequence(
        [point.beat - start for point in points], [point.note_text for point in points]))
    return (sample_index, start, end, tokens)


def _materialize(dataset: Any, entry: tuple[Any, ...]) -> dict[str, Any]:
    """Use the dataset's own audio/condition implementation for a synthetic window."""
    original = dataset.segments
    arrays = [getattr(dataset, name, None) for name in (
        "segment_quality_weights", "segment_quality_token_weights", "segment_quality_loss_starts",
    )]
    source_entries = getattr(dataset, "preference_segments", original)
    source_weights = getattr(dataset, "preference_segment_quality_weights", arrays[0] or [])
    source_token_weights = getattr(dataset, "preference_segment_quality_token_weights", arrays[1] or [])
    source_loss_starts = getattr(dataset, "preference_segment_quality_loss_starts", arrays[2] or [])
    try:
        source_index = next(i for i, candidate in enumerate(source_entries) if candidate is entry or candidate == entry)
    except StopIteration:
        source_index = -1
    dataset.segments = list(original) + [entry]
    if arrays[0] is not None:
        dataset.segment_quality_weights = list(arrays[0]) + [source_weights[source_index] if 0 <= source_index < len(source_weights) else 1.0]
    if arrays[1] is not None:
        dataset.segment_quality_token_weights = list(arrays[1]) + [source_token_weights[source_index] if 0 <= source_index < len(source_token_weights) else []]
    if arrays[2] is not None:
        dataset.segment_quality_loss_starts = list(arrays[2]) + [source_loss_starts[source_index] if 0 <= source_index < len(source_loss_starts) else 0]
    try:
        return dataset[len(dataset.segments) - 1]
    finally:
        dataset.segments = original
        for name, value in zip(("segment_quality_weights", "segment_quality_token_weights",
                                "segment_quality_loss_starts"), arrays):
            if value is not None:
                setattr(dataset, name, value)


def _encode_window(dataset: Any, chart: Any, start: Fraction, end: Fraction,
                   *, prefix: list[int] | None = None, prefix_len: int = 0,
                   block_start: Fraction | None = None) -> tuple[Tensor, Tensor, Tensor]:
    # Context windows use the same block origin as the existing dataset entry.
    block_start = start if block_start is None else Fraction(block_start)
    selected = [point for point in chart.points if start <= point.beat < end]
    sequence = EventSequence(
        beats=[point.beat - block_start for point in selected],
        note_texts=[point.note_text for point in selected],
    )
    window_tokens = dataset.vocabulary.encode(sequence)
    if prefix is None:
        ids = window_tokens
        mask = torch.ones(len(ids) - 1, dtype=torch.bool)
    else:
        ids = [dataset.vocabulary.bos_id] + list(prefix) + list(window_tokens[1:])
        # prefix_len includes context delimiters, exactly as loss_mask_after.
        mask = torch.zeros(len(ids) - 1, dtype=torch.bool)
        mask[prefix_len:] = True
    max_tokens = getattr(dataset, "max_tokens", None)
    if max_tokens is not None and len(ids) > int(max_tokens):
        raise ValueError(
            f"corrected preference window has {len(ids)} tokens; max_tokens={int(max_tokens)}"
        )
    tokens = torch.tensor(ids, dtype=torch.long)
    return tokens[:-1], tokens[1:], mask


def build_preference_pairs(
    dataset: Any,
    annotations: Any,
    *,
    max_pairs: int | None = None,
    seed: int = 0,
) -> tuple[list[PreferencePair], dict[str, Any]]:
    """Build corrected-vs-original pairs from an existing event dataset.

    Records without a correction are intentionally absent from preference
    training. Dataset quality policy already excludes those targets from SFT.
    """
    records = _records(annotations)
    pairs: list[PreferencePair] = []
    audit: dict[str, Any] = {"records": len(records), "windows": 0, "pairs": 0,
                             "skipped": [], "chart_filter_audit": [], "negative_without_correction": 0}
    parsed: dict[str, Any] = {}
    source_segments = list(getattr(dataset, "preference_segments", dataset.segments))
    negative_by_sample: dict[int, dict[str, Any]] = {}
    for sample_index, ref in enumerate(getattr(dataset, "samples", [])):
        record = _sample_record(dataset, ref, records)
        if record and record.get("label") == "negative" and record.get("corrected_path"):
            negative_by_sample[sample_index] = record
    candidates: list[tuple[int, tuple[Any, ...], int, Fraction, Fraction, dict[str, Any]]] = []
    for segment_index, entry in enumerate(source_segments):
        sample_index, start, end, _, _, _ = _window_info(SimpleNamespace(segments=[entry]), 0)
        record = negative_by_sample.get(sample_index)
        if record is None:
            continue
        ref = dataset.samples[sample_index]
        if not _region_hit(ref.chart, start, end, list(record.get("regions") or [])):
            continue
        candidates.append((segment_index, entry, sample_index, start, end, record))
    audit["candidate_windows"] = len(candidates)
    if max_pairs is not None and len(candidates) > int(max_pairs):
        rng = random.Random(seed)
        # Sample round-robin across charts so one long rejected chart cannot
        # consume the entire per-epoch preference budget.
        grouped: dict[str, list[tuple[int, tuple[Any, ...], int, Fraction, Fraction, dict[str, Any]]]] = {}
        for candidate in candidates:
            grouped.setdefault(str(candidate[5].get("chart_id") or candidate[5].get("song_id")), []).append(candidate)
        keys = list(grouped)
        rng.shuffle(keys)
        for values in grouped.values():
            rng.shuffle(values)
        selected = []
        while keys and len(selected) < int(max_pairs):
            remaining = []
            for key in keys:
                values = grouped[key]
                if values and len(selected) < int(max_pairs):
                    selected.append(values.pop())
                if values:
                    remaining.append(key)
            keys = remaining
        candidates = selected
    audit["selected_windows"] = len(candidates)
    for index, segment_entry, sample_index, start, end, record in candidates:
        block_start = start
        if len(segment_entry) >= 7:
            block_start = start - Fraction(segment_entry[6])
        prefix = list(segment_entry[4]) if len(segment_entry) >= 7 else None
        prefix_len = int(segment_entry[5]) if len(segment_entry) >= 7 else 0
        ref = dataset.samples[sample_index]
        audit["windows"] += 1
        corrected_path = _resolve(record["corrected_path"], Path(record.get("source_path", ref.source_path)).parent)
        key = str(corrected_path)
        try:
            corrected_difficulty = int(record.get("corrected_difficulty", record.get("difficulty_index", ref.chart.difficulty)))
            expected_hash = record.get("corrected_sha256")
            if expected_hash:
                from maichart.training.quality_annotations import chart_sha256
                actual_hash = chart_sha256(corrected_path, corrected_difficulty)
                if str(expected_hash) != actual_hash:
                    raise ValueError("corrected chart hash mismatch")
            if key not in parsed:
                corrected = parse_event_chart(
                    corrected_path, corrected_difficulty,
                    strict_notes=False)
                sanitized = sanitize_chart(
                    corrected, getattr(dataset, "chart_filter_policy", "legacy_strict_v1")
                )
                corrected = sanitized.chart
                if sanitized.removed:
                    audit["chart_filter_audit"].append({"source_path": str(corrected_path),
                        "difficulty_index": corrected_difficulty, **sanitized.to_dict()})
                    preview = ", ".join(f"{item['beat']}:{item['atom']}" for item in sanitized.removed[:3])
                    print(f"[chart_filter] preference={corrected_path} difficulty={corrected_difficulty} "
                          f"removed={len(sanitized.removed)} sample=[{preview}]", flush=True)
                inspect_chart(corrected)
                parsed[key] = corrected
            corrected = parsed[key]
            reason = _validate_corrected(ref.chart, corrected)
            if reason:
                raise ValueError(reason)
            chosen_input, chosen_target, chosen_mask = _encode_window(
                dataset, corrected, start, end, prefix=prefix, prefix_len=prefix_len,
                block_start=block_start)
            # Parse, validate and encode the correction before loading audio.
            # A stale path, changed tempo map or out-of-vocabulary note should
            # fail cheaply instead of decoding audio for an unusable pair.
            rejected = copy.deepcopy(_materialize(dataset, segment_entry))
            # Reuse the original causal prefix and action-state snapshot for both
            # candidates; deriving either from the corrected future leaks its edits.
            chosen = copy.deepcopy(rejected)
            chosen.update({"decoder_input": chosen_input, "target": chosen_target, "loss_mask": chosen_mask})
            # A pair must share the original audio, global memory and external
            # condition.  Keep those fields from the dataset item verbatim.
            meta = dict(rejected.get("meta", {}))
            meta.update({"preference": True, "source_index": index, "chart_id": record.get("chart_id"),
                         "label": "negative", "corrected_path": str(corrected_path)})
            action_state_profile = getattr(dataset, "action_state_profile", "none")
            if action_state_profile != "none":
                meta["action_state_profile"] = action_state_profile
            input_policies = {
                "context_budget_policy": getattr(dataset, "context_budget_policy", "legacy_dynamic_v1"),
                "chart_filter_policy": getattr(dataset, "chart_filter_policy", "legacy_strict_v1"),
            }
            legacy_policies = {"context_budget_policy": "legacy_dynamic_v1",
                               "chart_filter_policy": "legacy_strict_v1"}
            if input_policies != legacy_policies:
                meta["input_policies"] = input_policies
            pairs.append(PreferencePair(chosen=chosen, rejected=rejected, meta=meta))
        except (OSError, ValueError, UnicodeError) as exc:
            audit["skipped"].append({"index": index, "song_id": ref.song_id,
                                     "start_beat": str(start), "reason": str(exc)})
    # Negative records without corrections are reported, but never become
    # preference pairs or silently enter the preference objective.
    for record in records:
        if record.get("label") == "negative" and not record.get("corrected_path"):
            audit["negative_without_correction"] += 1
    audit["pairs"] = len(pairs)
    return pairs, audit


def _masked_logprob(logits: Tensor, target: Tensor, mask: Tensor, padding: Tensor | None = None) -> tuple[Tensor, int]:
    logp = F.log_softmax(logits.float(), dim=-1)
    token_logp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    valid = mask.to(device=logits.device, dtype=torch.bool)
    if padding is not None:
        valid = valid & ~padding.to(device=logits.device, dtype=torch.bool)
    count = valid.sum(dim=-1).clamp_min(1)
    # Average over valid tokens so preference scores do not automatically
    # punish longer Simai encodings (for example, compound Slide syntax).
    return (token_logp * valid).sum(dim=-1) / count, int(valid.sum().item())


def score_pair_logits(logits: Tensor, batch: Mapping[str, Tensor]) -> tuple[Tensor, dict[str, int]]:
    """Return per-example mean teacher-forced log probabilities."""
    scores, tokens = _masked_logprob(logits, batch["target"], batch["loss_mask"], batch.get("decoder_padding_mask"))
    return scores, {"tokens": tokens}


def precompute_reference(
    model: torch.nn.Module,
    pairs: list[PreferencePair],
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 2,
    dataset: Any | None = None,
) -> dict[int, tuple[float, float]]:
    """Compute frozen reference scores once; returned values are CPU floats."""
    if not pairs:
        return {}
    model_was_training = model.training
    model.eval()
    result: dict[int, tuple[float, float]] = {}
    with torch.no_grad():
        for begin in range(0, len(pairs), max(1, int(batch_size))):
            chunk = pairs[begin:begin + max(1, int(batch_size))]
            examples = [item for pair in chunk for item in (pair.chosen, pair.rejected)]
            batch = collate_events(examples)
            moved = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}
            logits = model(moved["audio"], moved["decoder_input"], moved["condition"],
                           condition_null=moved.get("condition_null"),
                           plan_audio_mask=moved.get("plan_audio_mask"),
                           audio_padding_mask=moved.get("audio_padding_mask"),
                           decoder_padding_mask=moved.get("decoder_padding_mask"),
                           global_audio=moved.get("global_audio"),
                           **action_state_kwargs(moved, batched=True))
            scores, _ = score_pair_logits(logits, moved)
            values = scores.detach().cpu().tolist()
            for offset, pair in enumerate(chunk):
                pair.reference = (float(values[offset * 2]), float(values[offset * 2 + 1]))
                result[begin + offset] = pair.reference
    if model_was_training:
        model.train()
    return result


def precompute_reference_scores(
    model: torch.nn.Module,
    pairs: list[PreferencePair],
    dataset: Any | None = None,
    device: torch.device | str = "cpu",
    *,
    batch_size: int = 2,
) -> dict[str, dict[str, float]]:
    """JSON-serializable frozen scores keyed by a stable pair digest."""
    precompute_reference(model, pairs, device=device, batch_size=batch_size, dataset=dataset)
    return {pair.key: {"chosen": float(pair.reference[0]), "rejected": float(pair.reference[1])}
            for pair in pairs if pair.reference is not None}


def _head_count(target: Tensor, vocabulary: Any, mask: Tensor) -> int:
    ids = target.detach().cpu().tolist()
    valid = mask.detach().cpu().tolist()
    return sum(bool(ok and str(vocabulary.id_to_token.get(int(token), "")).startswith("AT:"))
               for token, ok in zip(ids, valid))


def preference_epoch(
    model: torch.nn.Module,
    pairs: list[PreferencePair],
    dataset: Any,
    device: torch.device | str,
    optimizer: torch.optim.Optimizer | None,
    scaler: Any | None = None,
    *,
    max_pairs: int = 32,
    beta: float = 0.1,
    weight: float = 0.05,
    reference_model: torch.nn.Module | None = None,
    cached_reference: Mapping[int, tuple[float, float]] | None = None,
    reference_scores: Mapping[Any, Any] | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """Run a small DPO-plus-chosen-SFT pass and return auditable metrics."""
    if not pairs or max_pairs <= 0:
        return {"pairs": 0.0, "loss": 0.0, "win_rate": 0.0, "tokens": 0.0, "heads": 0.0,
                "chosen_length": 0.0, "rejected_length": 0.0}
    chosen_pairs = list(pairs)
    if len(chosen_pairs) > max_pairs:
        chosen_pairs = random.Random(seed).sample(chosen_pairs, int(max_pairs))
    if cached_reference is None and reference_model is not None:
        cached_reference = precompute_reference(reference_model, chosen_pairs, device=device)
    model.train()
    total_loss = total_tokens = total_heads = wins = 0.0
    chosen_lengths = rejected_lengths = 0.0
    action_optimizer_steps = 0
    def count_action_step(_optimizer, _args, _kwargs):
        nonlocal action_optimizer_steps
        action_optimizer_steps += 1
    for index, pair in enumerate(chosen_pairs):
        batch = collate_events([pair.chosen, pair.rejected])
        moved = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}
        logits = model(moved["audio"], moved["decoder_input"], moved["condition"],
                       condition_null=moved.get("condition_null"), audio_padding_mask=moved.get("audio_padding_mask"),
                       plan_audio_mask=moved.get("plan_audio_mask"),
                       decoder_padding_mask=moved.get("decoder_padding_mask"), global_audio=moved.get("global_audio"),
                       **action_state_kwargs(moved, batched=True))
        policy, stats = score_pair_logits(logits, moved)
        if pair.reference is not None:
            ref_chosen, ref_rejected = pair.reference
        elif reference_scores is not None and pair.key in reference_scores:
            value = reference_scores[pair.key]
            if isinstance(value, Mapping):
                ref_chosen, ref_rejected = float(value["chosen"]), float(value["rejected"])
            else:
                ref_chosen, ref_rejected = map(float, value)
        elif cached_reference and index in cached_reference:
            ref_chosen, ref_rejected = cached_reference[index]
        else:
            raise ValueError("preference pair has no frozen reference score; call precompute_reference first")
        margin = (policy[0] - policy[1]) - (policy.new_tensor(ref_chosen) - policy.new_tensor(ref_rejected))
        dpo = -F.logsigmoid(float(beta) * margin)
        chosen_nll = -policy[0]
        loss = float(weight) * dpo + 0.1 * chosen_nll
        if optimizer is not None:
            active = ("action_state_padding_mask" in moved
                      and bool((~moved["action_state_padding_mask"]).any()))
            hook = optimizer.register_step_post_hook(count_action_step) if active else None
            try:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            finally:
                if hook is not None:
                    hook.remove()
        total_loss += float(loss.detach().cpu())
        total_tokens += stats["tokens"]
        total_heads += _head_count(moved["target"][0], dataset.vocabulary, moved["loss_mask"][0])
        chosen_lengths += float(moved["loss_mask"][0].sum().item())
        rejected_lengths += float(moved["loss_mask"][1].sum().item())
        wins += float((margin.detach() > 0).item())
    count = float(len(chosen_pairs))
    return {"pairs": count, "loss": total_loss / count, "win_rate": wins / count,
            "tokens": total_tokens / count, "heads": total_heads / count,
            "action_state_optimizer_steps_successful": action_optimizer_steps,
            "chosen_length": chosen_lengths / count, "rejected_length": rejected_lengths / count}


def run_preference_epoch(*args: Any, reference_scores: Mapping[Any, Any] | None = None,
                         **kwargs: Any) -> dict[str, float]:
    """Naming-compatible entry point used by the event training loop."""
    if reference_scores is not None:
        kwargs["reference_scores"] = reference_scores
    return preference_epoch(*args, **kwargs)


def score_pair(logits: Tensor, batch_or_target: Mapping[str, Tensor] | Tensor,
               mask: Tensor | None = None) -> Tensor | tuple[Tensor, dict[str, int]]:
    """Score teacher-forced tokens; accepts a collated batch or tensors directly."""
    if isinstance(batch_or_target, Mapping):
        return score_pair_logits(logits, batch_or_target)
    if mask is None:
        raise ValueError("score_pair(logits, target, mask) requires a mask")
    scores, _ = _masked_logprob(logits, batch_or_target, mask)
    return scores

__all__ = ["PreferencePair", "build_preference_pairs", "score_pair", "score_pair_logits",
           "precompute_reference", "precompute_reference_scores", "preference_epoch",
           "run_preference_epoch"]
