"""Generate an exact-event Simai chart with an event_v1 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import torch

from maichart.event_timing import EventPoint, EventTempo, seconds_at_beat
from maichart.generation.acceptance import (
    audio_follow_metrics,
    evaluate,
    load_reference,
    metric_bundle,
    structural_metrics,
    timeline_series,
)
from maichart.generation.conditions import (
    GENERATION_DEFAULTS,
    GENERATION_SAMPLING_DEFAULTS,
    GenerationCondition,
)
from maichart.generation.event_audio import (
    GLOBAL_MEMORY_FRAMES,
    LEGACY_INPUT_DIM,
    audio_segment_tensor,
    compress_event_audio,
    extract_event_audio,
    legacy_feature_view,
    _time_to_beat, _normalize_tempo_map,
)
from maichart.generation.music_plan import (
    MUSIC_STRUCTURE_V3_PROFILE, is_music_structure,
)
from maichart.generation.model_compatibility import (
    checkpoint_capabilities, detect_model_family, get_generation_config_for_model, validate_args_for_model
)
from maichart.generation.post_process import (
    PostProcessConfig, detect_chart_conflicts, process_chart_points,
)
from maichart.generation.runtime_reference import run_reference
from maichart.generation.event_compile import compile_event_points
from maichart.generation.timeline import normalize_tempo_map, serialize_tempo_map, tempo_map_fingerprint, beat_at_audio_end
from maichart.generation.event_model import (
    EVENT_BOS,
    EVENT_END,
    EVENT_EOS,
    EVENT_SPECIALS,
    EventSequence,
    EventVocabulary,
)
from maichart.generation.event_ctx import CTX_CLOSE, CTX_OPEN  # noqa: E402: used by two helpers
from maichart.generation.event_quality import assess_event_chart, parse_event_notes
from maichart.notes import parse_timing_point_notes
from maichart.timing import RawTimingPoint
from maichart.training.event_checkpoints import load_event_model

# Re-roll budget for the empty-window guard: an <eos> before the first complete
# event is rolled back this many times (with fresh samples) before it is accepted.
EMPTY_WINDOW_REROLLS = 6


def _device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value)


def _candidate_path(path: Path, index: int, count: int) -> Path:
    return path if count == 1 else path.with_name(f"{path.stem}_candidate_{index + 1:03d}{path.suffix}")


def _musical_bpm(value: float) -> float:
    """Quantize a detected tempo onto the half-BPM grid authors write.

    A tempogram reports lag bins, so ``librosa`` returns values such as
    ``1500/11 = 136.36363636363637`` (= ``3000/22``) rather than a tempo.  In
    Simai the BPM is not decoration: it is the clock that converts every comma
    cell and every ``{division}`` directive into real time, so a lag bin written
    into ``(N)`` both reads as garbage and shifts the whole grid.

    Detected values therefore snap to the nearest 0.5 BPM, which is finer than
    the tracker's own resolution and matches the integer/half-integer tempos that
    dominate authored charts.  This normalizes the clock; it does not claim
    ground truth, so a caller who knows the real tempo must pass ``--bpm``,
    which is used verbatim and never reaches this function.
    """

    detected = float(value)
    if not math.isfinite(detected) or detected <= 0:
        raise ValueError("bpm must be a finite positive number.")
    return math.floor(detected * 2.0 + 0.5) / 2.0


def _file_identity(path: str | Path) -> dict[str, object]:
    source = Path(path)
    if not source.is_file():
        return {"path": str(source.resolve()), "sha256": None}
    stat = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(source.resolve()), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}


def validate_learned_curve_contract(curve: dict, *, checkpoint_path: str | Path,
                                    checkpoint_identity: dict[str, object],
                                    condition: GenerationCondition, first: float,
                                    tempo_map: list[EventTempo] | None,
                                    tempo_mode: str, bpm_input: float | None) -> None:
    """Reject a learned curve made for another generation context."""
    if not str(curve.get("method", "")).startswith("learned_music_structure_"):
        return
    curve_checkpoint = curve.get("checkpoint")
    if not curve_checkpoint:
        raise ValueError("Learned intensity curve has no checkpoint path; regenerate it")
    if Path(str(curve_checkpoint)).resolve() != Path(checkpoint_path).resolve():
        raise ValueError("Learned intensity curve was created with another checkpoint; regenerate it")
    curve_identity = curve.get("checkpoint_identity")
    if not isinstance(curve_identity, dict) or "size" not in curve_identity or "mtime_ns" not in curve_identity:
        raise ValueError("Learned intensity curve has unsupported checkpoint identity; regenerate it")
    same_size = curve_identity.get("size") == checkpoint_identity.get("size")
    curve_hash = curve_identity.get("sha256")
    current_hash = checkpoint_identity.get("sha256")
    same_hash = isinstance(curve_hash, str) and isinstance(current_hash, str) and curve_hash == current_hash
    # Old curves passed mtime_ns through JavaScript Number, which rounds this
    # ~1.8e18 value. Accept only its exact IEEE-754 round trip.
    old_browser_mtime_match = (curve_hash is None and isinstance(curve_identity.get("mtime_ns"), (int, float))
                               and float(curve_identity["mtime_ns"]) == float(checkpoint_identity.get("mtime_ns")))
    if not same_size or not (same_hash or old_browser_mtime_match):
        raise ValueError("Learned intensity curve checkpoint identity changed; regenerate it")
    try:
        if curve.get("condition") != condition.to_dict():
            raise ValueError("Learned intensity curve condition differs from generation condition; regenerate it")
        if float(curve.get("first", 0.0)) != float(first):
            raise ValueError("Learned intensity curve uses another first/audio offset; regenerate it")
    except (TypeError, ValueError) as exc:
        if "regenerate it" in str(exc):
            raise
        raise ValueError("Learned intensity curve has invalid condition/first metadata; regenerate it") from exc
    explicit_hash = tempo_map_fingerprint(serialize_tempo_map(tempo_map) if tempo_map is not None else None)
    if curve.get("tempo_map_sha256") != explicit_hash:
        raise ValueError("Learned intensity curve was created with another explicit tempo map; regenerate it")
    if explicit_hash is None:
        if str(curve.get("tempo_mode", "constant")) != str(tempo_mode):
            raise ValueError("Learned intensity curve uses another tempo mode; regenerate it")
        curve_bpm_input = curve.get("bpm_input")
        if curve_bpm_input != bpm_input:
            raise ValueError("Learned intensity curve uses another BPM input; regenerate it")


def _tempo_map(tempo_mode: str, audio, bpm: float, *, snap: bool = True) -> list[EventTempo]:
    """Build the chart tempo map for the requested mode.

    ``constant`` pins one tempo at beat zero.  ``auto`` promotes the
    high-confidence local estimates the extractor already computed instead of
    discarding them, which is the only way a chart can follow a song whose
    tempo changes.  Without this the ``--tempo-mode`` flag is inert and every
    song is forced to a single clock.

    ``snap`` quantizes detected tempos onto the authored grid; an explicit
    ``--bpm`` is passed with ``snap=False`` so the caller's value reaches the
    chart unchanged.
    """

    quantize = _musical_bpm if snap else (lambda value: float(value))
    changes = [EventTempo(Fraction(0), quantize(bpm))]
    if tempo_mode != "auto":
        return changes
    for candidate in getattr(audio, "tempo_changes", None) or ():
        try:
            beat = Fraction(str(candidate.get("beat"))).limit_denominator(1920)
            local = quantize(float(candidate["bpm"]))
            confidence = float(candidate.get("confidence", 0.0))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
        # A low-confidence or out-of-order anchor would move the clock without
        # evidence, so only confident, strictly later candidates are promoted.
        if confidence < 0.9 or beat <= changes[-1].beat or beat <= 0 or local <= 0:
            continue
        if abs(local - float(changes[-1].bpm)) < 1e-9:
            continue
        changes.append(EventTempo(beat, local))
    return changes


def _segment_windows(total_beats: Fraction, segment_beats: int, *, limit: int = 0,
                     overlap_beats: int = 0) -> list[tuple[Fraction, Fraction]]:
    """Split ``[0, total_beats)`` into the same windows the model trained on.

    Training targets are segment-relative (``event_dataset.py`` stores
    ``point.beat - start``), and the checkpoint vocabulary cannot express a
    position at or beyond ``segment_beats``.  Generation therefore has to walk
    the song one training window at a time and re-anchor each window's local
    beats onto the full-song clock; a single whole-song decode can only ever
    produce the first window.
    """

    if segment_beats <= 0:
        raise ValueError("segment_beats must be positive.")
    if overlap_beats < 0 or overlap_beats >= segment_beats:
        raise ValueError("overlap_beats must be in [0, segment_beats).")
    stride = segment_beats - overlap_beats
    windows: list[tuple[Fraction, Fraction]] = []
    start = Fraction(0)
    while start < total_beats:
        end = min(start + segment_beats, total_beats)
        if end <= start:
            break
        windows.append((start, end))
        if limit and len(windows) >= limit:
            break
        start += stride
    return windows


def _duration_beats(duration_seconds: float, bpm: float) -> Fraction:
    return (
        Fraction(str(max(0.0, float(duration_seconds)))).limit_denominator(100000)
        * Fraction(str(float(bpm))).limit_denominator(1000000)
        / 60
    ).limit_denominator(1920)


def _atom_heads(atom: str) -> frozenset[str]:
    """Buttons a single note atom strikes, as strings.

    Touch notes report a string position ("A4", "B7", "C") while taps and slide
    heads report 1..8, so both are normalised with ``str``.  Returning a frozenset
    lets the caller test collisions with a single set intersection.
    """

    try:
        parsed = parse_timing_point_notes(
            RawTimingPoint(index=0, raw=atom, note_text=atom, beat=Fraction(0), tick=0,
                           time_sec=0.0, division=4, bpm=120.0)
        )
    except Exception:
        return frozenset()
    return frozenset(str(note.position) for note in parsed.notes if str(note.position))


@dataclass(frozen=True)
class _NoteGrammar:
    """A prefix index built from note atoms seen in the training set."""

    atoms: frozenset[str]
    prefixes: frozenset[str]
    # Buttons each atom strikes, keyed by the atom text, precomputed once so the
    # decode loop never has to re-parse.  Positions are stored as strings because
    # touch notes use "A4"/"B7"/"C" while taps and slide heads use "1".."8".
    heads: dict[str, frozenset[str]] = field(default_factory=dict)
    # Every head reachable from a *partial* atom: prefix -> union of the heads of
    # all atoms that start with it.  `can_continue` uses it to reject a part that
    # can only ever collide with a button the cell already strikes, which is the
    # state that used to be reachable and then impossible to close (or impossible
    # to extend), producing cells such as `1/1-5[8:1]*-3[8:1]`.
    head_closure: dict[str, frozenset[str]] = field(default_factory=dict, compare=False)
    # Same, restricted to atoms that have the prefix *properly*: a completed but
    # colliding atom is only legal if some longer spelling of it strikes different
    # heads (``A`` -> ``A4``).  For ``4`` -> ``47`` that is impossible, so the
    # character is never offered.
    longer_head_closure: dict[str, frozenset[str]] = field(default_factory=dict, compare=False)

    @classmethod
    def from_atoms(cls, values: list[str] | tuple[str, ...] | None) -> "_NoteGrammar | None":
        atoms = frozenset("".join(str(value).split()) for value in (values or ()) if str(value).strip())
        if not atoms:
            return None
        prefixes = frozenset(prefix for atom in atoms for length in range(1, len(atom) + 1)
                             for prefix in (atom[:length],))
        heads = {atom: _atom_heads(atom) for atom in atoms}
        buckets: dict[str, set[str]] = {}
        longer_buckets: dict[str, set[str]] = {}
        for atom, atom_heads in heads.items():
            if not atom_heads:
                continue
            for length in range(1, len(atom) + 1):
                prefix = atom[:length]
                buckets.setdefault(prefix, set()).update(atom_heads)
                if length < len(atom):
                    longer_buckets.setdefault(prefix, set()).update(atom_heads)
        closure = {prefix: frozenset(values_) for prefix, values_ in buckets.items()}
        longer = {prefix: frozenset(values_) for prefix, values_ in longer_buckets.items()}
        return cls(atoms=atoms, prefixes=prefixes, heads=heads, head_closure=closure,
                   longer_head_closure=longer)

    def _parts_strike_distinct_buttons(self, parts: list[str]) -> bool:
        """Whether the completed parts avoid striking one button twice.

        ``is_complete`` refuses to close such a cell, so ``can_continue`` must
        refuse to build one.  Otherwise the decoder can reach a state where no
        character can be added (the last part is a complete atom, so no
        continuation exists) and ``<event_end>`` is withheld -- an empty mask,
        which ``_decode`` turns into a hard error instead of a chart.
        """
        seen: frozenset[str] = frozenset()
        for part in parts:
            buttons = self.heads.get(part)
            if buttons is None:
                buttons = _atom_heads(part)
            if buttons & seen:
                return False
            seen = seen | buttons
        return True

    @staticmethod
    def _split_slashes(text: str) -> list[str]:
        parts: list[str] = []
        current: list[str] = []
        square_depth = 0
        for char in text:
            if char == "[":
                square_depth += 1
            elif char == "]" and square_depth > 0:
                square_depth -= 1
            if char == "/" and square_depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(char)
        parts.append("".join(current))
        return parts

    def can_continue(self, text: str) -> bool:
        """Whether ``text`` is still a prefix of some legal cell.

        This rejects a repeated button, not only an unknown spelling -- including
        the button struck by the part currently being typed.  Without the latter
        check the decoder could build ``1/1-5[8:1]*-3[8:1]``: every character is a
        prefix of a catalog atom, so each one is accepted, but the finished cell
        strikes button 1 twice and ``is_complete`` therefore refuses to close it --
        leaving the decoder able to neither finish the event nor extend it, which
        wastes the window budget and surfaces as an invalid cell.  ``head_closure``
        makes that dead end unreachable.
        """
        if not text:
            return False
        parts = self._split_slashes(text)
        if text.endswith("/"):
            completed = parts[:-1]
            if not completed or not all(part in self.atoms for part in completed):
                return False
            if not self._parts_strike_distinct_buttons(completed):
                return False
            # A slash is only legal when some atom could still follow it, else the
            # cell is already unclosable and the mask would go empty.
            used = frozenset().union(*(self.heads[part] for part in completed))
            return any(atom_heads and not (atom_heads & used)
                       for atom_heads in self.heads.values())
        if not all(part in self.atoms for part in parts[:-1]):
            return False
        if not self._parts_strike_distinct_buttons(parts[:-1]):
            return False
        progress = parts[-1]
        if progress not in self.prefixes:
            return False
        if parts[:-1]:
            used = frozenset().union(*(self.heads[part] for part in parts[:-1]))
            if progress in self.atoms and (self.heads[progress] & used):
                # Closing right here would strike a button twice.  Only a longer
                # spelling that strikes different heads can still be legal.
                longer = self.longer_head_closure.get(progress)
                if not (longer and not (longer & used)):
                    return False
            reachable = self.head_closure.get(progress)
            if reachable and not (reachable - used):
                return False
        return True

    def is_complete(self, text: str) -> bool:
        if not text:
            return False
        parts = self._split_slashes(text)
        if not all(part in self.atoms for part in parts):
            return False
        parsed = parse_timing_point_notes(
            RawTimingPoint(index=0, raw=text, note_text=text, beat=Fraction(0), tick=0,
                           time_sec=0.0, division=4, bpm=120.0)
        )
        if parsed.unknown_tokens or not parsed.notes:
            return False
        # Every note kind must be counted, not just taps.  Tap and slide heads
        # report an int position (1..8) while touch notes report a string
        # ("A4", "B7", "C", ...), so filtering on int made the guard vacuous for
        # touch chords and let cells such as "A4/A4" and a ten-button
        # "A4/A5/A5f/A4/A4/A5/A5/A4/A5/D4f" through.
        heads = Counter(str(note.position) for note in parsed.notes)
        return not any(count > 1 for count in heads.values())


def _allowed_ids(vocabulary: EventVocabulary, state: str, chars: int = 0,
                 last_beat: Fraction | None = None, note_text: str = "",
                 note_grammar: _NoteGrammar | None = None) -> list[int]:
    result: list[int] = []
    for index, token in enumerate(vocabulary.tokens):
        if state == "position" and token.startswith("AT:"):
            if last_beat is None or Fraction(token[3:]) > last_beat:
                result.append(index)
        elif state == "position" and token == EVENT_EOS:
            result.append(index)
        elif state == "char" and token.startswith("CHAR:"):
            candidate = note_text + token[5:]
            if note_grammar is None or note_grammar.can_continue(candidate):
                result.append(index)
        elif state == "char" and token == EVENT_END and chars and (
            note_grammar is None or note_grammar.is_complete(note_text)
        ):
            result.append(index)
    if not result and state == "char" and chars:
        # No character can extend the current cell and the grammar will not close
        # it, which happens when the last part completed into an atom that
        # repeats a button already struck (e.g. "A4/A4").  Closing the cell here
        # keeps generation alive; the cell is then reported as a
        # ``duplicate_button_head`` fatal error by the quality pass instead of
        # aborting the whole song with "reached an invalid grammar state".
        end_id = vocabulary.token_to_id.get(EVENT_END)
        if end_id is not None:
            result.append(end_id)
    return result


def _is_control_token(token: str) -> bool:
    """Whether a vocabulary token is structural and exempt from repeat penalties.

    The pad/bos/eos/unk/``<event_end>`` markers and every ``AT:`` position token
    carry the chart's structure, not its wording: discounting them would make
    the decoder dodge the beat clock and the event length rather than repeated
    note text, so they are never penalised.
    """

    return token in EVENT_SPECIALS or token.startswith("AT:")


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus truncation on already grammar-masked logits.

    For ``0 < top_p < 1`` the probabilities are softmaxed, accumulated from the
    most probable token down, and the smallest prefix whose cumulative
    probability reaches ``top_p`` is kept; every other logit becomes ``-inf``.
    Any other value (0, >=1, non-finite) returns the logits untouched, so the
    switch is a strict no-op by default.  The nucleus is computed on the
    post-grammar-mask logits (temperature enters later, only for sampling), and
    the argmax path never calls this helper.
    """

    value = float(top_p)
    if not (0.0 < value < 1.0):
        return logits
    probabilities = torch.softmax(logits, dim=-1)
    sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
    cumulative = torch.cumsum(sorted_probabilities, dim=-1)
    # The first rank whose cumulative probability reaches ``top_p`` closes the
    # nucleus, so at least the most probable token always survives and the
    # grammar-masked tokens (probability exactly 0) never do.
    cutoff = int(torch.nonzero(cumulative >= value, as_tuple=False)[0].item())
    keep = sorted_indices[: cutoff + 1]
    filtered = torch.full_like(logits, float("-inf"))
    filtered[keep] = logits[keep]
    return filtered


def _apply_repetition_penalty(logits: torch.Tensor, generated_ids: list[int], penalty: float,
                              vocabulary: EventVocabulary) -> torch.Tensor:
    """CTRL-style repetition penalty on the tokens seen in the recent window.

    Every non-control id among ``generated_ids`` has its positive logit divided
    by ``penalty`` and its negative logit multiplied, applied once per id
    however often it repeats.  Callers pass the ids already intersected with
    the grammar mask: disallowed ids sit at ``-inf`` and stay there.  ``penalty
    == 1.0`` returns the logits untouched (no floating-point round-trip), which
    is what keeps the default off-switch byte-identical.
    """

    value = float(penalty)
    if value == 1.0:
        return logits
    seen: set[int] = set()
    for token_id in generated_ids:
        token_id = int(token_id)
        token = vocabulary.id_to_token.get(token_id)
        if token is None or _is_control_token(token):
            continue
        seen.add(token_id)
    if not seen:
        return logits
    index = torch.tensor(sorted(seen), dtype=torch.long, device=logits.device)
    updated = logits.clone()
    current = updated[index]
    updated[index] = torch.where(current < 0, current * value, current / value)
    return updated


def _decode(model, vocabulary, audio, global_audio, condition, device, *, max_tokens, temperature,
            sample, seed, guidance_scale=1.0, top_p: float = 0.0, repetition_penalty: float = 1.0,
            repetition_window: int = 128, atom_catalog=None, allow_truncated=False,
            prefix_ids: list[int] | None = None, position_offset: Fraction = Fraction(0), condition_default: float = 0.5,
            plan_override=None, plan_controls=None, plan_audio_mask=None, density_feedback=None,
            position_end: Fraction | None = None, onset_context: dict | None = None):
    """Sample one window, optionally seeded with a lead-in context block.

    ``prefix_ids`` is the cross-window family's ``<ctx> ... <ctx_end>`` block.
    Those tokens are given, not produced: they are pushed through the cache
    without sampling, so the first sampled token continues from them exactly the
    way training did.  The whole composite (prefix + window) must fit the
    checkpoint's positional grid, which is why the caller passes a reduced
    budget.

    ``position_offset`` is how many beats the lead-in block starts before the
    window.  Training positions every token relative to the block start so the
    sequence has one origin instead of two (Transformer-XL's "temporal
    confusion"), so the emitted positions carry that offset and it is removed
    here to give the caller window-relative beats again.
    """

    if max_tokens < 3:
        raise ValueError("max_tokens must be at least 3.")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("temperature must be finite and positive.")
    if not math.isfinite(float(guidance_scale)) or float(guidance_scale) < 0:
        raise ValueError("guidance_scale must be finite and nonnegative.")
    if not math.isfinite(float(repetition_penalty)) or float(repetition_penalty) <= 0:
        raise ValueError("repetition_penalty must be finite and positive.")
    if int(repetition_window) < 1:
        raise ValueError("repetition_window must be a positive integer.")
    torch.manual_seed(seed)
    model.eval()
    condition = condition.to(device).unsqueeze(0)
    unconditional_condition = None
    if guidance_scale != 1.0:
        unconditional_condition = condition.clone()
        unconditional_condition[:, 1:] = float(condition_default)
    audio = audio.to(device).unsqueeze(0)
    global_audio = global_audio.to(device).unsqueeze(0)
    prefix = list(prefix_ids or [])
    ids = [vocabulary.token_to_id[EVENT_BOS]] + prefix
    # The loop below feeds position `step` and predicts `step + 1`, so the last
    # prefix token occupies index len(ids) - 1 and is fed at that step.
    first_step = len(ids) - 1
    if first_step >= max_tokens:
        raise ValueError(
            f"context prefix ({len(prefix)} tokens) leaves no room in max_tokens={max_tokens}."
        )
    state, chars = "position", 0
    last_beat: Fraction | None = None
    note_text = ""
    eos_rerolls = 0
    note_grammar = _NoteGrammar.from_atoms(atom_catalog)
    finished = False
    # Index of the first token of the newest completed event.  A truncated
    # decode can only be salvaged up to this point, because a half-emitted
    # event has no position or note text to recover.
    committed = len(ids)
    cell_start = committed
    previous_beat = None
    # Guidance keeps a second, independent cache because its attention must use
    # the zeroed condition throughout, not just at the final layer.
    unconditional_caches: list | None = None

    def warm_up(caches, streaming_memory, conditioning):
        """Push every prefix token except the newest through the cache."""

        for step in range(first_step):
            _, caches = model.decode_step(
                streaming_memory, torch.tensor([[ids[step]]], dtype=torch.long, device=device),
                conditioning, caches, step, audio_mask,
            )
        return caches

    with torch.inference_mode():
        memory, audio_mask = model.encode_audio(
            audio, condition, global_audio=global_audio,
            condition_null=torch.zeros((audio.size(0),), dtype=torch.bool, device=device),
            **({"plan_audio_mask": plan_audio_mask} if plan_audio_mask is not None else {}),
        )
        onset_guidance = None
        onset_logits = None
        if onset_context is not None and onset_context.get("strength", 0) > 0:
            onset_logits = model.local_onset_logits(memory)[0]
        if hasattr(model, "append_plan_memory"):
            try:
                planner_kwargs = {}
                if plan_override is not None:
                    planner_kwargs["plan_override"] = plan_override
                if plan_controls:
                    planner_kwargs["plan_controls"] = plan_controls
                memory, audio_mask, _ = model.append_plan_memory(memory, audio_mask, **planner_kwargs)
            except TypeError:
                # Test doubles and pre-planner checkpoints expose the old
                # two-argument helper.  Their behaviour remains byte-compatible.
                memory, audio_mask, _ = model.append_plan_memory(memory, audio_mask)
        # Classifier-free guidance: the null branch must attend a memory encoded
        # with the zeroed condition as well.  The encoder folds the condition
        # into every memory frame, so reusing the conditional memory and only
        # zeroing the decoder-side input is not a real null pass -- the logits
        # difference that guidance is supposed to subtract is then mostly gone.
        unconditional_memory = None
        if guidance_scale != 1.0:
            unconditional_memory, unconditional_audio_mask = model.encode_audio(
                audio, unconditional_condition, global_audio=global_audio,
                condition_null=torch.ones((audio.size(0),), dtype=torch.bool, device=device),
                **({"plan_audio_mask": plan_audio_mask} if plan_audio_mask is not None else {}),
            )
            if onset_logits is not None:
                null_onsets = model.local_onset_logits(unconditional_memory)[0]
                onset_logits = null_onsets + float(guidance_scale) * (onset_logits - null_onsets)
            if hasattr(model, "append_plan_memory"):
                try:
                    planner_kwargs = {}
                    if plan_override is not None:
                        planner_kwargs["plan_override"] = plan_override
                    if plan_controls:
                        planner_kwargs["plan_controls"] = plan_controls
                    unconditional_memory, unconditional_audio_mask, _ = model.append_plan_memory(
                        unconditional_memory, unconditional_audio_mask, **planner_kwargs)
                except TypeError:
                    unconditional_memory, unconditional_audio_mask, _ = model.append_plan_memory(
                        unconditional_memory, unconditional_audio_mask)
            else:
                unconditional_audio_mask = audio_mask
        if onset_logits is not None:
            from maichart.generation.onset_guidance import LocalOnsetGuidance
            onset_guidance = LocalOnsetGuidance.from_predictions(
                onset_logits, vocabulary=vocabulary, **onset_context)
        caches: list = []
        if first_step > 0:
            caches = warm_up(caches, memory, condition)
            if guidance_scale != 1.0:
                unconditional_caches = warm_up([], unconditional_memory, unconditional_condition)
        decoder_input = torch.tensor([[ids[first_step]]], dtype=torch.long, device=device)
        while len(ids) < max_tokens:
            step = len(ids) - 1
            logits, caches = model.decode_step(
                memory, decoder_input, condition, caches, step, audio_mask,
            )
            logits = logits[:, -1, :].squeeze(0)
            if guidance_scale != 1.0:
                # The unconditional pass attends its own memory (encoded with
                # the zeroed condition) with its own cache: reusing the
                # conditional keys would leak the conditioning the guidance is
                # supposed to subtract.
                if unconditional_caches is None:
                    unconditional_caches = []
                unconditional, unconditional_caches = model.decode_step(
                    unconditional_memory, decoder_input, unconditional_condition,
                    unconditional_caches, step, unconditional_audio_mask,
                )
                unconditioned = unconditional[:, -1, :].squeeze(0)
                logits = unconditioned + float(guidance_scale) * (logits - unconditioned)
            allowed = _allowed_ids(vocabulary, state, chars, last_beat, note_text, note_grammar)
            if state == "position" and position_end is not None:
                # Context-family vocabulary spans lead-in + target. Particularly
                # at the song's first window, frequent AT:16+ tokens must not be
                # generated and then silently discarded by the merge.
                allowed = [idx for idx in allowed
                           if not vocabulary.id_to_token[idx].startswith("AT:")
                           or position_offset <= Fraction(vocabulary.id_to_token[idx][3:]) < position_end]
            if not allowed:
                raise ValueError(f"decoder reached an invalid grammar state: state={state!r}, chars={chars}")
            masked = torch.full_like(logits, float("-inf"))
            masked[allowed] = logits[allowed]
            if density_feedback is not None:
                masked = density_feedback.shape_logits(masked, vocabulary, state=state, last_beat=last_beat)
            if sample:
                # Both shapers act on the sampling branch only, so the argmax
                # path stays byte-identical whatever these flags are.  With the
                # defaults (top_p=0.0, repetition_penalty=1.0) each helper
                # returns the masked logits untouched, so the distribution below
                # is exactly the pre-switch one.  The penalty runs first, then
                # the nucleus is cut from the penalised distribution.
                penalized = _apply_repetition_penalty(
                    masked, ids[-int(repetition_window):], repetition_penalty, vocabulary)
                scaled = penalized / max(float(temperature), 0.05)
                if onset_guidance is not None:
                    scaled = onset_guidance.shape_logits(scaled, state=state)
                filtered = _apply_top_p(scaled, top_p)
                probabilities = torch.softmax(filtered, dim=-1)
                token_id = int(torch.multinomial(probabilities, 1).item())
            else:
                if onset_guidance is not None:
                    masked = onset_guidance.shape_logits(masked, state=state)
                token_id = int(masked.argmax().item())
            ids.append(token_id)
            decoder_input = torch.tensor([[token_id]], dtype=torch.long, device=device)
            token = vocabulary.id_to_token[token_id]
            if state == "position":
                if token == EVENT_EOS:
                    if position_end is not None and not any(vocabulary.id_to_token[i].startswith("AT:") for i in allowed):
                        finished = True
                        break
                    # Empty-window guard: a model can answer <eos> immediately
                    # for a window whose audio or lead-in looks quiet, which is
                    # exactly where the measured blank windows came from.  A
                    # window must contribute at least one complete event before
                    # <eos> is accepted; re-roll instead, but only a bounded
                    # number of times so a genuinely exhausted context cannot
                    # spin forever.
                    if density_feedback is not None and density_feedback.allow_empty_end():
                        finished = True
                        break
                    if len([i for i in ids[len(prefix) + 1:]
                            if vocabulary.id_to_token.get(i) == EVENT_END]) > 0:
                        finished = True
                        break
                    if eos_rerolls >= EMPTY_WINDOW_REROLLS:
                        finished = True
                        break
                    eos_rerolls += 1
                    # A rejected EOS is not part of the chart or its context.
                    # Keeping it made vocabulary.decode stop before any later
                    # re-roll notes. Refeed the retained final token at the
                    # same position, using only its preceding cache entries.
                    ids.pop()
                    cache_length = len(ids) - 1
                    caches = [(key[:, :cache_length, :], value[:, :cache_length, :]) for key, value in caches]
                    if unconditional_caches is not None:
                        unconditional_caches = [(key[:, :cache_length, :], value[:, :cache_length, :])
                                                for key, value in unconditional_caches]
                    decoder_input = torch.tensor([[ids[-1]]], dtype=torch.long, device=device)
                    continue
                previous_beat = last_beat
                cell_start = len(ids) - 1
                last_beat = Fraction(token[3:])
                committed = len(ids) - 1
                state, chars, note_text = "char", 0, ""
            elif state == "char":
                if token == EVENT_END:
                    if density_feedback is not None and not density_feedback.accept_cell(last_beat, note_text):
                        # Complete configurations that exceed a hard quota are
                        # rejected as a whole. Rewind tokens AND attention cache
                        # so later events do not attend invisible notes.
                        ids = ids[:cell_start]
                        caches = [(key[:, :cell_start - 1, :], value[:, :cell_start - 1, :]) for key, value in caches]
                        if unconditional_caches is not None:
                            unconditional_caches = [(key[:, :cell_start - 1, :], value[:, :cell_start - 1, :])
                                                    for key, value in unconditional_caches]
                        last_beat = previous_beat
                        state, chars, note_text = "position", 0, ""
                        committed = len(ids)
                        decoder_input = torch.tensor([[ids[-1]]], dtype=torch.long, device=device)
                        continue
                    state, chars, note_text = "position", 0, ""
                    committed = len(ids)
                else:
                    chars += 1
                    note_text += token[5:]
            if len(ids) >= max_tokens:
                break
    if not finished:
        if not allow_truncated:
            raise ValueError("decoder reached max_tokens before emitting <eos>.")
        # Drop any partially emitted event and keep every completed one, so a
        # window that exhausted its budget still contributes its real notes.
        ids = ids[:committed]
    # Only the window is returned.  The lead-in block is given context, so
    # decoding it back would report the previous window's notes as this one's.
    sequence = vocabulary.decode(ids[len(prefix) + 1:])
    if position_offset:
        # Undo the block origin: callers re-anchor beats onto the song clock from
        # the window start, so they need window-relative positions.
        sequence = EventSequence(beats=[Fraction(beat) - position_offset for beat in sequence.beats],
                                 note_texts=list(sequence.note_texts))
    return sequence


def _clamp_decode_budget(requested: int, model) -> int:
    """Clamp a decode budget to the checkpoint's positional grid.

    ``EventConditionedTransformer.decode_step`` rejects any position at or beyond
    ``config.max_tokens``, and that number comes from the checkpoint rather than
    from this run's flags.  A larger request therefore cannot be honoured: it
    aborts the window, and with it the whole candidate, after minutes of work.
    Using the model's own limit emits as much as the model can represent and then
    moves on to the next window.
    """

    limit = int(getattr(getattr(model, "config", None), "max_tokens", requested))
    if int(requested) > limit:
        print(f"warning: requested max_tokens={int(requested)} exceeds the checkpoint position "
              f"grid ({limit}); clamping to {limit}", flush=True)
        return limit
    return int(requested)


@dataclass(frozen=True)
class _DecodedWindow:
    """One training-window decode plus the report of how it ended."""

    sequence: EventSequence
    start_beat: Fraction
    end_beat: Fraction
    closed: bool
    intensity_curve: dict | None = None
    playability: dict | None = None
    original_sequence: EventSequence | None = None


def _effective_curve_profile(profile: dict, *, strength: float, mode: str,
                             span: float = 0.5) -> dict:
    """One effective curve shared by decoding, the guard and final reports."""
    if not math.isfinite(float(strength)) or not 0.0 <= float(strength) <= 1.0:
        raise ValueError("intensity curve strength must be finite and in 0..1")
    if mode not in {"follow", "hard", "soft", "strict"}:
        raise ValueError("intensity curve mode must be follow, hard, soft or strict")
    limit = max(0.0, min(1.0, float(span)))
    result = dict(profile)
    points = []
    for point in profile["points"]:
        factor = 2.0 * float(point["intensity"])
        if mode != "strict":
            factor = min(1.0 + limit, max(1.0 - limit, factor))
        effective = 1.0 + float(strength) * (factor - 1.0)
        points.append({"time_sec": float(point["time_sec"]), "intensity": effective * 0.5})
    result["points"] = points
    return result


def _rest_filter_sequence(sequence, *, start_beat, tempo_changes, first, profile):
    regions = (profile or {}).get("rest_regions", [])
    if not regions:
        return sequence, 0
    selected = [(beat, text) for beat, text in zip(sequence.beats, sequence.note_texts)
                if not any(region["start_sec"] <= float(seconds_at_beat(start_beat + beat, tempo_changes, first)) < region["end_sec"]
                           for region in regions)]
    return EventSequence([beat for beat, _ in selected], [text for _, text in selected]), len(sequence.beats) - len(selected)


def _curve_plan_controls(model, condition, factor: float, duration_sec: float,
                         *, strength: float = 1.0) -> dict[str, float] | None:
    """Translate a curve factor into planner count/density probabilities.

    Planner controls are deliberately soft: the decoder still chooses exact
    event cells and configuration logits.  The hard upper guard below is what
    makes a reduced curve auditable when an old checkpoint under-responds to
    planner conditioning.
    """

    if not getattr(model, "planner_enabled", False):
        return None
    radar = getattr(condition, "radar", {}) or {}
    base = max(0.0, min(1.0, float(radar.get("note_density", 0.5))))
    scaled = max(0.0, min(1.0, base * max(0.0, float(factor))))
    # count_proxy is log1p(note_count)/log1p(64); note_density is normalized
    # to 16 heads/sec by the training contract.
    desired_count = max(0.0, float(radar.get("note_density", 0.5)) * 16.0
                        * max(0.0, float(duration_sec)) * max(0.0, float(factor)))
    count = min(1.0, math.log1p(desired_count) / math.log1p(64.0))
    # The v2 planner reserves the last control for the requested local
    # workload.  Keep the legacy two-control payload when the checkpoint has
    # no planner head wide enough to consume it.
    controls = {"count": count, "density": scaled}
    if getattr(model, "planner_enabled", False):
        controls["intensity"] = max(0.0, min(1.0, float(factor) / 2.0))
        controls["__strength"] = max(0.0, min(1.0, float(strength)))
    return controls


def _curve_filter_sequence(sequence: EventSequence, *, start_beat: Fraction,
                           tempo_changes, first: float, curve_profile, base_density: float,
                           window_start_sec: float, window_end_sec: float,
                           audio=None,
                           tolerance: float = 0.15, apply_guard: bool = True) -> tuple[EventSequence, dict]:
    """Apply a local 2-second density ceiling without splitting event cells."""

    from maichart.generation.music_intensity import curve_window_values

    profile_times = [float(point["time_sec"]) for point in curve_profile["points"]]
    profile_values = [float(point["intensity"]) for point in curve_profile["points"]]

    def point_intensity(second: float) -> float:
        import numpy as np
        return float(np.interp(float(second), profile_times, profile_values))

    grid_start = math.floor(float(window_start_sec) / 2.0) * 2.0
    grid_end = max(grid_start + 2.0, math.ceil(float(window_end_sec) / 2.0) * 2.0)
    cells = []
    absolute_points = [EventPoint(Fraction(beat) + start_beat, text)
                       for beat, text in zip(sequence.beats, sequence.note_texts)]
    parsed = parse_event_notes(absolute_points, tempo_changes, first)
    by_beat = {Fraction(point.beat): len(point.notes) for point in parsed.points}
    for beat, text in zip(sequence.beats, sequence.note_texts):
        absolute = Fraction(beat) + start_beat
        when = float(seconds_at_beat(absolute, tempo_changes, first))
        heads = max(0, int(by_beat.get(absolute, 0)))
        cells.append((when, Fraction(beat), text, heads))
    kept = {i for i, item in enumerate(cells)
            if float(window_start_sec) <= item[0] < float(window_end_sec)}
    bins = []
    removed = 0
    under_target = 0
    raw_cells = [{"time_sec": item[0], "heads": item[3]} for item in cells
                 if float(window_start_sec) <= item[0] < float(window_end_sec)]
    # A cell at an exact boundary belongs to the later bin.  Curve values are
    # trapezoid/time-interpolated over the *real* two-second slice, while the
    # last partial window receives only its actual duration quota.
    cursor = grid_start
    while cursor < grid_end - 1e-9:
        end = min(cursor + 2.0, float(window_end_sec))
        if end <= float(window_start_sec):
            cursor += 2.0
            continue
        begin = max(cursor, float(window_start_sec))
        duration = max(0.0, end - begin)
        indexes = [i for i, item in enumerate(cells) if begin <= item[0] < end]
        intensity = max(0.0, min(1.0, float(curve_window_values(curve_profile, begin, end))))
        factor = 2.0 * intensity
        desired = max(0.0, float(base_density) * 16.0 * duration * factor)
        raw_actual = sum(cells[i][3] for i in indexes)
        if indexes:
            zero_local = ({i for i in indexes if point_intensity(cells[i][0]) <= 1e-9}
                          if apply_guard else set())
            kept.difference_update(zero_local)
            cap = int(math.floor(desired * (1.0 + tolerance) + 1e-9))
            if factor <= 1e-9:
                cap = 0
            remaining = [i for i in indexes if i in kept]
            actual = sum(cells[i][3] for i in remaining)
            if apply_guard and actual > cap:
                # Keep beat-aligned cells first, then earliest onsets.  A cell
                # remains whole: chords, holds, slides and touches are never
                # rewritten or split by this guard.
                budget = cap
                keep_local = set()
                onset_values = {}
                if audio is not None and len(getattr(audio, "features", ())) and len(getattr(audio, "times", ())):
                    try:
                        import numpy as np

                        onset_column = np.asarray(audio.features)[:, 92]
                        for i in remaining:
                            onset_values[i] = float(np.interp(cells[i][0], audio.times, onset_column))
                    except (IndexError, TypeError, ValueError):
                        onset_values = {}
                onset_low = min(onset_values.values(), default=0.0)
                onset_span = max(onset_values.values(), default=0.0) - onset_low
                if onset_span > 1e-9:
                    onset_values = {i: (value - onset_low) / onset_span
                                    for i, value in onset_values.items()}
                candidates = set(remaining)
                while candidates:
                    def priority(i):
                        onset = onset_values.get(i, 0.0)
                        spacing = (0.5 if not keep_local else min(
                            abs(cells[i][0] - cells[j][0]) for j in keep_local) / max(duration, 1e-9))
                        beat_grid = 0.03 if cells[i][1].denominator == 1 else 0.0
                        return onset + 0.2 * min(0.5, spacing) + beat_grid
                    i = max(candidates, key=lambda item: (priority(item), -cells[item][0]))
                    candidates.remove(i)
                    if cells[i][3] <= budget:
                        keep_local.add(i)
                        budget -= cells[i][3]
                kept.difference_update(set(indexes) - keep_local)
                actual = sum(cells[i][3] for i in keep_local)
            removed += raw_actual - actual
            if actual < desired:
                under_target += int(math.ceil(desired - actual))
        else:
            actual = 0
            if desired > 0:
                under_target += int(math.ceil(desired))
        bins.append({"start_sec": begin, "end_sec": end,
                         "factor": factor, "desired_heads": desired,
                         "raw_heads": raw_actual,
                         "actual_heads": actual,
                         "actual_density": actual / max(duration, 1e-9),
                         "removed": max(0, raw_actual - actual),
                         "under_target": max(0.0, desired - actual)})
        cursor += 2.0
    selected = [cells[i] for i in sorted(kept)]
    return EventSequence([item[1] for item in selected], [item[2] for item in selected]), {
        "local_bins": bins, "removed": removed, "under_target": under_target,
        "raw_cells": raw_cells,
    }


def _final_curve_bins(points: list[EventPoint], profile: dict, *, tempo_changes,
                      first: float, base_density: float, windows: list[_DecodedWindow]) -> list[dict]:
    """Recount curve adherence on the final merged, audible-span-filtered chart."""

    from maichart.generation.music_intensity import curve_window_values

    duration = float(profile["duration_sec"])
    parsed = parse_event_notes(points, tempo_changes, first)
    final_points = [point for point in parsed.points if point.notes]
    raw_at_time: dict[float, int] = {}
    for index, window in enumerate(windows):
        # Shared time belongs to the earlier window, matching _merge_windows.
        low = float(seconds_at_beat(window.start_beat, tempo_changes, first))
        if index:
            low = max(low, float(seconds_at_beat(windows[index - 1].end_beat, tempo_changes, first)))
        high = float(seconds_at_beat(window.end_beat, tempo_changes, first))
        for item in (window.intensity_curve or {}).get("raw_cells", []):
            key = float(item["time_sec"])
            if low <= key < high and key not in raw_at_time:
                raw_at_time[key] = int(item.get("heads", 0))
    result = []
    begin = 0.0
    while begin < duration - 1e-9:
        end = min(begin + 2.0, duration)
        desired_intensity = max(0.0, min(1.0, float(curve_window_values(profile, begin, end))))
        factor = 2.0 * desired_intensity
        desired = max(0.0, float(base_density) * 16.0 * (end - begin) * factor)
        actual = sum(len(point.notes) for point in final_points
                     if begin <= float(point.time_sec) < end)
        raw = sum(heads for second, heads in raw_at_time.items() if begin <= second < end)
        result.append({
            "start_sec": begin, "end_sec": end, "factor": factor,
            "desired_heads": desired, "raw_heads": raw, "actual_heads": actual,
            "actual_density": actual / max(end - begin, 1e-9),
            "removed": max(0, raw - actual),
            "under_target": max(0.0, desired - actual),
        })
        begin += 2.0
    return result


def _expected_input_dim(model) -> int | None:
    """Feature width the model was built with, if it exposes one.

    Stub models used by tests have no config; nothing is adapted for them.
    """

    config = getattr(model, "config", None)
    value = getattr(config, "input_dim", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _fit_feature_dim(tensor, expected: int, *, where: str):
    """Adapt a feature tensor to the dimension a checkpoint was trained with.

    Release v4 grew the input from 100 to 112 columns (structure, phase, and
    beat-normalized rhythm channels). A checkpoint trained before that expects
    100 or 108, so
    generation reduces the tensor to the old layout instead of failing with
    "Expected 100 audio features, got 108" -- the model is still perfectly
    usable, it simply never saw the newer channels.
    """

    actual = int(tensor.shape[-1])
    if actual == expected:
        return tensor
    if expected == LEGACY_INPUT_DIM:
        print(f"feature_compat=legacy {where}: checkpoint wants {expected} features, "
              f"pipeline produced {actual}; using the pre-v3 layout "
              f"(consider retraining to use the structure and phase channels)", flush=True)
        return legacy_feature_view(tensor)
    if expected == 108:
        # v3 checkpoints consumed acoustic + timing context but not the four
        # beat-normalized rhythm channels introduced in v4.
        print(f"feature_compat=v3 {where}: checkpoint wants 108 features; dropping rhythm channels",
              flush=True)
        return tensor[..., :108]
    raise ValueError(
        f"checkpoint expects {expected} audio features but the pipeline produces {actual}; "
        "retrain the model with the current feature release."
    )


def _vocabulary_position_limit(vocabulary: EventVocabulary) -> Fraction | None:
    """Largest beat position the checkpoint's alphabet can express."""

    limit: Fraction | None = None
    for token in vocabulary.tokens:
        if isinstance(token, str) and token.startswith("AT:"):
            try:
                value = Fraction(token[3:])
            except (ValueError, ZeroDivisionError):
                continue
            if limit is None or value > limit:
                limit = value
    return limit


def _trim_sequence(sequence: EventSequence, start: Fraction, *, tempo_changes, first: float,
                   low_sec: float, high_sec: float, tolerance: float = 0.02) -> EventSequence:
    """Drop events outside the audible span, so silence gets no notes.

    The clock is the file: a song's silent lead-in and its trailing silence are
    cut into ordinary 16-beat windows, and nothing in the model's input says
    "the music has not started".  Measured on a real run, the chart wrote 8 notes
    before the music's first audible frame (1.41 s onward, over relative_rms
    0.000) and 9 notes after its last (up to 146.12 s).  Trimming here also keeps
    the `<ctx>` lead-in consistent: the next window is conditioned on exactly the
    notes that survive.

    ``tolerance`` is one feature frame (20 ms at 50 Hz): a note landing exactly on
    the boundary must survive float rounding, while a note in real silence is
    dropped.
    """

    if not sequence.beats:
        return sequence
    low = float(low_sec) - max(0.0, float(tolerance))
    high = float(high_sec) + max(0.0, float(tolerance))
    kept_beats: list[Fraction] = []
    kept_texts: list[str] = []
    for beat, text in zip(sequence.beats, sequence.note_texts):
        when = float(seconds_at_beat(Fraction(beat) + start, tempo_changes, first))
        if low <= when <= high:
            kept_beats.append(beat)
            kept_texts.append(text)
    if len(kept_beats) == len(sequence.beats):
        return sequence
    return EventSequence(kept_beats, kept_texts)


def _decode_windows(model, vocabulary, audio, global_audio, condition, device, *, windows, tempo_changes,
                    first, song_duration, bpm, max_tokens, temperature, sample, seed, guidance_scale=1.0,
                    top_p: float = 0.0, repetition_penalty: float = 1.0, repetition_window: int = 128,
                    atom_catalog=None, min_window_sec=0.25, context_beats=0, audio_margin_beats=0,
                    context_tokens=0, intensity_follow: bool = False, intensity_span: float = 0.5,
                    intensity_mode: str = "loudness", intensity_curve=None,
                    intensity_curve_strength: float = 1.0,
                    intensity_curve_mode: str = "follow",
                    tail_damping: bool = False, tail_damping_windows: int = 4,
                    tail_damping_strength: float = 0.6, audio_bounds_trim: bool = False,
                    audio_bounds_lead: float = 0.5, silence_threshold: float = 0.15,
                    silence_window_share: float = 0.05, condition_default: float = 0.5,
                    fine_note_strength: float = 0.5,
                    local_onset_bias_strength: float = 0.0,
                    post_process_config: PostProcessConfig | None = None) -> list[_DecodedWindow]:
    """Generate the whole song one training window at a time.

    This mirrors ``EventDataset.__getitem__`` exactly: each window samples the
    audio slice between its own start and end beat, passes the full-song beat
    at the slice start as ``start_beat`` so the chart clock never resets, and
    the decoder emits segment-relative positions that are re-anchored here.
    A whole-song single decode is what limited output to the first window.

    A window whose audio slice is too short to carry any beat information ends
    the generation rather than emitting an ungrounded window.

    With ``context_beats > 0`` (cross-window family) each window is also
    *seeded* with the notes already generated for the preceding beats, and its
    audio slice grows by ``audio_margin_beats`` on both sides -- exactly the
    layout training used.  ``audio_margin_beats`` alone is meaningful too: it
    lets the encoder look at the music either side of the window.

    The model sees local audio as well as song-level controls. Intensity
    following additionally scales density/peak using loudness or multifeature
    musical tension. An edited curve takes priority over automatic following
    and tail damping, and supplies planner targets and a local density ceiling.
    """

    results: list[_DecodedWindow] = []
    feedback_cells = []
    # Clamp once here as well: the salvage retry below repeats the decode, so a
    # budget beyond the checkpoint's positional grid would fail twice.  The helper
    # is idempotent and only warns when the value actually changes.
    budget = _clamp_decode_budget(max_tokens, model)
    # `condition` is the user's GenerationCondition here (not a vector): the
    # per-window intensity factor has to be applied to named controls.
    base_vector = condition.to_vector()
    intensity = [1.0] * len(windows)
    curve_values = [1.0] * len(windows)
    curve_api = None
    curve_strength = float(intensity_curve_strength)
    if not math.isfinite(curve_strength) or not 0.0 <= curve_strength <= 1.0:
        raise ValueError("intensity curve strength must be finite and in 0..1")
    raw_curve = intensity_curve
    control_curve = None
    if intensity_curve is not None:
        intensity_curve = _effective_curve_profile(intensity_curve, strength=curve_strength,
                                                   mode=intensity_curve_mode, span=intensity_span)
        control_curve = _effective_curve_profile(raw_curve, strength=1.0,
                                                 mode=intensity_curve_mode, span=intensity_span)
    if intensity_curve is not None and curve_strength > 0.0:
        from maichart.generation import music_intensity as curve_api
        for index, (start, end) in enumerate(windows):
            start_sec = float(seconds_at_beat(start, tempo_changes, first))
            end_sec = float(seconds_at_beat(end, tempo_changes, first))
            effective = max(0.0, min(2.0, 2.0 * float(curve_api.curve_window_values(
                intensity_curve, start_sec, end_sec))))
            curve_values[index] = max(0.0, min(2.0, effective))
        intensity = curve_values[:]
        print("intensity_curve=on factors: " + ", ".join(f"{v:.2f}" for v in intensity), flush=True)
    elif intensity_follow or tail_damping:
        from maichart.generation import intensity as intensity_module

        if intensity_follow:
            resolved_mode = intensity_mode
            if resolved_mode == "auto":
                profile = str(getattr(model, "planner_profile", ""))
                resolved_mode = "model" if is_music_structure(profile) else "multifeature"
            if resolved_mode == "model":
                active_profile = str(getattr(model, "planner_profile", "legacy_v1"))
                if not is_music_structure(active_profile):
                    raise ValueError(
                        "intensity-mode=model requires a music_structure_v2/v3 checkpoint")
                resolved_mode = None
                print("intensity_reference=model_" + active_profile
                      + " (authored local workload learned from audio)", flush=True)
            else:
                print(f"intensity_reference={resolved_mode}", flush=True)
            kwargs = dict(audio=audio, windows=windows, tempo_changes=tempo_changes,
                          first=first, span=intensity_span, song_duration=song_duration)
            if resolved_mode is not None:
                if resolved_mode != "loudness":
                    kwargs["mode"] = resolved_mode
                intensity = intensity_module.window_factors(**kwargs)
        if tail_damping:
            intensity = intensity_module.combine(
                intensity,
                intensity_module.tail_factors(len(windows), windows=tail_damping_windows,
                                              strength=tail_damping_strength),
            )
        print(f"intensity_follow={'on' if intensity_follow else 'off'} "
              f"span={intensity_span:.2f} tail_damping={'on' if tail_damping else 'off'} "
              f"factors: {intensity_module.describe(intensity)}", flush=True)

    # Audible span: the generation clock is the file, so without this the silent
    # lead-in and the trailing silence are asked for notes like any other window.
    # Loudness alone is not enough -- a quiet attack stays under the loudness
    # threshold (measured: the first attack at 1.98s while loudness only crossed
    # 15% at 2.40s), so the bounds take loudness *or* onset evidence.
    shares: list[float] | None = None
    span: tuple[float, float] | None = None
    if audio_bounds_trim:
        from maichart.generation import intensity as bounds_module

        activity = bounds_module.music_activity(audio, loudness_threshold=silence_threshold)
        found = activity.bounds() if activity is not None else None
        span = bounds_module.trimmed_span(found, lead=audio_bounds_lead, song_duration=song_duration)
        if activity is not None:
            shares = activity.window_shares(
                windows, tempo_changes=tempo_changes, first=first, song_duration=song_duration,
            )
        if found is None:
            print("audio_bounds music=none (整轨既无响度也无起音：不裁剪)", flush=True)
        else:
            print(f"audio_bounds music={found[0]:.2f}-{found[1]:.2f}s "
                  f"trimmed={span[0]:.2f}-{span[1]:.2f}s lead={audio_bounds_lead:.2f}s "
                  f"{activity.describe()} "
                  f"window_skip_below={silence_window_share:.2f}", flush=True)
    for index, (start, end) in enumerate(windows):
        # A window with no audible music at all gets no notes and no decode: the
        # chart is supposed to be silent there, and skipping also saves the whole
        # forward pass (a long silent outro used to cost one decode per window).
        share = shares[index] if shares is not None else None
        if share is not None and share < silence_window_share:
            print(f"event_window={index + 1}/{len(windows)} beats=[{start},{end}) "
                  f"events=0 skipped=silence audible={share:.0%}", flush=True)
            continue
        begin_beat = max(Fraction(0), start - Fraction(audio_margin_beats)) if audio_margin_beats else start
        end_beat = end + Fraction(audio_margin_beats) if audio_margin_beats else end
        begin_sec = max(0.0, seconds_at_beat(begin_beat, tempo_changes, first))
        end_sec = max(begin_sec, seconds_at_beat(end_beat, tempo_changes, first))
        if song_duration:
            end_sec = min(end_sec, float(song_duration))
        if index and end_sec - begin_sec < min_window_sec:
            break
        # The lead-in is what the model has already written for those beats.
        prefix_ids: list[int] = []
        offset = Fraction(0)
        if context_beats > 0:
            from maichart.generation.event_ctx import context_start_beat

            offset = start - context_start_beat(start, context_beats)
            if offset:
                # A checkpoint trained before the position origin was unified
                # numbers its window from 0 and its alphabet stops at
                # segment_beats.  Asking it for position 8 (the block offset)
                # would leave the grammar with no legal position token at all,
                # which is what made it emit <eos> after a couple of notes.
                limit = _vocabulary_position_limit(vocabulary)
                if limit is None or offset > limit:
                    if not results:
                        print(f"context_position_convention=legacy (alphabet tops out at {limit}; "
                              f"window positions stay window-relative)", flush=True)
                    offset = Fraction(0)
            prefix_ids = _context_prefix(vocabulary, results, start, context_beats, context_tokens)
        window_budget = budget - len(prefix_ids)
        if window_budget < 3:
            print(f"event_window={index + 1}/{len(windows)} skipped: context prefix "
                  f"({len(prefix_ids)} tokens) leaves no decoding budget", flush=True)
            break
        if intensity_curve is not None and curve_strength > 0.0 and intensity[index] <= 1e-9:
            sequence, curve_stats = _curve_filter_sequence(
                EventSequence([], []), start_beat=start, tempo_changes=tempo_changes, first=first,
                curve_profile=intensity_curve,
                base_density=float(condition.radar.get("note_density", 0.5)),
                window_start_sec=max(0.0, float(seconds_at_beat(start, tempo_changes, first))),
                window_end_sec=min(float(song_duration), max(0.0, float(seconds_at_beat(end, tempo_changes, first)))),
                audio=audio)
            curve_stats["factor"] = intensity[index]
            results.append(_DecodedWindow(sequence=sequence, start_beat=start, end_beat=end,
                                          closed=True, intensity_curve=curve_stats))
            print(f"event_window={index + 1}/{len(windows)} beats=[{start},{end}) "
                  "events=0 skipped=intensity_curve_rest", flush=True)
            continue
        segment_audio = audio_segment_tensor(
            audio, begin_sec, end_sec, song_duration=song_duration, bpm=bpm,
            start_beat=_time_to_beat(float(begin_sec) - first, _normalize_tempo_map(tempo_changes, bpm)),
            first=first, tempo_changes=tempo_changes,
        )
        expected_dim = _expected_input_dim(model)
        if expected_dim is not None:
            segment_audio = _fit_feature_dim(segment_audio, expected_dim, where="segment")
        # Apply this window's density intention alongside its local audio.
        window_condition = condition
        window_vector = base_vector
        active_profile = str(getattr(model, "planner_profile", "legacy_v1"))
        structure_manual = (intensity_curve is not None and curve_strength > 0
                            and is_music_structure(active_profile))
        if intensity[index] != 1.0 and not structure_manual:
            from maichart.generation.conditions import scale_intensity

            window_condition = scale_intensity(condition, intensity[index])
            window_vector = window_condition.to_vector()
        plan_controls = None
        # Fine-rhythm bias is a planner control, so it needs a checkpoint whose
        # planner was trained to predict the fine-rhythm channel.  For any other
        # profile the flag is inert and the model-family check already warned.
        if active_profile == MUSIC_STRUCTURE_V3_PROFILE and fine_note_strength != 0.5:
            plan_controls = {"__fine_rhythm_bias": float(fine_note_strength)}
        if intensity_curve is not None and curve_strength > 0.0:
            core_duration = max(0.0, float(seconds_at_beat(end, tempo_changes, first)
                                           - seconds_at_beat(start, tempo_changes, first)))
            # The planner itself blends with its learned baseline. Supply the
            # unblended (but range-limited) target so strength is applied once.
            control_factor = 2.0 * curve_api.curve_window_values(
                control_curve, float(seconds_at_beat(start, tempo_changes, first)),
                float(seconds_at_beat(end, tempo_changes, first)))
            curve_controls = _curve_plan_controls(model, condition, control_factor, core_duration,
                                                 strength=curve_strength)
            if curve_controls is not None:
                plan_controls = {**(plan_controls or {}), **curve_controls}
        planner_kwargs = {"plan_controls": plan_controls} if plan_controls is not None else {}
        if local_onset_bias_strength > 0:
            from maichart.generation.event_audio import segment_frame_times
            planner_kwargs["onset_context"] = {
                "frame_times": segment_frame_times(begin_sec, end_sec),
                "start_beat": start, "position_offset": offset,
                "tempo_changes": tempo_changes, "first": first,
                "strength": local_onset_bias_strength,
            }
        feedback = None
        if intensity_curve is not None and ((curve_strength > 0 and intensity_curve_mode in {"follow", "hard"})
                                            or intensity_curve.get("rest_regions")):
            from maichart.generation.density_feedback import CurveDensityFeedback
            def make_feedback():
                return CurveDensityFeedback(intensity_curve,
                    base_density=float(condition.radar.get("note_density", .5)),
                    start_beat=start, end_beat=end, tempo_changes=tempo_changes, first=first,
                    position_offset=offset, mode=intensity_curve_mode, strength=curve_strength,
                    prior_cells=feedback_cells,
                    owned_start_sec=(float(seconds_at_beat(results[-1].end_beat, tempo_changes, first))
                                     if results else None))
            feedback = make_feedback()
            planner_kwargs["density_feedback"] = feedback
        if is_music_structure(active_profile):
            from maichart.generation.event_audio import FEATURE_RATE_HZ
            core_low = float(seconds_at_beat(start, tempo_changes, first))
            core_high = float(seconds_at_beat(end, tempo_changes, first))
            sample_times = begin_sec + torch.arange(segment_audio.size(0)) / FEATURE_RATE_HZ
            planner_kwargs["plan_audio_mask"] = ((sample_times < core_low) | (sample_times >= core_high)).to(device).unsqueeze(0)
        window_started = time.perf_counter()
        def decode_attempt(attempt):
            nonlocal feedback
            if feedback is not None:
                feedback = make_feedback()
                planner_kwargs["density_feedback"] = feedback
            try:
                sequence = _decode(
                    model, vocabulary, segment_audio, global_audio, window_vector, device,
                    max_tokens=window_budget, temperature=temperature,
                    sample=sample, seed=seed + index + attempt * 1000003, guidance_scale=guidance_scale,
                    top_p=top_p, repetition_penalty=repetition_penalty,
                    repetition_window=repetition_window,
                    atom_catalog=atom_catalog, prefix_ids=prefix_ids, position_offset=offset, condition_default=condition_default,
                    position_end=offset + end - start,
                    **planner_kwargs,
                )
                closed = True
            except ValueError as exc:
                # Only an exhausted budget is recoverable: the window ran out of room
                # instead of reaching a grammar dead end.  A real grammar error must
                # still surface, so it is re-raised.
                #
                # The salvage decode repeats the window with truncation allowed.  It
                # now runs against a budget sized to real charts, so a window that
                # exhausts it is rare; previously this retry repeated a 2048-step
                # decode on top of the 2048-step failure.
                message = str(exc)
                if "before emitting <eos>" not in message and "exceeds max_tokens" not in message:
                    raise
                if feedback is not None:
                    feedback = make_feedback()
                    planner_kwargs["density_feedback"] = feedback
                sequence = _decode(
                    model, vocabulary, segment_audio, global_audio, window_vector, device,
                    max_tokens=window_budget, temperature=temperature,
                    sample=sample, seed=seed + index + attempt * 1000003, guidance_scale=guidance_scale,
                    top_p=top_p, repetition_penalty=repetition_penalty,
                    repetition_window=repetition_window,
                    atom_catalog=atom_catalog, allow_truncated=True, prefix_ids=prefix_ids,
                    position_offset=offset, condition_default=condition_default,
                    position_end=offset + end - start,
                    **planner_kwargs,
                )
                closed = False
            curve_stats = None
            if intensity_curve is not None and curve_strength > 0.0:
                sequence, curve_stats = _curve_filter_sequence(
                    sequence, start_beat=start, tempo_changes=tempo_changes, first=first,
                    curve_profile=intensity_curve,
                    base_density=float(condition.radar.get("note_density", 0.5)),
                    window_start_sec=max(0.0, float(seconds_at_beat(start, tempo_changes, first))),
                    window_end_sec=min(float(song_duration), max(0.0, float(seconds_at_beat(end, tempo_changes, first)))),
                    audio=audio, apply_guard=intensity_curve_mode == "strict")
                curve_stats["factor"] = intensity[index]
            sequence, rest_removed = _rest_filter_sequence(sequence, start_beat=start, tempo_changes=tempo_changes,
                                                            first=first, profile=intensity_curve)
            if feedback is not None or rest_removed:
                curve_stats = curve_stats or {}
                curve_stats["density_feedback"] = dict(feedback.stats) if feedback is not None else {}
                curve_stats["rest_removed_cells"] = rest_removed
            # Stay inside the audible span: the silent lead-in and the trailing
            # silence must not receive notes, and trimming before the sequence is
            # stored keeps the next window's `<ctx>` lead-in consistent with it.
            removed = 0
            if span is not None:
                trimmed = _trim_sequence(sequence, start, tempo_changes=tempo_changes, first=first,
                                         low_sec=span[0], high_sec=span[1])
                removed = len(sequence.beats) - len(trimmed.beats)
                sequence = trimmed
            return sequence, closed, curve_stats, removed
        sequence, closed, curve_stats, removed, playability, original_sequence = _select_playable_window(
            decode_attempt, results, start=start, end=end, tempo_changes=tempo_changes,
            first=first, config=post_process_config, seed=seed + index)
        if feedback is not None:
            absolute_points = [EventPoint(start + beat, text) for beat, text in zip(sequence.beats, sequence.note_texts)]
            parsed_cells = parse_event_notes(absolute_points, tempo_changes, first)
            # Earlier-window ownership mirrors merge: overlapping later windows
            # must not count the same heads twice in an absolute-time quota.
            previous_high = float(seconds_at_beat(results[-1].end_beat, tempo_changes, first)) if results else 0.0
            feedback_cells.extend({"time_sec": float(point.time_sec), "heads": len(point.notes)}
                                  for point in parsed_cells.points if point.notes and float(point.time_sec) >= previous_high - 1e-8)
        results.append(_DecodedWindow(sequence=sequence, start_beat=start, end_beat=end,
                                      closed=closed, intensity_curve=curve_stats,
                                      playability=playability, original_sequence=original_sequence))
        # Per-window progress.  Generation is one forward pass per token, so a
        # long silence here previously looked like a hang; reporting each window
        # as it closes makes the remaining work visible.  The intensity factor
        # and the density it produced are printed too, so a run that claims to
        # follow the music can be audited from the log alone.
        # `share` is None whenever the audible-span trim is off (or the track has
        # no measurable loudness), so it must not be formatted unconditionally.
        audible_text = f"audible={share:.0%} " if share is not None else ""
        trimmed_text = f"trimmed_out_of_bounds={removed} " if removed else ""
        print(
            f"event_window={index + 1}/{len(windows)} beats=[{start},{end}) "
            f"events={len(sequence.beats)} closed={closed} context_tokens={len(prefix_ids)} "
            f"intensity={intensity[index]:.2f} note_density={window_condition.radar['note_density']:.3f} "
            f"{audible_text}{trimmed_text}"
            f"seconds={time.perf_counter() - window_started:.2f}",
            flush=True,
        )
    return results


def _select_playable_window(decode_attempt, prior, *, start, end, tempo_changes, first, config, seed):
    """Retry only a conflicting new window, preserving all preceding context.

    Every attempt passes the same density/rest/audio-boundary filters. A later
    window sees the chosen/repaired result. No optional policy means one decode.
    """
    if config is None or not config.enable_conflict_detection:
        sequence, closed, curve_stats, removed = decode_attempt(0)
        return sequence, closed, curve_stats, removed, None, None
    attempts = 1 + (config.conflict_retries if config and config.enable_conflict_detection
                    and config.conflict_repair_mode != "none" else 0)
    floor = max(start, prior[-1].end_beat) if prior else start
    best = None
    history = []
    for attempt in range(attempts):
        sequence, closed, curve_stats, removed = decode_attempt(attempt)
        trial = _DecodedWindow(sequence, start, end, closed, curve_stats)
        merged = _merge_windows([*prior, trial], overlap_beats=1)
        result = detect_chart_conflicts(merged, tempo_changes, first, config) if config else None
        hard = [issue for issue in result.conflicts_detected if issue.severity == "hard"
                and any(note.beat >= floor for note in issue.notes)] if result else []
        history.append({"attempt": attempt, "seed": seed + attempt * 1000003, "hard": len(hard)})
        if best is None or len(hard) < best[0]:
            best = (len(hard), sequence, closed, curve_stats, removed, merged, result)
        if not hard:
            break
    _, sequence, closed, curve_stats, removed, merged, result = best
    original = sequence
    report = result.to_dict() if result is not None else None
    if config and config.enable_conflict_detection and config.conflict_repair_mode == "local":
        repaired = process_chart_points(merged, tempo_changes, first, config, repair_after_beat=floor)
        sequence = EventSequence([p.beat - start for p in repaired.points if floor <= p.beat < end],
                                 [p.note_text for p in repaired.points if floor <= p.beat < end])
        report = repaired.to_dict()
    if report is not None:
        report["attempts"] = history
        report["scope"] = "previous_context_and_current_window"
    return sequence, closed, curve_stats, removed, report, original


def _context_prefix(vocabulary: EventVocabulary, windows: list[_DecodedWindow], start: Fraction,
                    context_beats: int, context_tokens: int) -> list[int]:
    """Token ids of the lead-in block for a window starting at ``start``.

    Built from the notes already generated -- never from ground truth -- and
    using the same position convention as training: positions are relative to
    the start of the lead-in block.
    """

    from maichart.generation.event_ctx import lead_in_events, prefix_token_ids

    # Match final merge ownership: an overlap belongs to the earlier window.
    # Duplicate AT positions otherwise make prefix encoding fail and drop all
    # context exactly at the seam where it is needed.
    points = _merge_windows(windows, overlap_beats=1)
    if not points:
        return _empty_prefix(vocabulary)
    events = lead_in_events(points, start, context_beats)
    if not events:
        return _empty_prefix(vocabulary)
    try:
        return prefix_token_ids(vocabulary, events, context_tokens)
    except ValueError as exc:
        # An event outside the checkpoint vocabulary must not abort a song that
        # is otherwise fine: decode this window without context instead.  Losing
        # the lead-in is what makes a window forget what came before it (visible
        # as a configuration switch or a repeated block), so the drop must be
        # visible rather than silent.
        if not getattr(_context_prefix, "warned", False):
            _context_prefix.warned = True
            print(f"context_prefix_fallback lead-in dropped once: {str(exc)[:160]} "
                  f"(window beat {start})", flush=True)
        _context_prefix.dropped = getattr(_context_prefix, "dropped", 0) + 1
        return _empty_prefix(vocabulary)


def _empty_prefix(vocabulary: EventVocabulary) -> list[int]:
    """The empty lead-in layout training always used.

    ``prefix_token_ids`` returns ``<ctx> <ctx_end>`` even for an empty lead-in,
    so a song's first window (and any window saved by the vocabulary fallback)
    must still open with the delimiter pair: without them the decoder sees a
    layout no training window ever had.
    """

    return [vocabulary.token_to_id[CTX_OPEN], vocabulary.token_to_id[CTX_CLOSE]]


def apply_contract_condition_guard(condition: GenerationCondition, contract_params: dict,
                                   *, default_level: float) -> tuple[GenerationCondition, str]:
    """Neutralise condition values a checkpoint was never trained to read.

    ``condition_mask_prob=1.0`` training replaces every control but ``level``
    with a constant default, so profile-supplied radar/style values are out of
    distribution -- and feeding them did exactly what the mask was introduced to
    prevent (the decoder holding one arbitrary style for hundreds of beats).
    The guard replaces masked slots with the value the contract recorded and
    reports what it dropped; it is skipped entirely when the supplied condition
    is already neutral.

    Returns ``(condition, note)``; a non-empty note means something was dropped.
    """

    params = dict(contract_params or {})
    # Accept either the whole contract dict or its condition_params sub-dict:
    # callers have passed both shapes during the rollout, and guessing wrong
    # would silently fire the guard on checkpoints that learned the controls.
    if isinstance(params.get("condition_params"), dict):
        params = dict(params["condition_params"])
    default = float(params.get("default", 0.5))
    # Old checkpoints may omit condition_params entirely.  Such checkpoints
    # were trained before the contract was recorded; do not silently assume
    # mask_prob=1.0 and destroy a supplied profile.  The guard is only strict
    # when the checkpoint explicitly records its masking policy.
    if "mask_prob" not in params or float(params["mask_prob"]) < 1.0:
        return condition, ""
    supplied = {**(condition.radar or {}), **(condition.style or {})}
    non_default = {key: value for key, value in supplied.items()
                   if abs(float(value) - default) > 1e-6}
    if not non_default:
        return condition, ""
    names = ", ".join(sorted(non_default)[:4])
    print(f"condition_profile_ignored mask_prob=1.0 keeps only `level`; "
          f"dropped {names} (values the checkpoint never trained on)", flush=True)
    neutral = GenerationCondition(
        level=float(condition.level) or default_level, difficulty=condition.difficulty,
        radar={key: default for key in condition.radar},
        style={key: default for key in condition.style})
    return neutral, "neutralised"


def resolve_generation_condition(*, level, condition_file=None, condition_data=None,
                                 condition_profile=None, contract_params=None):
    """Overlay only explicit fields on generation defaults, then apply the contract.

    A level-only file is a partial request: omitted controls must keep their
    calibrated baseline rather than becoming GenerationCondition's numeric null.
    """
    if condition_file is not None and condition_data is not None:
        raise ValueError("provide condition_file or condition_data, not both")
    condition = GenerationCondition(level=level, radar=GENERATION_DEFAULTS,
                                    style=GENERATION_DEFAULTS)
    source = "corpus_defaults"
    if condition_profile:
        from maichart.training.condition_profile import condition_from_profile, load_profile

        condition = condition_from_profile(load_profile(condition_profile), level=level)
        source = f"condition_profile={condition_profile}"
    if condition_file is not None or condition_data is not None:
        raw = (json.loads(Path(condition_file).read_text(encoding="utf-8-sig"))
               if condition_file is not None else condition_data)
        if not isinstance(raw, dict):
            raise ValueError("condition file must contain a JSON object")
        explicit = GenerationCondition.from_dict(raw)
        condition = GenerationCondition(
            level=explicit.level if "level" in raw else condition.level,
            difficulty=explicit.difficulty if "difficulty" in raw else condition.difficulty,
            radar={**condition.radar, **{key: explicit.radar[key]
                   for key in (raw.get("radar") or {}) if key in explicit.radar}},
            style={**condition.style, **{key: explicit.style[key]
                   for key in (raw.get("style") or {}) if key in explicit.style}},
        )
        source += "+condition_file"
    condition, note = apply_contract_condition_guard(
        condition, contract_params, default_level=condition.level)
    if note:
        source += "+contract_mask_1.0"
    return condition, source


def _merge_windows(windows: list[_DecodedWindow], *, overlap_beats: int = 0) -> list[EventPoint]:
    """Re-anchor segment-relative positions onto the full-song beat clock.

    Training stores ``point.beat - start``; the checkpoint vocabulary tops out
    just under ``segment_beats``.  Adding each window's start beat back is what
    turns "one 16-beat fragment" into a song-wide timeline.  Positions are also
    clamped into their own window so a stray relative value cannot collide with
    the next window, and the first of two equal beats wins because the
    compiler rejects duplicate beats outright.

    With overlapping windows (``overlap_beats > 0``) the earlier window keeps
    the shared beats: its output is what the next window was conditioned on, so
    letting the next window overwrite them would contradict its own context.
    """

    points: list[EventPoint] = []
    seen: set[Fraction] = set()
    for index, window in enumerate(windows):
        floor = window.start_beat
        if overlap_beats > 0 and index:
            floor = max(floor, windows[index - 1].end_beat)
        for beat, text in zip(window.sequence.beats, window.sequence.note_texts):
            absolute = Fraction(beat) + window.start_beat
            if absolute < floor or absolute >= window.end_beat:
                continue
            if absolute in seen:
                continue
            seen.add(absolute)
            points.append(EventPoint(absolute, text))
    points.sort(key=lambda point: point.beat)
    return points


def _filter_points_to_audio_span(points: list[EventPoint], *, tempo_changes, first: float,
                                 span: tuple[float, float] | None, tolerance: float = 0.02) -> list[EventPoint]:
    if span is None:
        return points
    low, high = float(span[0]) - max(0.0, tolerance), float(span[1]) + max(0.0, tolerance)
    return [point for point in points
            if low <= float(seconds_at_beat(point.beat, tempo_changes, first)) <= high]


def _filter_points_to_file_span(points: list[EventPoint], *, tempo_changes, first: float,
                                duration_sec: float) -> list[EventPoint]:
    """Keep note heads strictly in the half-open audio-file interval [0, end).

    Audible-span trimming has its own tolerance; it must not leak through this
    unconditional file boundary, including a negative chart/audio offset.
    """

    high = float(duration_sec)
    kept = []
    for point in points:
        # Generated event positions use a nonnegative chart-beat domain.
        # The file-clock check below separately handles negative audio offsets.
        if Fraction(point.beat) < 0:
            continue
        when = float(seconds_at_beat(point.beat, tempo_changes, first))
        if 0.0 <= when < high:
            kept.append(point)
    return kept


def _candidate_quality_score(quality: dict) -> float:
    """Rank candidates without hiding structural or syntax failures."""
    if not quality or not quality.get("syntax_valid", False):
        return float("-inf")
    score = 0.0
    if quality.get("playable_verified", False):
        score += 2.0
    score -= 10.0 * float(quality.get("note_warning_count", 0) or 0)
    audio_follow = quality.get("audio_follow") or {}
    curve = quality.get("intensity_curve") or {}
    correlation = None if curve else audio_follow.get("rms_density_correlation")
    if correlation is not None and math.isfinite(float(correlation)):
        score += max(-1.0, min(1.0, float(correlation)))
    if curve:
        bins = curve.get("local_bins", [])
        errors = [abs(float(b.get("actual_heads", 0.0)) - float(b.get("desired_heads", 0.0))) /
                  max(float(b.get("end_sec", 0.0)) - float(b.get("start_sec", 0.0)), 1e-9)
                  for b in bins]
        if errors:
            score -= min(2.0, sum(errors) / len(errors)) * 0.25
    control = quality.get("control") or {}
    deltas = control.get("actual_minus_requested") or {}
    for section in ("radar", "style"):
        for value in (deltas.get(section) or {}).values():
            if value is not None and math.isfinite(float(value)):
                score -= min(1.0, abs(float(value))) * 0.1
    return float(score)


def _failure_payload(error: Exception, *, candidate: int | None = None) -> dict:
    message = str(error)
    truncated = len(message) > 2000
    if truncated:
        message = message[:2000] + " ... [message truncated]"
    return {
        "schema": "maichart-event-generation-quality-v1",
        "generation_status": "failed",
        "syntax_valid": False,
        "playable_verified": False,
        "candidate": candidate,
        "fatal_errors": [{"code": "generation_failed", "message": message}],
        "error_type": type(error).__name__,
        "error_message_truncated": truncated,
    }


def _write_failure_report(output: Path, quality_path: Path, error: Exception, *, candidate: int | None = None,
                          diagnostics: dict | None = None) -> None:
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _failure_payload(error, candidate=candidate)
    if diagnostics is not None:
        payload["diagnostics"] = diagnostics
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    quality_path.write_text(serialized, encoding="utf-8")
    output.with_suffix(".error.json").write_text(serialized, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-ir")
    parser.add_argument("--quality-report")
    parser.add_argument("--gold-stats",
                        help="gold_stats.json from maichart.training.gold_set; when given, the quality "
                             "report gains an acceptance verdict comparing this chart against the gold "
                             "percentiles.")
    parser.add_argument("--condition-file")
    parser.add_argument("--level", type=float, default=GENERATION_DEFAULTS["level"],
                        help=f"Target difficulty (corpus median {GENERATION_DEFAULTS['level']}).")
    parser.add_argument("--bpm", type=float)
    parser.add_argument("--tempo-map", help="Explicit tempo-map JSON file; takes priority over --bpm and automatic tempo.")
    parser.add_argument("--first", type=float, default=0.0)
    parser.add_argument("--tempo-mode", choices=("auto", "constant"),
                        default=GENERATION_SAMPLING_DEFAULTS["tempo_mode"],
                        help="'constant' pins the detected global tempo for the whole song; "
                             "'auto' also promotes high-confidence local changes. Constant is "
                             "the default because the local candidates are derived from the "
                             "onset tracker's beat intervals, which are unreliable in the "
                             "clock dimension even when they place beats well, and a spurious "
                             "change desynchronises every later note. It also matches the "
                             "corpus: only 11.6%% (231/1991) of the charts change tempo at all.")
    parser.add_argument("--guidance-scale", type=float, default=GENERATION_SAMPLING_DEFAULTS["guidance_scale"])
    parser.add_argument("--local-onset-bias-strength", type=float, default=0.0,
                        help="Soft AT-position guidance from a trained chart-onset head, 0..1; 0 disables. "
                             "Requires an onset-enabled checkpoint. Existing rest/density masks still apply.")
    parser.add_argument("--candidates", type=int, default=GENERATION_SAMPLING_DEFAULTS["candidates"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=GENERATION_SAMPLING_DEFAULTS["temperature"],
                        help="Sampling temperature. The panel, the server and this CLI all use "
                             f"{GENERATION_SAMPLING_DEFAULTS['temperature']}, so the same request produces the "
                             "same chart wherever it is launched from.")
    # Keep shared sampling defaults. Character reuse is not the same as a
    # repeated musical configuration: these controls cannot detect circles,
    # alternating buttons, or legitimate recurring musical motifs.
    parser.add_argument("--top-p", type=float, default=GENERATION_SAMPLING_DEFAULTS["top_p"],
                        help="Nucleus truncation for sampling: softmax the post-grammar-mask logits and "
                             "keep the smallest set of tokens whose cumulative probability reaches this "
                             "value; 0 or >=1 disables.")
    parser.add_argument("--repetition-penalty", type=float,
                        default=GENERATION_SAMPLING_DEFAULTS["repetition_penalty"],
                        help="CTRL-style penalty on non-control tokens seen inside the window: positive "
                             "logits divided, negative ones multiplied, once per distinct token; "
                             "sampling only, 1.0 disables. AT positions are exempt. This is a character "
                             "reuse penalty, not a detector of repeated chart configurations.")
    parser.add_argument("--repetition-window", type=int,
                        default=GENERATION_SAMPLING_DEFAULTS["repetition_window"],
                        help="How many tokens (context prefix included) to scan for character reuse. "
                             "This length has no fixed beat or time equivalent; note text length and "
                             "density determine how much music it covers.")
    parser.add_argument("--sample", dest="sample", action="store_true",
                        default=GENERATION_SAMPLING_DEFAULTS["sample"],
                        help="Sample instead of argmax (default). Sampling is what the panel does; greedy "
                             "decoding loops on the highest-probability cell and produces flat charts.")
    parser.add_argument("--greedy", dest="sample", action="store_false",
                        help="Argmax decoding. Deterministic, but repetitive.")
    parser.add_argument("--max-tokens", type=int, default=GENERATION_SAMPLING_DEFAULTS["max_tokens"],
                        help="Per-window decoder budget, and *not* the training contract's --max-tokens "
                             "(1024 = longest training target). A window's budget must cover the context "
                             "prefix (up to --context-tokens=320) plus the window itself (measured p50=155, "
                             "p100=444 tokens), which is 320+444=764, so 768 is the smallest safe value; "
                             "raise it if --context-tokens is raised.")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--intensity-follow", dest="intensity_follow", action="store_true",
                        default=GENERATION_SAMPLING_DEFAULTS["intensity_follow"],
                        help="Scale per-window note_density/peak using the reference selected by "
                             "--intensity-mode. Multifeature mode combines rhythm, spectral/timbral "
                             "activity and harmonic/local contrast with a limited loudness contribution. "
                             "The model also receives local audio; style controls remain unchanged.")
    parser.add_argument("--no-intensity-follow", dest="intensity_follow", action="store_false",
                        help="Disable intensity following (the pre-flag behaviour).")
    parser.add_argument("--intensity-mode", choices=("auto", "model", "loudness", "multifeature"),
                        default=GENERATION_SAMPLING_DEFAULTS.get("intensity_mode", "multifeature"),
                        help="Automatic intensity reference: multifeature musical tension or legacy loudness.")
    parser.add_argument("--intensity-curve",
                        help="Validated maichart-intensity-curve-v1 profile. When supplied it "
                             "replaces RMS following and tail damping and applies a local density guard.")
    parser.add_argument("--intensity-curve-strength", type=float, default=0.65,
                        help="曲线影响强度；0 完全旁路手工曲线，1 完全采用曲线因子。")
    parser.add_argument("--intensity-curve-mode", choices=("follow", "hard", "soft", "strict"), default="follow",
                        help="follow 生成中柔性密度反馈；hard 生成中严苛上限；soft/strict 保留旧规划/后过滤行为。")
    parser.add_argument("--intensity-follow-span", type=float,
                        default=GENERATION_SAMPLING_DEFAULTS["intensity_follow_span"],
                        help="How far a window may deviate from the requested density, as a fraction: "
                             "0.5 means the factor stays inside 0.5x..1.5x.")
    parser.add_argument("--tail-damping", dest="tail_damping", action="store_true",
                        default=GENERATION_SAMPLING_DEFAULTS["tail_damping"],
                        help="Linearly taper the density condition over the last --tail-damping-windows "
                             "windows.  A musical prior about outros (songs usually wind down after "
                             "their last note even when the audio is still loud), off by default.")
    parser.add_argument("--tail-damping-windows", type=int,
                        default=GENERATION_SAMPLING_DEFAULTS["tail_damping_windows"],
                        help="How many trailing windows take part in the taper.")
    parser.add_argument("--tail-damping-strength", type=float,
                        default=GENERATION_SAMPLING_DEFAULTS["tail_damping_strength"],
                        help="Density multiplier reached at the very last window: 0.6 => the final "
                             "window is asked for 40%% of the requested density.")
    parser.add_argument("--audio-bounds-trim", dest="audio_bounds_trim", action="store_true",
                        default=GENERATION_SAMPLING_DEFAULTS["audio_bounds_trim"],
                        help="Keep the chart inside the audible part of the song.  The clock is the "
                             "file, so a silent lead-in and trailing silence are cut into ordinary "
                             "windows and asked for notes: measured on a real run, the music was "
                             "audible 2.44-142.94s while the chart wrote 8 notes before it and 9 "
                             "notes after it.  Windows with no audible music are emitted empty, and "
                             "events outside the audible span are dropped (which also keeps the "
                             "`<ctx>` lead-in consistent).")
    parser.add_argument("--no-audio-bounds-trim", dest="audio_bounds_trim", action="store_false",
                        help="Disable the audible-span trim (the pre-flag behaviour).")
    parser.add_argument("--audio-bounds-lead", type=float,
                        default=GENERATION_SAMPLING_DEFAULTS["audio_bounds_lead"],
                        help="Seconds of tolerance beyond the audible span.  Default 0 = the chart may "
                             "not write a single note over silence; raise it only when a pickup "
                             "before the first onset or a decaying final chord must be charted "
                             "(0.5s of tolerance put a real run's first note 0.46s before the music).")
    parser.add_argument("--silence-threshold", type=float,
                        default=GENERATION_SAMPLING_DEFAULTS["silence_threshold"],
                        help="A frame counts as audible when its song-relative loudness is above this "
                             "fraction of the song's median (1.0 = median).  Measured on a real song, "
                             "0.10-0.40 all put the first audible frame at 2.4s, so the exact value is "
                             "not critical.")
    parser.add_argument("--silence-window-share", type=float,
                        default=GENERATION_SAMPLING_DEFAULTS["silence_window_share"],
                        help="A window whose audible-frame share is below this is emitted empty "
                             "without decoding at all.")
    parser.add_argument("--condition-profile",
                        help="Corpus condition profile (condition_profile.json). Used as the default control "
                             "vector when no --condition-file is given; without it every control falls back "
                             "to 0.5, which is far outside the range the corpus actually contains.")
    # Cross-window family.  All four default to 0/off, which reproduces the
    # event_v1 behaviour exactly; the family entry point
    # (`maichart.generation.event_ctx_cli`) fills in its own defaults.
    parser.add_argument("--context-beats", type=int, default=0,
                        help="Seed each window with the notes already written for the preceding "
                             "N beats (cross-window family). 0 disables.")
    parser.add_argument("--audio-margin-beats", type=int, default=0,
                        help="Sample N extra beats of audio before and after each window, so the "
                             "encoder can see the surrounding music (cross-window family).")
    parser.add_argument("--context-tokens", type=int, default=192,
                        help="Token budget for the seeded lead-in block (cross-window family).")
    parser.add_argument("--overlap-beats", type=int, default=0,
                        help="Generate with overlapping windows of this overlap; the earlier window "
                             "keeps the shared beats (cross-window family).")
    # Rhythm bias and playability constraints (from GOAL_COMPLEX_RHYTHM_PLAYABILITY_FINAL.md)
    parser.add_argument("--rhythm-bias", choices=("auto", "uniform", "triplet", "dotted", "mixed"),
                        default="auto",
                        help="Deprecated rhythm-class switch. Only auto is implemented; "
                             "use fine-note-strength for relative per-window preference.")
    parser.add_argument("--fine-note-strength", type=float, default=0.5,
                        help="Relative fine-rhythm planner preference: 0 reduces, 0.5 follows each "
                             "window prediction, 1 increases. v3 only; no forced subdivision.")
    parser.add_argument("--enable-conflict-detection", action="store_true",
                        help="Enable playability conflict detection (hold-tap overlap, slide-tap collision, etc).")
    parser.add_argument("--detect-hard-conflicts", action=argparse.BooleanOptionalAction, default=True,
                        help="Detect hard conflicts (game judgment system conflicts). Default True.")
    parser.add_argument("--detect-soft-conflicts", action="store_true",
                        help="Detect soft conflicts (ergonomic issues, large jumps). Default False.")
    parser.add_argument("--conflict-repair-mode", choices=("none", "local", "reject"),
                        default="none",
                        help="Conflict repair mode: none (detect only), local (remove conflicting notes), "
                             "reject (abort on hard conflicts). Default none.")
    parser.add_argument("--conflict-retries", type=int, default=0,
                        help="Optional conflicting-window retries, 0 disables, maximum 5.")
    parser.add_argument("--runtime-reference-mode", choices=("off", "report", "reject"), default="off",
                        help="Optional MiaCode runtime-reference sidecar; default off.")
    parser.add_argument("--runtime-reference-executable", type=Path,
                        help="Path to the user-provided miacode_muri_dump executable.")
    parser.add_argument("--runtime-reference-timeout-sec", type=float, default=30.0,
                        help="Per-candidate reference timeout in (0, 60] seconds.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--title", default="Generated maimai chart")
    parser.add_argument("--artist", default="")
    parser.add_argument("--chart-level", default="")
    parser.add_argument("--designer", default="AI")
    parser.add_argument("--strict-notes", action="store_true",
                        help="Reject the whole chart if any note cell repeats a button or uses "
                             "an unknown token. Off by default: a handful of dubious cells in a "
                             "500-note chart is not a reason to throw the chart away, so the "
                             "problems are reported in --quality-report and the chart is still "
                             "written.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not math.isfinite(args.fine_note_strength) or not 0 <= args.fine_note_strength <= 1:
        raise ValueError("fine-note-strength must be finite in 0..1")
    if not math.isfinite(args.local_onset_bias_strength) or not 0 <= args.local_onset_bias_strength <= 1:
        raise ValueError("local-onset-bias-strength must be finite in 0..1")
    if args.rhythm_bias != "auto":
        raise ValueError("rhythm-bias rhythm classes are not implemented; use auto with fine-note-strength")
    if args.runtime_reference_mode != "off" and not args.runtime_reference_executable:
        raise ValueError("runtime-reference-executable is required when runtime-reference-mode is report or reject")
    if not math.isfinite(float(args.runtime_reference_timeout_sec)) or not 0 < float(args.runtime_reference_timeout_sec) <= 60:
        raise ValueError("runtime-reference-timeout-sec must be finite and in (0, 60].")
    if args.runtime_reference_mode != "off" and not Path(args.runtime_reference_executable).expanduser().is_file():
        raise ValueError(f"runtime reference executable does not exist: {args.runtime_reference_executable}")
    post_process_config = PostProcessConfig(
        enable_conflict_detection=args.enable_conflict_detection,
        detect_hard_conflicts=args.detect_hard_conflicts,
        detect_soft_conflicts=args.detect_soft_conflicts,
        conflict_repair_mode=args.conflict_repair_mode,
        conflict_retries=args.conflict_retries,
    )
    if int(args.candidates) < 1:
        raise SystemExit("candidates must be positive.")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output_ir = Path(args.output_ir) if args.output_ir else output.with_suffix(".chart.json")
    quality_path = Path(args.quality_report) if args.quality_report else output.with_suffix(".quality.json")
    intensity_curve = None
    # Load the gold reference up front so a bad path fails before any GPU work.
    gold_reference = load_reference(args.gold_stats) if args.gold_stats else None
    if gold_reference is not None:
        print(f"gold_reference={args.gold_stats} charts={gold_reference.get('charts')}", flush=True)
    try:
        device = _device(args.device)
        model, vocabulary, checkpoint = load_event_model(args.checkpoint, device)
        capabilities = checkpoint_capabilities(checkpoint)
        if capabilities["errors"]:
            raise ValueError("checkpoint capability contract invalid: " + "; ".join(capabilities["errors"]))
        # 检测模型类型和兼容性
        model_family = detect_model_family(checkpoint)
        is_valid, warnings = validate_args_for_model(model_family, args, checkpoint=checkpoint)
        if warnings:
            for warning in warnings:
                print(f"warning: {warning}", flush=True)
        
        # 获取该模型支持的生成配置
        generation_config = get_generation_config_for_model(model_family, args, checkpoint=checkpoint)
        print(f"model_family={model_family.value}", flush=True)
        # The position embedding table is sized by the training contract, so the
        # checkpoint - not this run's flags - is the hard ceiling on how many
        # tokens one window can emit.  Requesting more used to raise "decoder
        # position exceeds max_tokens" partway through a window and discard the
        # whole candidate after minutes of GPU work.
        decode_budget = _clamp_decode_budget(args.max_tokens, model)
        audio = extract_event_audio(args.audio, sample_rate=args.sample_rate)
        if args.intensity_curve:
            from maichart.generation import music_intensity as curve_api
            intensity_curve = curve_api.validate_intensity_profile(
                curve_api.load_intensity_profile(args.intensity_curve),
                duration_sec=float(audio.duration_sec),
            )
            expected_hash = str(intensity_curve.get("audio_sha256") or "")
            actual_hash = str(curve_api.audio_sha256(args.audio))
            if expected_hash and expected_hash != actual_hash:
                raise ValueError("intensity curve audio_sha256 does not match --audio")
            if not expected_hash:
                raise ValueError("intensity curve must include audio_sha256 to prevent applying it to another song")
            print(f"intensity_curve={args.intensity_curve} schema={intensity_curve.get('schema')} "
                  f"audio_sha256={actual_hash[:12]} duration={audio.duration_sec:.3f}s", flush=True)
        explicit_tempo = normalize_tempo_map(json.loads(Path(args.tempo_map).read_text(encoding="utf-8-sig"))) if args.tempo_map else None
        detected_bpm = float(explicit_tempo[0].bpm if explicit_tempo else args.bpm or audio.tempo_bpm)
        if detected_bpm <= 0:
            raise ValueError("BPM could not be estimated. Provide --bpm.")
        # The tracker reports tempogram lag bins such as 3000/22, and in Simai
        # the BPM is the clock that turns every comma cell and every
        # {division} directive into real time.  An off-grid clock displaces
        # every note in the song, so an explicit --bpm is used verbatim while a
        # detected estimate is snapped to a musical value.
        bpm = detected_bpm if args.bpm is not None else _musical_bpm(detected_bpm)
        tempo_changes = explicit_tempo or _tempo_map(args.tempo_mode, audio, bpm, snap=args.bpm is None)
        bpm = float(tempo_changes[0].bpm)
        if intensity_curve is not None and str(intensity_curve.get("method", "")).startswith("learned_music_structure_"):
            if intensity_curve.get("tempo_map_sha256") != tempo_map_fingerprint(explicit_tempo):
                raise ValueError("Learned intensity curve was created with another explicit tempo map; regenerate it")
        contract_params = (checkpoint.get("data_contract", {}) or {}).get("condition_params") or {}
        condition, condition_source = resolve_generation_condition(
            level=args.level, condition_file=args.condition_file,
            condition_profile=args.condition_profile, contract_params=contract_params)
        print(f"condition_source={condition_source} level={condition.level:.2f} "
              f"radar={{{', '.join(f'{k}={v:.3f}' for k, v in condition.radar.items())}}} "
              f"style={{{', '.join(f'{k}={v:.3f}' for k, v in condition.style.items())}}}", flush=True)
        # The full-song memory must be built with the same helper training uses,
        # which re-evaluates the chart clock at each memory centre.  A raw
        # adaptive_avg_pool1d over the whole song averages beat_phase_sin/cos
        # across a full beat period, destroying the phase channels.
        #
        # The frame count comes from the checkpoint, not from a constant: the
        # encoder rejects a memory whose length differs from global_audio_tokens,
        # so a checkpoint trained with 256 frames cannot be generated with 64.
        global_frames = int(
            checkpoint.get("model_config", {}).get("global_audio_tokens") or GLOBAL_MEMORY_FRAMES
        )
        global_audio = compress_event_audio(
            audio, song_duration=audio.duration_sec, bpm=bpm, first=args.first,
            tempo_changes=tempo_changes, frames=global_frames,
        )
        expected_dim = _expected_input_dim(model)
        if expected_dim is not None:
            global_audio = _fit_feature_dim(global_audio, expected_dim, where="global_memory")
        duration_beats = beat_at_audio_end(audio.duration_sec, tempo_changes, args.first)
        segment_beats = int(checkpoint.get("data_contract", {}).get("segment_beats") or 16)
        overlap_beats = int(checkpoint.get("data_contract", {}).get("family_params", {}).get("overlap_beats", args.overlap_beats) or 0)
        windows = _segment_windows(duration_beats, segment_beats, overlap_beats=overlap_beats)
        if not windows:
            raise ValueError("Audio duration is too short to place any chart beat.")
        audio_identity = _file_identity(args.audio)
        checkpoint_identity = _file_identity(args.checkpoint)
        if intensity_curve is not None:
            validate_learned_curve_contract(
                intensity_curve, checkpoint_path=args.checkpoint,
                checkpoint_identity=checkpoint_identity, condition=condition,
                first=args.first, tempo_map=explicit_tempo,
                tempo_mode=args.tempo_mode, bpm_input=args.bpm,
            )
    except Exception as exc:
        _write_failure_report(output, quality_path, exc)
        print(f"generation_status=failed error={exc}", flush=True)
        return 1
    failures = 0
    candidate_records: list[tuple[Path, Path, float]] = []
    for candidate in range(args.candidates):
        target = _candidate_path(output, candidate, args.candidates)
        target_ir = _candidate_path(output_ir, candidate, args.candidates)
        target_quality = _candidate_path(quality_path, candidate, args.candidates)
        # A rejected candidate must never leave a previous successful chart at
        # its advertised output path. Preserve prior artifacts under history.
        previous = None
        artifacts = [target, target_ir, target_quality, target.with_suffix(".original.txt"),
                     target.with_suffix(".original.chart.json"), target.with_suffix(".repaired.txt")]
        for artifact in dict.fromkeys(artifacts):
            if artifact.exists():
                if artifact.resolve() in {Path(args.audio).resolve(), Path(args.checkpoint).resolve()}:
                    raise ValueError("Output must not overwrite the source audio or checkpoint")
                if previous is None:
                    previous = target.parent / ".history" / f"{target.stem}-{time.time_ns()}"
                    previous.mkdir(parents=True)
                artifact.replace(previous / artifact.name)
        quality = None
        candidate_seed = args.seed + candidate * 1000
        sample = args.sample or args.candidates > 1
        print(f"candidate={candidate + 1}/{args.candidates} seed={candidate_seed} "
              f"windows={len(windows)} max_tokens={decode_budget} device={device}", flush=True)
        try:
            decoded = _decode_windows(
                model, vocabulary, audio, global_audio, condition, device,
                windows=windows, tempo_changes=tempo_changes, first=args.first,
                song_duration=audio.duration_sec, bpm=bpm, max_tokens=decode_budget,
                temperature=args.temperature, sample=sample,
                seed=args.seed + candidate * 1000, guidance_scale=args.guidance_scale,
                top_p=args.top_p, repetition_penalty=args.repetition_penalty,
                repetition_window=args.repetition_window,
                atom_catalog=checkpoint.get("atom_catalog"),
                context_beats=args.context_beats, audio_margin_beats=args.audio_margin_beats,
                context_tokens=args.context_tokens,
                intensity_follow=args.intensity_follow,
                intensity_span=args.intensity_follow_span,
                intensity_mode=args.intensity_mode,
                intensity_curve=intensity_curve,
                intensity_curve_strength=args.intensity_curve_strength,
                intensity_curve_mode=args.intensity_curve_mode,
                tail_damping=args.tail_damping,
                tail_damping_windows=args.tail_damping_windows,
                tail_damping_strength=args.tail_damping_strength,
                audio_bounds_trim=args.audio_bounds_trim,
                audio_bounds_lead=args.audio_bounds_lead,
                silence_threshold=args.silence_threshold,
                silence_window_share=args.silence_window_share,
                condition_default=float(contract_params.get("default", 0.5)),
                fine_note_strength=float(generation_config.get("fine_note_strength", 0.5)),
                local_onset_bias_strength=float(generation_config.get("local_onset_bias_strength", 0.0)),
                post_process_config=post_process_config if args.enable_conflict_detection else None,
            )
            points = _merge_windows(decoded, overlap_beats=overlap_beats)
            # File-time is an unconditional hard boundary.  Audible-span
            # trimming is a musical heuristic and may be disabled, but a chart
            # event outside the audio file can never be valid.
            before_file = len(points)
            points = _filter_points_to_file_span(
                points, tempo_changes=tempo_changes, first=args.first,
                duration_sec=audio.duration_sec,
            )
            audible_span = None
            boundary_removed = before_file - len(points)
            if args.audio_bounds_trim:
                from maichart.generation.intensity import music_activity, trimmed_span

                activity = music_activity(audio, loudness_threshold=args.silence_threshold)
                audible_span = trimmed_span(
                    activity.bounds() if activity is not None else None,
                    lead=args.audio_bounds_lead, song_duration=audio.duration_sec,
                )
                before = len(points)
                points = _filter_points_to_audio_span(
                    points, tempo_changes=tempo_changes, first=args.first, span=audible_span,
                )
                boundary_removed += before - len(points)
            intentional_rest_only = intensity_curve is not None and any(
                region["start_sec"] <= 0 and region["end_sec"] >= audio.duration_sec
                for region in intensity_curve.get("rest_regions", []))
            if not points and not intentional_rest_only:
                raise ValueError("Generation produced no notes for any window inside the audible span.")
            original_points = points
            if args.conflict_repair_mode == "local":
                original_windows = [_DecodedWindow(w.original_sequence or w.sequence,
                                    w.start_beat, w.end_beat, w.closed) for w in decoded]
                original_points = _filter_points_to_file_span(
                    _merge_windows(original_windows, overlap_beats=overlap_beats),
                    tempo_changes=tempo_changes, first=args.first, duration_sec=audio.duration_sec)
                if args.audio_bounds_trim:
                    original_points = _filter_points_to_audio_span(original_points,
                        tempo_changes=tempo_changes, first=args.first, span=audible_span)
            conflict_result = process_chart_points(points, tempo_changes, args.first, post_process_config)
            points = conflict_result.points
            quality = assess_event_chart(points, tempo_changes, first=args.first,
                                         duration_sec=max(float(audio.duration_sec - args.first), 1e-3),
                                         target_condition=condition)
            quality["audio_sha256"] = audio_identity["sha256"]
            quality["checkpoint_identity"] = checkpoint_identity
            quality["audio_duration_sec"] = audio.duration_sec
            quality["timing_inputs"] = {"explicit_tempo_map": serialize_tempo_map(explicit_tempo) if explicit_tempo else None,
                                        "tempo_map_sha256": tempo_map_fingerprint(explicit_tempo),
                                        "first": args.first, "tempo_mode": args.tempo_mode}
            if previous is not None:
                quality["previous_output_backup"] = str(previous)
            if intensity_curve is not None:
                curve_windows = []
                for item in decoded:
                    stats = item.intensity_curve or {}
                    curve_windows.append({
                        "start_beat": str(item.start_beat), "end_beat": str(item.end_beat),
                        "factor": stats.get("factor", 0.0),
                        "local_bins": stats.get("local_bins", []),
                        "removed": int(stats.get("removed", 0) or 0),
                        "under_target": int(stats.get("under_target", 0) or 0),
                        "density_feedback": stats.get("density_feedback", {}),
                        "rest_removed_cells": stats.get("rest_removed_cells", 0),
                    })
                quality["intensity_curve"] = {
                    "schema": str(intensity_curve.get("schema", "maichart-intensity-curve-v1")),
                    "audio_sha256": intensity_curve.get("audio_sha256"),
                    "mode": str(args.intensity_curve_mode),
                    "strength": float(args.intensity_curve_strength),
                    "rest_regions": intensity_curve.get("rest_regions", []),
                    "target": "manual_curve_effective_factor",
                    "method": ({"follow": "planner_and_decode_soft_density_feedback",
                                "hard": "decode_hard_density_cap_with_whole_cell_guard",
                                "strict": "profile_trapezoid_2s_density_upper_guard",
                                "soft": "planner_guidance_without_postfilter"}[args.intensity_curve_mode]),
                    "limitations": [
                        ("Strict control caps complete event cells; it does not fill under-target density."
                         if args.intensity_curve_mode in {"hard", "strict"} else
                         "Soft control guides the plan and preserves generated cells; it does not guarantee target density."),
                        "Learned or audio-reference workload is not a verified difficulty rating or emotion label.",
                    ],
                    "local_bins": _final_curve_bins(
                        points, _effective_curve_profile(
                            intensity_curve, strength=args.intensity_curve_strength,
                            mode=args.intensity_curve_mode, span=args.intensity_follow_span),
                        tempo_changes=tempo_changes, first=args.first,
                        base_density=float(condition.radar.get("note_density", 0.5)), windows=decoded),
                    "windows": curve_windows,
                }
            quality["boundary_trim"] = {
                "enabled": bool(args.audio_bounds_trim),
                "span_seconds": list(audible_span) if audible_span is not None else None,
                "removed_after_merge": int(boundary_removed),
            }
            # one tail_step after the last note.  Padding to the audio-derived
            # duration emitted hundreds of empty {1} cells and forced pathological
            # divisions such as {3528} from the fractional remainder.
            if args.enable_conflict_detection:
                quality["playability"] = conflict_result.to_dict()
                quality["playability"]["scope"] = "complete_merged_song"
                quality["playability"]["windows"] = [
                    {"start_beat": str(w.start_beat), "end_beat": str(w.end_beat),
                     **(w.playability or {})} for w in decoded]
                quality["playability"]["repairs"] = [
                    repair for w in decoded for repair in (w.playability or {}).get("repairs", [])
                ] + conflict_result.repairs_applied
                # Preserve original diagnostics as well as the final recheck.
                before_repair = detect_chart_conflicts(original_points, tempo_changes, args.first,
                                                       post_process_config)
                quality["playability"]["original_details"] = [
                    c.to_dict() for c in before_repair.conflicts_detected]
                if not conflict_result.success:
                    quality["generation_status"] = "rejected_by_playability"
                    target_ir.write_text(json.dumps({"schema": "maichart-event-chart-v1",
                        "points": [p.to_dict() for p in points],
                        "tempo_changes": [{"beat": str(t.beat), "bpm": t.bpm} for t in tempo_changes],
                        "first": args.first, "end_beat": str(duration_beats),
                        "audio_duration_sec": audio.duration_sec, "audio_sha256": audio_identity["sha256"], "checkpoint_identity": checkpoint_identity, "quality": quality}, ensure_ascii=False, indent=2), encoding="utf-8")
                    target_quality.write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
                    print("candidate rejected by static playability references", flush=True)
                    failures += 1
                    continue
            try:
                text, chart = compile_event_points(
                    points, tempo_changes=tempo_changes, bpm=bpm, first=args.first,
                    title=args.title, artist=args.artist,
                    level=args.chart_level or str(condition.level), designer=args.designer,
                    difficulty=condition.difficulty, strict_notes=args.strict_notes,
                )
            except ValueError as compile_error:
                # Keep the exact event and quality artifacts even when the
                # round-trip compiler rejects a rare fractional grouping. This
                # distinguishes model quality from serialization failure and
                # lets the caller inspect the generated event timeline.
                quality["generation_status"] = "compiled_with_error"
                quality["compile_error"] = str(compile_error)
                quality["windows_decoded"] = len(decoded)
                quality["windows_closed"] = sum(1 for window in decoded if window.closed)
                quality["coverage_beats"] = str(points[-1].beat if points else Fraction(0))
                quality["song_beats"] = str(duration_beats)
                quality["structure"] = structural_metrics(
                    points, tempo_changes, args.first, audio_duration_sec=audio.duration_sec)
                quality["audio_follow"] = audio_follow_metrics(
                    points, audio, tempo_changes, args.first, duration_sec=audio.duration_sec)
                target_ir.write_text(json.dumps({
                    "schema": "maichart-event-chart-v1",
                    "points": [point.to_dict() for point in points],
                    "quality": quality,
                    "compile_error": str(compile_error),
                }, ensure_ascii=False, indent=2), encoding="utf-8")
                target_quality.write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"candidate={candidate + 1}/{args.candidates} generation_status=compiled_with_error "
                      f"error={compile_error}", flush=True)
                failures += 1
                continue
            if args.runtime_reference_mode != "off":
                runtime_reference = run_reference(
                    args.runtime_reference_executable,
                    text,
                    difficulty=condition.difficulty,
                    first=float(args.first),
                    timeout_sec=float(args.runtime_reference_timeout_sec),
                    mode=args.runtime_reference_mode,
                )
                quality["runtime_reference"] = runtime_reference
                if runtime_reference["status"] in {"rejected", "failed"}:
                    if args.runtime_reference_mode == "reject":
                        quality["generation_status"] = (
                            "rejected_by_runtime_reference"
                            if runtime_reference["status"] == "rejected"
                            else "runtime_reference_failed"
                        )
                        target_ir.write_text(json.dumps({
                            "schema": "maichart-event-chart-v1",
                            "points": [point.to_dict() for point in points],
                            "tempo_changes": [{"beat": str(t.beat), "bpm": t.bpm} for t in tempo_changes],
                            "first": args.first, "end_beat": str(duration_beats),
                            "audio_duration_sec": audio.duration_sec,
                            "audio_sha256": audio_identity["sha256"],
                            "checkpoint_identity": checkpoint_identity, "quality": quality,
                        }, ensure_ascii=False, indent=2), encoding="utf-8")
                        target_quality.write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
                        print(f"candidate rejected by runtime reference: {runtime_reference['reject_reason']}", flush=True)
                        failures += 1
                        continue
                    quality["generation_status"] = "success_with_warnings"
                if runtime_reference["status"] == "rejected":
                    quality["generation_status"] = "rejected_by_runtime_reference"
            # Report the note-level problems instead of discarding the chart.
            # A cell that strikes one button twice (or uses an unknown token) is a
            # real defect, but a handful of them among hundreds of good notes must
            # not cost the user the whole generation.  They are surfaced here, in
            # the quality report, so the chart is still usable and the defect is
            # explicit rather than silent.
            note_warnings = [issue for issue in quality.get("fatal_errors", [])
                             if issue.get("code") in ("duplicate_button_head", "unsupported_token")]
            # Always record the count, even when zero, so consumers can branch on
            # the field without having to check for its absence.
            quality["note_warnings"] = note_warnings
            quality["note_warning_count"] = len(note_warnings)
            for issue in note_warnings[:20]:
                print(f"warning: {issue.get('message') or issue}", flush=True)
            if len(note_warnings) > 20:
                print(f"warning: ... {len(note_warnings) - 20} more note-level warnings", flush=True)
            quality["seed"] = args.seed
            quality["candidate_seed"] = args.seed + candidate * 1000
            # Decode provenance: A/B comparisons must be able to state exactly
            # how a chart was sampled, or two arms cannot be told apart later.
            quality["decoding"] = {
                "sample": bool(sample), "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "repetition_penalty": float(args.repetition_penalty),
                "repetition_window": int(args.repetition_window),
                "guidance_scale": float(args.guidance_scale),
                "model_family": str((checkpoint.get("data_contract", {}) or {}).get("target_schema") or "event_v1"),
                "planner_profile": str(getattr(model, "planner_profile", "legacy_v1")),
                "fine_note_strength_requested": float(args.fine_note_strength),
                "fine_note_strength_effective": float(generation_config.get("fine_note_strength", 0.5)),
                "local_onset_bias_strength": float(generation_config.get("local_onset_bias_strength", 0.0)),
                "onset_guidance_policy": "bounded_position_logprob_mass_v1",
                "fine_note_control": "relative_window_log_odds_v1",
            }
            playability_warning = args.enable_conflict_detection and (
                conflict_result.coverage.get("status") == "partial"
                or bool(conflict_result.remaining_conflicts))
            runtime_warning = (quality.get("runtime_reference", {}).get("status") == "failed")
            quality["generation_status"] = (
                "success_with_warnings" if note_warnings or playability_warning or runtime_warning else "success"
            )
            quality["windows_decoded"] = len(decoded)
            quality["windows_closed"] = sum(1 for window in decoded if window.closed)
            quality["coverage_beats"] = str(points[-1].beat if points else Fraction(0))
            quality["song_beats"] = str(duration_beats)
            # Structure, audio-follow and (when a gold reference is supplied)
            # the acceptance verdict.  These are what tell a user whether the
            # chart is worth playing, as opposed to merely syntactically valid.
            quality["structure"] = structural_metrics(
                points, tempo_changes, args.first, audio_duration_sec=audio.duration_sec)
            quality["audio_follow"] = audio_follow_metrics(
                points, audio, tempo_changes, args.first, duration_sec=audio.duration_sec)
            quality["candidate_score"] = _candidate_quality_score(quality)
            candidate_records.append((target, target_quality, float(quality["candidate_score"])))
            if gold_reference:  # noqa: SIM108 - keep the two branches explicit
                acceptance_metrics = metric_bundle(
                    points, audio, tempo_changes, args.first,
                    duration_sec=audio.duration_sec, quality=quality)
                acceptance = evaluate(acceptance_metrics, gold_reference)
                acceptance["timeline"] = timeline_series(
                    points, audio, tempo_changes, args.first, duration_sec=audio.duration_sec)
                quality["acceptance"] = acceptance
                print(f"acceptance_verdict={acceptance['verdict']} "
                      f"failures={acceptance['failures']} warnings={acceptance['warnings']}", flush=True)
                for item in acceptance["criteria"]:
                    if item["verdict"] in {"fail", "warn"}:
                        print(f"  acceptance_{item['verdict']}: {item['label']} — {item['message']}", flush=True)
            else:
                quality["timeline"] = timeline_series(
                    points, audio, tempo_changes, args.first, duration_sec=audio.duration_sec)
            if [p.to_dict() for p in original_points] != [p.to_dict() for p in points]:
                original_text, _ = compile_event_points(original_points, tempo_changes=tempo_changes,
                    bpm=bpm, first=args.first, title=args.title, artist=args.artist,
                    level=args.chart_level or str(condition.level), designer=args.designer,
                    difficulty=condition.difficulty, strict_notes=False)
                target.with_suffix(".original.txt").write_text(original_text, encoding="utf-8")
                target.with_suffix(".original.chart.json").write_text(json.dumps({
                    "schema": "maichart-event-chart-v1", "points": [p.to_dict() for p in original_points],
                    "tempo_changes": [{"beat": str(t.beat), "bpm": t.bpm} for t in tempo_changes],
                    "first": args.first, "end_beat": str(duration_beats),
                    "audio_duration_sec": audio.duration_sec, "audio_sha256": audio_identity["sha256"], "checkpoint_identity": checkpoint_identity}, ensure_ascii=False, indent=2), encoding="utf-8")
                target.with_suffix(".repaired.txt").write_text(text, encoding="utf-8")
            target.write_text(text, encoding="utf-8")
            target_ir.write_text(json.dumps({"schema": "maichart-event-chart-v1", "points": [point.to_dict() for point in points],
                                             "tempo_changes": [{"beat": str(t.beat), "bpm": t.bpm} for t in tempo_changes],
                                             "first": args.first, "end_beat": str(duration_beats),
                                             "audio_duration_sec": audio.duration_sec,
                                             "audio_sha256": audio_identity["sha256"], "checkpoint_identity": checkpoint_identity,
                                             "title": args.title, "difficulty": condition.difficulty, "level": str(condition.level),
                                             "chart_ir": json.loads(__import__("maichart.serialization", fromlist=["chart_to_json"]).chart_to_json(chart)),
                                             "quality": quality}, ensure_ascii=False, indent=2), encoding="utf-8")
            target_quality.write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            failures += 1
            _write_failure_report(target, target_quality, exc, candidate=candidate + 1,
                                  diagnostics=quality)
            print(f"candidate={candidate + 1}/{args.candidates} generation_status=failed error={exc}", flush=True)
            continue
        print(f"candidate={candidate + 1}/{args.candidates} output={target} events={len(points)} "
              f"windows={len(decoded)} open_windows={sum(1 for window in decoded if not window.closed)} "
              f"last_beat={points[-1].beat if points else 0} bpm={bpm} checkpoint_epoch={checkpoint.get('epoch')}", flush=True)
    if candidate_records:
        ranked = sorted(candidate_records, key=lambda item: item[2], reverse=True)
        rank_by_quality = {str(report_path): rank for rank, (_, report_path, _) in enumerate(ranked, start=1)}
        for target, report_path, score in candidate_records:
            try:
                payload = json.loads(report_path.read_text(encoding="utf-8"))
                payload["candidate_score"] = score
                payload["candidate_rank"] = rank_by_quality[str(report_path)]
                payload["candidate_best"] = payload["candidate_rank"] == 1
                report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            except (OSError, ValueError, TypeError):
                pass
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())







