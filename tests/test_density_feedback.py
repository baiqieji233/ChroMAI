from fractions import Fraction

import pytest
import torch

from maichart.event_timing import EventTempo
from maichart.generation.density_feedback import CurveDensityFeedback
from maichart.generation.event_cli import _decode, _effective_curve_profile, _rest_filter_sequence
from maichart.generation.event_model import EventSequence, EventVocabulary
from maichart.generation.music_intensity import curve_window_values, validate_intensity_profile


def profile(points=None, rests=None):
    result = {"schema": "maichart-intensity-curve-v1", "duration_sec": 8.0,
              "points": [{"time_sec": time, "intensity": value} for time, value in (points or [(0, .5), (8, .5)])]}
    if rests is not None:
        result["rest_regions"] = rests
    return result


def feedback(curve=None, *, density=.125, mode="follow", strength=1, prior=(), offset=Fraction(0)):
    return CurveDensityFeedback(curve or profile(), base_density=density, start_beat=Fraction(0),
                                end_beat=Fraction(16), tempo_changes=[EventTempo(Fraction(0), 120)],
                                position_offset=offset, mode=mode, strength=strength, prior_cells=prior)


def vocabulary():
    return EventVocabulary.from_event_charts([], segment_beats=16)


def test_soft_feedback_penalizes_crowding_and_keeps_later_positions():
    control = feedback()
    assert control.accept_cell(Fraction(0), "1/2")
    vocab = vocabulary()
    logits = torch.zeros(len(vocab))
    shaped = control.shape_logits(logits, vocab, state="position", last_beat=Fraction(0))
    assert shaped[vocab.token_to_id["AT:1"]] < shaped[vocab.token_to_id["AT:4"]]
    assert torch.isfinite(shaped[vocab.token_to_id["AT:1"]])
    assert control.accept_cell(Fraction(1), "3/4")
    assert control.stats["rejected_cells"] == 0


def test_hard_budget_counts_all_heads_and_continues_later_regions():
    control = feedback(density=.0625, mode="hard")  # 2 heads per 2-second region
    assert control.accept_cell(Fraction(0), "1/2")
    vocab = vocabulary()
    shaped = control.shape_logits(torch.zeros(len(vocab)), vocab, state="position", last_beat=Fraction(0))
    assert torch.isneginf(shaped[vocab.token_to_id["AT:1"]])
    assert torch.isfinite(shaped[vocab.token_to_id["AT:4"]])
    assert torch.isneginf(shaped[vocab.eos_id])
    assert not control.accept_cell(Fraction(4), "3/4/5")
    assert control.stats["rejected_heads"] == 3
    assert control.counts[1] == 0
    assert control.accept_cell(Fraction(8), "6")


def test_hard_budget_includes_heads_from_previous_partial_window():
    control = feedback(mode="hard", prior=[{"time_sec": 1.0, "heads": 4}])
    vocab = vocabulary()
    shaped = control.shape_logits(torch.zeros(len(vocab)), vocab, state="position", last_beat=None)
    assert torch.isneginf(shaped[vocab.token_to_id["AT:0"]])
    assert torch.isfinite(shaped[vocab.token_to_id["AT:4"]])
    assert torch.isneginf(shaped[vocab.eos_id])


def test_overlap_owned_by_previous_window_cannot_consume_later_quota():
    control = CurveDensityFeedback(profile(), base_density=.125, start_beat=Fraction(2),
        end_beat=Fraction(16), tempo_changes=[EventTempo(Fraction(0), 120)],
        mode="hard", owned_start_sec=3)
    vocab = vocabulary()
    shaped = control.shape_logits(torch.zeros(len(vocab)), vocab, state="position", last_beat=None)
    # Local beat 3 -> absolute beat 5 -> 2.5sec belongs to the previous window.
    assert torch.isneginf(shaped[vocab.token_to_id["AT:3"]])
    assert torch.isfinite(shaped[vocab.token_to_id["AT:4"]])
    assert control.counts[1] == 0


def test_tempo_context_offset_and_positive_first_keep_absolute_clock_and_quota():
    control = CurveDensityFeedback(profile(rests=[{"start_sec": 5, "end_sec": 6}]),
        base_density=.125, start_beat=Fraction(8), end_beat=Fraction(16),
        tempo_changes=[EventTempo(Fraction(0), 120), EventTempo(Fraction(8), 240)],
        first=1, position_offset=Fraction(16), mode="hard")
    assert control.time_for(Fraction(16)) == pytest.approx(5)
    assert control.time_for(Fraction(20)) == pytest.approx(6)
    assert not control._can_start(5)
    assert control._can_start(6)
    assert control.budget(0) == pytest.approx(2)  # only 1..2s can receive heads


def test_explicit_rest_is_half_open_and_independent_of_curve_strength():
    curve = profile(rests=[{"start_sec": 0, "end_sec": 2}])
    control = feedback(curve, mode="soft", strength=0)
    vocab = vocabulary()
    shaped = control.shape_logits(torch.zeros(len(vocab)), vocab, state="position", last_beat=None)
    assert torch.isneginf(shaped[vocab.token_to_id["AT:0"]])
    assert torch.isfinite(shaped[vocab.token_to_id["AT:4"]])
    assert torch.isneginf(shaped[vocab.eos_id])
    assert not control.accept_cell(Fraction(3), "1")
    assert control.accept_cell(Fraction(4), "2")
    seq = EventSequence([Fraction(2), Fraction(4)], ["1h[4:8]", "2"])
    filtered, removed = _rest_filter_sequence(seq, start_beat=Fraction(0),
        tempo_changes=[EventTempo(Fraction(0), 120)], first=0, profile=curve)
    assert removed == 1 and filtered.note_texts == ["2"]


def test_rest_validation_merges_intervals_and_integrates_exactly():
    curve = profile(rests=[{"start_sec": 2, "end_sec": 4}, {"start_sec": 1, "end_sec": 3}])
    assert validate_intensity_profile(curve)["rest_regions"] == [{"start_sec": 1.0, "end_sec": 4.0}]
    assert curve_window_values(curve, 0, 8) == pytest.approx(.5 * 5 / 8)
    for rest in ({"start_sec": -1, "end_sec": 2}, {"start_sec": 3, "end_sec": 2},
                 {"start_sec": 0, "end_sec": float("nan")}, {"start_sec": 0, "end_sec": 9}):
        with pytest.raises(ValueError):
            validate_intensity_profile(profile(rests=[rest]))


def test_low_opening_does_not_end_a_window_with_later_strong_demand():
    control = feedback(profile([(0, .05), (2, .05), (3, .7), (8, .7)]))
    vocab = vocabulary()
    shaped = control.shape_logits(torch.zeros(len(vocab)), vocab, state="position", last_beat=None)
    assert torch.isneginf(shaped[vocab.eos_id])
    quiet = feedback(profile([(0, .02), (8, .02)]))
    shaped_quiet = quiet.shape_logits(torch.zeros(len(vocab)), vocab, state="position", last_beat=None)
    assert quiet.allow_empty_end()
    assert torch.isfinite(shaped_quiet[vocab.eos_id])
    loud = feedback(profile([(0, .7), (8, .7)]))
    loud.shape_logits(torch.zeros(len(vocab)), vocab, state="position", last_beat=None)
    assert not loud.allow_empty_end()


def test_follow_and_hard_share_soft_curve_range_and_strength():
    curve = profile([(0, 0), (8, 1)])
    soft = _effective_curve_profile(curve, strength=.7, span=.6, mode="soft")
    assert _effective_curve_profile(curve, strength=.7, span=.6, mode="follow") == soft
    assert _effective_curve_profile(curve, strength=.7, span=.6, mode="hard") == soft


class ChordModel:
    """Chooses a three-head cell wherever possible, with realistic cache lengths."""
    def __init__(self, vocab):
        self.vocab = vocab
        self.cache_positions = []

    def eval(self): return self

    def encode_audio(self, audio, condition, **kwargs): return torch.zeros(1, 1, 8), None

    def decode_step(self, memory, token, condition, caches, position, audio_mask=None):
        assert not caches or caches[0][0].size(1) == position
        history = [] if not caches else caches[0][0][0, :, 0].int().tolist()
        history.append(int(token.item()))
        texts = [self.vocab.id_to_token[index] for index in history]
        last = texts[-1]
        logits = torch.full((1, 1, len(self.vocab)), -100.)
        if last.startswith("AT:"):
            logits[0, 0, self.vocab.token_to_id["CHAR:1"]] = 10
        elif last == "CHAR:1":
            logits[0, 0, self.vocab.token_to_id["CHAR:2"]] = 10
        elif last == "CHAR:2":
            logits[0, 0, self.vocab.token_to_id["CHAR:3"]] = 10
        elif last == "CHAR:3":
            logits[0, 0, self.vocab.token_to_id["<event_end>"]] = 10
        else:
            positions = [Fraction(text[3:]) for text in texts if text.startswith("AT:")]
            for beat in (Fraction(0), Fraction(4), Fraction(8), Fraction(12)):
                if not positions or beat > max(positions):
                    logits[0, 0, self.vocab.token_to_id[f"AT:{beat}"]] = 20 - float(beat)
            logits[0, 0, self.vocab.eos_id] = -5
        key = torch.tensor(history, dtype=torch.float32).reshape(1, -1, 1)
        self.cache_positions.append(position)
        return logits, [(key, key.clone())]


class ActualCacheChordModel(ChordModel):
    """Scripted choices passed through the real Transformer's cache contract."""
    def __init__(self, vocab):
        super().__init__(vocab)
        from maichart.generation.event_model import EventConditionedTransformer, EventTransformerConfig
        self.transformer = EventConditionedTransformer(EventTransformerConfig(
            input_dim=8, condition_dim=16, vocab_size=len(vocab), d_model=8, nhead=2,
            num_layers=1, dim_feedforward=16, dropout=0, max_tokens=128)).eval()
        self.script_cache = []

    def decode_step(self, memory, token, condition, caches, position, audio_mask=None):
        assert not caches or caches[0][0].shape == (1, position, 8)
        _, real_caches = self.transformer.decode_step(memory, token, condition, caches, position, audio_mask)
        self.script_cache = [(key[:, :position], value[:, :position]) for key, value in self.script_cache]
        logits, self.script_cache = super().decode_step(memory, token, condition, self.script_cache, position, audio_mask)
        return logits, real_caches


class EosOnceModel(ChordModel):
    def __init__(self, vocab):
        super().__init__(vocab)
        self.initial_eos = True

    def decode_step(self, *args, **kwargs):
        logits, caches = super().decode_step(*args, **kwargs)
        if self.initial_eos:
            self.initial_eos = False
            logits.fill_(-100)
            logits[0, 0, self.vocab.eos_id] = 30
        return logits, caches


@pytest.mark.parametrize("use_feedback", [False, True])
def test_empty_high_window_reroll_removes_rejected_eos_from_cache_and_output(use_feedback):
    vocab = vocabulary()
    control = feedback(profile([(0, .7), (8, .7)])) if use_feedback else None
    sequence = _decode(EosOnceModel(vocab), vocab, torch.zeros(2, 8), torch.zeros(2, 8), torch.zeros(16),
        torch.device("cpu"), max_tokens=100, temperature=.65, sample=False, seed=1, density_feedback=control)
    assert sequence.beats and sequence.beats[0] == 0


def test_empty_low_window_does_not_force_reroll():
    vocab = vocabulary()
    sequence = _decode(EosOnceModel(vocab), vocab, torch.zeros(2, 8), torch.zeros(2, 8), torch.zeros(16),
        torch.device("cpu"), max_tokens=100, temperature=.65, sample=False, seed=1,
        density_feedback=feedback(profile([(0, .1), (8, .1)])))
    assert sequence.beats == []


@pytest.mark.parametrize("model_class", [ChordModel, ActualCacheChordModel])
def test_rejected_chord_rewinds_cfg_caches_and_does_not_leak_invisible_context(model_class):
    vocab = vocabulary()
    model = model_class(vocab)
    # First 2s allows only two heads, later regions allow at least three.
    control = feedback(profile([(0, .125), (2, .125), (2.01, .5), (8, .5)]), mode="hard")
    sequence = _decode(model, vocab, torch.zeros(2, 8), torch.zeros(2, 8), torch.zeros(16),
                       torch.device("cpu"), max_tokens=100, temperature=.65, sample=False, seed=1,
                       guidance_scale=1.1, density_feedback=control)
    assert control.stats["rejected_cells"] == 1
    assert sequence.beats and sequence.beats[0] >= 4
    assert all(text == "123" for text in sequence.note_texts)


@pytest.mark.parametrize("strength", [0, .85])
def test_all_rest_cli_compiles_empty_chart_and_writes_reports(tmp_path, monkeypatch, strength):
    import json
    import numpy as np
    from maichart.generation import event_cli
    from maichart.generation.event_audio import ACOUSTIC_DIM, EventAudio
    from maichart.generation.music_intensity import audio_sha256
    from maichart.event_timing import parse_event_chart_text
    audio_path = tmp_path / "song.mp3"
    audio_path.write_bytes(b"test audio identity")
    audio = EventAudio(np.zeros((400, ACOUSTIC_DIM)), np.arange(400) * .02, 8, 120)
    monkeypatch.setattr(event_cli, "extract_event_audio", lambda *a, **kw: audio)
    monkeypatch.setattr(event_cli, "load_event_model", lambda *a, **kw: (ChordModel(vocabulary()), vocabulary(), {"epoch": 1}))
    curve = profile(rests=[{"start_sec": 0, "end_sec": 8}])
    curve["audio_sha256"] = audio_sha256(audio_path)
    curve_path = tmp_path / "curve.json"
    curve_path.write_text(json.dumps(curve), encoding="utf-8")
    output = tmp_path / "maidata.txt"
    code = event_cli.main(["--checkpoint", str(tmp_path / "model.pt"), "--audio", str(audio_path),
        "--output", str(output), "--device", "cpu", "--bpm", "120", "--max-tokens", "100",
        "--intensity-curve", str(curve_path), "--intensity-curve-strength", str(strength)])
    assert code == 0
    assert not parse_event_chart_text(output.read_text(encoding="utf-8"), 5).points
    report = json.loads(output.with_suffix(".quality.json").read_text(encoding="utf-8"))
    assert report["generation_status"] == "success"
    assert report["intensity_curve"]["rest_regions"] == [{"start_sec": 0.0, "end_sec": 8.0}]
