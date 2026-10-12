"""Sampling-shaping switches for the event decoder: ``--top-p`` and repetition penalty.

These tests exercise only pure logic -- the two logits shapers extracted next to
``_decode`` in ``event_cli.py``, the argparse surface, and the pass-through
chain.  No network, no checkpoints, no mocked model beyond a fixed-logits stub.

The contract under test: with the *function-level* defaults (``top_p=0.0``,
``repetition_penalty=1.0``) both shapers are strict no-ops, so the sampled
distribution is byte-identical to the pre-switch behaviour; the shapers act on
the grammar-masked logits of the sampling branch only and never touch the
argmax path.  The CLI deliberately ships corpus-measured values instead
(conditions.GENERATION_SAMPLING_DEFAULTS), which is asserted separately.
"""

from __future__ import annotations

import inspect
from fractions import Fraction

import pytest
import torch

from maichart.event_timing import EventTempo
from maichart.generation import event_cli
from maichart.generation.event_ctx_cli import _with_defaults
from maichart.generation.conditions import GenerationCondition, RADAR_KEYS, STYLE_KEYS
from maichart.generation.event_model import EVENT_BOS, EventSequence, EventVocabulary

# Small real alphabet: the five structural specials first, then the two clock
# tokens and three note characters, so the control-token exemption is tested
# against the genuine vocabulary layout.
_CONDITION = GenerationCondition(level=13.0, radar={key: 0.3 for key in RADAR_KEYS}, style={key: 0.4 for key in STYLE_KEYS})

VOCABULARY = EventVocabulary(["AT:0", "AT:1", "CHAR:1", "CHAR:2", "CHAR:3"])


def _token(name: str) -> int:
    return VOCABULARY.token_to_id[name]


def _masked_like(values: dict[int, float]) -> torch.Tensor:
    """A grammar-masked logits vector: everything not listed sits at -inf."""

    tensor = torch.full((len(VOCABULARY),), float("-inf"))
    for token_id, value in values.items():
        tensor[token_id] = value
    return tensor


def test_top_p_disabled_values_are_strict_no_ops():
    logits = torch.tensor([3.0, 2.9, 2.8, 2.0, float("-inf")])
    snapshot = logits.clone()
    for top_p in (0.0, 1.0, -0.5, 2.0):
        assert torch.equal(event_cli._apply_top_p(logits, top_p), snapshot), top_p
        assert torch.equal(logits, snapshot)  # nothing is mutated in place either


def test_top_p_keeps_the_minimal_nucleus_prefix():
    # softmax([3.0, 2.9, 2.8, 2.0]) ~= [0.3235, 0.2927, 0.2648, 0.1190];
    # the descending cumulative is [0.3235, 0.6162, 0.8810, 1.0].
    logits = torch.tensor([3.0, 2.9, 2.8, 2.0])

    only_top = event_cli._apply_top_p(logits, 0.3)  # 0.3235 >= 0.3 already
    assert torch.equal(only_top, torch.tensor([3.0, float("-inf"), float("-inf"), float("-inf")]))

    top_two = event_cli._apply_top_p(logits, 0.5)  # 0.6162 >= 0.5 > 0.3235
    assert torch.equal(top_two, torch.tensor([3.0, 2.9, float("-inf"), float("-inf")]))

    # 0.8810 < 0.885, so the whole finite support is the minimal prefix and the
    # logits come back element-wise unchanged.
    everything = event_cli._apply_top_p(logits, 0.885)
    assert torch.equal(everything, logits)


def test_top_p_never_keeps_a_zero_probability_token():
    # A grammar-masked shape: the clock and structure tokens sit at -inf, so
    # their softmax probability is exactly 0 and they must never join the
    # nucleus, whatever top_p is.
    logits = torch.full((len(VOCABULARY),), float("-inf"))
    logits[_token("CHAR:1")] = 3.0
    logits[_token("CHAR:2")] = 2.9
    logits[_token("CHAR:3")] = 2.0

    filtered = event_cli._apply_top_p(logits, 0.999999)
    assert filtered[_token("CHAR:1")] == 3.0
    assert filtered[_token("CHAR:2")] == 2.9
    assert filtered[_token("CHAR:3")] == 2.0
    for name in ("<pad>", "<bos>", "<eos>", "<unk>", "<event_end>", "AT:0", "AT:1"):
        assert filtered[_token(name)] == float("-inf")


def test_repetition_penalty_downweights_window_tokens_once():
    # CTRL applies the penalty once per id in the window however often it
    # repeats, so the single-occurrence ``CHAR:2`` is penalised alongside the
    # thrice-repeated ``CHAR:1``; ``CHAR:3`` is outside the window and unchanged.
    first, second, third = _token("CHAR:1"), _token("CHAR:2"), _token("CHAR:3")
    logits = torch.zeros(len(VOCABULARY))
    logits[first], logits[second], logits[third] = 2.0, 4.0, -1.5

    filtered = event_cli._apply_repetition_penalty(logits, [first, first, first, second], 1.2, VOCABULARY)
    assert filtered[first].item() == pytest.approx(2.0 / 1.2)  # positive: divided
    assert filtered[second].item() == pytest.approx(4.0 / 1.2)  # once per id, not per occurrence
    assert filtered[third].item() == -1.5  # never generated in this window
    assert filtered[_token("AT:0")].item() == 0.0  # untouched logit stays untouched


def test_repetition_penalty_multiplies_negative_logits():
    repeated = _token("CHAR:1")
    logits = torch.zeros(len(VOCABULARY))
    logits[repeated] = -2.0

    filtered = event_cli._apply_repetition_penalty(logits, [repeated], 1.5, VOCABULARY)
    assert filtered[repeated].item() == pytest.approx(-2.0 * 1.5)  # even more negative


def test_repetition_penalty_never_touches_the_grammar_mask():
    # A grammar-masked vector: the penalty>1 must lower the seen token's logit
    # while every masked position stays exactly -inf (no NaN, no resurrection).
    first, second = _token("CHAR:1"), _token("CHAR:2")
    logits = _masked_like({first: 2.0, second: 4.0})
    snapshot = logits.clone()

    filtered = event_cli._apply_repetition_penalty(logits, [first], 1.6, VOCABULARY)
    assert filtered[first].item() == pytest.approx(2.0 / 1.6)
    assert filtered[second].item() == 4.0  # not in the window: untouched
    for name in ("<pad>", "<bos>", "<eos>", "<unk>", "<event_end>", "AT:0", "AT:1"):
        assert filtered[_token(name)] == float("-inf")
    assert not torch.isnan(filtered).any()
    assert torch.equal(logits, snapshot)  # the input tensor is not mutated

    # A previously generated token that the grammar now disallows sits at -inf;
    # penalising it must not bring it back into the distribution.
    masked_out = _masked_like({second: 4.0})
    penalized = event_cli._apply_repetition_penalty(masked_out, [first], 1.6, VOCABULARY)
    assert penalized[first] == float("-inf")
    assert not torch.isnan(penalized).any()


def test_repetition_penalty_skips_control_tokens():
    note = _token("CHAR:1")
    controls = tuple(_token(name) for name in ("<pad>", "<bos>", "<eos>", "<unk>", "<event_end>", "AT:0", "AT:1"))
    logits = torch.zeros(len(VOCABULARY))
    logits[note] = 2.0
    for offset, token_id in enumerate(controls, start=1):
        logits[token_id] = float(offset)
    snapshot = logits.clone()

    filtered = event_cli._apply_repetition_penalty(logits, [note, *controls], 2.0, VOCABULARY)
    assert filtered[note].item() == pytest.approx(1.0)
    for token_id in controls:
        assert filtered[token_id].item() == snapshot[token_id].item()


def test_repetition_window_one_penalises_only_the_last_token():
    first, second = _token("CHAR:1"), _token("CHAR:2")
    logits = torch.zeros(len(VOCABULARY))
    logits[first], logits[second] = 3.0, 6.0
    history = [first, first, first, second]

    # ``_decode`` slices ``ids[-repetition_window:]`` before calling the helper,
    # so a window of 1 sees exactly the newest token.
    filtered = event_cli._apply_repetition_penalty(logits, history[-1:], 1.5, VOCABULARY)
    assert filtered[first].item() == 3.0
    assert filtered[second].item() == pytest.approx(6.0 / 1.5)


def test_decode_penalises_only_the_recent_window(monkeypatch):
    # Full ``_decode`` path: whatever the penalty sees must be exactly the last
    # ``min(generated, window)`` ids.  The scripted model emits a long CHAR:1
    # run ("the stuck decoder" the feature exists for), so a wide window
    # penalises those ids at every later step, while a window of 1 only ever
    # sees the single newest token.
    script = [_token(name) for name in ("AT:0", "CHAR:1", "CHAR:1", "CHAR:1",
                                        "<event_end>", "<eos>")]
    first, AT0 = _token("CHAR:1"), _token("AT:0")
    captured: list[list[int]] = []
    real_penalty = event_cli._apply_repetition_penalty

    def spy(logits, generated_ids, penalty, vocabulary):
        captured.append(list(generated_ids))
        return real_penalty(logits, generated_ids, penalty, vocabulary)

    monkeypatch.setattr(event_cli, "_apply_repetition_penalty", spy)
    bos = VOCABULARY.token_to_id[EVENT_BOS]
    # ``_decode`` seeds ``ids`` with ``<bos>`` and appends every emitted token;
    # the penalty is called once per proposal with ``ids[-window:]`` as its
    # history, so at proposal k (0-based) the window is ``[bos]+script[:k]``.
    expected_window = lambda window: [
        list(([VOCABULARY.token_to_id[EVENT_BOS], *script][:step])[-window:])
        for step, _ in enumerate(script, start=1)
    ]
    wide_ids = expected_window(128)
    narrow_ids = expected_window(2)

    def run(window: int) -> list[list[int]]:
        captured.clear()
        _scripted_decode(sample=True, repetition_window=window, script_override=script)
        return [list(ids) for ids in captured]

    assert run(128) == wide_ids
    assert run(2) == narrow_ids
    # At the step proposing the SECOND CHAR:1 the wide window (history so far)
    # already contains the FIRST CHAR:1.  A window of 2 holds only the two newest
    # tokens: ``<bos>`` has dropped out, and AT:0 is one step from dropping out.
    narrow_tokens = {VOCABULARY.id_to_token[i] for i in narrow_ids[2]}
    assert first not in wide_ids[1] and first in wide_ids[2][1:]
    assert EVENT_BOS not in narrow_tokens and "AT:0" in narrow_tokens and "CHAR:1" in narrow_tokens
    assert EVENT_BOS not in narrow_ids[3] and "AT:0" not in {
        VOCABULARY.id_to_token[i] for i in narrow_ids[3]}
    assert wide_ids[3][:3] == [bos, AT0, first]
    # The very first window is the ``<bos>`` seed alone, and it is never empty.
    assert wide_ids[0] == narrow_ids[0] == [bos]


def test_repetition_penalty_unity_is_a_byte_identical_no_op():
    logits = _masked_like({_token("CHAR:1"): 2.0, _token("CHAR:2"): -0.5})
    for generated in ([_token("CHAR:1"), _token("CHAR:2")], [_token("AT:0")], []):
        snapshot = logits.clone()
        assert torch.equal(
            event_cli._apply_repetition_penalty(logits, generated, 1.0, VOCABULARY), snapshot
        )
        assert torch.equal(logits, snapshot)


def test_temperature_changes_nucleus_support():
    logits = torch.tensor([0.0, -0.4, -0.8])
    cold = event_cli._apply_top_p(logits / 0.2, 0.7)
    warm = event_cli._apply_top_p(logits / 2.0, 0.7)
    assert torch.isfinite(cold).sum().item() == 1
    assert torch.isfinite(warm).sum().item() == 2


    # penalty 2.0 turns [3.0, 2.9, 2.8] into [1.5, 2.9, 2.8]; the nucleus of
    # THAT distribution at top_p=0.5 keeps {2.9, 2.8}, whereas a top-p applied
    # before the penalty would have kept {3.0, 2.9} -- so the order is visible.
    logits = _masked_like({_token("CHAR:1"): 3.0, _token("CHAR:2"): 2.9, _token("CHAR:3"): 2.8})
    penalized = event_cli._apply_repetition_penalty(logits, [_token("CHAR:1")], 2.0, VOCABULARY)
    filtered = event_cli._apply_top_p(penalized, 0.5)
    assert filtered[_token("CHAR:1")] == float("-inf")
    assert filtered[_token("CHAR:2")] == 2.9
    assert filtered[_token("CHAR:3")] == 2.8


class _ScriptedModel:
    """Fixed-logits stub: the k-th call proposes ``script[k]`` at logit 10."""

    def __init__(self, script, vocab_size):
        self.script = script
        self.vocab_size = vocab_size
        self.decode_calls = 0
        self.plan_calls = 0

    def eval(self):
        return self

    def encode_audio(self, audio, condition, global_audio=None, **kwargs):
        return torch.zeros((1, 1, 8), device=audio.device), None

    def append_plan_memory(self, memory, audio_mask):
        self.plan_calls += 1
        return memory, audio_mask, torch.zeros((memory.size(0), 9), device=memory.device)

    def decode_step(self, memory, token, condition, caches, position, audio_mask=None):
        self.decode_calls += 1
        token_id = self.script[min(self.decode_calls - 1, len(self.script) - 1)]
        logits = torch.full((1, 1, self.vocab_size), -100.0, device=token.device)
        logits[0, 0, token_id] = 10.0
        return logits, caches


def _scripted_decode(sample: bool, seed: int = 7, script_override: list[int] | None = None,
                     **overrides) -> EventSequence:
    script = script_override or [
        _token(name)
        for name in ("AT:0", "CHAR:1", "CHAR:2", "CHAR:1", "<event_end>", "<eos>")
    ]
    model = _ScriptedModel(script, len(VOCABULARY))
    return event_cli._decode(
        model,
        VOCABULARY,
        torch.zeros((2, 8)),
        torch.zeros((4, 8)),
        torch.zeros(24),
        torch.device("cpu"),
        max_tokens=32,
        temperature=0.8,
        sample=sample,
        seed=seed,
        **overrides,
    )


def test_decode_sample_branch_wires_both_shapers_on_the_masked_logits(monkeypatch):
    top_p_calls: list[tuple[float, torch.Tensor]] = []
    penalty_calls: list[tuple[torch.Tensor, list[int], float, EventVocabulary]] = []
    real_top_p = event_cli._apply_top_p
    real_penalty = event_cli._apply_repetition_penalty

    def spy_top_p(logits, top_p):
        top_p_calls.append((top_p, logits))
        return real_top_p(logits, top_p)

    def spy_penalty(logits, generated_ids, penalty, vocabulary):
        penalty_calls.append((logits, list(generated_ids), penalty, vocabulary))
        return real_penalty(logits, generated_ids, penalty, vocabulary)

    monkeypatch.setattr(event_cli, "_apply_top_p", spy_top_p)
    monkeypatch.setattr(event_cli, "_apply_repetition_penalty", spy_penalty)

    result = _scripted_decode(sample=True)

    # One shaper pass per sampled token, with the no-op defaults flowing through.
    assert result == EventSequence([Fraction(0)], ["121"])
    assert len(top_p_calls) == len(penalty_calls) == 6
    for (top_p_value, scaled_logits), (penalized, generated_ids, penalty, vocabulary) in zip(
        top_p_calls, penalty_calls
    ):
        assert top_p_value == 0.0
        assert penalty == 1.0
        assert vocabulary is VOCABULARY
        assert generated_ids  # a nonempty history slice at every step
        # Temperature scaling precedes nucleus selection, so top-p sees the
        # distribution that will actually be sampled.
        assert torch.equal(scaled_logits, penalized / 0.8)
    # The generated-id windows track the growing history, prefix-free here.
    bos = VOCABULARY.token_to_id[EVENT_BOS]
    assert penalty_calls[0][1] == [bos]
    assert penalty_calls[1][1] == [bos, _token("AT:0")]
    assert penalty_calls[-1][1][-1] == _token("<event_end>")


def test_decode_with_defaults_matches_explicitly_disabled_flags():
    default = _scripted_decode(sample=True)
    explicit = _scripted_decode(sample=True, top_p=0.0, repetition_penalty=1.0, repetition_window=128)
    assert default == explicit


def test_incremental_decode_injects_planner_for_each_cfg_branch():
    script = [_token(name) for name in ("AT:0", "CHAR:1", "<event_end>", "<eos>")]
    model = _ScriptedModel(script, len(VOCABULARY))
    event_cli._decode(
        model, VOCABULARY, torch.zeros((2, 8)), torch.zeros((4, 8)),
        torch.zeros(24), torch.device("cpu"), max_tokens=16, temperature=0.8,
        sample=False, seed=1, guidance_scale=1.0,
    )
    assert model.plan_calls == 1

    guided = _ScriptedModel(script, len(VOCABULARY))
    event_cli._decode(
        guided, VOCABULARY, torch.zeros((2, 8)), torch.zeros((4, 8)),
        torch.zeros(24), torch.device("cpu"), max_tokens=16, temperature=0.8,
        sample=False, seed=1, guidance_scale=1.5,
    )
    assert guided.plan_calls == 2


def test_decode_rejects_invalid_penalty_and_window():
    with pytest.raises(ValueError, match="repetition_penalty"):
        _scripted_decode(sample=True, repetition_penalty=0.0)
    with pytest.raises(ValueError, match="repetition_window"):
        _scripted_decode(sample=True, repetition_window=0)


def test_argparse_new_flags_exist_with_defaults_and_types():
    from maichart.generation.conditions import GENERATION_SAMPLING_DEFAULTS as SAMPLING

    parser = event_cli.build_parser()
    defaults = parser.parse_args(["--checkpoint", "m.pt", "--audio", "a.mp3", "--output", "o.txt"])
    # The CLI ships the corpus-measured defaults, not no-ops: the panel, the
    # server and this CLI must produce the same chart from the same request.
    assert defaults.temperature == SAMPLING["temperature"]
    assert defaults.top_p == SAMPLING["top_p"]
    assert defaults.repetition_penalty == SAMPLING["repetition_penalty"]
    assert defaults.repetition_window == SAMPLING["repetition_window"]
    assert defaults.sample is SAMPLING["sample"], "sampling is the default decode mode"
    assert parser.parse_args(["--checkpoint", "m.pt", "--audio", "a.mp3", "--output", "o.txt",
                              "--greedy"]).sample is False
    assert type(defaults.top_p) is float
    assert type(defaults.repetition_penalty) is float
    assert type(defaults.repetition_window) is int

    explicit = parser.parse_args([
        "--checkpoint", "m.pt", "--audio", "a.mp3", "--output", "o.txt",
        "--top-p", "0.9", "--repetition-penalty", "1.2", "--repetition-window", "16",
    ])
    assert explicit.top_p == pytest.approx(0.9)
    assert explicit.repetition_penalty == pytest.approx(1.2)
    assert explicit.repetition_window == 16
    assert {action.dest for action in parser._actions} >= {"top_p", "repetition_penalty", "repetition_window"}


def test_decode_signatures_carry_the_new_kwargs_with_defaults():
    for function in (event_cli._decode, event_cli._decode_windows):
        parameters = inspect.signature(function).parameters
        assert parameters["top_p"].default == 0.0
        assert parameters["repetition_penalty"].default == 1.0
        assert parameters["repetition_window"].default == 128
        for name in ("top_p", "repetition_penalty", "repetition_window"):
            assert parameters[name].kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )


def test_decode_windows_forwards_the_flags_to_both_call_sites(monkeypatch):
    recorded: list[dict] = []

    def fake_decode(*args, **kwargs):
        recorded.append(kwargs)
        if len(recorded) == 1:
            raise ValueError("decoder reached max_tokens before emitting <eos>.")
        return EventSequence([Fraction(0)], ["1"])

    monkeypatch.setattr(event_cli, "_decode", fake_decode)
    monkeypatch.setattr(
        event_cli, "audio_segment_tensor",
        lambda *args, **kwargs: torch.zeros((2, 8)),
    )

    results = event_cli._decode_windows(
        object(), VOCABULARY, torch.zeros((2, 8)), torch.zeros((4, 8)), _CONDITION,        torch.device("cpu"),
        windows=[(Fraction(0), Fraction(4))],
        tempo_changes=[EventTempo(Fraction(0), 120.0)],
        first=0.0, song_duration=0, bpm=120.0, max_tokens=8, temperature=0.8,
        sample=True, seed=1, guidance_scale=1.0,
        top_p=0.9, repetition_penalty=1.3, repetition_window=7,
    )

    assert len(results) == 1
    assert not results[0].closed
    assert len(recorded) == 2  # normal path, then the salvage retry
    for kwargs in recorded:
        assert kwargs["top_p"] == pytest.approx(0.9)
        assert kwargs["repetition_penalty"] == pytest.approx(1.3)
        assert kwargs["repetition_window"] == 7
    assert "allow_truncated" not in recorded[0]  # normal call keeps the default
    assert recorded[1]["allow_truncated"] is True  # salvage call flips it


def test_ctx_entry_point_passes_the_new_flags_through():
    argv = _with_defaults([
        "--checkpoint", "m.pt", "--audio", "a.mp3", "--output", "o.txt",
        "--top-p", "0.9", "--repetition-penalty", "1.2", "--repetition-window", "16",
    ])
    assert argv[argv.index("--top-p") + 1] == "0.9"
    assert argv[argv.index("--repetition-penalty") + 1] == "1.2"
    assert argv[argv.index("--repetition-window") + 1] == "16"


def test_large_context_is_not_subtracted_twice_from_decode_budget(monkeypatch):
    recorded = []
    def decode(*args, **kwargs):
        recorded.append(kwargs)
        if len(recorded) == 1:
            raise ValueError("decoder reached max_tokens before emitting <eos>.")
        return EventSequence([Fraction(0)], ["1"])
    monkeypatch.setattr(event_cli, "_decode", decode)
    monkeypatch.setattr(event_cli, "audio_segment_tensor", lambda *a, **k: torch.zeros((2, 8)))
    monkeypatch.setattr(event_cli, "_context_prefix", lambda *a, **k: [0] * 600)
    result = event_cli._decode_windows(
        object(), VOCABULARY, torch.zeros((2, 8)), torch.zeros((4, 8)), _CONDITION,
        torch.device("cpu"), windows=[(Fraction(0), Fraction(4))],
        tempo_changes=[EventTempo(Fraction(0), 120)], first=0, song_duration=0,
        bpm=120, max_tokens=1024, temperature=.8, sample=True, seed=1,
        context_beats=16, context_tokens=768)
    assert len(result) == 1 and len(recorded) == 2
    assert all(call["max_tokens"] == 1024 and len(call["prefix_ids"]) == 600 for call in recorded)


def test_decode_windows_argmax_path_ignores_the_flags(monkeypatch):
    # The shapers are sampling-branch-only: with sample=False (argmax path) the
    # window decoder must return the same chart whether or not the flags are on.
    monkeypatch.setattr(event_cli, "audio_segment_tensor", lambda *args, **kwargs: torch.zeros((2, 8)))
    script = [_token(name) for name in ("AT:0", "CHAR:1", "<event_end>", "<eos>")]
    common = dict(
        windows=[(Fraction(0), Fraction(4))],
        tempo_changes=[EventTempo(Fraction(0), 120.0)],
        first=0.0, song_duration=0, bpm=120.0, max_tokens=8, temperature=0.8,
        sample=False, seed=1, guidance_scale=1.0,
    )
    off = event_cli._decode_windows(
        _ScriptedModel(script, len(VOCABULARY)), VOCABULARY,
        torch.zeros((2, 8)), torch.zeros((4, 8)), _CONDITION, torch.device("cpu"), **common)
    on = event_cli._decode_windows(
        _ScriptedModel(script, len(VOCABULARY)), VOCABULARY,
        torch.zeros((2, 8)), torch.zeros((4, 8)), _CONDITION, torch.device("cpu"),        top_p=0.9, repetition_penalty=4.0, repetition_window=8, **common)
    assert [window.sequence for window in off] == [window.sequence for window in on]
    assert off[0].sequence.note_texts == ["1"]
