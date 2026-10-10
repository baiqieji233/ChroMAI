from fractions import Fraction as F

import torch

from maichart.generation import event_cli
from maichart.generation.event_audio import INPUT_DIM
from maichart.generation.event_model import EventVocabulary, EventSequence
from maichart.generation.event_ctx import CTX_OPEN, CTX_CLOSE


class PositionModel:
    """Strongly prefers an illegal adjacent-window beat over the legal one."""
    def __init__(self, vocabulary, outside, inside):
        self.vocabulary, self.outside, self.inside = vocabulary, outside, inside

    def eval(self): return self

    def encode_audio(self, *a, **kw): return torch.zeros(1, 1, 4), None

    def decode_step(self, memory, token, condition, caches, position, audio_mask=None):
        logits = torch.full((1, 1, len(self.vocabulary)), -100.)
        def set_value(name, value): logits[0, 0, self.vocabulary.token_to_id[name]] = value
        if position == 0:
            set_value(self.outside, 20.)
            set_value(self.inside, 10.)
        else:
            set_value({1: "CHAR:1", 2: "<event_end>"}.get(position, "<eos>"), 20.)
        return logits, caches


def test_first_window_cannot_emit_the_next_windows_position():
    vocab = EventVocabulary.from_event_charts([], segment_beats=32)
    model = PositionModel(vocab, "AT:16", "AT:2")
    seq = event_cli._decode(model, vocab, torch.zeros(2, INPUT_DIM), torch.zeros(2, INPUT_DIM),
                           torch.zeros(16), torch.device("cpu"), max_tokens=20,
                           temperature=1, sample=False, seed=1, position_end=F(16))
    assert seq.beats == [F(2)]


def test_later_window_cannot_emit_lead_in_notes_again():
    vocab = EventVocabulary.from_event_charts([], segment_beats=32)
    model = PositionModel(vocab, "AT:2", "AT:17")
    seq = event_cli._decode(model, vocab, torch.zeros(2, INPUT_DIM), torch.zeros(2, INPUT_DIM),
                           torch.zeros(16), torch.device("cpu"), max_tokens=20,
                           temperature=1, sample=False, seed=1, position_offset=F(16), position_end=F(32))
    assert seq.beats == [F(1)]


def test_context_prefix_uses_the_same_overlap_owner_as_final_chart():
    base = EventVocabulary.from_event_charts([], segment_beats=32)
    vocab = EventVocabulary([*base.tokens, CTX_OPEN, CTX_CLOSE])
    windows = [event_cli._DecodedWindow(EventSequence([F(14)], ["1"]),F(0),F(16),True),
               event_cli._DecodedWindow(EventSequence([F(2),F(6)], ["8","2"]),F(12),F(28),True)]
    prefix = event_cli._context_prefix(vocab, windows, F(24), 16, 192)
    text = [vocab.id_to_token[token] for token in prefix]
    assert "CHAR:1" in text and "CHAR:2" in text
    assert "CHAR:8" not in text
    assert text.count("AT:6") == 1
