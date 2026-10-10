from fractions import Fraction

import torch

from maichart.generation.event_model import (
    EVENT_END,
    EVENT_EOS,
    EVENT_PAD,
    EVENT_SPECIALS,
    EventVocabulary,
    char_token,
    position_token,
)
from maichart.generation.event_ctx import CTX_OPEN, CTX_CLOSE
from tools.evaluate_event_tokens import score_token_batch, shuffle_acoustic_frames
from maichart.generation.event_audio import ACOUSTIC_DIM, CONTEXT_KEYS, RHYTHM_CONTEXT_KEYS


def test_scoring_excludes_context_and_padding_and_separates_gap_groups():
    tokens = list(EVENT_SPECIALS) + [CTX_OPEN, CTX_CLOSE]
    tokens += [char_token(char) for char in "1h-"]
    tokens += [position_token(Fraction(0)), position_token(Fraction(1, 2)),
               position_token(Fraction(1, 3)), position_token(Fraction(3, 2))]
    vocab = EventVocabulary(tokens)
    truth = [vocab.token_to_id[token] for token in (
        position_token(Fraction(3, 2)), char_token("1"), EVENT_END,
        position_token(Fraction(0)), char_token("1"), EVENT_END,
        position_token(Fraction(1, 2)), char_token("h"), EVENT_END, EVENT_EOS,
    )]
    guess = truth.copy()
    guess[6] = vocab.token_to_id[position_token(Fraction(1, 3))]
    guess[7] = vocab.token_to_id[EVENT_END]
    guess[9] = vocab.token_to_id[EVENT_PAD]
    probabilities = torch.zeros((len(truth), len(vocab)))
    for i, token_id in enumerate(guess):
        probabilities[i, token_id] = 0.8
        probabilities[i, truth[i]] = max(float(probabilities[i, truth[i]]), 0.2)
    loss_mask = torch.tensor([False, False, False, True, True, True, True, True, True, True])
    padding_mask = torch.tensor([False] * 9 + [True])

    result = score_token_batch(vocab, probabilities, torch.tensor(truth), loss_mask, padding_mask)

    assert result["token_count"] == 6
    assert result["token_correct"] == 4
    assert result["position_groups"]["first"]["count"] == 1
    assert result["position_groups"]["first"]["top1_accuracy"] == 1.0
    assert result["position_groups"]["1/2"]["count"] == 1
    assert result["position_groups"]["1/2"]["top1_accuracy"] == 0.0
    assert result["character_groups"]["hold"]["top1_accuracy"] == 0.0
    assert result["premature_event_end"] == {
        "count": 1, "among_true_note_characters": 2, "ratio": 0.5,
    }


def test_ordinary_syntax_is_reported_separately_from_type_markers():
    vocab = EventVocabulary(list(EVENT_SPECIALS) + [char_token(c) for c in "1h/"])
    truth = torch.tensor([vocab.token_to_id[char_token("1")], vocab.token_to_id[char_token("h")],
                          vocab.token_to_id[char_token("/")]])
    probabilities = torch.zeros((3, len(vocab)))
    probabilities[0, truth[0]] = 0.7
    probabilities[1, vocab.token_to_id[EVENT_END]] = 0.9
    probabilities[2, truth[2]] = 0.8

    result = score_token_batch(vocab, probabilities, truth, torch.ones(3, dtype=torch.bool),
                               torch.zeros(3, dtype=torch.bool))

    assert result["character_groups"]["ordinary_syntax"]["count"] == 1
    assert result["character_groups"]["hold"]["count"] == 1
    assert result["character_groups"]["ordinary_syntax"]["top1_accuracy"] == 1.0
    assert result["character_groups"]["hold"]["top1_accuracy"] == 0.0
    assert result["character_groups"]["chord_separator"]["top1_accuracy"] == 1.0
    assert "slide" not in result["character_groups"]


def test_shuffle_acoustic_ablation_preserves_time_columns_and_padding():
    feature_dim = ACOUSTIC_DIM + len(RHYTHM_CONTEXT_KEYS) + len(CONTEXT_KEYS)
    audio = torch.arange(2 * 4 * feature_dim, dtype=torch.float32).reshape(2, 4, feature_dim)
    padding = torch.tensor([[False, False, False, False], [False, False, True, True]])

    shuffled = shuffle_acoustic_frames(audio, padding, seed=17)

    acoustic_end = ACOUSTIC_DIM + len(RHYTHM_CONTEXT_KEYS)
    assert torch.equal(shuffled[..., acoustic_end:], audio[..., acoustic_end:])
    assert torch.equal(shuffled[1, :2, :acoustic_end].sort(dim=0).values,
                       audio[1, :2, :acoustic_end].sort(dim=0).values)
    assert torch.equal(shuffled[1, 2:], audio[1, 2:])
    assert torch.equal(audio, torch.arange(2 * 4 * feature_dim, dtype=torch.float32).reshape(2, 4, feature_dim))
