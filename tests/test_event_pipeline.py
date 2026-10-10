import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch

from maichart.event_timing import EventTempo
from maichart.generation import event_cli
from maichart.generation.event_audio import (
    ACOUSTIC_DIM,
    GLOBAL_MEMORY_FRAMES,
    INPUT_DIM,
    CONTEXT_OFFSET,
    EventAudio,
    audio_segment_tensor,
)
from maichart.generation.event_model import EventSequence, EventVocabulary
from maichart.training.event_dataset import EventDataset


def test_audio_context_accepts_event_tempo_objects_and_integrates_active_bpm():
    audio = EventAudio(
        np.zeros((2, ACOUSTIC_DIM), dtype=np.float32),
        np.asarray([0.0, 1.5], dtype=np.float32),
        1.5,
        120.0,
    )
    tensor = audio_segment_tensor(
        audio,
        0.0,
        1.5,
        song_duration=1.5,
        first=0.0,
        start_beat=0,
        tempo_changes=[EventTempo(Fraction(0), 120.0), EventTempo(Fraction(1), 60.0)],
    )
    assert tensor.shape[1] == INPUT_DIM
    # 0 -> 1 beat consumes 0.5 s at 120 BPM; the next beat consumes 1 s at
    # 60 BPM.  The second half must therefore use the second tempo, rather
    # than accidentally integrating the whole map at the first BPM.
    # local_tempo_bpm is context column 2, i.e. offset by ACOUSTIC_DIM.
    assert tensor[-1, CONTEXT_OFFSET + 2].item() == pytest.approx(60.0 / 240.0, abs=1e-5)


def test_event_dataset_skips_one_bad_difficulty_without_poisoning_good_one(tmp_path, capsys):
    valid_dir = tmp_path / "valid"
    invalid_dir = tmp_path / "invalid"
    valid_dir.mkdir()
    invalid_dir.mkdir()
    (valid_dir / "maidata.txt").write_text(
        "&wholebpm=120\n&first=0.25\n&lv_5=13+\n&inote_5=(120){4}1,2,E\n", encoding="utf-8"
    )
    (invalid_dir / "maidata.txt").write_text(
        "&wholebpm=120\n&lv_5=13+\n&inote_5=(120){4}1/1,E\n", encoding="utf-8"
    )
    for directory in (valid_dir, invalid_dir):
        (directory / "track.mp3").write_bytes(b"placeholder")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "songs": [
            {
                "song_id": "valid",
                "maidata_path": "valid/maidata.txt",
                "audio_path": "valid/track.mp3",
                "difficulties": [
                    {"difficulty_index": 1},
                    {"difficulty_index": 5, "level_raw": "13+"},
                ],
            },
            {
                "song_id": "invalid",
                "maidata_path": "invalid/maidata.txt",
                "audio_path": "invalid/track.mp3",
                "difficulties": [{"difficulty_index": 5, "level_raw": "13"}],
            },
        ]
    }), encoding="utf-8")

    dataset = EventDataset(manifest, cache_dir=tmp_path / "cache", eager_audio=False, segment_beats=4, max_tokens=128)
    audit = dataset.audit()
    assert audit["samples"] == 1
    assert audit["segments"] >= 1
    assert not any(item.get("difficulty_index") == 1 for item in audit["skipped"])
    assert any(item["song_id"] == "invalid" and item["reason"] == "invalid_chart" for item in audit["skipped"])
    fake_audio = EventAudio(
        np.zeros((2, ACOUSTIC_DIM), dtype=np.float32),
        np.asarray([0.0, 2.0], dtype=np.float32),
        2.0,
        120.0,
    )
    dataset._load_audio = lambda ref: fake_audio
    item = dataset[0]
    assert item["audio"].shape[1] == INPUT_DIM
    assert item["global_audio"].shape == (GLOBAL_MEMORY_FRAMES, INPUT_DIM)
    output = capsys.readouterr().out
    assert "event_dataset_summary" in output
    assert len(output) < 4000


class _ScriptedModel:
    def __init__(self, script, vocab_size):
        self.script = script
        self.vocab_size = vocab_size
        self.decode_calls = 0

    def eval(self):
        return self

    def encode_audio(self, audio, condition, **kwargs):
        return torch.zeros((1, 1, 8), device=audio.device), None

    def decode_memory(self, memory, decoder_input, condition, audio_mask=None):
        self.decode_calls += 1
        step = decoder_input.size(1) - 1
        token_id = self.script[min(step, len(self.script) - 1)]
        logits = torch.full(
            (1, decoder_input.size(1), self.vocab_size), -100.0, device=decoder_input.device
        )
        logits[:, -1, token_id] = 10.0
        return logits

    def decode_step(self, memory, token, condition, caches, position, audio_mask=None):
        """Cached counterpart of ``decode_memory`` for one new token."""

        self.decode_calls += 1
        token_id = self.script[min(position, len(self.script) - 1)]
        logits = torch.full(
            (1, 1, self.vocab_size), -100.0, device=token.device
        )
        logits[:, -1, token_id] = 10.0
        return logits, caches


def test_decoder_preserves_multi_character_note_and_emits_eos():
    vocabulary = EventVocabulary.from_event_charts([], segment_beats=4)
    script = [
        vocabulary.token_to_id[token]
        for token in ("AT:0", "CHAR:4", "CHAR:-", "CHAR:6", "<event_end>",
                      "AT:1", "CHAR:1", "<event_end>", "<eos>")
    ]
    model = _ScriptedModel(script, len(vocabulary))
    result = event_cli._decode(
        model,
        vocabulary,
        torch.zeros((2, INPUT_DIM)),
        torch.zeros((4, INPUT_DIM)),
        torch.zeros(24),
        torch.device("cpu"),
        max_tokens=32,
        temperature=0.8,
        sample=False,
        seed=1,
    )
    assert result == EventSequence([Fraction(0), Fraction(1)], ["4-6", "1"])
    assert model.decode_calls == len(script)


def test_decoder_fails_loudly_when_max_tokens_cuts_an_event():
    vocabulary = EventVocabulary.from_event_charts([], segment_beats=4)
    script = [vocabulary.token_to_id[token] for token in ("AT:0", "CHAR:4", "CHAR:-", "CHAR:6")]
    model = _ScriptedModel(script, len(vocabulary))
    with pytest.raises(ValueError, match="max_tokens"):
        event_cli._decode(
            model,
            vocabulary,
            torch.zeros((2, INPUT_DIM)),
            torch.zeros((4, INPUT_DIM)),
            torch.zeros(24),
            torch.device("cpu"),
            max_tokens=4,
            temperature=0.8,
            sample=False,
            seed=1,
        )


def test_note_grammar_rejects_random_character_prefixes_and_duplicate_heads():
    grammar = event_cli._NoteGrammar.from_atoms(["4-6[8:1]", "1", "2"])
    assert grammar is not None
    assert grammar.can_continue("4-")
    assert not grammar.can_continue("4-A")
    assert grammar.is_complete("4-6[8:1]")
    assert not grammar.is_complete("1/1")
    chord_grammar = event_cli._NoteGrammar.from_atoms(["1", "2"])
    assert chord_grammar is not None and chord_grammar.can_continue("1/")
    assert chord_grammar.is_complete("1/2")
    assert not chord_grammar.is_complete("1/1")


def test_decoder_uses_checkpoint_atom_catalog_to_keep_note_text_bounded():
    vocabulary = EventVocabulary.from_event_charts([], segment_beats=4)
    script = [
        vocabulary.token_to_id[token]
        for token in ("AT:0", "CHAR:4", "CHAR:-", "CHAR:6", "CHAR:[", "CHAR:8",
                      "CHAR::", "CHAR:1", "CHAR:]", "<event_end>", "<eos>")
    ]
    model = _ScriptedModel(script, len(vocabulary))
    result = event_cli._decode(
        model, vocabulary, torch.zeros((2, INPUT_DIM)), torch.zeros((4, INPUT_DIM)), torch.zeros(24),
        torch.device("cpu"), max_tokens=32, temperature=0.8, sample=False, seed=1,
        atom_catalog=["4-6[8:1]"],
    )
    assert result.note_texts == ["4-6[8:1]"]


def test_cli_writes_explicit_failure_report_and_no_fake_chart(tmp_path, monkeypatch):
    audio = EventAudio(
        np.zeros((1, ACOUSTIC_DIM), dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
        1.0,
        120.0,
    )
    vocabulary = EventVocabulary.from_event_charts([], segment_beats=4)
    monkeypatch.setattr(event_cli, "load_event_model", lambda path, device: (object(), vocabulary, {"epoch": 2, "schema": "maichart-event-generator-checkpoint-v1", "data_contract": {"target_schema": "event_v1"}}))
    monkeypatch.setattr(event_cli, "extract_event_audio", lambda path, sample_rate: audio)
    monkeypatch.setattr(event_cli, "_decode", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic decoder failure")))
    output = tmp_path / "generated" / "maidata.txt"

    result = event_cli.main([
        "--checkpoint", str(tmp_path / "model.pt"),
        "--audio", str(tmp_path / "song.mp3"),
        "--output", str(output),
        "--device", "cpu",
    ])

    assert result == 1
    assert not output.exists()
    quality = json.loads(output.with_suffix(".quality.json").read_text(encoding="utf-8"))
    error = json.loads(output.with_suffix(".error.json").read_text(encoding="utf-8"))
    assert quality["generation_status"] == "failed"
    assert error["fatal_errors"][0]["code"] == "generation_failed"


def test_cli_success_path_writes_chart_ir_and_quality_report(tmp_path, monkeypatch):
    audio = EventAudio(
        np.zeros((1, ACOUSTIC_DIM), dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
        1.0,
        120.0,
    )
    vocabulary = EventVocabulary.from_event_charts([], segment_beats=4)
    sequence = EventSequence([Fraction(0)], ["4-6[8:1]"])
    monkeypatch.setattr(event_cli, "load_event_model", lambda path, device: (object(), vocabulary, {"epoch": 2, "schema": "maichart-event-generator-checkpoint-v1", "data_contract": {"target_schema": "event_v1"}}))
    monkeypatch.setattr(event_cli, "extract_event_audio", lambda path, sample_rate: audio)
    monkeypatch.setattr(event_cli, "_decode", lambda *args, **kwargs: sequence)
    output = tmp_path / "generated" / "maidata.txt"

    result = event_cli.main([
        "--checkpoint", str(tmp_path / "model.pt"),
        "--audio", str(tmp_path / "song.mp3"),
        "--output", str(output),
        "--device", "cpu",
        "--max-tokens", "32",
    ])

    assert result == 0
    assert output.exists()
    assert "4-6[8:1]" in output.read_text(encoding="utf-8")
    chart_ir = json.loads(output.with_suffix(".chart.json").read_text(encoding="utf-8"))
    quality = json.loads(output.with_suffix(".quality.json").read_text(encoding="utf-8"))
    assert chart_ir["schema"] == "maichart-event-chart-v1"
    assert quality["generation_status"] == "success"
