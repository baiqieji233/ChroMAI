"""The gold workflow must split the corpus correctly and preserve the schema.

Two things matter and both were broken before by accident:

* an event manifest marks usable difficulties with ``event_usable_for_training``
  (the legacy helper only knows ``usable_for_training``), so a split built with
  the legacy helper silently produced empty manifests;
* the held-out gold validation songs must be **absent from the pretrain
  manifest**, otherwise the acceptance signal is contaminated by songs the model
  already memorised during stage 1.
"""

import json
from pathlib import Path

import pytest

from maichart.training.gold_set import (
    build_gold_set,
    discover_gold_songs,
    exclude_gold_songs,
    scan_gold_folder,
    select_gold_songs,
    songs_with_usable_difficulties,
    split_gold,
)


def _song(song_id, designer, notes=500, usable=True, title=None):
    return {
        "song_id": song_id,
        "title": title or f"Song {song_id}",
        "artist": "Tester",
        "maidata_path": f"{song_id}/maidata.txt",
        "audio_path": f"{song_id}/track.mp3",
        "wholebpm": 150,
        "dataset_usable": usable,
        "audio": {"readable": True, "status": "source_present"},
        "difficulties": [{
            "difficulty_index": 5,
            "difficulty_name": "Re:Master",
            "level_raw": "13",
            "level": 13.0,
            "designer": designer,
            "has_chart": True,
            "note_count": notes,
            "event_usable_for_training": usable,
            "filter_reasons": [],
        }],
    }


def _manifest(songs):
    return {
        "schema": "maichart-training-manifest-v1",
        "source_root": ".",
        "cache_dir": "cache",
        "song_count": len(songs),
        "difficulty_count": len(songs),
        "usable_difficulty_count": sum(
            1 for song in songs for item in song["difficulties"] if item.get("event_usable_for_training")
        ),
        "preprocessing": {
            "schema": "maichart-preprocessing-contract-v1",
            "target_schema": "event_v1",
            "feature_profile": "event_audio_v1",
            "sample_rate": 22050,
            "hop_length": 441,
        },
        "songs": songs,
        "summary": {},
        "warnings": [],
        "errors": [],
    }


@pytest.fixture
def corpus(tmp_path):
    songs = []
    for index in range(20):
        designer = "GoodArtist" if index % 2 == 0 else "OtherDude"
        songs.append(_song(f"song{index:03d}", designer, notes=400 + index * 10))
        # Materialise the files: a written split is verified against the disk, so
        # a manifest full of placeholder paths is exactly what the check rejects.
        folder = tmp_path / f"song{index:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "maidata.txt").write_text(
            f"&title=song{index:03d}\n&wholebpm=134\n&first=0\n&lv_4=13\n&inote_4=(134){{4}}"
            + "1,2,3,4," * 16 + "\n",
            encoding="utf-8",
        )
        (folder / "track.mp3").write_bytes(b"\x00" * 32)
    path = tmp_path / "training_manifest.json"
    path.write_text(json.dumps(_manifest(songs), ensure_ascii=False), encoding="utf-8")
    return path


def test_selection_requires_a_criterion(corpus):
    manifest = json.loads(corpus.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="谱师"):
        select_gold_songs(manifest, designers=[], song_ids=[])


def test_selection_by_designer_and_song_id(corpus):
    manifest = json.loads(corpus.read_text(encoding="utf-8"))
    by_designer, summary = select_gold_songs(manifest, designers=["GoodArtist"])
    assert len(by_designer) == 10
    assert summary["gold_songs"] == 10
    assert all(song["gold_reason"].startswith("designer=") for song in by_designer)

    by_id, _ = select_gold_songs(manifest, song_ids=["song001"])
    assert [song["song_id"] for song in by_id] == ["song001"]
    # Case-insensitive designer matching.
    mixed_case, _ = select_gold_songs(manifest, designers=["goodartist"])
    assert len(mixed_case) == 10


def test_selection_respects_min_note_count(corpus):
    manifest = json.loads(corpus.read_text(encoding="utf-8"))
    selected, summary = select_gold_songs(manifest, designers=["GoodArtist"], min_note_count=450)
    # Even-indexed songs carry 400 + index * 10 notes, so 006 (460) and above pass.
    assert [song["song_id"] for song in selected] == [
        "song006", "song008", "song010", "song012", "song014", "song016", "song018",
    ]
    assert summary["matched_but_rejected"]["note_count_low"] == 3


def test_legacy_usable_flag_is_accepted(tmp_path):
    songs = [_song("a", "Legacy", usable=False)]
    songs[0]["difficulties"][0]["event_usable_for_training"] = False
    songs[0]["difficulties"][0]["usable_for_training"] = True
    manifest = _manifest(songs)
    assert len(songs_with_usable_difficulties(manifest["songs"])) == 1


def test_split_holds_out_validation_songs(corpus):
    summary = build_gold_set(corpus, out_dir=corpus.parent / "gold", designers=["GoodArtist"],
                             val_ratio=0.4, seed=7, write_stats=False)
    assert summary["selection"]["gold_songs"] == 10
    assert summary["gold_finetune"]["songs"] + summary["gold_val"]["songs"] == 10
    assert summary["gold_val"]["songs"] == 4
    assert summary["pretrain"]["songs"] == 20 - 4

    pretrain = json.loads((corpus.parent / "gold" / "pretrain_manifest.json").read_text(encoding="utf-8"))
    validation = json.loads((corpus.parent / "gold" / "gold_val_manifest.json").read_text(encoding="utf-8"))
    finetune = json.loads((corpus.parent / "gold" / "gold_finetune_manifest.json").read_text(encoding="utf-8"))

    val_ids = {song["song_id"] for song in validation["songs"]}
    pretrain_ids = {song["song_id"] for song in pretrain["songs"]}
    finetune_ids = {song["song_id"] for song in finetune["songs"]}
    assert not (val_ids & pretrain_ids), "held-out gold songs must not be in the pretrain set"
    assert not (val_ids & finetune_ids), "held-out gold songs must not be trainable at all"
    assert pretrain_ids | val_ids == {song["song_id"] for song in json.loads(
        corpus.read_text(encoding="utf-8"))["songs"]}


def test_split_is_deterministic_and_preserves_the_event_schema(corpus):
    first = build_gold_set(corpus, out_dir=corpus.parent / "gold_a", designers=["GoodArtist"],
                           val_ratio=0.4, seed=11, write_stats=False)
    second = build_gold_set(corpus, out_dir=corpus.parent / "gold_b", designers=["GoodArtist"],
                            val_ratio=0.4, seed=11, write_stats=False)
    a = json.loads((corpus.parent / "gold_a" / "gold_val_manifest.json").read_text(encoding="utf-8"))
    b = json.loads((corpus.parent / "gold_b" / "gold_val_manifest.json").read_text(encoding="utf-8"))
    assert [song["song_id"] for song in a["songs"]] == [song["song_id"] for song in b["songs"]]
    assert first["gold_val"]["songs"] == second["gold_val"]["songs"]

    # The WebUI decides whether a manifest is an event manifest from these
    # fields, so the split must carry them through unchanged.
    assert a["preprocessing"]["target_schema"] == "event_v1"
    assert a["preprocessing"]["feature_profile"] == "event_audio_v1"
    assert a["schema"] == "maichart-training-manifest-v1"
    assert a["cache_dir"]


def test_split_gold_never_empties_the_training_side():
    songs = [_song("only", "Solo")]
    train, val = split_gold(songs, val_ratio=0.5, seed=1)
    assert len(train) == 1 and not val
    with pytest.raises(ValueError):
        split_gold(songs, val_ratio=1.5, seed=1)


# --------------------------------------------------------------- gold folder


def _write_raw_chart(directory, notes=("1,2,3,4," * 16), level="13", audio="track.mp3"):
    """A real chart folder: 64 notes so the preprocessor accepts it.

    The title embeds the folder name so every chart has *distinct* content --
    chart identity is compared by maidata digest, and identical text would make
    different songs look like copies of each other.
    """

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "maidata.txt").write_text(
        f"&title={directory.name}\n&artist=Tester\n&wholebpm=134\n&first=0\n"
        f"&lv_4={level}\n&inote_4=(134){{4}}{notes}\n",
        encoding="utf-8",
    )
    if audio:
        (directory / audio).write_bytes(b"\x00" * 32)
    return directory


@pytest.fixture
def raw_corpus(tmp_path):
    """A corpus root with two normal songs and a `gold` folder holding three."""

    root = tmp_path / "raw"
    _write_raw_chart(root / "song_a")
    _write_raw_chart(root / "song_b")
    _write_raw_chart(root / "gold" / "GoodOne")
    _write_raw_chart(root / "gold" / "GoodTwo")
    # Nested grouping is allowed, and a chart without audio must be skipped.
    _write_raw_chart(root / "gold" / "set2024" / "GoodThree")
    _write_raw_chart(root / "gold" / "NoAudio", audio=None)
    return root


def test_discover_gold_songs_finds_nested_folders_and_skips_broken(raw_corpus):
    songs, summary = discover_gold_songs(raw_corpus / "gold", raw_corpus)
    # GoodOne, GoodTwo and set2024/GoodThree are usable; NoAudio has no audio.
    assert summary["found_directories"] == 4
    assert summary["songs"] == 3
    assert summary["skipped"] == 1
    assert summary["skipped_examples"][0]["reason"] == "no_usable_difficulty"
    assert "audio_missing" in summary["skipped_examples"][0]["message"]
    # Paths stay relative to the corpus root so a manifest can be written beside it.
    paths = sorted(song["maidata_path"] for song in songs)
    assert paths == ["gold/GoodOne/maidata.txt", "gold/GoodTwo/maidata.txt",
                     "gold/set2024/GoodThree/maidata.txt"]
    assert all(song["difficulties"][0]["event_usable_for_training"] for song in songs)


def test_scan_gold_folder_reports_without_writing(raw_corpus, tmp_path):
    manifest_path = tmp_path / "training_manifest.json"
    manifest_path.write_text(json.dumps(_manifest([])), encoding="utf-8")
    report = scan_gold_folder(manifest_path, gold_root=raw_corpus / "gold")
    assert report["exists"] is True
    assert report["songs"] == 3
    assert len(report["titles"]) == 3
    assert not (tmp_path / "gold_set").exists()

    missing = scan_gold_folder(manifest_path, gold_root=raw_corpus / "nope")
    assert missing["exists"] is False
    assert missing["songs"] == 0


def test_exclude_gold_songs_matches_copies_by_content(raw_corpus):
    """A gold chart copied under a different path is still recognised."""

    import shutil

    corpus_songs = [
        {"song_id": "song_a", "maidata_path": "song_a/maidata.txt", "audio_path": "song_a/track.mp3"},
        {"song_id": "song_b", "maidata_path": "song_b/maidata.txt", "audio_path": "song_b/track.mp3"},
    ]
    # Same chart text, different folder: must still be excluded.
    copied = raw_corpus / "elsewhere" / "CopyOfA"
    shutil.copytree(raw_corpus / "song_a", copied)
    gold_songs = [{"song_id": "CopyOfA", "maidata_path": "elsewhere/CopyOfA/maidata.txt",
                   "audio_path": "elsewhere/CopyOfA/track.mp3"}]
    kept, dropped = exclude_gold_songs(corpus_songs, gold_songs, raw_corpus)
    assert dropped == ["song_a"]
    assert [song["song_id"] for song in kept] == ["song_b"]


def test_build_gold_set_from_folder(raw_corpus, tmp_path):
    manifest = _manifest([_song("song_a", "Other"), _song("song_b", "Other")])
    # The manifest lives inside the corpus root, exactly like the real datasets,
    # so its relative song paths resolve against its own directory.
    manifest_path = raw_corpus / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    summary = build_gold_set(
        manifest_path, out_dir=tmp_path / "gold_set", gold_root=raw_corpus / "gold",
        val_ratio=0.34, seed=5, write_stats=False,
    )
    assert summary["mode"] == "folder"
    assert summary["selection"]["gold_songs"] == 3
    assert summary["gold_val"]["songs"] == 1
    assert summary["gold_finetune"]["songs"] == 2

    out = tmp_path / "gold_set"
    for name in ("pretrain_manifest.json", "gold_finetune_manifest.json",
                 "gold_val_manifest.json", "gold_summary.json"):
        assert (out / name).is_file(), name
    pretrain = json.loads((out / "pretrain_manifest.json").read_text(encoding="utf-8"))
    validation = json.loads((out / "gold_val_manifest.json").read_text(encoding="utf-8"))
    finetune = json.loads((out / "gold_finetune_manifest.json").read_text(encoding="utf-8"))
    # The gold charts live outside the corpus manifest, so nothing is removed
    # from pretraining; every split keeps the event schema.
    assert pretrain["song_count"] == 2
    assert validation["song_count"] == 1
    assert finetune["song_count"] == 2
    for split in (pretrain, validation, finetune):
        assert split["preprocessing"]["target_schema"] == "event_v1"
        assert split["cache_dir"]


def test_build_gold_set_from_folder_excludes_held_out_songs_from_pretrain(raw_corpus, tmp_path):
    """When gold charts are also in the corpus, the held-out ones must disappear."""

    manifest = _manifest([
        _song("song_a", "Other"),
        _song("gold/GoodOne", "Other"),
        _song("gold/GoodTwo", "Other"),
        _song("gold/set2024/GoodThree", "Other"),
    ])
    manifest_path = raw_corpus / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    summary = build_gold_set(
        manifest_path, out_dir=tmp_path / "gold_set2", gold_root=raw_corpus / "gold",
        val_ratio=0.34, seed=5, write_stats=False,
    )
    assert summary["gold_val"]["songs"] == 1
    assert summary["excluded_from_pretrain"]["songs"] == 1
    assert summary["pretrain"]["songs"] == 3
    pretrain = json.loads((tmp_path / "gold_set2" / "pretrain_manifest.json").read_text(encoding="utf-8"))
    val = json.loads((tmp_path / "gold_set2" / "gold_val_manifest.json").read_text(encoding="utf-8"))
    held_out = {song["maidata_path"] for song in val["songs"]}
    assert held_out.isdisjoint({song["maidata_path"] for song in pretrain["songs"]})


def test_stale_corpus_entries_are_skipped_not_fatal(raw_corpus, tmp_path, capsys):
    """A manifest can outlive its files: moving songs into ``gold`` used to abort.

    The user's real case: the corpus manifest still lists songs whose folders were
    moved into the gold folder (or deleted).  The pretrain subset must skip them
    and say so -- the previous behaviour was a hard failure whose message blamed
    the gold folder, even though every gold chart was perfectly fine.
    """

    songs = [_song("song_a", "Other"), _song("song_b", "Other"), _song("gold/GoodOne", "Other")]
    manifest_path = raw_corpus / "training_manifest.json"
    manifest_path.write_text(json.dumps(_manifest(songs)), encoding="utf-8")
    # song_b's files vanish from the corpus tree (for example: moved away).
    for name in ("maidata.txt", "track.mp3"):
        (raw_corpus / "song_b" / name).unlink()

    summary = build_gold_set(
        manifest_path, out_dir=tmp_path / "gold_set_stale", gold_root=raw_corpus / "gold",
        val_ratio=0.34, seed=5, write_stats=False,
    )
    assert summary["pretrain"]["songs"] == 1, "the two surviving songs minus the gold one"
    skipped = summary["pretrain_skipped_missing_source"]
    assert skipped["songs"] == 1
    assert "song_b" in skipped["examples"][0]
    assert "不存在" in skipped["examples"][0]
    pretrain = json.loads((tmp_path / "gold_set_stale" / "pretrain_manifest.json").read_text(encoding="utf-8"))
    assert [song["song_id"] for song in pretrain["songs"]] == ["song_a"]
    log = capsys.readouterr().out
    assert "跳过 1 首" in log and "song_b" in log
    assert "重新预处理" in log, "the log must say how to actually fix the stale manifest"


def test_missing_gold_chart_still_fails_with_a_precise_message(tmp_path):
    """A broken *gold* chart is a real error: name the song and the file.

    ``discover_gold_songs`` already skips charts without audio, so this path
    matters when a path vanishes between discovery and writing (or when gold
    songs come from the corpus by designer).  The old message blamed the whole
    gold folder without naming anything; that is what made the real failure
    ("91 stale corpus entries") so hard to read.
    """

    from maichart.training import gold_set

    base = tmp_path / "corpus"
    base.mkdir()
    song = _song("gone", "GoodArtist")
    manifest = _manifest([song])
    with pytest.raises(ValueError) as excinfo:
        gold_set._write_manifest(manifest, [song], tmp_path / "out3" / "gold_val_manifest.json",
                                 source_base=base)
    message = str(excinfo.value)
    assert "gone" in message, "the offending song id must be named"
    assert "maidata_path" in message, "the offending field must be named"
    assert "金标歌本身" in message, "the message must scope the failure to gold charts"


def test_build_gold_set_rejects_an_empty_gold_folder(tmp_path):
    root = tmp_path / "raw"
    _write_raw_chart(root / "song_a")
    (root / "gold").mkdir()
    manifest_path = tmp_path / "training_manifest.json"
    manifest_path.write_text(json.dumps(_manifest([])), encoding="utf-8")
    with pytest.raises(ValueError, match="gold 文件夹"):
        build_gold_set(manifest_path, out_dir=tmp_path / "out", gold_root=root / "gold",
                       write_stats=False)


def test_gold_manifests_stay_resolvable_from_a_nested_output_folder(raw_corpus, tmp_path):
    """A split written into a subfolder must still find its charts.

    Writing the outputs into the charts folder itself (or into a folder inside
    it) used to leave every song path measured from the wrong base: training then
    failed with "Source maidata/audio is missing" for every song.
    """

    manifest_path = raw_corpus / "training_manifest.json"
    manifest_path.write_text(json.dumps(_manifest([_song("song_a", "Other")])), encoding="utf-8")
    out = raw_corpus / "gold" / "gold_set"  # nested inside the charts folder
    summary = build_gold_set(manifest_path, out_dir=out, gold_root=raw_corpus / "gold",
                             val_ratio=0.34, seed=3, write_stats=False)
    assert summary["gold_val"]["songs"] + summary["gold_finetune"]["songs"] >= 1

    for name in ("pretrain_manifest.json", "gold_finetune_manifest.json", "gold_val_manifest.json"):
        data = json.loads((out / name).read_text(encoding="utf-8"))
        for song in data["songs"]:
            path = Path(song["maidata_path"])
            resolved = path if path.is_absolute() else (out / path).resolve()
            assert resolved.is_file(), f"{name}: {song['maidata_path']} does not resolve"


def test_gold_manifests_use_absolute_paths_and_survive_being_moved(raw_corpus, tmp_path):
    """A derived manifest must keep working after it is copied elsewhere.

    Relative paths are tied to the folder the manifest was written into; moving
    the file then breaks every song with "Source maidata/audio is missing" and no
    hint that the manifest, not the data, moved.
    """

    manifest_path = raw_corpus / "training_manifest.json"
    manifest_path.write_text(json.dumps(_manifest([_song("song_a", "Other")])), encoding="utf-8")
    out = tmp_path / "gold_set"
    build_gold_set(manifest_path, out_dir=out, gold_root=raw_corpus / "gold",
                   val_ratio=0.34, seed=3, write_stats=False)

    moved = tmp_path / "elsewhere" / "deep" / "nested"
    moved.mkdir(parents=True)
    for name in ("pretrain_manifest.json", "gold_finetune_manifest.json", "gold_val_manifest.json"):
        source = out / name
        target = moved / name
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        data = json.loads(target.read_text(encoding="utf-8"))
        for song in data["songs"]:
            assert Path(song["maidata_path"]).is_absolute(), f"{name} stored a relative path"
            assert Path(song["maidata_path"]).is_file()
            assert Path(song["audio_path"]).is_file()


def test_build_gold_set_fails_loudly_when_a_chart_disappeared(raw_corpus, tmp_path):
    """A manifest that cannot find its data must not be written at all."""

    from maichart.training.gold_set import _absolutize_manifest_paths

    # Unit level: a path that does not resolve is reported, not silently stored.
    missing = _absolutize_manifest_paths(
        {"songs": [{"song_id": "gone", "maidata_path": "gone/maidata.txt"}]}, tmp_path)
    assert missing and "gone" in missing[0]
    assert missing[0].endswith("maidata_path=gone/maidata.txt")

    # Integration level: after a build, every stored path exists on disk.
    manifest_path = raw_corpus / "training_manifest.json"
    manifest_path.write_text(json.dumps(_manifest([_song("song_a", "Other")])), encoding="utf-8")
    out = tmp_path / "gold_set"
    build_gold_set(manifest_path, out_dir=out, gold_root=raw_corpus / "gold",
                   val_ratio=0.34, seed=3, write_stats=False)
    for name in ("pretrain_manifest.json", "gold_finetune_manifest.json", "gold_val_manifest.json"):
        data = json.loads((out / name).read_text(encoding="utf-8"))
        for song in data["songs"]:
            assert Path(song["maidata_path"]).is_file(), name
            assert Path(song["audio_path"]).is_file(), name


def test_dataset_recovers_a_manifest_moved_one_level_up(tmp_path, monkeypatch):
    """The manifest sits in the dataset root while paths are relative to gold/."""

    from maichart.generation.event_audio import EventAudio
    from maichart.training import event_dataset as module
    from maichart.training.event_dataset import EventDataset

    raw = tmp_path / "raw"
    _write_raw_chart(raw / "gold" / "GoodOne")
    manifest_path = raw / "gold_val_manifest.json"
    manifest_path.write_text(json.dumps({
        "schema": "maichart-training-manifest-v1",
        "source_root": str(raw),
        "cache_dir": str(raw / "cache"),
        "preprocessing": {"target_schema": "event_v1", "feature_profile": "event_audio_v1"},
        "songs": [{
            "song_id": "GoodOne", "title": "GoodOne", "maidata_path": "GoodOne/maidata.txt",
            "audio_path": "GoodOne/track.mp3", "wholebpm": 134,
            "difficulties": [{"difficulty_index": 4, "level_raw": "13", "level": 13.0,
                              "event_usable_for_training": True, "note_count": 64}],
        }],
        "summary": {}, "warnings": [], "errors": [],
    }, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(module, "extract_event_audio", lambda path, **kwargs: EventAudio(
        features=np.zeros((500, 98), dtype=np.float32), times=np.arange(500, dtype=np.float32) * 0.02,
        duration_sec=10.0, tempo_bpm=134.0, tempo_changes=[], metadata={}))
    dataset = EventDataset(manifest_path, cache_dir=tmp_path / "cache", segment_beats=16,
                           max_tokens=512, eager_audio=False)
    assert len(dataset.samples) == 1, "the subfolder holding the charts must be discovered"
    assert dataset._extra_base == (raw / "gold").resolve() or str(dataset._extra_base).endswith("gold")


def test_dataset_finds_songs_when_the_manifest_lives_in_a_subfolder(tmp_path, monkeypatch):
    """The dataset must survive a manifest whose paths are relative to the root.

    This is the failure that produced "No usable event segments": every song was
    reported as missing because only the manifest's own folder was tried.
    """

    from maichart.training import event_dataset as module
    from maichart.training.event_dataset import EventDataset

    raw = tmp_path / "raw"
    _write_raw_chart(raw / "song_a")
    manifest_dir = raw / "gold_set"
    manifest_dir.mkdir()
    manifest_path = manifest_dir / "gold_val_manifest.json"
    manifest_path.write_text(json.dumps({
        "schema": "maichart-training-manifest-v1",
        "source_root": str(raw),
        # Relative to the dataset root, not to the manifest's folder.
        "cache_dir": "cache",
        "preprocessing": {"target_schema": "event_v1", "feature_profile": "event_audio_v1"},
        "songs": [{
            "song_id": "song_a", "title": "Song a", "maidata_path": "song_a/maidata.txt",
            "audio_path": "song_a/track.mp3", "wholebpm": 134,
            "difficulties": [{"difficulty_index": 4, "level_raw": "13", "level": 13.0,
                              "event_usable_for_training": True, "note_count": 64}],
        }],
        "summary": {}, "warnings": [], "errors": [],
    }, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(module, "extract_event_audio",
                        lambda path, **kwargs: __import__("maichart.generation.event_audio",
                                                          fromlist=["EventAudio"]).EventAudio(
                            features=np.zeros((500, 98), dtype=np.float32),
                            times=np.arange(500, dtype=np.float32) * 0.02,
                            duration_sec=10.0, tempo_bpm=134.0, tempo_changes=[], metadata={}))
    dataset = EventDataset(manifest_path, cache_dir=tmp_path / "cache", segment_beats=16, max_tokens=512,
                           eager_audio=False)
    assert len(dataset.samples) == 1, "the song must be found via the declared source root"
    assert dataset.samples[0].source_path.is_file()
    assert len(dataset) > 0
