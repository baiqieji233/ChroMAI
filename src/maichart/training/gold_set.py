"""Turn selected high-quality charts into a training group and an acceptance bar.

The gold workflow produces three manifests from one full training manifest:

``pretrain_manifest.json``
    The whole corpus **minus the held-out gold validation songs**.  Stage 1
    trains here, so the acceptance songs are never seen during training.
``gold_finetune_manifest.json``
    The gold songs that are allowed to be trained on (stage 2, ``finetune``).
``gold_val_manifest.json``
    Held-out gold songs.  Used as the validation manifest for **both** stages,
    so "which checkpoint is best" is decided by how well it predicts the kind of
    chart a human would call good.

It also computes ``gold_stats.json``: per-metric percentiles over the held-out
gold charts, which :mod:`maichart.generation.acceptance` uses to judge a
generated chart.

Selection is deliberately objective and reviewable: a song is gold when its
designer is on the supplied list or its ``song_id`` is listed explicitly, and
(when ``--min-note-count`` is set) one of its usable difficulties has at least
that many notes.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

from maichart.generation.acceptance import gold_reference, metric_bundle
from maichart.preprocess import _rebase_training_manifest_paths, _subset_training_manifest


def _difficulty_usable(difficulty: dict[str, Any]) -> bool:
    """Usable flag for either manifest flavour.

    Event manifests set ``event_usable_for_training`` while the legacy
    preprocessor sets ``usable_for_training``; the legacy split helper only
    understands the latter, which is why splitting an event manifest used to
    produce empty files.
    """

    return bool(difficulty.get("event_usable_for_training") or difficulty.get("usable_for_training"))


def usable_difficulties(song: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in song.get("difficulties", []) if _difficulty_usable(item)]


def songs_with_usable_difficulties(songs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for song in songs:
        difficulties = usable_difficulties(song)
        if difficulties:
            copied = dict(song)
            copied["difficulties"] = difficulties
            result.append(copied)
    return result


def _max_note_count(song: dict[str, Any]) -> int:
    counts = [int(item.get("note_count") or 0) for item in usable_difficulties(song)]
    return max(counts) if counts else 0


def _designer_names(song: dict[str, Any]) -> set[str]:
    return {
        str(item.get("designer") or "").strip().casefold()
        for item in usable_difficulties(song)
        if str(item.get("designer") or "").strip()
    }


def select_gold_songs(
    manifest: dict[str, Any],
    *,
    designers: Sequence[str] = (),
    song_ids: Sequence[str] = (),
    min_note_count: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pick the gold songs and explain why each one was or was not selected."""

    wanted_designers = {str(name).strip().casefold() for name in designers if str(name).strip()}
    wanted_ids = {str(value).strip() for value in song_ids if str(value).strip()}
    if not wanted_designers and not wanted_ids:
        raise ValueError("请至少提供一位谱师名或一个 song_id：金标集必须来自明确的筛选条件")

    candidates = songs_with_usable_difficulties(manifest.get("songs", []))
    selected: list[dict[str, Any]] = []
    rejected: dict[str, int] = {"song_id_missing": 0, "designer_missing": 0, "note_count_low": 0, "no_usable_chart": 0}
    reasons: list[dict[str, Any]] = []
    by_id = {str(song.get("song_id")): song for song in manifest.get("songs", [])}
    for song in manifest.get("songs", []):
        song_id = str(song.get("song_id"))
        if not usable_difficulties(song):
            continue
        entry = by_id.get(song_id, song)
        matched_id = song_id in wanted_ids
        matched_designer = bool(_designer_names(entry) & wanted_designers)
        note_count = _max_note_count(entry)
        if not matched_id and not matched_designer:
            continue
        if min_note_count and note_count < min_note_count:
            rejected["note_count_low"] += 1
            reasons.append({"song_id": song_id, "reason": "note_count_low", "note_count": note_count})
            continue
        copied = dict(entry)
        copied["gold_reason"] = (
            "song_id" if matched_id else f"designer={sorted(_designer_names(entry) & wanted_designers)[0]}"
        )
        selected.append(copied)
    selected.sort(key=lambda item: str(item.get("song_id")))
    summary = {
        "criteria": {
            "designers": sorted(wanted_designers),
            "song_ids": sorted(wanted_ids),
            "min_note_count": int(min_note_count),
        },
        "gold_songs": len(selected),
        "gold_difficulties": sum(len(usable_difficulties(song)) for song in selected),
        "matched_but_rejected": rejected,
        "rejected_examples": reasons[:10],
    }
    return selected, summary


def split_gold(
    gold_songs: Sequence[dict[str, Any]], *, val_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split gold songs into (finetune, held-out validation) by song."""

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val-ratio 必须在 0 和 1 之间")
    songs = list(gold_songs)
    random.Random(int(seed)).shuffle(songs)
    val_count = max(1, int(round(len(songs) * float(val_ratio)))) if len(songs) > 1 else 0
    val_count = min(val_count, max(0, len(songs) - 1))
    return songs[val_count:], songs[:val_count]


def _resolved_manifest_path(value: Any, base: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def _drop_missing_source_songs(
    songs: Sequence[dict[str, Any]], base: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    """Drop corpus songs whose source files no longer exist.

    A manifest outlives the files it describes: songs are moved into the ``gold``
    folder, deleted, or the corpus is edited after preprocessing.  Such an entry
    can never be trained on, so the pretrain subset skips it and reports it
    instead of failing the whole build.  Only ``maidata_path`` / ``audio_path``
    are fatal; auxiliary paths are cleared and the song is kept.
    """

    kept: list[dict[str, Any]] = []
    missing: list[str] = []
    for song in songs:
        fatal = [key for key in ("maidata_path", "audio_path")
                 if song.get(key) and not _resolved_manifest_path(song[key], base).exists()]
        if fatal:
            missing.append(f"{song.get('song_id', '?')}（{'、'.join(fatal)} 指向的文件不存在）")
            continue
        optional = [key for key in ("convert_report_path",)
                    if song.get(key) and not _resolved_manifest_path(song[key], base).exists()]
        if optional:
            song = dict(song)
            for key in optional:
                song.pop(key, None)
        kept.append(song)
    return kept, missing


def _write_manifest(
    manifest: dict[str, Any],
    songs: Sequence[dict[str, Any]],
    target: Path,
    *,
    source_base: Path,
) -> dict[str, Any]:
    subset = _subset_training_manifest(manifest, [[song] for song in songs])
    _rebase_training_manifest_paths(subset, source_base=source_base, target_base=target.parent)
    target.parent.mkdir(parents=True, exist_ok=True)
    # A split is a derived file: it gets copied into a dataset folder, kept as a
    # backup, or pointed at from somewhere else.  Relative paths are tied to the
    # folder the manifest happened to be written into, so moving it silently
    # breaks every song ("Source maidata/audio is missing") with no hint that the
    # file, not the data, moved.  Store absolute paths instead.
    missing = _absolutize_manifest_paths(subset, target.parent)
    if missing:
        raise ValueError(
            f"金标清单里有 {len(missing)} 条路径指向不存在的文件，清单不会写出：\n  "
            + "\n  ".join(missing[:8])
            + "\n以上是**金标歌本身**。请确认 gold 文件夹里每首歌都有 maidata.txt 与真实存在的音频"
              "（track.mp3/wav/ogg）；若这些谱面来自训练集目录，请检查它们是否被移动或删除，"
              "或重新预处理训练集以刷新清单。"
        )
    target.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")
    return subset


def _absolutize_manifest_paths(manifest: dict[str, Any], base: Path) -> list[str]:
    """Rewrite relative paths as absolute; return the ones that do not exist.

    Paths that resolve against ``base`` are frozen as absolute, so the manifest
    keeps working wherever it is later moved to.
    """

    missing: list[str] = []
    for key in ("source_root", "cache_dir"):
        value = manifest.get(key)
        if value and not str(value).startswith("<"):
            path = Path(str(value))
            manifest[key] = str(path if path.is_absolute() else (base / path).resolve())
    for song in manifest.get("songs", []):
        for key in ("maidata_path", "audio_path", "convert_report_path"):
            value = song.get(key)
            if not value:
                continue
            path = Path(str(value))
            resolved = path if path.is_absolute() else (base / path).resolve()
            if not resolved.exists():
                missing.append(f"{song.get('song_id', '?')} {key}={value}")
                continue
            song[key] = str(resolved)
        audio = song.get("audio")
        if isinstance(audio, dict) and audio.get("audio_features_path"):
            features = Path(str(audio["audio_features_path"]))
            resolved = features if features.is_absolute() else (base / features).resolve()
            audio["audio_features_path"] = str(resolved)
    return missing


def _chart_path(song: dict[str, Any], base: Path) -> Path:
    raw = Path(str(song.get("maidata_path")))
    return raw if raw.is_absolute() else (base / raw).resolve()


def _audio_path(song: dict[str, Any], base: Path) -> Path | None:
    value = song.get("audio_path")
    if not value:
        return None
    raw = Path(str(value))
    return raw if raw.is_absolute() else (base / raw).resolve()


def gold_stats_for_songs(
    songs: Sequence[dict[str, Any]],
    *,
    base: Path,
    cache_dir: Path,
    progress: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measure every held-out gold chart and turn it into percentile bands."""

    from maichart.event_timing import parse_event_chart
    from maichart.generation.event_audio import audio_cache_dir, extract_event_audio

    metric_sets: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for song in songs:
        source = _chart_path(song, base)
        audio_file = _audio_path(song, base)
        for difficulty in usable_difficulties(song):
            index = difficulty.get("difficulty_index", difficulty.get("index"))
            if index is None:
                continue
            try:
                chart = parse_event_chart(source, int(index))
            except Exception as exc:  # noqa: BLE001 - reported, not fatal
                failures.append({"song_id": song.get("song_id"), "difficulty": index,
                                 "stage": "parse", "error": str(exc)})
                continue
            if not chart.points:
                continue
            audio = None
            if audio_file is not None and audio_file.is_file():
                try:
                    audio = extract_event_audio(audio_file, cache_dir=audio_cache_dir(cache_dir))
                except Exception as exc:  # noqa: BLE001
                    failures.append({"song_id": song.get("song_id"), "difficulty": index,
                                     "stage": "audio", "error": str(exc)})
            duration = float(audio.duration_sec) if audio is not None else None
            try:
                metrics = metric_bundle(chart.points, audio, chart.tempo_changes,
                                        first=chart.first, duration_sec=duration)
            except Exception as exc:  # noqa: BLE001
                failures.append({"song_id": song.get("song_id"), "difficulty": index,
                                 "stage": "metrics", "error": str(exc)})
                continue
            metrics["song_id"] = song.get("song_id")
            metrics["difficulty_index"] = int(index)
            metrics["title"] = song.get("title")
            metric_sets.append(metrics)
            if progress:
                print(f"gold_measured={song.get('song_id')}:{index} "
                      f"notes={metrics.get('notes')} blank={metrics.get('blank_fraction')}", flush=True)
    reference = gold_reference(metric_sets, label="gold")
    reference["charts_detail"] = metric_sets
    reference["failures"] = failures[:20]
    return reference, metric_sets


def discover_gold_songs(
    gold_root: str | Path,
    corpus_root: str | Path,
    *,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse every chart folder under ``gold_root`` into manifest song records.

    A gold folder is an ordinary raw dataset folder: one directory per song, each
    holding ``maidata.txt`` and an audio file named ``track.mp3`` / ``track.wav``
    / ``track.ogg`` (the same layout the corpus uses).  Nesting is fine --
    ``_discover_sample_dirs`` walks recursively and skips editor/autosave
    directories -- so ``gold/setA/GoodChart/maidata.txt`` also works.

    Paths are stored relative to ``corpus_root`` so the resulting manifest can be
    written next to the corpus manifest and still resolve.
    """

    from maichart.preprocess import _discover_sample_dirs
    from maichart.training.prepare_events import _inspect_song

    gold = Path(gold_root).resolve()
    root = Path(corpus_root).resolve()
    directories = _discover_sample_dirs(gold) if gold.exists() else []
    songs: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for directory in directories:
        try:
            record = _inspect_song(directory, root, limit=limit)
        except Exception as exc:  # noqa: BLE001 - every failure is a review record
            skipped.append({"path": str(directory), "reason": "parse_error", "message": str(exc)})
            continue
        if record is None:
            continue
        usable = usable_difficulties(record)
        if not usable:
            reasons = sorted({
                reason for difficulty in record.get("difficulties", [])
                for reason in difficulty.get("filter_reasons", [])
            })
            skipped.append({
                "path": str(directory), "reason": "no_usable_difficulty",
                "message": "、".join(reasons) or "没有可训练难度",
            })
            continue
        songs.append(record)
    summary = {
        "gold_root": str(gold),
        "found_directories": len(directories),
        "songs": len(songs),
        "usable_difficulties": sum(len(usable_difficulties(song)) for song in songs),
        "skipped": len(skipped),
        "skipped_examples": skipped[:10],
        "titles": [str(song.get("title") or song.get("song_id")) for song in songs[:40]],
    }
    return songs, summary


def _manifest_song_digest(song: dict[str, Any], base: Path) -> str | None:
    """Content hash of a song's maidata file: the same chart == the same song."""

    import hashlib

    path = _chart_path(song, base)
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def exclude_gold_songs(
    corpus_songs: Sequence[dict[str, Any]],
    gold_songs: Sequence[dict[str, Any]],
    base: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Drop corpus songs that are the same chart as a held-out gold song.

    Matching is by resolved path first and by maidata content hash second, so a
    copied file (different folder, identical chart) is still recognised.  Without
    this, stage 1 would train on the very songs the acceptance bar is measured
    against, and the validation loss would be meaningless.
    """

    gold_paths = {str(_chart_path(song, base)) for song in gold_songs}
    gold_digests = {digest for song in gold_songs if (digest := _manifest_song_digest(song, base))}
    kept: list[dict[str, Any]] = []
    dropped: list[str] = []
    for song in corpus_songs:
        path = str(_chart_path(song, base))
        digest = _manifest_song_digest(song, base)
        if path in gold_paths or (digest is not None and digest in gold_digests):
            dropped.append(str(song.get("song_id")))
            continue
        kept.append(song)
    return kept, dropped


def _corpus_root(manifest: dict[str, Any], source_base: Path) -> Path:
    value = manifest.get("source_root")
    if not value:
        return source_base
    root = Path(str(value))
    return root.resolve() if root.is_absolute() else (source_base / root).resolve()


def build_gold_set(
    manifest_path: str | Path,
    *,
    out_dir: str | Path,
    cache_dir: str | Path | None = None,
    gold_root: str | Path | None = None,
    designers: Sequence[str] = (),
    song_ids: Sequence[str] = (),
    val_ratio: float = 0.3,
    seed: int = 42,
    min_note_count: int = 0,
    write_stats: bool = True,
    progress: bool = True,
) -> dict[str, Any]:
    """Build the three manifests plus the gold reference statistics.

    Two ways to choose the gold songs:

    ``gold_root``
        preferred: read the charts the user placed in a ``gold`` folder;
    ``designers`` / ``song_ids``
        fallback: pick songs out of the existing corpus by chart author or id.
    """

    source = Path(manifest_path).resolve()
    manifest = json.loads(source.read_text(encoding="utf-8"))
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    source_base = source.parent
    corpus_root = _corpus_root(manifest, source_base)

    if gold_root is not None:
        gold, folder_summary = discover_gold_songs(gold_root, corpus_root)
        if min_note_count:
            kept = [song for song in gold if _max_note_count(song) >= int(min_note_count)]
            folder_summary["rejected_note_count_low"] = len(gold) - len(kept)
            gold = kept
        if not gold:
            raise ValueError(
                f"gold 文件夹里没有找到可用谱面：{Path(gold_root).resolve()}\n"
                "请确认每首歌是独立文件夹、里面有 maidata.txt 与 track.mp3/track.wav/track.ogg"
            )
        folder_summary["criteria"] = {"min_note_count": int(min_note_count)}
        # Keep one shared key name across both modes so callers (and the WebUI)
        # do not have to know which one produced the set.
        folder_summary["gold_songs"] = len(gold)
        folder_summary["gold_difficulties"] = sum(len(usable_difficulties(song)) for song in gold)
        selection = dict(folder_summary)
        selection["mode"] = "folder"
    else:
        gold, selection = select_gold_songs(
            manifest, designers=designers, song_ids=song_ids, min_note_count=min_note_count
        )
        if not gold:
            raise ValueError("按当前条件没有选中任何歌曲：请检查谱师名拼写，或放宽 --min-note-count")
        selection = dict(selection)
        selection["mode"] = "designers"

    finetune_songs, val_songs = split_gold(gold, val_ratio=val_ratio, seed=seed)
    pretrain_songs, excluded = exclude_gold_songs(manifest.get("songs", []), val_songs, source_base)
    # The corpus manifest can outlive the files it describes (songs moved into
    # the gold folder, deleted, or the tree edited after preprocessing).  Those
    # entries cannot be trained on, so they are skipped and reported here rather
    # than aborting the build with a message about the gold folder, which was
    # never the problem.
    pretrain_songs, pretrain_missing = _drop_missing_source_songs(pretrain_songs, source_base)
    if pretrain_missing:
        print(
            f"warning: 预训练清单跳过 {len(pretrain_missing)} 首歌——源文件已不存在"
            f"（清单过期，或文件被移入 gold 文件夹）",
            flush=True,
        )
        for item in pretrain_missing[:5]:
            print(f"  skipped {item}", flush=True)
        if len(pretrain_missing) > 5:
            print(f"  ... 另有 {len(pretrain_missing) - 5} 首同样跳过", flush=True)
        print("  提示：重新预处理训练集可生成不含这些过期条目的新清单。", flush=True)

    pretrain_path = out / "pretrain_manifest.json"
    finetune_path = out / "gold_finetune_manifest.json"
    val_path = out / "gold_val_manifest.json"
    pretrain = _write_manifest(manifest, pretrain_songs, pretrain_path, source_base=source_base)
    finetune = _write_manifest(manifest, finetune_songs, finetune_path, source_base=source_base)
    validation = _write_manifest(manifest, val_songs, val_path, source_base=source_base)

    summary: dict[str, Any] = {
        "schema": "maichart-gold-set-v1",
        "mode": selection.get("mode"),
        "source_manifest": str(source),
        "out_dir": str(out),
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "selection": selection,
        "excluded_from_pretrain": {"songs": len(excluded), "song_ids": excluded[:20]},
        "pretrain_skipped_missing_source": {
            "songs": len(pretrain_missing),
            "examples": pretrain_missing[:10],
            "note": "这些歌的 maidata/audio 在训练集目录里已不存在；清单可能过期。",
        },
        "pretrain": {"path": str(pretrain_path), "songs": pretrain.get("song_count"),
                     "difficulties": pretrain.get("difficulty_count")},
        "gold_finetune": {"path": str(finetune_path), "songs": finetune.get("song_count"),
                          "difficulties": finetune.get("difficulty_count")},
        "gold_val": {"path": str(val_path), "songs": validation.get("song_count"),
                     "difficulties": validation.get("difficulty_count")},
    }
    if write_stats:
        cache = Path(cache_dir).resolve() if cache_dir else (out / "cache")
        reference, _ = gold_stats_for_songs(val_songs, base=source_base, cache_dir=cache, progress=progress)
        stats_path = out / "gold_stats.json"
        stats_path.write_text(json.dumps(reference, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["gold_stats"] = {
            "path": str(stats_path),
            "charts": reference.get("charts"),
            "metrics": sorted((reference.get("metrics") or {}).keys()),
        }
        summary["holdout_metric_summary"] = {
            key: {"p10": band["p10"], "p50": band["p50"], "p90": band["p90"]}
            for key, band in (reference.get("metrics") or {}).items()
        }
    summary_path = out / "gold_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def scan_gold_folder(
    manifest_path: str | Path,
    *,
    gold_root: str | Path | None = None,
    min_note_count: int = 0,
) -> dict[str, Any]:
    """Report what a gold folder contains, without writing anything.

    The WebUI calls this before building so the user can see which charts were
    found, which were skipped and why -- a mistyped folder name or a missing
    ``track.mp3`` should be visible before a build is attempted.
    """

    source = Path(manifest_path).resolve()
    manifest = json.loads(source.read_text(encoding="utf-8"))
    corpus_root = _corpus_root(manifest, source.parent)
    root = Path(gold_root).resolve() if gold_root else corpus_root / "gold"
    songs, summary = discover_gold_songs(root, corpus_root)
    if min_note_count:
        kept = [song for song in songs if _max_note_count(song) >= int(min_note_count)]
        summary["rejected_note_count_low"] = len(songs) - len(kept)
        summary["songs"] = len(kept)
        summary["usable_difficulties"] = sum(len(usable_difficulties(song)) for song in kept)
    summary["exists"] = root.exists()
    summary["default_root"] = str(corpus_root / "gold")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build gold training/validation manifests and acceptance statistics.",
    )
    parser.add_argument("manifest", help="full training manifest produced by preprocessing")
    parser.add_argument("--out-dir", required=True, help="where the gold manifests are written")
    parser.add_argument("--cache-dir", help="audio feature cache (defaults to <out-dir>/cache)")
    parser.add_argument("--gold-root",
                        help="folder holding the gold charts (default: <dataset root>/gold). When set, "
                             "the gold set comes from this folder instead of the designer filter.")
    parser.add_argument("--scan", action="store_true",
                        help="only report what the gold folder contains, then exit")
    parser.add_argument("--designers", default="", help="comma or semicolon separated designer names")
    parser.add_argument("--song-ids", default="", help="comma or semicolon separated song ids")
    parser.add_argument("--val-ratio", type=float, default=0.3, help="share of gold songs held out")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-note-count", type=int, default=0,
                        help="skip gold candidates whose densest usable chart has fewer notes")
    parser.add_argument("--no-stats", action="store_true", help="skip gold_stats.json (no audio needed)")
    return parser


def _split_names(value: str) -> list[str]:
    return [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.scan:
        summary = scan_gold_folder(args.manifest, gold_root=args.gold_root,
                                   min_note_count=args.min_note_count)
        print(f"gold_root={summary['gold_root']} exists={summary['exists']}", flush=True)
        print(f"chart_folders={summary['found_directories']} usable_songs={summary['songs']} "
              f"usable_difficulties={summary['usable_difficulties']} skipped={summary['skipped']}", flush=True)
        for title in summary["titles"][:20]:
            print(f"  found: {title}", flush=True)
        for item in summary["skipped_examples"]:
            print(f"  skipped: {item['path']} ({item['reason']}: {item.get('message', '')})", flush=True)
        return 0
    summary = build_gold_set(
        args.manifest,
        out_dir=args.out_dir,
        cache_dir=args.cache_dir,
        gold_root=args.gold_root,
        designers=_split_names(args.designers),
        song_ids=_split_names(args.song_ids),
        val_ratio=args.val_ratio,
        seed=args.seed,
        min_note_count=args.min_note_count,
        write_stats=not args.no_stats,
    )
    selection = summary["selection"]
    gold_count = selection.get("gold_songs", selection.get("songs"))
    print(f"gold_mode={summary['mode']} gold_songs={gold_count} "
          f"finetune={summary['gold_finetune']['songs']} val={summary['gold_val']['songs']} "
          f"pretrain={summary['pretrain']['songs']} "
          f"excluded_from_pretrain={summary['excluded_from_pretrain']['songs']}", flush=True)
    print(f"pretrain_manifest={summary['pretrain']['path']}", flush=True)
    print(f"gold_finetune_manifest={summary['gold_finetune']['path']}", flush=True)
    print(f"gold_val_manifest={summary['gold_val']['path']}", flush=True)
    if summary.get("gold_stats"):
        print(f"gold_stats={summary['gold_stats']['path']} charts={summary['gold_stats']['charts']}", flush=True)
    print(f"gold_summary={summary['summary_path']}", flush=True)
    if not summary["pretrain"]["songs"]:
        print("warning: 预训练集为空，请检查 manifest 与筛选条件", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
