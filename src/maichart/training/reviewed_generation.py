"""Export explicitly reviewed regions as an ordinary, masked training source.

The complete chart is retained for context. Only approved regions receive
positive supervision; comparisons and whole-chart praise are not approvals.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from maichart.maidata import parse_maidata_file
from maichart.training.quality_annotations import chart_sha256, audio_sha256, region_component_weights


def approved_regions(records):
    # Partition overlapping reviews once at export, not at every training step.
    rows = [row for row in records if row.get("scope") == "region"]
    bounds = sorted({float(row[key]) for row in rows for key in ("start_sec", "end_sec")})
    result = []
    for start, end in zip(bounds, bounds[1:]):
        timing = components = 0.
        entry_start = None
        for row in rows:
            if float(row["start_sec"]) > start or float(row["end_sec"]) < end:
                continue
            label, reason = row["label"], row.get("reason", "")
            if label == "good":
                use = row.get("training_use")
                if use == "none":
                    timing = components = 0.; entry_start = None
                elif use in {"both", "timing", "layout"}:
                    # A new explicit choice replaces the previous scope.
                    timing = float(use != "layout")
                    components = float(use != "timing")
                    entry_start = (float(row.get('entry_start_sec', row['start_sec']))
                        if timing and (row.get('learn_entry') is True or 'entry_start_sec' in row) else None)
            elif label == "uncertain":
                timing = components = 0.; entry_start = None
            elif label == "problem":
                if reason not in {"layout", "comfort"}: timing = 0.; entry_start = None
                if reason != "timing": components = 0.
        if not (timing or components): continue
        region = dict(start_sec=start, end_sec=end, label="good", reason="", weight=1.,
                      timing_weight=timing, component_weight=components)
        if entry_start is not None:
            # Keep the ORIGINAL onset boundary after later reviews split this
            # interval. Training resolves its unique first event, not the first
            # event of every remaining piece.
            region['entry_start_sec'] = entry_start
        if result and result[-1]["end_sec"] == start and region_component_weights(result[-1]) == (timing, components) and result[-1].get('entry_start_sec') == entry_start:
            result[-1]["end_sec"] = end
        else:
            result.append(region)
    return result


def export_reviewed(candidates, output_root, *, inspect_only=False, allow_empty=False):
    # Reuse the same file/content checks as saving a review. No GPU/audio decode.
    from maichart.webui.quality_api import _feedback_chart, _feedback_audio, _feedback_span
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("请先载入完整谱面")
    charts, ignored, seen, reviews = [], [], set(), {}
    for candidate in candidates:
        identity = _feedback_chart(candidate)
        key = (identity["path"], identity["difficulty_index"])
        if key in seen: continue
        seen.add(key)
        source = Path(identity["path"])
        sidecar = source.with_name("generation_feedback.json")
        if not sidecar.is_file(): continue
        feedback = json.loads(sidecar.read_text(encoding="utf-8"))
        if feedback.get("schema") != "maichart-generation-feedback-v1":
            raise ValueError("反馈文件格式不兼容")
        records = []
        for row in feedback.get("records", []):
            if row.get("path") != identity["path"] or row.get("difficulty_index") != identity["difficulty_index"]:
                continue
            if row.get("chart_sha256") != identity["chart_sha256"] or row.get("file_sha256") != identity["file_sha256"]:
                ignored.append("谱面已变化的旧反馈"); continue
            if 'training_use' not in row:
                span=_feedback_span(row)
                review_key=(identity['path'],identity['difficulty_index'],span.get('start_sec'),span.get('end_sec'))
                reviews[review_key]=dict(**identity,**span,label=row['label'],reason=row.get('reason',''),title=candidate.get('title',''))
            if row.get("scope") != "region":
                ignored.append("整谱评价（未逐段确认）"); continue
            span = _feedback_span(row)
            if row.get("label") not in {"good", "problem", "uncertain"} or not span:
                raise ValueError("片段反馈内容无效，请重新保存")
            records.append(row)
        regions = approved_regions(records)
        if not regions: continue
        audio = _feedback_audio(candidate)
        if not audio: raise ValueError("校准素材需要原任务对应的音频")
        if any(row.get('training_use') in {'both','timing','layout'} and row.get("audio_sha256") != audio["audio_sha256"] for row in records):
            raise ValueError("已标记片段的音频已变化，请重新试听确认")
        parsed = parse_maidata_file(source)
        slot = identity["difficulty_index"]
        difficulty = next(item for item in parsed.difficulties if item.index == slot)
        if not difficulty.level: raise ValueError("校准谱面缺少等级，请先补齐等级")
        # An approved empty range cannot supply an event target on its own.
        # It remains in the sidecar; honest rests between approved events stay.
        from maichart.event_timing import parse_event_chart
        event_chart = parse_event_chart(source, difficulty_index=slot)
        from maichart.tempo_clock import TempoClock
        clock = TempoClock(event_chart.tempo_changes, event_chart.audio_first)
        seconds = clock.at_floats([float(point.beat) for point in event_chart.points])
        if not any(r["start_sec"] <= second < r["end_sec"] for r in regions for second in seconds):
            ignored.append("仅含空白、没有已确认音符的范围"); continue
        charts.append(dict(identity=identity, audio=audio, regions=regions,
                           title=parsed.title or source.parent.name, level=difficulty.level))
    summary = dict(charts=len(charts), regions=sum(len(chart["regions"]) for chart in charts),
                   items=[dict(**chart["identity"], **chart["audio"], title=chart["title"], **region)
                          for chart in charts for region in chart["regions"]], ignored=ignored)
    summary['reviews']=list(reversed(reviews.values()))
    fingerprint = hashlib.sha256(json.dumps(charts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]
    summary['collection_id']=fingerprint
    if inspect_only: return summary
    if not charts:
        if allow_empty:return dict(**summary,manifest=None)
        raise ValueError("尚未收集训练示范。请圈选有音符的片段，选择学习内容，再点“收为训练示范”。普通好评、整谱好评或A/B胜出不会直接进入训练。")
    root = Path(output_root).resolve() / fingerprint
    manifest = root / "training_manifest.json"
    root.mkdir(parents=True, exist_ok=True)
    songs, annotations = [], []
    for number, chart in enumerate(charts):
        identity, audio = chart["identity"], chart["audio"]
        destination = root / f"chart_{number:03d}" / "maidata.txt"
        destination.parent.mkdir(exist_ok=True)
        # Freeze the chart bytes so a later generation cannot silently replace it.
        if not destination.exists(): shutil.copyfile(identity["path"], destination)
        if audio_sha256(destination) != identity["file_sha256"]:
            raise ValueError("校准素材快照校验失败，请保留当前目录并重新整理")
        song_id = f"reviewed_{fingerprint}_{number}"
        slot = identity["difficulty_index"]
        songs.append(dict(song_id=song_id, title=chart["title"], maidata_path=str(destination),
                          audio_path=audio["audio_path"], reviewed_only=True,
                          difficulties=[dict(difficulty_index=slot, level_raw=chart["level"],
                                             event_usable_for_training=True, usable_for_training=True)]))
        annotations.append(dict(song_id=song_id, difficulty_index=slot, source_path=str(destination),
                                **audio, chart_sha256=chart_sha256(destination, slot),
                                label="normal", weight=1., unmarked_weight=0., regions=chart["regions"]))
    payload = dict(schema="maichart-training-manifest-v1", source_root=str(root), songs=songs,
                   preprocessing={"target_schema": "event_v1"},
                   reviewed_training=dict(policy="approved_regions_only_v1",
                                          unmarked="context_only", whole_chart_reviews="ignored"))
    # Publish annotations first; an incomplete export is never a usable manifest.
    from maichart.training.quality_annotations import save_annotations
    save_annotations(manifest, annotations)
    from maichart.training.dataset_catalog import _write_json
    _write_json(manifest,payload)
    return dict(**summary, manifest=str(manifest), name="人工确认的生成片段",
                policy="approved_regions_only_v1")


def library_state(root, task_id, collection):
    path=Path(root)/'library.json'
    data=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    saved=data.get('tasks',{}).get(task_id)
    return dict(published=bool(saved),needs_update=(saved or {}).get('collection_id')!=collection['collection_id'])


def update_library(collection, task_id, root):
    """Publish one logical source; previous snapshots stay immutable."""
    from maichart.training.quality_annotations import _lock_for, save_annotations
    from maichart.training.dataset_catalog import merge_training_manifests, _write_json
    if not isinstance(task_id,str) or not task_id or len(task_id)>256:
        raise ValueError('示范库缺少有效的生成任务标识')
    root=Path(root).resolve();index=root/'library.json';root.mkdir(parents=True,exist_ok=True)
    with _lock_for(index):
        tasks=json.loads(index.read_text(encoding='utf-8')).get('tasks',{}) if index.exists() else {}
        if collection.get('manifest'):
            tasks[task_id]=dict(manifest=collection['manifest'],collection_id=collection['collection_id'],
                               regions=collection['regions'],charts=collection['charts'])
        else:tasks.pop(task_id,None)
        digest=hashlib.sha256(json.dumps(tasks,sort_keys=True).encode()).hexdigest()[:20]
        manifest=root/'library'/digest/'training_manifest.json'
        regions=sum(row['regions'] for row in tasks.values())
        if not manifest.exists():
            if tasks:
                merged=merge_training_manifests([row['manifest'] for _,row in sorted(tasks.items())],manifest,model_family='chart_composer_v2')
            else:
                merged=dict(schema='maichart-training-manifest-v1',songs=[],source_root=str(manifest.parent),preprocessing={'target_schema':'event_v1'})
                save_annotations(manifest,[])
            merged['reviewed_training']=dict(policy='approved_regions_only_v1',regions=regions)
            _write_json(manifest,merged)
        _write_json(index,dict(tasks=tasks,manifest=str(manifest)))
    return dict(manifest=str(manifest),name='人工确认示范库',review_group='composer-v2-reviewed-library',
                regions=regions,charts=sum(row['charts'] for row in tasks.values()),
                task_regions=collection['regions'],policy='approved_regions_only_v1')
