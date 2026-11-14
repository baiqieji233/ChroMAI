"""Resolve the quality sidecars and merge rules used by Composer v2."""
from __future__ import annotations

from functools import cache
from pathlib import Path

from maichart.portable_paths import material_path
from maichart.training.quality_annotations import (
    annotation_path, annotations_fingerprint, chart_sha256, load_quality_annotations,
    merge_annotation_records, region_component_weights, resolve_auto_annotations,
    split_annotation_source, zero_unmarked_region_weights,
)


class TrainingQuality:
    def __init__(self, manifest, payload, *, enabled, use_auto, audio_digest=None, resolve_path=None,
                 chart_digest=chart_sha256):
        from maichart.form.data import _preparation_annotation_index

        self.manual_path = annotation_path(manifest)
        if not self.manual_path.exists():
            origin = split_annotation_source(Path(manifest), payload)
            if origin is not None:
                self.manual_path = annotation_path(origin)
        self.manual = load_quality_annotations(self.manual_path) if enabled and self.manual_path.is_file() else None
        self.auto_path, self.auto = resolve_auto_annotations(manifest, payload) if use_auto else (None, None)
        self.auto_fingerprint = annotations_fingerprint(self.auto) if self.auto is not None else None
        self.resolve_path = resolve_path or cache(lambda value: material_path(value).resolve())
        self.manual_index = _preparation_annotation_index(self.manual, resolve_path=self.resolve_path)
        self.auto_index = _preparation_annotation_index(self.auto, resolve_path=self.resolve_path)
        self.audio_digest = audio_digest
        self.chart_digest = cache(chart_digest)

    def records_for(self, source, slot, audio):
        from maichart.form.data import _preparation_annotation

        key = slot, str(source)
        manual = (_preparation_annotation(self.manual, source, slot, audio, None, self.manual_index,
                                          audio_digest=self.audio_digest, resolve_path=self.resolve_path,
                                          chart_digest=self.chart_digest)
                  if key in self.manual_index else None)
        auto = (_preparation_annotation(self.auto, source, slot, audio, None, self.auto_index,
                                        audio_digest=self.audio_digest, resolve_path=self.resolve_path,
                                        chart_digest=self.chart_digest)
                if key in self.auto_index else None)
        return manual, auto

    def record_for(self, source, slot, audio):
        manual, auto = self.records_for(source, slot, audio)
        return merge_annotation_records(manual, auto) if auto is not None else manual


def quality_excludes_chart(record):
    return bool(record and (record.get("label") in ("excluded", "negative") or float(record.get("weight", 1.)) <= 0.))


def stream_region_weight_fn(unmarked, legacy_zero_unmarked=False):
    return zero_unmarked_region_weights if unmarked == 0 and not legacy_zero_unmarked else region_component_weights
