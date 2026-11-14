"""WebUI review history, kept outside the training annotation contract."""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from maichart.training.quality_annotations import annotations_fingerprint


_SCHEMA = """
CREATE TABLE IF NOT EXISTS chart_decisions (
    source TEXT PRIMARY KEY, signature TEXT NOT NULL, kind TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS prompt_decisions (
    source TEXT NOT NULL, prompt_id TEXT NOT NULL, reason TEXT NOT NULL,
    start_sec REAL NOT NULL, end_sec REAL NOT NULL, decision TEXT NOT NULL,
    reviewed_at TEXT NOT NULL, audio_sha256 TEXT,
    PRIMARY KEY (source, prompt_id)
);
"""


def source_key(manifest, row):
    # Catalog paths are already resolved. Keep the location/slot stable across
    # source edits and independent of the combined manifest's chart namespace.
    return json.dumps([os.path.normcase(str(manifest)),
                       os.path.normcase(row.source_path), row.difficulty_index], ensure_ascii=False)


def decision_signature(record):
    return annotations_fingerprint({key: record.get(key) for key in (
        "chart_sha256", "audio_sha256", "label", "weight", "reason", "chart_override", "corrected_path", "corrected_sha256",
    )})


class ReviewStore:
    def __init__(self, workspace):
        self.path = Path(workspace).absolute() / "materials_review.sqlite3"

    def _write(self, statement, values):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=5)) as db, db:
            db.executescript(_SCHEMA)
            db.executemany(statement, values)

    def read(self):
        charts, prompts = {}, {}
        if not self.path.is_file():
            return charts, prompts
        with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=5)) as db:
            db.row_factory = sqlite3.Row
            # Both SELECTs see the same version even if another WebUI writes.
            db.execute("BEGIN")
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "chart_decisions" in tables:
                for row in db.execute("SELECT * FROM chart_decisions"):
                    charts[row["source"]] = (row["signature"], row["kind"])
            if "prompt_decisions" in tables:
                for row in db.execute("SELECT * FROM prompt_decisions"):
                    prompts.setdefault(row["source"], {})[row["prompt_id"]] = dict(row)
        return charts, prompts

    def save_charts(self, entries):
        self._write("INSERT OR REPLACE INTO chart_decisions VALUES (?, ?, ?)", entries)

    def save_prompt(self, source, prompt, decision, audio_sha256):
        self._write("INSERT OR REPLACE INTO prompt_decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [(
            source, prompt["prompt_id"], prompt["reason"], prompt["start_sec"], prompt["end_sec"],
            decision, datetime.now(timezone.utc).isoformat(), audio_sha256,
        )])


def chart_provenance(source, record, saved):
    if record is None or record.get("chart_override") is False or record.get("unmarked_weight") == 0:
        return None
    event = saved.get(source)
    if event is not None and event[0] == decision_signature(record):
        return event[1]
    if (record["label"], record["weight"]) != ("normal", 1.0) or record.get("chart_override") is True:
        return "legacy"
    return None


def report_version(report):
    return annotations_fingerprint({key: report.get(key) for key in (
        "manifest", "manifest_sha256", "audit_id", "preview", "settings", "scope", "review_regions",
    )})


def review_prompts(row, source, spans, settings, audio_version, saved, annotation, *, audio_sha256=None):
    previous = saved.get(source, {})
    prompts = {}
    rules = {key: value for key, value in settings.items() if key not in {
        "audit_id", "manifest_sha256", "comfort_cutoff", "comfort_cutoffs_by_tier", "good_cutoffs_by_tier",
    }}
    for span in spans:
        reason = str(span.get("reason") or "")
        start, end = float(span["start_sec"]), float(span["end_sec"])
        prompt_id = annotations_fingerprint({
            "source": source, "chart": row.chart_sha256, "audio_path": os.path.normcase(row.audio_path) if row.audio_path else None,
            "audio_version": audio_version, "rules": rules, "diagnostic": span,
        })
        prior = previous.get(prompt_id)
        if (prior is not None and prior.get("audio_sha256")
                and (audio_sha256 is None or prior["audio_sha256"] == audio_sha256)):
            decision = prior["decision"]
        elif any(item["reason"] == reason and start < item["end_sec"] and item["start_sec"] < end
                 and item["decision"] in {"kept", "annotated"} for item in previous.values()):
            decision = "needs_recheck"
        else:
            decision = "pending"
        overlaps = any(start < region["end_sec"] and region["start_sec"] < end
                       for region in (annotation or {}).get("regions", ()))
        prompts[prompt_id] = {
            **span, "reason": reason, "start_sec": start, "end_sec": end, "prompt_id": prompt_id,
            "review_status": decision, "reviewed": decision in {"kept", "annotated"},
            "overlaps_annotation": overlaps,
        }
    return list(prompts.values())
