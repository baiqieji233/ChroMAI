"""Repair compatibility facade; production repairs retain EventPoint provenance.

Use post_process.process_chart_points for actual exported chart repair.
"""
from __future__ import annotations


class ConflictRepairer:
    def __init__(self, repair_mode="none"):
        if repair_mode not in {"none", "local", "reject"}:
            raise ValueError("repair_mode must be none/local/reject")
        self.mode = repair_mode

    def repair(self, notes, conflicts):
        hard = [c for c in conflicts if c.severity == "hard"]
        if self.mode == "reject" and hard:
            raise ValueError("Rejected by static playability references")
        if self.mode != "local":
            return notes, []
        targets = {c.repair_target for c in hard if c.repair_target not in {None, (-1, -1)}}
        removed = [n for n in notes if n.source_id in targets]
        return [n for n in notes if n.source_id not in targets], [
            f"Removed {n.raw} at {n.time:.6f}s" for n in removed]
