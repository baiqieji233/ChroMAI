"""Read-only, CPU-friendly action capability metrics for Composer logits."""
from __future__ import annotations

import json
import math
from typing import Any

import torch
from torch.nn import functional as F


_KIND_BITS = {"tap": 1, "hold": 2, "slide": 4, "touch": 8, "touch_hold": 16}
_KINDS = ("STOP", "tap", "hold", "slide", "touch", "mixed")


def atom_kind_labels(actions: Any) -> tuple[str, ...]:
    """Return action-id labels using parser-backed CompositionalActions factors."""
    if not hasattr(actions, "atom_kind_mask") or "kind_mask" not in actions.factor_values:
        raise TypeError("expected CompositionalActions-like kind factor tables")
    values = actions.factor_values["kind_mask"]
    ids = actions.atom_kind_mask.detach().cpu().tolist()
    if not ids or int(ids[0]) != 0:
        raise ValueError("action id 0 must be STOP")
    labels = ["STOP"]
    for factor_id in ids[1:]:
        if int(factor_id) < 0 or int(factor_id) >= len(values):
            raise ValueError("atom_kind_mask contains an invalid factor id")
        try:
            mask = int(json.loads(values[int(factor_id)]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("kind_mask factor value is not an integer bitmask") from exc
        if mask <= 0 or mask & ~sum(_KIND_BITS.values()):
            raise ValueError(f"invalid kind mask: {mask}")
        kinds = {"touch" if name == "touch_hold" else name
                 for name, bit in _KIND_BITS.items() if mask & bit}
        if len(kinds) != 1:
            labels.append("mixed")
        else:
            labels.append(next(iter(kinds)))
    return tuple(labels)


def _finite(value: torch.Tensor) -> float | None:
    v = float(value)
    return v if math.isfinite(v) else None


def _batch_summary(logits: torch.Tensor, targets: torch.Tensor, labels: tuple[str, ...],
                   weights: torch.Tensor) -> dict[str, Any]:
    bsz, slots, vocab = logits.shape
    device = logits.device
    targets = targets.to(device=device, dtype=torch.long)
    weights = weights.to(device=device, dtype=torch.float64)
    active = torch.ones_like(targets, dtype=torch.bool)
    if slots > 1:
        active[:, 1:] = targets[:, :-1].ne(0)
    stop_factor = torch.where(targets.eq(0), .35, 1.).to(torch.float64)
    effective = stop_factor * active
    ce = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1), reduction="none").reshape(bsz, slots).double()
    row_den = effective.sum(1)
    row_loss = (ce * effective).sum(1) / row_den.clamp_min(1e-30)
    kind_id = {name: i for i, name in enumerate(_KINDS)}
    atom_kind = torch.tensor([kind_id[x] for x in labels], device=device)

    def base(mask: torch.Tensor, value: torch.Tensor, weight: torch.Tensor) -> dict[str, Any]:
        count = int(mask.sum())
        den = weight[mask].sum()
        return {"row_count": count, "weight_sum": _finite(den),
                "metric_weight_sum": _finite(den),
                "action_loss": (_finite((value[mask] * weight[mask]).sum() / den)
                                if count and float(den) > 0 else None)}

    def slot0_stats(mask: torch.Tensor, weight: torch.Tensor) -> dict[str, Any]:
        out = base(mask, row_loss, weight)
        count = int(mask.sum())
        den = weight[mask].sum()
        out['type_row_counts'] = {name: int(atom_kind[targets[mask, 0]].eq(kind_id[name]).sum())
                                  for name in _KINDS}
        if not count or float(den) <= 0:
            out.update({"exact_atom_ce": None, "exact_atom_top1": None,
                        "type_ce": None, "type_top1": None,
                        "type_probability_by_target": {k: None for k in _KINDS}})
            return out
        t = targets[mask, 0]
        l = logits[mask, 0].float()
        w = weight[mask]
        logp = F.log_softmax(l, -1)
        type_mass = []
        for name in _KINDS:
            ids = atom_kind.eq(kind_id[name])
            type_mass.append(torch.logsumexp(logp[:, ids], -1) if ids.any()
                             else torch.full((count,), -torch.inf, device=device))
        type_logp = torch.stack(type_mass, -1)
        true_kind = atom_kind[t]
        true_type_ce = -type_logp.gather(1, true_kind[:, None]).squeeze(1).double()
        out["exact_atom_ce"] = _finite((-logp.gather(1, t[:, None]).squeeze(1).double() * w).sum() / den)
        out["exact_atom_top1"] = _finite((l.argmax(-1).eq(t).double() * w).sum() / den)
        out["type_ce"] = _finite((true_type_ce * w).sum() / den)
        out["type_top1"] = _finite((type_logp.argmax(-1).eq(true_kind).double() * w).sum() / den)
        confusion = {}
        for true_name in _KINDS:
            rows = true_kind.eq(kind_id[true_name])
            cden = w[rows].sum()
            if not rows.any() or float(cden) <= 0:
                confusion[true_name] = None
            else:
                confusion[true_name] = {
                    "row_count": int(rows.sum()), "weight_sum": _finite(cden),
                    "probabilities": {pred: _finite((type_logp[rows].exp()[:, kind_id[pred]] * w[rows]).sum() / cden)
                                      for pred in _KINDS}}
        out["type_probability_by_target"] = confusion
        return out

    slot0 = active[:, 0] if slots else torch.zeros(bsz, dtype=torch.bool, device=device)
    first_stop = torch.zeros_like(active)
    if slots:
        first_stop = active & targets.eq(0)
        first_stop[:, 0] &= targets[:, 0].eq(0)
    later = active.clone()
    if slots:
        later[:, 0] = False
    later &= targets.ne(0)

    # Token-group loss/accuracy use effective token weights. row_count counts
    # unique rows; token_count and weight_sum describe the actual token sample.
    def token_group(mask: torch.Tensor) -> dict[str, Any]:
        row_mask = mask.any(1)
        tok_w = weights[:, None] * stop_factor * mask
        den = tok_w.sum()
        n_tok = int(mask.sum())
        out = {"row_count": int(row_mask.sum()), "token_count": n_tok,
               "weight_sum": _finite((weights[:, None] * mask).sum()),
               "effective_weight_sum": _finite(den)}
        if not n_tok or float(den) <= 0:
            out.update({"action_ce": None, "exact_atom_top1": None})
            return out
        correct = logits.argmax(-1).eq(targets)
        out["action_ce"] = _finite((ce * tok_w).sum() / den)
        out["exact_atom_top1"] = _finite((correct.double() * tok_w).sum() / den)
        return out

    slots_out = []
    for s in range(slots):
        m = active[:, s]
        ew = weights * effective[:, s]
        den = ew[m].sum()
        slots_out.append({"row_count": int(m.sum()), "weight_sum": _finite(weights[m].sum()),
                          "effective_weight_sum": _finite(den),
                          "action_ce": (_finite((ce[m, s] * ew[m]).sum() / den)
                                        if m.any() and float(den) > 0 else None),
                          "exact_atom_top1": (_finite((logits[m, s].argmax(-1).eq(targets[m, s]).double() * ew[m]).sum() / den)
                                              if m.any() and float(den) > 0 else None)})
    return {"slot0": slot0_stats(slot0, weights),
            "later_nonstop": token_group(later),
            "first_stop": token_group(first_stop),
            "all_rows": slot0_stats(torch.ones(bsz, dtype=torch.bool, device=device), weights),
            "slots": slots_out}


def _merge_summaries(parts: list[dict[str, Any]]) -> dict[str, Any]:
    if not parts:
        return {}
    result: dict[str, Any] = {}
    for group in ("slot0", "later_nonstop", "first_stop", "all_rows"):
        rows = sum(p[group]["row_count"] for p in parts)
        weights = [p[group].get("weight_sum") or 0. for p in parts]
        out = {"row_count": rows, "weight_sum": _finite(torch.tensor(sum(weights), dtype=torch.float64))}
        for metric in ("action_loss", "action_ce", "exact_atom_ce", "exact_atom_top1", "type_ce", "type_top1"):
            vals = []
            for p in parts:
                item = p[group]
                den = item.get("effective_weight_sum", item.get("metric_weight_sum", item.get("weight_sum")))
                val = item.get(metric)
                if den is not None and val is not None and den > 0:
                    vals.append((val, den))
            out[metric] = sum(v*d for v, d in vals) / sum(d for _, d in vals) if vals else None
        if group in ("slot0", "all_rows"):
            out['type_row_counts'] = {name: sum(p[group]['type_row_counts'][name] for p in parts)
                                      for name in _KINDS}
            confusion = {}
            for true_name in _KINDS:
                items = [p[group]["type_probability_by_target"].get(true_name) for p in parts]
                good = [x for x in items if x is not None and x["weight_sum"] > 0]
                den = sum(x["weight_sum"] for x in good)
                confusion[true_name] = (None if not good else {
                    "row_count": out['type_row_counts'][true_name], "weight_sum": den,
                    "probabilities": {pred: sum(x["probabilities"][pred] * x["weight_sum"] for x in good) / den
                                      for pred in _KINDS}})
            out["type_probability_by_target"] = confusion
        else:
            out["token_count"] = sum(p[group].get("token_count", 0) for p in parts)
            out["effective_weight_sum"] = sum(p[group].get("effective_weight_sum") or 0. for p in parts)
        result[group] = out
        if group == 'slot0':
            out['sequence_action_loss'] = out.pop('action_loss')
    max_slots = max(len(p["slots"]) for p in parts)
    slots_out = []
    for s in range(max_slots):
        items = [p["slots"][s] for p in parts if s < len(p["slots"])]
        den = sum(x["effective_weight_sum"] or 0. for x in items)
        slots_out.append({"row_count": sum(x["row_count"] for x in items),
                          "weight_sum": sum(x["weight_sum"] or 0. for x in items),
                          "effective_weight_sum": den or None,
                          "action_ce": (sum(x["action_ce"] * x["effective_weight_sum"] for x in items if x["action_ce"] is not None and x["effective_weight_sum"])
                                        / den if den else None),
                          "exact_atom_top1": (sum(x["exact_atom_top1"] * x["effective_weight_sum"] for x in items if x["exact_atom_top1"] is not None and x["effective_weight_sum"])
                                               / den if den else None)})
    result["slots"] = slots_out
    return result


class CapabilityMetrics:
    """Accumulate batched ``[B,S,V]`` logits without retaining model outputs."""

    def __init__(self, action_factors: Any):
        self.labels = atom_kind_labels(action_factors)
        self._parts: list[dict[str, Any]] = []

    def update(self, action_logits: torch.Tensor, batch: dict[str, Any]) -> None:
        targets = batch["action_target"]
        if action_logits.ndim != 3 or targets.ndim != 2 or action_logits.shape[:2] != targets.shape:
            raise ValueError("expected action_logits [B,S,V] and batch action_target [B,S]")
        if action_logits.shape[1] < 1:
            raise ValueError("action_logits must contain at least slot 0")
        if action_logits.shape[-1] != len(self.labels):
            raise ValueError("logit vocabulary does not match action factors")
        if not torch.isfinite(action_logits).all():
            raise ValueError("action_logits contain non-finite values")
        if targets.numel() and (targets.min() < 0 or targets.max() >= len(self.labels)):
            raise ValueError("action_target contains an invalid action id")
        if targets.dtype not in (torch.int32, torch.int64):
            raise ValueError("action_target must contain integer action ids")
        if targets.numel() and (targets.eq(0).cumsum(-1).gt(0) & targets.ne(0)).any():
            raise ValueError("action_target contains non-STOP actions after the first STOP")
        bsz = targets.shape[0]
        weights = torch.ones(bsz, dtype=torch.float64, device=action_logits.device)
        for key in ("sample_weight", "transition_weight", "action_weight"):
            if key in batch:
                value = torch.as_tensor(batch[key], dtype=torch.float64, device=action_logits.device).reshape(-1)
                if value.numel() != bsz or not torch.isfinite(value).all() or (value < 0).any():
                    raise ValueError(f"{key} must be finite, nonnegative, and one value per row")
                weights *= value
        if not torch.isfinite(weights).all():
            raise ValueError('combined action weights overflowed')
        self._parts.append(_batch_summary(action_logits.detach(), targets.detach(), self.labels, weights))

    def report(self) -> dict[str, Any]:
        """Return JSON-compatible metrics; empty groups have null metrics."""
        if not self._parts:
            empty = {"row_count": 0, "weight_sum": 0.0, "action_loss": None,
                     "exact_atom_ce": None, "exact_atom_top1": None,
                     "type_ce": None, "type_top1": None,
                     "type_probability_by_target": {name: None for name in _KINDS}}
            return {"slot0": dict(empty), "later_nonstop": {
                        "row_count": 0, "token_count": 0, "weight_sum": 0.0,
                        "effective_weight_sum": 0.0, "action_ce": None,
                        "exact_atom_top1": None},
                    "first_stop": {"row_count": 0, "token_count": 0,
                        "weight_sum": 0.0, "effective_weight_sum": 0.0,
                        "action_ce": None, "exact_atom_top1": None},
                    "all_rows": dict(empty), "slots": []}
        return _merge_summaries(self._parts)
