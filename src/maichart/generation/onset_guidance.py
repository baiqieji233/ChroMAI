"""Soft position guidance from learned chart-onset scores, on the exact chart clock."""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import torch
import torch.nn.functional as F

from maichart.event_timing import seconds_at_beat


class LocalOnsetGuidance:
    """Redistribute legal AT probability without changing character or EOS logits.

    Position-group mass is preserved before nucleus truncation. The subsequent
    top-p selection and changed event history can still change total density;
    existing rest and density masks always take precedence.
    """

    def __init__(self, token_ids, log_scores, strength: float):
        self.strength = float(strength)
        if not math.isfinite(self.strength) or not 0 <= self.strength <= 1:
            raise ValueError("local onset bias strength must be finite and in [0, 1]")
        self.token_ids = torch.as_tensor(token_ids, dtype=torch.long)
        self.log_scores = torch.as_tensor(log_scores, dtype=torch.float32)
        if self.token_ids.shape != self.log_scores.shape or self.token_ids.ndim != 1:
            raise ValueError("onset guidance requires one score per position token")
        if not torch.isfinite(self.log_scores).all():
            raise ValueError("onset guidance scores must be finite")
        self._buffers = {}

    @classmethod
    def from_predictions(cls, logits, frame_times, vocabulary, *, start_beat,
                         position_offset, tempo_changes, first: float, strength: float):
        predicted = logits.detach().float().reshape(-1).cpu()
        if not torch.isfinite(predicted).all():
            raise ValueError("onset head returned non-finite scores")
        times = np.asarray(frame_times, dtype=np.float32)
        if not len(times) or not predicted.numel() or predicted.numel() > len(times):
            raise ValueError("onset head and local audio time grid do not align")
        if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
            raise ValueError("onset frame times must be finite and ordered")
        if predicted.numel() != len(times):
            # Single-window inference pools precisely these real local frames.
            times = F.adaptive_avg_pool1d(
                torch.from_numpy(times).reshape(1, 1, -1), predicted.numel()
            ).reshape(-1).numpy()
        probability = predicted.sigmoid().numpy()
        clock_slack = max(1e-6, float(np.abs(np.spacing(times[[0, -1]])).max()))
        ids, scores = [], []
        for index, token in enumerate(vocabulary.tokens):
            if not token.startswith("AT:"):
                continue
            beat = Fraction(start_beat) + Fraction(token[3:]) - Fraction(position_offset)
            stamp = float(seconds_at_beat(beat, tempo_changes, first))
            # Outside the observed local grid we have no alignment evidence.
            # Leave those tokens untouched rather than inventing a negative.
            if stamp < float(times[0]) - clock_slack or stamp > float(times[-1]) + clock_slack:
                continue
            value = float(np.interp(stamp, times, probability))
            ids.append(index)
            scores.append(math.log(max(1e-4, min(1.0, value))))
        return cls(ids, scores, strength)

    def shape_logits(self, logits: torch.Tensor, *, state: str) -> torch.Tensor:
        if self.strength == 0 or state != "position" or self.token_ids.numel() == 0:
            return logits
        key = (logits.device, logits.dtype)
        if key not in self._buffers:
            self._buffers[key] = (self.token_ids.to(logits.device),
                                  self.log_scores.to(device=logits.device, dtype=logits.dtype))
        ids, scores = self._buffers[key]
        legal = torch.isfinite(logits[ids])
        active = ids[legal]
        evidence = scores[legal]
        if active.numel() <= 1:
            return logits
        bias = (evidence - evidence.mean()).clamp(-2.0, 2.0) * self.strength
        before = logits[active]
        after = before + bias
        after = after - (torch.logsumexp(after, 0) - torch.logsumexp(before, 0))
        result = logits.clone()
        result[active] = after
        return result
