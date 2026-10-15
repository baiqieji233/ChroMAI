"""Causal timing features for cached generation, including prefix rewinds."""
from fractions import Fraction

import torch

from maichart.form.rhythm_geometry import CausalGeometry


class DecodeGeometry:
    def __init__(self, model, codec, origin=Fraction(0)):
        self.enabled = getattr(model.config, 'rhythm_geometry_profile', 'none') != 'none'
        self.codec, self.origin = codec, Fraction(origin)
        self.state = CausalGeometry(codec, origin=self.origin)
        self.consumed = 0
        self.last = None

    def kwargs(self, ids, index, device):
        if not self.enabled:
            return {}
        if index != self.consumed:
            # Canonical Fraction replacement, search backtracking and note
            # fallback all rewind the KV cache. Replay the retained *actual*
            # token prefix so the numeric state agrees with that same cache.
            self.state = CausalGeometry(self.codec, origin=self.origin)
            for token in ids[:index]:
                self.state.consume(token)
        self.last = self.state.consume(ids[index])
        self.consumed = index + 1
        return {'timing_features': torch.tensor([[self.last]], device=device, dtype=torch.float32)}

    def search_features(self, device):
        if not self.enabled:
            return None
        if self.last is None:
            raise ValueError('Position search needs a consumed causal prefix')
        # Speculative P: digits do not reveal their complete coordinate until
        # <note>; every candidate thus shares the last committed beat here.
        return torch.tensor([[self.last]], device=device, dtype=torch.float32)
