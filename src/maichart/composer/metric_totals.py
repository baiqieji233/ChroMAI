"""Accumulate scalar diagnostics with one device vector addition per batch."""
from __future__ import annotations

import torch


class MetricTotals:
    """Keep accumulation order and rejected-window isolation without tiny kernels.

    Values stay on the training device until a report is requested. The first
    addition owns a clone, so committing a window never aliases its totals.
    """

    def __init__(self):
        self.names = ()
        self.values = None

    @torch.no_grad()
    def add(self, metrics):
        if isinstance(metrics, MetricTotals):
            if metrics.values is None:
                return
            names, values = metrics.names, metrics.values
        else:
            names = tuple(metrics)
            if not names:
                return
            values = torch.stack([metrics[name].detach().reshape(()) for name in names]).float()
        if self.values is None:
            self.names, self.values = names, values.clone()
        elif self.names == names:
            self.values.add_(values)
        else:
            # Optional diagnostics can appear/disappear. Preserve dictionary
            # addition semantics without transferring any values to the CPU.
            combined = dict(zip(self.names, self.values.unbind()))
            for name, value in zip(names, values.unbind()):
                combined[name] = combined[name] + value if name in combined else value
            self.names = tuple(combined)
            self.values = torch.stack(list(combined.values()))

    def as_floats(self):
        if self.values is None:
            return {}
        return dict(zip(self.names, self.values.cpu().tolist()))
