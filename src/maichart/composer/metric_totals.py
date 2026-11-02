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
        self._indices = {}

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
            # Names keep their first-seen order. One packed indexed addition
            # handles optional/recovery subsets without a kernel per scalar.
            index=self._indices.get(names)
            if index is None:
                positions={name:i for i,name in enumerate(self.names)}
                added=tuple(name for name in names if name not in positions)
                if added:
                    offset=len(self.names)
                    positions.update((name,offset+i) for i,name in enumerate(added))
                    self.names=self.names+added
                    self.values=torch.cat((self.values,self.values.new_zeros(len(added))))
                index=torch.tensor([positions[name] for name in names],device=self.values.device,dtype=torch.long)
                self._indices[names]=index
            self.values.index_add_(0,index,values)

    def as_floats(self):
        if self.values is None:
            return {}
        return dict(zip(self.names, self.values.cpu().tolist()))
