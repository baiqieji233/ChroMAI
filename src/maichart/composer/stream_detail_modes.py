"""Small mode-dependent normalized-output biases, with no recurrent mode axis."""
import torch
from torch import nn


class ModeDetailControl(nn.Module):
    def __init__(self,modes,width=32,rank=8):
        super().__init__()
        self.modes=int(modes)
        self.features=nn.Linear(width,rank,bias=False)
        self.choices=nn.Parameter(torch.zeros(modes,rank))
        self.stop=nn.Parameter(torch.zeros(modes,5))
    def bias(self,candidates,mode=None):
        values=self.choices@self.features(candidates).T
        return values if mode is None else values[int(mode)]
    def stop_bias(self,type_id,mode=None):
        return self.stop[:,type_id] if mode is None else self.stop[int(mode),type_id]
