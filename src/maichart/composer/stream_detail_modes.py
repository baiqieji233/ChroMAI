"""Small mode-dependent normalized-output biases, with no recurrent mode axis."""
import torch
from torch import nn


class ModeDetailControl(nn.Module):
    def __init__(self,modes,width=32,rank=8,*,parameterization='centered_v2'):
        super().__init__()
        if parameterization not in ('legacy_v1','centered_v2'):raise ValueError('Invalid detail mode parameterization')
        self.parameterization=parameterization
        self.modes=int(modes)
        self.features=nn.Linear(width,rank,bias=False)
        self.choices=nn.Parameter(torch.zeros(modes,rank))
        self.stop=nn.Parameter(torch.zeros(modes,5))
    def bias(self,candidates,mode=None):
        choices=self.choices-self.choices.mean(0,keepdim=True) if self.parameterization=='centered_v2' else self.choices
        values=choices@self.features(candidates).T
        return values if mode is None else values[int(mode)]
    def stop_bias(self,type_id,mode=None):
        stop=self.stop-self.stop.mean(0,keepdim=True) if self.parameterization=='centered_v2' else self.stop
        return stop[:,type_id] if mode is None else stop[int(mode),type_id]
