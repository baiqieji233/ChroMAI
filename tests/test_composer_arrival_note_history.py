"""v6 history contracts; execution is deferred to the coordinated test run."""
import torch

from maichart.composer.stream_model import StreamComposer, BOS, NOTE, WAIT


def test_note_clocks_ignore_wait_subdivision():
    original=StreamComposer._note_history_clock(torch.tensor([[BOS,NOTE,WAIT,NOTE]]),
        torch.tensor([[0.,.5,1.5,2.]]),torch.tensor([[0.,1.,3.,4.]]))
    split=StreamComposer._note_history_clock(torch.tensor([[BOS,NOTE,WAIT,WAIT,NOTE]]),
        torch.tensor([[0.,.5,1.,1.5,2.]]),torch.tensor([[0.,1.,2.,3.,4.]]))
    for field in ('ordinal','gap','elapsed','since_note','since_note_beats'):
        torch.testing.assert_close(original[field],split[field][:,[0,1,3,4]])
    assert original['gap'][0,-1]==3
    assert original['elapsed'][0,-1]==1.5


def test_note_clock_empty_memory_uses_origin_and_note_ordinal():
    result=StreamComposer._note_history_clock(torch.tensor([[WAIT,WAIT,NOTE,WAIT]]),
        torch.tensor([[1.,2.,3.,4.]]),torch.tensor([[2.,4.,6.,8.]]))
    assert result['last'].tolist()==[[-1,-1,2,2]]
    assert result['ordinal'].tolist()==[[0,0,1,1]]
    assert result['since_note'].tolist()==[[1.,2.,0.,1.]]
    assert result['gap'].tolist()==[[2.,4.,6.,2.]]
