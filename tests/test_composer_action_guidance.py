from types import SimpleNamespace
import pytest
import torch

from maichart.composer.action_guidance import constrain_touch_history,action_logits_with_touch_guard


def test_touch_history_cap_keeps_stop_and_group_conditionals():
    full=torch.tensor([2.,0.,1.,8.,7.])
    audio=torch.tensor([3.,2.,1.,-1.,-2.])
    touch=torch.tensor([False,False,False,True,True])
    result=constrain_touch_history(full,audio,touch)
    p=full.softmax(0);q=result.softmax(0)
    torch.testing.assert_close(q[0],p[0])
    torch.testing.assert_close(result[3]-result[4],full[3]-full[4])
    torch.testing.assert_close(result[1]-result[2],full[1]-full[2])
    expected=torch.logsumexp(audio[3:],0)-torch.logsumexp(audio[1:3],0)+1.
    actual=torch.logsumexp(result[3:],0)-torch.logsumexp(result[1:3],0)
    torch.testing.assert_close(actual,expected)
    assert q[3:].sum()<p[3:].sum()


@pytest.mark.parametrize('touch_logit',[-20.,-1.,0.])
def test_lower_or_already_bounded_touch_logits_are_exactly_unchanged(touch_logit):
    full=torch.tensor([1.,0.,touch_logit]);audio=torch.tensor([1.,0.,0.])
    result=constrain_touch_history(full,audio,torch.tensor([False,False,True]))
    assert torch.equal(full,result)


@pytest.mark.parametrize('touch',[[False,False,False],[False,True,True]])
def test_absent_group_keeps_original_logits(touch):
    full=torch.tensor([1.,0.,3.]);audio=full.clone()
    assert torch.equal(constrain_touch_history(full,audio,torch.tensor(touch)),full)


@pytest.mark.parametrize('full,audio',[
    ([0.,float('-inf'),2.],[0.,float('-inf'),1.]),
    ([0.,1.,float('-inf')],[0.,1.,2.]),
    ([0.,1.,2.],[0.,1.,float('-inf')]),
    ([1000.,-1000.,1000.],[1000.,1000.,-1000.]),
])
def test_degenerate_and_extreme_logits_never_introduce_nan(full,audio):
    full=torch.tensor(full);audio=torch.tensor(audio)
    result=constrain_touch_history(full,audio,torch.tensor([False,False,True]))
    assert not result.isnan().any()
    assert result[0]==full[0]


@pytest.mark.parametrize('bound',[-1.,float('nan'),float('inf')])
def test_invalid_bounds_rejected(bound):
    with pytest.raises(ValueError,match='bound'):
        constrain_touch_history(torch.zeros(3),torch.zeros(3),torch.tensor([False,False,True]),log_odds_bound=bound)


def test_zero_bound_uses_audio_touch_odds_without_changing_detail_or_stop():
    full=torch.tensor([.5,0.,5.]);audio=torch.tensor([9.,1.,-2.])
    result=constrain_touch_history(full,audio,torch.tensor([False,False,True]),log_odds_bound=0.)
    torch.testing.assert_close(result[2]-result[1],audio[2]-audio[1])
    torch.testing.assert_close(result.softmax(0)[0],full.softmax(0)[0])


def test_decoder_uses_same_candidate_and_same_prefix_and_can_disable_guard():
    calls=[]
    full=torch.tensor([0.,0.,8.]);audio=torch.tensor([0.,2.,-2.])
    def action_slot(context,slot,selected_ids,**kwargs):
        calls.append((context,slot,list(selected_ids),kwargs))
        return full if context=='full' else audio
    model=SimpleNamespace(action_slot=action_slot);mask=torch.tensor([False,False,True])
    diagnostics={}
    result=action_logits_with_touch_guard(model,'full',1,[2],audio_context='audio',enabled=True,touch_mask=mask,diagnostics=diagnostics)
    assert calls[0][:3]==('full',1,[2]) and calls[1][:3]==('audio',1,[2])
    assert result[2]<full[2]
    assert diagnostics['applied']
    calls.clear()
    assert action_logits_with_touch_guard(model,'full',0,[],audio_context='audio',enabled=False) is full
    assert len(calls)==1


def test_guidance_reduces_runaway_touch_without_overriding_physical_masks():
    from maichart.composer.generate import _sample_actions,_head_metadata
    from maichart.generation.event_cli import _NoteGrammar
    atoms=['1','E1'];grammar=_NoteGrammar.from_atoms(atoms);metadata=_head_metadata(atoms,grammar,touch_layout_policy='open')
    full=torch.tensor([0.,0.,8.]);audio=torch.tensor([0.,3.,-3.])
    corrected=constrain_touch_history(full,audio,torch.tensor([False,False,True]))
    rng=torch.Generator().manual_seed(717)
    samples=[_sample_actions(corrected[None],atoms,grammar,.9,rng,head_metadata=metadata)[0] for _ in range(200)]
    assert samples.count('E1')<10 and samples.count('1')>190
    assert _sample_actions(corrected[None],atoms,grammar,.9,rng,blocked={'1'},head_metadata=metadata)==['E1']


def test_cli_allows_replaying_previous_generation_policy():
    from maichart.composer.generate import parser
    required=['--audio','unused','--checkpoint','unused','--output','unused']
    assert parser().parse_args(required).touch_history_guard
    assert not parser().parse_args(required+['--no-touch-history-guard']).touch_history_guard
