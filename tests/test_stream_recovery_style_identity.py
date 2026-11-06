"""A recovery rollout's chosen song component also supplies its actual controls."""
from copy import deepcopy
from types import SimpleNamespace

import torch
from torch import nn

from maichart.composer.stream_phrase_runtime import (
    attach_draft_runtime,ensure_song_response,freeze_recovery_song_style,
)
from maichart.composer.stream_song_profile import SongProfileCompleter


class RecoveryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(draft_policy='beat_ordered_draft_v1',
            song_response_policy='unified_profile_response_v2',plan_kind_policy='unified_first_arrival_v5')
        self.song_profile=SongProfileCompleter(4,components=2)
        self.profile_reads=self.plan_reads=self.forward_reads=0
        final=self.song_profile.network[-1]
        with torch.no_grad():
            final.weight.zero_();final.bias.zero_()
            bias=final.bias.view(2,-1);bias[1,0]=-40
            bins=bias[:,1:].view(2,15,self.song_profile.bins)
            bins.fill_(-15);bins[0,:,32]=15;bins[1,:,96]=15
        self.eval()

    def prepare_plan_cache(self,cache,condition,inverse,end):
        self.plan_reads+=1
        return {**cache,'conditioned_plan':{'condition':condition.clone()}}

    def forward(self,batch,audio_cache):
        self.forward_reads+=1
        frozen=audio_cache['unified_style']
        assert torch.equal(batch['condition'],frozen['execution_controls'])
        return {'song_response_logits':frozen['prior_logits'],'selected_component':frozen['choice']}


def inputs(choice=None):
    source=torch.full((16,),.95);source[0]=.8
    context=SimpleNamespace(condition=source.clone(),song_response_choice=choice,_committed=[])
    batch=dict(condition=source[None].clone(),song_inverse=torch.tensor([0]),audio_end_sec=torch.tensor([10.]),
        phrase_start_beat=torch.tensor([[0.]]),phrase_end_beat=torch.tensor([[8.]]))
    cache={'global':torch.ones(1,3,4),'global_mask':torch.ones(1,3,dtype=torch.bool),
        'conditioned_plan':{'condition':source[None].clone()}}
    return context,batch,cache


def test_style_is_frozen_before_first_forward_and_reused_by_phrase_prior():
    model=RecoveryModel();context,batch,shared=inputs(choice=1);generator=torch.Generator().manual_seed(7)
    prepared,local=freeze_recovery_song_style(model,context,batch,shared,generator)
    assert model.forward_reads==0
    assert 'unified_style' not in shared
    assert local is not shared and local['global'] is shared['global']
    frozen=local['unified_style'];record=context._recovery_frozen_style
    assert frozen['choice']==record['choice']==1
    assert frozen['prior_logits'].argmax(-1).item()==0
    assert record['known_mask'].tolist()==[True]+[False]*15
    assert torch.equal(record['execution_controls'],context.condition)
    assert torch.equal(prepared['condition'],local['conditioned_plan']['condition'])
    assert not torch.equal(prepared['condition'],batch['condition'])
    result=model(prepared,audio_cache=local)
    assert ensure_song_response(model,context,result,generator)==result['selected_component']==1
    packed=attach_draft_runtime(context,prepared,model)
    first_key=packed['_draft_runtime_key']
    before=generator.get_state().clone()
    again,cached=freeze_recovery_song_style(model,context,prepared,shared,generator)
    assert cached is local and torch.equal(again['condition'],prepared['condition'])
    assert model.plan_reads==1 and torch.equal(before,generator.get_state())
    assert attach_draft_runtime(context,again,model)['_draft_runtime_key']==first_key


def test_song_shared_audio_does_not_share_episode_choice_or_completed_controls():
    model=RecoveryModel();first,batch,shared=inputs(choice=0);second,_,_=inputs(choice=1)
    # A previously frozen foreign episode in a caller's cache is never adopted.
    foreign={'choice':99};shared['unified_style']=foreign
    a,cache_a=freeze_recovery_song_style(model,first,batch,shared,torch.Generator())
    b,cache_b=freeze_recovery_song_style(model,second,batch,shared,torch.Generator())
    assert cache_a['unified_style']['choice']==0 and cache_b['unified_style']['choice']==1
    assert not torch.equal(a['condition'],b['condition'])
    assert shared['unified_style'] is foreign
    assert torch.equal(shared['conditioned_plan']['condition'],batch['condition'])
    assert cache_a['global'] is cache_b['global'] is shared['global']
    assert cache_a['conditioned_plan'] is not cache_b['conditioned_plan']


def test_saved_episode_keeps_written_control_identity_after_weight_update():
    model=RecoveryModel();context,batch,shared=inputs(choice=1)
    _,local=freeze_recovery_song_style(model,context,batch,shared,torch.Generator())
    saved=deepcopy(context._recovery_frozen_style)
    with torch.no_grad():
        model.song_profile.network[-1].bias.add_(.7)
        model.song_profile.network[-1].bias.view(2,-1)[1,0]=20
    same_batch,refreshed=freeze_recovery_song_style(model,context,batch,shared,torch.Generator())
    assert refreshed is not local
    assert refreshed['unified_style']['prior_logits'].argmax(-1).item()==1
    assert torch.equal(same_batch['condition'][0],saved['execution_controls'])
    restored,_,_=inputs();restored._recovery_frozen_style=deepcopy(saved)
    prepared,current=freeze_recovery_song_style(model,restored,batch,shared,torch.Generator())
    assert current is not local
    assert current['unified_style']['prior_logits'].argmax(-1).item()==1
    assert restored.song_response_choice==saved['choice']==1
    assert torch.equal(prepared['condition'][0],saved['execution_controls'])
    assert torch.equal(restored._recovery_frozen_style['known_mask'],saved['known_mask'])


def test_reference_unknown_controls_do_not_enter_recovery_prior_or_completion():
    model=RecoveryModel();a,batch,shared=inputs(choice=0);b,other,_=inputs(choice=0)
    b.condition[1:]=.01;other['condition']=b.condition[None].clone()
    first,cache_a=freeze_recovery_song_style(model,a,batch,shared,torch.Generator())
    second,cache_b=freeze_recovery_song_style(model,b,other,shared,torch.Generator())
    assert torch.equal(first['condition'],second['condition'])
    assert torch.equal(cache_a['unified_style']['prior_logits'],cache_b['unified_style']['prior_logits'])
