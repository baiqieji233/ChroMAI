import numpy as np
import pytest
import torch
from maichart.composer.stream_song_profile import SongProfileCompleter, feasible_profile, pool_audio


def test_profile_type_locks_include_overlap_and_support_endpoints():
    values=np.full(16,.3);known=np.zeros(16,dtype=bool)
    values[8:12]=[0,0,1,1];known[8:12]=True
    result=feasible_profile(values,known)
    assert np.allclose(result[8:12],[0,0,1,1])
    assert result[7]==result[12]


def test_profile_conflicting_type_and_chord_locks_rejected():
    values=np.full(16,.3);known=np.zeros(16,dtype=bool)
    values[7]=0;values[12]=1;known[[7,12]]=True
    with pytest.raises(ValueError,match='chord'):feasible_profile(values,known)
    known[:]=False;known[[8,9]]=True;values[8:10]=1
    with pytest.raises(ValueError,match='note-type'):feasible_profile(values,known)


def test_profile_sampling_bounded_reproducible_and_locked():
    torch.manual_seed(4);module=SongProfileCompleter(8,4)
    audio=torch.randn(3,8);condition=torch.full((3,16),.4)
    known=torch.zeros_like(condition,dtype=torch.bool);known[:,[0,8]]=True;condition[:,8]=0
    a=module.sample(audio,condition,known,torch.Generator().manual_seed(9))
    b=module.sample(audio,condition,known,torch.Generator().manual_seed(9))
    assert torch.equal(a,b)
    assert torch.equal(a[:,8],condition[:,8])
    assert torch.equal(a[:,7],a[:,12])
    assert bool(((a>=0)&(a<=1)).all())
    # hold and touch overlap is a valid touch-hold probability.
    overlap=a[:,8:12].sum(-1)-1
    assert bool((overlap>=-1e-6).all())
    assert bool((overlap<=torch.minimum(a[:,10],a[:,11])+1e-6).all())


def test_profile_distribution_loss_and_masked_pool():
    module=SongProfileCompleter(8,3)
    cache={'global':torch.randn(2,4,8),'global_mask':torch.tensor([[True,True,False,False],[True,False,False,False]])}
    assert torch.allclose(pool_audio(cache)[1],cache['global'][1,0])
    conditions=torch.rand(2,16);mask=torch.zeros_like(conditions,dtype=torch.bool);mask[:,0]=True
    loss=module.loss(cache,conditions,mask,reduction='none')
    assert loss.shape==(2,) and bool(torch.isfinite(loss).all())
    loss.mean().backward()
    assert module.network[-1].weight.grad is not None


def test_phrase_runtime_uses_frozen_song_choice():
    from types import SimpleNamespace
    from maichart.composer.stream_phrase_runtime import execution_options
    seen=[]
    class Model:
        config=SimpleNamespace(phrase_execution_policy='persistent_modes_v1',phrase_modes=2)
        def execution_prior(self,output,batch,*,song_choice=None):
            seen.append(song_choice)
            return torch.tensor([[[0.,1.]]])
        def execution_proposal(self,action,output,batch,mode):return {'mode':mode}
    context=SimpleNamespace(_committed=[],song_response_choice=3)
    batch={'phrase_start_beat':torch.tensor([[0.]]),'step_beat':torch.tensor([[1.]])}
    first=execution_options(Model(),context,{},batch,None,torch.Generator(),temperature=0)
    second=execution_options(Model(),context,{},batch,None,torch.Generator(),temperature=0)
    assert seen==[3] and first==second=={'mode':1}


def test_tied_mask_binding_cannot_leak_hidden_chord():
    module=SongProfileCompleter(8,2)
    condition=torch.full((2,16),.25);condition[:,12]=.8
    mask=torch.zeros_like(condition,dtype=torch.bool);mask[:,7]=True
    bound,known=module.bind_known(condition,mask)
    assert torch.equal(bound[:,7],bound[:,12]) and bool(known[:,12].all())
    assert bool(known[:,0].all())


def test_slide_radar_is_algebraically_bounded():
    values=np.full(16,.3);known=np.zeros(16,dtype=bool)
    values[9]=0;known[9]=True
    result=feasible_profile(values,known)
    assert np.allclose(result[3:6],0)
    known[4]=True;values[4]=.2
    with pytest.raises(ValueError):feasible_profile(values,known)


def test_recovery_song_response_frozen_after_first_output():
    from types import SimpleNamespace
    from maichart.composer.stream_phrase_runtime import ensure_song_response
    context=SimpleNamespace()
    first=ensure_song_response(None,context,{'song_response_logits':torch.tensor([[0.,-float('inf')]])},torch.Generator())
    second=ensure_song_response(None,context,{'song_response_logits':torch.tensor([[-float('inf'),0.]])},torch.Generator())
    assert first==second==context.song_response_choice==0


@pytest.mark.parametrize('source,explicit_file',[('manual',False),('auto',True),('checkpoint',False)])
def test_legacy_preferences_preserved_without_probability_reinterpretation(source,explicit_file):
    from types import SimpleNamespace
    from maichart.composer.stream_song_profile import resolve_song_profile
    from maichart.generation.conditions import GenerationCondition,GENERATION_DEFAULTS,RADAR_KEYS,STYLE_KEYS
    condition=GenerationCondition(level=13.3,radar={k:GENERATION_DEFAULTS[k] for k in RADAR_KEYS},style={k:GENERATION_DEFAULTS[k] for k in STYLE_KEYS})
    class ForbiddenProfile:
        def sample(self,*args,**kwargs):raise AssertionError('manual/file must bypass learned completion')
    model=SimpleNamespace(config=SimpleNamespace(song_profile_policy='masked_joint_v1'),song_profile=ForbiddenProfile())
    args=SimpleNamespace(song_style_source=source,condition_file='explicit.json' if explicit_file else None,seed=4)
    actual,report=resolve_song_profile(model,{'global':torch.zeros(1,1,8)},condition,args,{})
    assert actual.to_dict()==condition.to_dict()
    assert report['effective_source']!='learned_masked_joint_v1'


def test_actual_generation_condition_default_preserved_and_explicit_lock_conflict_rejected():
    from types import SimpleNamespace
    from maichart.composer.stream_song_profile import resolve_song_profile
    from maichart.generation.conditions import GenerationCondition
    model=SimpleNamespace(config=SimpleNamespace(song_profile_policy='off'))
    args=SimpleNamespace(song_style_source='manual',seed=0)
    condition=GenerationCondition()
    cache={'global':torch.zeros(1,1,8)}
    actual,_=resolve_song_profile(model,cache,condition,args,{})
    assert actual.to_dict()==condition.to_dict()
    args.song_style_locks={'chord':0.,'chord_bias':1.}
    with pytest.raises(ValueError,match='chord'):resolve_song_profile(model,cache,condition,args,{})
