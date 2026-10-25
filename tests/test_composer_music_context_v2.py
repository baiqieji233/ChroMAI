from fractions import Fraction
import copy

import numpy as np
import pytest
import torch

from maichart.composer.acoustics import AcousticContext
from maichart.composer.candidates import MUSIC_CANDIDATE_POLICY, candidate_beats
from maichart.composer.checkpoint import migrate_music_context, save, load, inspect_payload
from maichart.composer.model import ChartComposer, ComposerConfig
from maichart.composer.workload import causal_ledger, prefix_ledger, trailing_workload_targets
from maichart.event_timing import EventTempo


def small_model(**options):
    return ChartComposer(ComposerConfig(atom_vocab_size=7,d_model=16,nhead=4,layers=1,
        dim_feedforward=32,history_events=8,action_slots=3,dropout=0.,
        architecture_profile='music_context_v2',**options))


def inputs():
    return dict(candidate_features=torch.randn(2,6,141),history=torch.randint(1,7,(2,4,3)),
        history_features=torch.randn(2,4,8),condition=torch.randn(2,16),action_state=torch.randn(2,50),
        candidate_mask=torch.zeros(2,6,dtype=torch.bool),history_padding_mask=torch.zeros(2,4,dtype=torch.bool),
        workload_audio=torch.randn(2,9,104),trailing_audio=torch.randn(2,9,52),
        interval_audio=torch.randn(2,6,16),ledger=torch.rand(2,6,6))


def test_cached_prefix_is_equivalent_and_cannot_see_future():
    times=np.arange(0.,100.,.25);heads=np.arange(len(times))%3+1
    sums=np.r_[0.,heads.cumsum()];query=np.array([0.,2.,4.,20.,110.])
    for count in (0,1,20,len(times)):
        np.testing.assert_allclose(prefix_ledger(times,sums,query,committed_count=count),
                                   causal_ledger(times[:count],heads[:count],query))
    with pytest.raises(ValueError):prefix_ledger(times,sums,query,committed_count=-1)
    np.testing.assert_allclose(prefix_ledger(times.tolist(),sums.tolist(),query),causal_ledger(times,heads,query))


def test_trailing_labels_and_interval_audio_have_matching_seconds():
    times=np.linspace(0.,10.,1001);values=np.ones((1001,120),np.float32)
    values[(times>=3.)&(times<4.),92]=4.
    context=AcousticContext(values,times,10.)
    before=context.interval(1.,np.array([2.,4.,5.]))
    after=context.interval(1.,np.array([1.5,2.,4.,5.,8.]))
    np.testing.assert_allclose(before,after[[1,2,3]])
    assert before[1,0]>before[0,0]
    assert context.trailing(np.array([2.,4.])).shape==(2,52)
    events=np.array([1.,2.,3.]);heads=np.array([0.,1.,3.,4.])
    target=trailing_workload_targets(events,heads,[2.,4.])
    np.testing.assert_allclose(np.expm1(target),prefix_ledger(events,heads,[2.,4.])[:,:4],atol=1e-6)
    fast=AcousticContext(values,times,10.,validated_bank=True)
    np.testing.assert_array_equal(fast.at(np.array([2.,4.])),context.at(np.array([2.,4.])))


@pytest.mark.parametrize('source_length',[0,1,201])
@pytest.mark.parametrize('queries',[[],[-2.,0.,.005,1.,4.,9.99,10.,15.]])
def test_batched_acoustic_integrals_preserve_window_order_boundaries_and_empty_inputs(source_length,queries):
    rng=np.random.default_rng(71)
    times=np.linspace(0.,10.,source_length)
    values=rng.normal(size=(source_length,120)).astype(np.float32)
    context=AcousticContext(values,times,10.)
    seconds=np.asarray(queries,dtype=np.float64)
    rows=[]
    for width in (.5,2.,8.):
        lo=np.clip(seconds-width/2.,0.,context.duration)
        hi=np.clip(seconds+width/2.,0.,context.duration)
        rows.append((context._integral(hi)-context._integral(lo))/np.maximum(hi-lo,1e-6)[:,None])
    rows.append(np.broadcast_to(context.global_mean,(len(seconds),26)))
    expected=np.nan_to_num(np.concatenate(rows,axis=-1),nan=0.,posinf=0.,neginf=0.).astype(np.float32)
    np.testing.assert_array_equal(context.at(seconds),expected)
    expected_trailing=np.concatenate([(context._integral(seconds)-context._integral(seconds-width))/width
                                      for width in (1.,4.)],axis=-1).astype(np.float32)
    np.testing.assert_array_equal(context.trailing(seconds),expected_trailing)
    assert context.interval(1.,seconds).shape==(len(seconds),16)


def test_symbolic_dropout_hides_all_chart_representations_together():
    torch.manual_seed(10);model=small_model().eval();original=inputs()
    changed={key:value.clone() for key,value in original.items()}
    changed['history']=torch.randint(1,7,original['history'].shape)
    changed['history_features']*=10;changed['action_state']*=10;changed['ledger']*=10
    with torch.no_grad():
        left=model(**original,symbolic_keep=torch.zeros(2,1),decode_actions=False)
        right=model(**changed,symbolic_keep=torch.zeros(2,1),decode_actions=False)
    torch.testing.assert_close(left['candidate_logits'],right['candidate_logits'])
    torch.testing.assert_close(left['candidate_context'],right['candidate_context'])
    torch.testing.assert_close(left['audio_candidate_logits'],right['audio_candidate_logits'])
    assert not torch.allclose(model(**original,decode_actions=False)['candidate_logits'],
                              model(**changed,decode_actions=False)['candidate_logits'])
    full=model(**original,decode_actions=False)
    fast=model(**original,decode_actions=False,audio_auxiliary=False)
    torch.testing.assert_close(full['candidate_logits'],fast['candidate_logits'])
    assert 'audio_candidate_logits' not in fast


def test_shared_slots_train_like_inference_and_retain_stop_gradients():
    torch.manual_seed(12);model=small_model().eval();context=torch.randn(2,16)
    target=torch.tensor([[1,4,0],[2,0,0]])
    result=model.decode_candidate(context,action_targets=target)
    for row in range(2):
        for slot in range(3):
            prefix=target[row,:slot];prefix=prefix[prefix!=0].tolist()
            torch.testing.assert_close(result['action_logits'][row,slot],model.action_slot(context[row],slot,prefix))
    torch.nn.functional.cross_entropy(result['action_logits'].reshape(-1,7),target.reshape(-1)).backward()
    assert model.action_head.weight.grad[0].abs().sum()>0
    assert model.action_head.out_features==7


def test_new_heads_have_supervision_and_curve_remains_detached_from_decisions():
    from maichart.composer.train import _loss
    torch.manual_seed(2);model=small_model().eval();features=inputs()
    labels=torch.tensor([[1,2,0],[3,0,0]])
    output=model(**features,candidate_indices=torch.tensor([1,2]),action_targets=labels)
    batch={'candidate_target':torch.tensor([1,2]),'action_target':labels,
        'eos_target':torch.zeros(2),'future_target':torch.zeros(2,8),
        'workload_target':torch.zeros(2,9,5),'workload_weight':torch.ones(2,9),
        'workload_seconds':torch.arange(9)[None].expand(2,-1).float(),
        'trailing_target':torch.zeros(2,9,4),'trailing_weight':torch.ones(2,9,4)}
    loss,metrics=_loss(output,batch,7);loss.backward()
    assert metrics['audio_pointer_loss']>0 and metrics['trailing_workload_loss']>0
    assert model.workload_head[-1].weight.grad.abs().sum()>0
    assert model.trailing_head[-1].weight.grad.abs().sum()>0
    model.zero_grad(set_to_none=True)
    output=model(**features,decode_actions=False)
    output['candidate_logits'].sum().backward()
    assert model.workload_head[-1].weight.grad is None
    assert model.trailing_head[-1].weight.grad is None
    assert model.workload_projection.weight.grad.abs().sum()>0


def test_old_checkpoint_migration_is_explicit_and_new_roundtrip(tmp_path):
    old=ChartComposer(ComposerConfig(atom_vocab_size=7,d_model=16,nhead=4,layers=1,
        dim_feedforward=32,history_events=8,action_slots=3))
    old_weight=old.action_head.weight.detach().clone();new=migrate_music_context(old)
    torch.testing.assert_close(new.action_head.weight,old_weight.reshape(3,7,16).mean(0))
    torch.testing.assert_close(new.audio_projection.weight,old.audio_projection.weight)
    assert old.config.architecture_profile=='audio_workload_v1'
    path=tmp_path/'new.pt'
    save(path,new,torch.optim.AdamW(new.parameters()),epoch=1,args={},manifest='manifest.json',atom_catalog=list('123456'))
    restored,payload=load(path,include_optimizer=True)
    assert inspect_payload(payload)['can_resume']
    assert restored.config.architecture_profile=='music_context_v2'


def test_phase_recovery_remains_available_after_drift():
    from maichart.composer.diagnose import phase_recovery
    from maichart.tempo_clock import TempoClock
    tempo=[EventTempo(Fraction(0),120.)];clock=TempoClock(tempo,0.)
    for previous in (Fraction(11,48),Fraction(1,12),Fraction(2,3)):
        values=candidate_beats(previous,tempo=tempo,first=0.,audio_end_sec=30.,candidate_count=96,
                               policy=MUSIC_CANDIDATE_POLICY)
        assert values==sorted(set(values)) and len(values)<=96
        assert any(x.denominator==1 for x in values)
        target=Fraction(4)
        assert phase_recovery(previous,target,tempo=tempo,first=0.,end_sec=30.,count=96,
                              policy=MUSIC_CANDIDATE_POLICY,clock=clock) in (1,2)
    extended=candidate_beats(None,tempo=tempo,first=0.,audio_end_sec=90.,candidate_count=96,
                              policy=MUSIC_CANDIDATE_POLICY)
    assert Fraction(64) in extended and Fraction(128) in extended
    # Existing checkpoints retain their original bounded candidate horizon.
    legacy=candidate_beats(None,tempo=tempo,first=0.,audio_end_sec=90.,candidate_count=96)
    assert max(legacy)<=16


def test_intervention_report_is_finite_and_no_quality_gate():
    from maichart.composer.diagnose import interventions
    model=small_model().eval();report=interventions(model,inputs())
    assert 'workload_constant' in report and 'ledger_zero' in report
    assert report['symbolic_unread']['pointer_js']>=0.
    assert all(np.isfinite(value) for item in report.values() for value in item.values())


def test_small_batches_with_many_source_songs_do_not_overrun_sampler():
    from types import SimpleNamespace
    from maichart.composer.train import SongLocalBatchSampler
    dataset=SimpleNamespace(samples=[SimpleNamespace(song_index=i//3) for i in range(30)])
    sampler=SongLocalBatchSampler(dataset,2,torch.Generator().manual_seed(1),locality=8)
    batches=list(sampler)
    assert all(len(batch)<=2 for batch in batches)
    assert len(batches)==len(sampler)
    assert sorted(i for batch in batches for i in batch)==list(range(30))
