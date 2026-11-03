"""Music owns occurrence; missed entrances must not become permanent rests."""
import copy
import math
import random
from dataclasses import replace
from fractions import Fraction as F
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as TF

from maichart.composer import stream_augmentation as aug
from maichart.composer.stream_data import NOTE, WAIT, EOS, StreamStep, collate_stream, recovery_sample, second_at_beat
from maichart.composer.stream_model import StreamComposer, StreamComposerConfig
from maichart.composer.stream_train import upgrade_learning_stage, resolve_learning_policies


def hazard_fixture():
    from test_composer_stream_model import fixture
    old,batch=fixture(timing_output_policy='absolute_grid_v1')
    model=upgrade_learning_stage(old,{'plan_kind_policy':'audio_hazard_v3'}).eval()
    return model,batch


def test_music_occurrence_survives_a_wait_only_symbolic_history():
    model,batch=hazard_fixture()
    original=model(batch)
    altered=copy.deepcopy(batch)
    altered['history_kind'][:,1:]=WAIT
    for key in ('history_atoms','history_shape','history_phase','history_phase_mask'):
        altered[key][:,1:]=0
    altered['action_state'].normal_()
    altered['action_state_mask'].zero_()
    changed=model(altered)
    torch.testing.assert_close(original['kind_logits'],changed['kind_logits'],rtol=0,atol=0)
    # The fix does not erase symbol-sensitive configuration and time contexts.
    assert not torch.equal(original['hidden'],changed['hidden'])


def test_occurrence_is_the_same_interval_law_in_the_terminal_window():
    model,batch=hazard_fixture()
    with torch.no_grad():
        model.plan_head.weight.zero_()
        model.plan_head.bias[2]=math.log(math.expm1(.7))
    output=model(batch)
    expected=.7*torch.minimum(torch.full_like(batch['step_second'],1.),
                              batch['audio_end_sec'][:,None]-batch['step_second'])
    probabilities=output['kind_logits'].softmax(-1)
    torch.testing.assert_close(probabilities[...,NOTE],-torch.expm1(-expected),rtol=1e-5,atol=1e-6)
    remaining=batch['audio_end_beat'][:,None]-batch['step_beat']
    tail=torch.where(remaining>2,probabilities[...,WAIT],probabilities[...,EOS])
    torch.testing.assert_close(tail,torch.exp(-expected),rtol=1e-5,atol=1e-6)
    assert probabilities[0,-1,WAIT]==0 and probabilities[0,-1,EOS]>0


@pytest.mark.parametrize('kind,sign',[(NOTE,-1),(WAIT,1),(EOS,1)])
def test_real_occurrence_nll_trains_music_rate_in_both_directions(kind,sign):
    model,batch=hazard_fixture()
    output=model(batch)
    index=-1 if kind==EOS else 0
    nll=-output['kind_logits'][0,index].log_softmax(-1)[kind]
    nll.backward()
    gradient=model.plan_head.bias.grad[2]
    assert torch.isfinite(gradient) and gradient*sign>0
    assert model.kind_head.weight.grad is None
    assert model.plan_head.weight.grad is not None and model.plan_head.weight.grad.abs().sum()>0


def test_phrase_mode_cannot_reinstate_an_arbitrary_wait_veto():
    model,batch=hazard_fixture()
    groups=model.timing_grid.num_groups
    output=model(batch)
    model.config.phrase_modes=2
    model.phrase_execution=SimpleNamespace(timing=lambda intent:
        intent.new_tensor([0.,0.,-500.,500.,500.,*([0.]*groups),
                           0.,0.,500.,-500.,-500.,*([0.]*groups)]).expand(*intent.shape[:-1],-1))
    kinds,times=model.execution_timing_outputs({'phrase_intent':torch.zeros(1,4,16)},
        output['kind_logits'],output['delta_logits'],output['timing_grid_ids'])
    torch.testing.assert_close(kinds[:,:,0],output['kind_logits'],rtol=0,atol=0)
    torch.testing.assert_close(kinds[:,:,1],output['kind_logits'],rtol=0,atol=0)


def test_new_stage_selects_new_occurrence_and_entry_curriculum_without_rewriting_old_config():
    old=StreamComposerConfig.from_dict({'plan_kind_policy':'rate_integral_v2'})
    fresh=resolve_learning_policies(SimpleNamespace(),old,resume=False)
    assert fresh['plan_kind_policy']=='audio_hazard_v3'
    assert fresh['prefix_augmentation_input_policy']==aug.MUSICAL_ENTRY_INPUT_POLICY
    saved=resolve_learning_policies(SimpleNamespace(),old,resume=True)
    assert saved['plan_kind_policy']=='rate_integral_v2'
    assert old.plan_kind_policy=='rate_integral_v2'


def test_missed_prefix_has_only_exact_two_beat_waits_and_does_not_use_the_target():
    source=[];previous=F(0)
    for i in range(20):
        beat=F(i+1)
        source.append(StreamStep(NOTE,previous,beat,float(previous)/2,float(beat)/2,str(i%8+1)))
        previous=beat
    prefix,keep,count,gap=aug._missed_entry_prefix(source,12,F(8))
    changed=list(source);changed[12]=replace(changed[12],text='8')
    again=aug._missed_entry_prefix(changed,12,F(8))
    assert again==(prefix,keep,count,gap)
    assert prefix[:keep]==source[:keep] and count>0
    assert all(step.kind==WAIT and step.end-step.start==2 for step in prefix[keep:])
    assert prefix[-1].end==source[12].start


@pytest.mark.parametrize('rest',[False,True])
def test_entry_training_retains_authored_target_and_all_learning_budgets(monkeypatch,rest):
    from test_stream_augmentation import dataset
    ds=dataset(monkeypatch);row=ds[3]
    row['_training_chart_id']=17
    row['_profile_chart_importance']=.03125
    row['_profile_reference_batch_size']=16
    row['_song_response_group_size']=4
    result,mode,count=aug._augment_entry_row(ds,3,rng=random.Random(7),row=row,rest=rest)
    assert mode==('rest_control' if rest else 'missed_entry')
    at=result['loss_mask'].nonzero().flatten()
    assert at.tolist()==[len(result['history_kind'])-1]
    actual=int(result['target_kind'][-1])
    assert actual in ((WAIT,EOS) if rest else (NOTE,))
    source=next(s for s in ds.streams[0] if float(s.start)==float(result['step_beat'][-1])
                and float(s.end)==float(result['target_beat'][-1]) and s.kind==actual)
    assert result['target_second'][-1]==source.end_second
    assert not result['configuration_mask'].any()
    torch.testing.assert_close(result['sample_weight'].sum(),row['sample_weight'].sum(),rtol=0,atol=0)
    before,after=collate_stream([row]),collate_stream([result])
    for key in ('audio_plan_weight','audio_plan_blank_weight','profile_chart_weights'):
        torch.testing.assert_close(after[key],before[key],rtol=0,atol=0)
    assert result['_training_chart_id']==17
    assert not rest and count>0 or rest


def test_changed_target_atom_never_leaks_into_entry_history(monkeypatch):
    from test_stream_augmentation import dataset
    ds=dataset(monkeypatch);original=ds[3]
    before,_,_=aug._augment_entry_row(ds,3,rng=random.Random(2),row=original)
    selected=next(i for i,s in enumerate(ds.streams[0]) if s.kind==NOTE
                  and float(s.start)==float(before['step_beat'][-1])
                  and float(s.end)==float(before['target_beat'][-1]))
    ds.streams[0][selected]=replace(ds.streams[0][selected],text='8')
    after,_,_=aug._augment_entry_row(ds,3,rng=random.Random(2),row=original)
    for key in ('history_kind','history_atoms','history_delta','step_beat','action_state','continuation_features'):
        torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)


def test_entry_replay_retains_an_older_hold_across_waits_and_bpm_changes(monkeypatch):
    from test_stream_phase_recovery import _dataset
    from maichart.composer.stream_recovery import _context
    ds=_dataset(monkeypatch,native=True)
    source=ds.streams[0];target=source[12]
    prefix,keep,_,_=aug._missed_entry_prefix(source,12,F(2))
    assert keep>0 and prefix[0].text==source[0].text
    clock=ds.clocks[0]
    prefix=prefix[:keep]+[replace(s,start_second=second_at_beat(clock,s.start),
        end_second=second_at_beat(clock,s.end)) for s in prefix[keep:]]
    row=recovery_sample(ds,0,prefix,target,configuration_trusted=False)
    context=_context(ds,0,ds._bank(0));raw=context.batch(prefix,target.start)
    torch.testing.assert_close(row['action_state'][-1],raw['action_state'][0,-1])
    torch.testing.assert_close(row['action_state_mask'][-1],raw['action_state_mask'][0,-1])
    assert bool((~row['action_state_mask'][-1]).any())


def test_changed_learning_objective_cannot_silently_resume_v47_optimizer():
    from maichart.composer.stream_checkpoint import training_objective_compatible,UNIFIED_OBJECTIVE_VERSION
    payload=dict(model_config=dict(song_response_policy='unified_profile_response_v2'),
        args=dict(training_objective_recipe=dict(version='stream_balanced_hierarchical_configuration_v9')))
    assert not training_objective_compatible(payload)
    payload['args']['training_objective_recipe']['version']=UNIFIED_OBJECTIVE_VERSION
    assert training_objective_compatible(payload)
