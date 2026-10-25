from fractions import Fraction
from types import SimpleNamespace
import copy
import json

import numpy as np
import pytest
import torch

from maichart.composer.model import ComposerConfig,build_model,GROUNDED_PROFILE
from maichart.composer.grounded_model import hierarchical_log_probs
from maichart.composer.checkpoint import save,load,inspect_payload,migrate_audio_grounded
from maichart.composer import GROUNDED_TRAINING_SEMANTICS

ATOMS=['1','2','3h[4:1]','4-8[4:1]']


@pytest.mark.parametrize('profile', ['audio_workload_v1', 'music_context_v2', GROUNDED_PROFILE])
def test_finetune_expands_slots_preserving_old_weights(profile):
    from maichart.composer.checkpoint import migrate_music_context
    source = build_model(ComposerConfig(atom_vocab_size=5, d_model=16, nhead=4,
        layers=1, dim_feedforward=32, action_slots=8, architecture_profile=profile,
        head_timing_dim=99 if profile != 'audio_workload_v1' else 0, lens_width=8), ATOMS)
    targets = [GROUNDED_PROFILE] if profile == GROUNDED_PROFILE else ['music_context_v2', GROUNDED_PROFILE]
    for target in targets:
        upgraded = (migrate_audio_grounded(source, ATOMS, action_slots=9)
            if target == GROUNDED_PROFILE else migrate_music_context(source, action_slots=9))
        assert upgraded.config.action_slots == 9
        assert upgraded.slot_embedding.weight.shape == (9, 16)
        torch.testing.assert_close(upgraded.audio_projection.weight, source.audio_projection.weight)
        if profile != 'audio_workload_v1':
            torch.testing.assert_close(upgraded.slot_embedding.weight[:8], source.slot_embedding.weight)
        else:
            torch.testing.assert_close(upgraded.action_head.weight,
                source.action_head.weight.reshape(8, 5, 16).mean(0))
        with pytest.raises(ValueError, match='reduce action_slots'):
            if target == GROUNDED_PROFILE:
                migrate_audio_grounded(upgraded, ATOMS, action_slots=8)
            else:
                migrate_music_context(upgraded, action_slots=8)


def config(**kwargs):
    return ComposerConfig(atom_vocab_size=5,d_model=16,nhead=4,layers=1,dim_feedforward=32,
        dropout=0.,history_events=8,action_slots=3,architecture_profile=GROUNDED_PROFILE,
        head_timing_dim=99,lens_width=8,lens_chunk_size=3,**kwargs)


def inputs():
    x=torch.randn(2,6,141)
    x[:,:,120]=torch.tensor([.25,.5,1.,2.,6.,12.])
    return dict(candidate_features=x,history=torch.tensor([[[1,0,0],[2,0,0]],[[2,0,0],[1,0,0]]]),
        history_features=torch.randn(2,2,8),history_padding_mask=torch.zeros(2,2,dtype=torch.bool),
        condition=torch.randn(2,16),action_state=torch.randn(2,50),head_timing=torch.rand(2,6,99),
        candidate_mask=torch.zeros(2,6,dtype=torch.bool),workload_audio=torch.randn(2,8,104),
        ledger=torch.rand(2,6,6),interval_audio=torch.randn(2,6,16),trailing_audio=torch.randn(2,8,52),
        audio_lens=torch.randn(2,8,31,26),audio_lens_mask=torch.ones(2,8,31,dtype=torch.bool),
        planner_audio=torch.randn(2,4,104),planner_mask=torch.zeros(2,4,dtype=torch.bool),
        planner_indices=torch.tensor([[[0,1],[0,1],[1,2],[1,2],[2,3],[2,3]]]*2),
        planner_mix=torch.full((2,6),.5))


def test_hierarchical_pointer_removes_candidate_count_prior_and_masks_empty_bins():
    fine=torch.zeros(1,5);coarse=torch.zeros(1,6)
    bins=torch.tensor([[0,0,0,0,5]]);valid=torch.ones(1,5,dtype=torch.bool)
    probability=hierarchical_log_probs(fine,coarse,bins,valid).exp()
    torch.testing.assert_close(probability[0,:4].sum(),torch.tensor(.5))
    torch.testing.assert_close(probability[0,4],torch.tensor(.5))
    valid[0,0]=False
    probability=hierarchical_log_probs(fine,coarse,bins,valid).exp()
    assert probability[0,0]==0
    torch.testing.assert_close(probability.sum(),torch.tensor(1.))
    torch.testing.assert_close(probability[0,4],torch.tensor(.5))


def test_trailing_history_stop_elision_preserves_outputs_and_all_gradients():
    torch.manual_seed(63)
    dense=build_model(config(history_dropout=0.,ledger_dropout=0.,symbolic_dropout=0.),ATOMS).train()
    compact=copy.deepcopy(dense)
    data=inputs()
    full=torch.zeros(2,2,33,dtype=torch.long)
    full[...,:3]=data['history']
    full[0,0,3]=3
    labels=torch.tensor([[1,2,0],[2,0,0]])
    outputs=[];grads=[]
    for model,history in ((dense,full),(compact,full[...,:4])):
        result=model(**{**data,'history':history},candidate_indices=torch.tensor([1,2]),action_targets=labels)
        outputs.append({k:v.detach() for k,v in result.items() if torch.is_tensor(v)})
        sum(result[k].square().mean() for k in ('action_logits','audio_action_logits','candidate_logits','plan','workload')).backward()
        grads.append({k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None})
    for key in outputs[0]:
        torch.testing.assert_close(outputs[0][key],outputs[1][key],rtol=1e-5,atol=1e-6,msg=lambda msg:f'{key}: {msg}')
    assert grads[0].keys()==grads[1].keys()
    for key in grads[0]:
        torch.testing.assert_close(grads[0][key],grads[1][key],rtol=1e-4,atol=1e-6,msg=lambda msg:f'{key}: {msg}')


def test_audio_path_and_final_history_odds_bound_are_structural():
    model=build_model(config(),ATOMS).eval()
    data=inputs()
    with torch.no_grad():
        model.history_pointer.weight.normal_(0,20)
        model.history_bin.weight.normal_(0,20)
        model.action_relation.weight.normal_()
        output=model(**data,decode_actions=False)
    delta=output['candidate_logits']-output['audio_candidate_logits']
    assert torch.all(delta.max(-1).values-delta.min(-1).values<=2*model.config.history_residual_bound+1e-5)
    changed={k:v.clone() for k,v in data.items()}
    changed['history']=torch.flip(changed['history'],[1])
    changed['ledger']*=100;changed['head_timing']*=100;changed['action_state']*=100
    with torch.no_grad():
        left=model(**data,symbolic_keep=torch.zeros(2,1),decode_actions=False)
        right=model(**changed,symbolic_keep=torch.zeros(2,1),decode_actions=False)
    torch.testing.assert_close(left['candidate_context'],right['candidate_context'])
    torch.testing.assert_close(left['audio_candidate_logits'],right['audio_candidate_logits'])


def test_lens_plan_and_compositional_heads_receive_their_supervision():
    from maichart.composer.train import _loss
    torch.manual_seed(2)
    model=build_model(config(),ATOMS)
    data=inputs();targets=torch.tensor([[1,2,0],[2,0,0]])
    output=model(**data,candidate_indices=torch.tensor([1,2]),action_targets=targets)
    batch={'candidate_target':torch.tensor([1,2]),'action_target':targets,
        'future_target':torch.zeros(2,8),'eos_target':torch.zeros(2),
        'workload_target':torch.ones(2,8,5),'workload_weight':torch.ones(2,8),
        'workload_seconds':torch.arange(8)[None].repeat(2,1),
        'trailing_target':torch.zeros(2,8,4),'trailing_weight':torch.ones(2,8,4),
        'placement_target':torch.ones(2,8,5),'placement_weight':torch.ones(2,8,5),
        'planner_target':torch.ones(2,4,6),'planner_weight':torch.ones(2,4)}
    loss,metrics=_loss(output,batch,5)
    loss.backward()
    assert metrics['placement_loss']>0 and metrics['planner_loss']>0
    for weight in (model.lens.input_projection.weight,model.placement_head.weight,
                   model.planner.output.weight,model.action_factors.factors['kind_mask'].weight):
        assert weight.grad is not None and weight.grad.abs().sum()>0
    model.zero_grad(set_to_none=True)
    output=model(**data,decode_actions=False)
    output['candidate_logits'].sum().backward()
    assert model.planner.output.weight.grad is None
    assert model.workload_head[-1].weight.grad is None


def test_compositional_slot_training_matches_autoregressive_decoding():
    model=build_model(config(),ATOMS).eval()
    with torch.no_grad():
        for item in model.action_factors.factors.values():item.weight.normal_(0,.01)
        model.action_factors.relative_factors.weight.normal_(0,.01)
    context=torch.randn(2,16)
    targets=torch.tensor([[1,2,0],[3,0,0]])
    result=model.decode_candidate(context,action_targets=targets)
    for row in range(2):
        for slot in range(3):
            selected=targets[row,:slot];selected=selected[selected.ne(0)].tolist()
            torch.testing.assert_close(result['action_logits'][row,slot],model.action_slot(context[row],slot,selected))


def test_planner_padded_suffix_does_not_change_real_prefix():
    model=build_model(config(),ATOMS).eval()
    audio=torch.randn(1,3,104);condition=torch.randn(1,16)
    with torch.no_grad():
        short=model.planner(audio,condition,torch.zeros(1,3,dtype=torch.bool))
        padded=model.planner(torch.cat((audio,torch.randn(1,4,104)),1),condition,
            torch.tensor([[False,False,False,True,True,True,True]]))
    torch.testing.assert_close(short,padded[:,:3])


def test_new_checkpoint_roundtrip_and_explicit_partial_migration(tmp_path):
    model=build_model(config(),ATOMS)
    target=tmp_path/'cp.pt'
    payload=save(target,model,torch.optim.AdamW(model.parameters()),epoch=1,args={},manifest=tmp_path/'m.json',atom_catalog=ATOMS)
    assert payload['training_semantics']==GROUNDED_TRAINING_SEMANTICS
    assert inspect_payload(payload)['can_resume']
    restored,_=load(target,include_optimizer=True)
    for key,value in model.state_dict().items():
        if torch.is_tensor(value):torch.testing.assert_close(value,restored.state_dict()[key])
    oldconfig=ComposerConfig(**{**config().to_dict(),'architecture_profile':'music_context_v2'})
    old=build_model(oldconfig)
    migrated=migrate_audio_grounded(old,ATOMS)
    torch.testing.assert_close(old.action_head.weight,migrated.action_head.weight)
    assert migrated.config.architecture_profile==GROUNDED_PROFILE


def test_plan_targets_are_condition_specific_and_unknown_spans_are_masked():
    from maichart.composer.planning import PlanGrid
    from maichart.composer.event_views import EventViews
    from maichart.composer.acoustics import AcousticContext
    from maichart.event_timing import EventTempo,EventPoint
    from maichart.tempo_clock import TempoClock
    tempo=[EventTempo(Fraction(0),120.)]
    points=[EventPoint(Fraction(0),'1'),EventPoint(Fraction(1),'2'),EventPoint(Fraction(5),'3')]
    views=EventViews.build(points,tempo,0.,6.,excluded=[(2.2,2.3)])
    acoustic=AcousticContext(np.ones((61,120),np.float32),np.linspace(0,6,61),6.)
    plan=PlanGrid(TempoClock(tempo,0.),6.,acoustic,views)
    assert plan.targets.shape==(3,6)
    assert plan.targets[0,0]==pytest.approx(np.log1p(1.))
    assert plan.targets[2,5]==1
    assert plan.weights.tolist()==[1,0,1]
    indices,mix=plan.locate([.5,1.,2.,5.5])
    assert np.all((mix>=0)&(mix<=1))
    assert indices.max()<3


def test_real_dataset_grounded_forward_and_diagnostics(tmp_path,monkeypatch):
    from test_composer_event_views import dataset_fixture
    from maichart.composer.data import collate_composer
    from maichart.composer.train import _forward_batch,_loss
    from maichart.composer.diagnose import _inputs,interventions
    ds=dataset_fixture(tmp_path,monkeypatch)
    ds.architecture_profile=GROUNDED_PROFILE
    batch=collate_composer([ds[0],ds[1]])
    cfg=ComposerConfig(atom_vocab_size=len(ds.codec_atoms)+1,d_model=16,nhead=4,layers=1,
        dim_feedforward=32,history_events=2,architecture_profile=GROUNDED_PROFILE,
        head_timing_dim=99,lens_width=8,lens_chunk_size=8)
    model=build_model(cfg,ds.codec_atoms)
    output=_forward_batch(model,batch)
    loss,_=_loss(output,batch,cfg.atom_vocab_size)
    assert torch.isfinite(loss)
    assert batch['audio_lens'].shape==(2,112,31,26)
    report=interventions(model,_inputs(batch,'cpu'))
    assert 'symbolic_unread' in report


def test_grounded_training_checkpoint_generation_and_recovery_are_connected(tmp_path):
    import soundfile as sf
    from maichart.composer.train import main
    from maichart.composer.generate import parser,generate
    sr=8000;time=np.arange(sr*4)/sr
    audio=tmp_path/'track.wav'
    sf.write(audio,(.1*np.sin(2*np.pi*220*time)).astype(np.float32),sr)
    chart=tmp_path/'maidata.txt'
    chart.write_text('&wholebpm=120\n&lv_5=13\n&inote_5=(120){4}1,2,3,4,5,6,7,8,E\n',encoding='utf-8')
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'songs':[{'song_id':'grounded-e2e','maidata_path':str(chart),
        'audio_path':str(audio),'difficulties':[{'difficulty_index':5,'level_raw':'13'}]}]}),encoding='utf-8')
    output=tmp_path/'model'
    argv=['--manifest',str(manifest),'--checkpoint-dir',str(output),'--cache-dir',str(tmp_path/'cache'),
        '--epochs','1','--batch-size','2','--max-steps','2','--device','cpu','--precision','fp32',
        '--d-model','16','--layers','1','--nhead','4','--dim-feedforward','32','--candidate-count','16',
        '--action-slots','2','--history-events','8','--lens-width','8','--lens-chunk-size','8',
        '--prefetch-batches','0','--prepare-workers','1','--recovery-interval','1','--recovery-samples','1']
    assert main(argv)==0
    checkpoint=output/'composer_last.pt'
    loaded,payload=load(checkpoint,include_optimizer=True)
    assert loaded.config.architecture_profile==GROUNDED_PROFILE
    assert payload['optimizer_steps']==2
    assert 'planner_loss' in payload['metrics']
    report=generate(parser().parse_args(['--checkpoint',str(checkpoint),'--audio',str(audio),
        '--output',str(tmp_path/'generated.txt'),'--bpm','120','--max-events','3','--device','cpu',
        '--pointer-history-strength','.6','--seed','0']))
    assert report['pointer_history_strength']==.6
    assert report['coarse_audio_plan']['source']=='audio_and_condition_only'
    assert report['audio_lens_cache']['misses']>0


def test_new_optional_migration_parameters_are_applied():
    model=build_model(config(),ATOMS)
    upgraded=migrate_audio_grounded(model,ATOMS,lens_width=12,history_residual_bound=.6)
    assert upgraded.config.lens_width==12
    assert upgraded.config.history_residual_bound==.6
    torch.testing.assert_close(upgraded.action_head.weight,model.action_head.weight)


def test_factor_inference_cache_refreshes_after_weight_update():
    model=build_model(config(),ATOMS).eval()
    with torch.no_grad():
        before=model.action_factors.composed_weight().clone()
        model.action_factors.factors['kind_mask'].weight.add_(1.)
        after=model.action_factors.composed_weight()
    assert not torch.equal(before,after)


def test_active_slot_loss_is_invariant_to_trailing_stop_padding():
    from maichart.composer.train import _loss
    base_logits = torch.randn(2, 3, 5)
    pad_logits = torch.cat((base_logits, torch.randn(2, 6, 5)), dim=1)
    base_targets = torch.tensor([[1, 2, 0], [3, 0, 0]])
    pad_targets = torch.cat((base_targets, torch.zeros(2, 6, dtype=torch.long)), dim=1)
    common = {'candidate_logits': torch.zeros(2, 4), 'future': torch.zeros(2, 8), 'eos_logits': torch.zeros(2)}
    batch_base = {'candidate_target': torch.tensor([0, 1]), 'action_target': base_targets,
                  'future_target': torch.zeros(2, 8), 'eos_target': torch.zeros(2)}
    batch_pad = dict(batch_base, action_target=pad_targets)
    _, m_base = _loss(dict(common, action_logits=base_logits), batch_base, 5)
    _, m_pad = _loss(dict(common, action_logits=pad_logits), batch_pad, 5)
    torch.testing.assert_close(m_base['action_loss'], m_pad['action_loss'])


def test_enriched_head_timing_and_bounded_action_residual_match_slot_decoding():
    from maichart.composer.events import enrich_head_timing
    zeros = torch.zeros(2, 6, 99)
    assert torch.count_nonzero(enrich_head_timing(zeros)) == 0
    model = build_model(config(), ATOMS).eval()
    data = inputs()
    targets = torch.tensor([[1, 2, 0], [3, 0, 0]])
    with torch.no_grad():
        out = model(**data, candidate_indices=torch.tensor([1, 2]), action_targets=targets)
        assert 'audio_action_logits' in out
        for row in range(2):
            for slot in range(3):
                selected = targets[row, :slot]
                selected = selected[selected.ne(0)].tolist()
                step = model.action_slot(out['action_context'][row], slot, selected,
                                         audio_context=out['audio_action_context'][row])
                torch.testing.assert_close(out['action_logits'][row, slot], step)


@pytest.mark.parametrize('semantics',['chart_composer_training_v8_boundary_plans_recovery',
                                    'chart_composer_training_v9_active_slots_phrase_tail'])
def test_older_grounded_contract_requires_finetune_and_macro_category_mass_stays_calibrated(tmp_path,semantics):
    from maichart.composer.checkpoint import inspect_payload, save
    from maichart.composer.generate import _category_truncate_actions
    model = build_model(config(), ATOMS)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    ckpt_file = tmp_path / 'v8_contract_check.pt'
    payload = save(ckpt_file, model, opt, epoch=1, args={'lr': 2e-4}, manifest='manifest.json',
                   atom_catalog=ATOMS, training_state={'epoch_complete': True})
    assert inspect_payload(payload)['can_resume']
    bad=dict(payload,action_sampling_policy='pooled_touch_v1')
    assert not inspect_payload(bad)['valid']
    payload['training_semantics'] = semantics
    inspected = inspect_payload(payload)
    assert inspected['valid'] and inspected['can_generate'] and inspected['can_init']
    assert not inspected['can_resume']
    logits = torch.tensor([-1e9, 2.5, 2.4, -0.5, -0.6, -1.2, -1.3, -2.0])
    groups = (
        torch.tensor([False, True, True, False, False, False, False, False]),
        torch.tensor([False, False, False, True, True, False, False, False]),
        torch.tensor([False, False, False, False, False, True, True, False]),
        torch.tensor([False, False, False, False, False, False, False, True]),
    )
    expected = logits.softmax(-1)
    for temp in (0.6, 0.85, 0.9):
        probs = _category_truncate_actions(logits, temp, groups)
        for g in groups:
            torch.testing.assert_close(probs[g].sum(), expected[g].sum(), atol=1e-5, rtol=1e-5)
