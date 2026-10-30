import torch
from maichart.composer.checkpoint import migrate_audio_grounded,save,load
from test_composer_echo_stage2_integration import _echo_fixture


def fixture():
    old,data=_echo_fixture('song_paired_v3')
    model=migrate_audio_grounded(old,old.action_factors.atom_catalog,
        action_head_policy='compact_grouped_v3',button_load_policy='per_button_v1',
        future_horizon_policy='candidate_horizon_v1',timing_distribution_policy='survival_v1').eval()
    data['candidate_features'][...,121]=data['candidate_features'][...,120]*.5
    data['history_second_ages']=data['history_beat_ages']*.5
    data['quality_inputs'].update(quality_future_audio=torch.randn(2,6,10,104),
        quality_future_mask=torch.ones(2,6,10,dtype=torch.bool),
        quality_future_offsets=torch.tensor([[.25,.5,.75,1.,1.5,2.,3.,4.,6.,8.]]*2))
    data['empty_gap_audio']=torch.randn(2,4,104)
    return model,data


def test_compact_training_generation_and_checkpoint(tmp_path):
    model,data=fixture()
    assert not hasattr(model,'action_head') and not hasattr(model,'future_horizon')
    assert model.quality_context.future_evidence_projection is None
    with torch.no_grad():
        model.conserved_action_head.macro_head.weight.normal_(std=.1)
        model.button_load.output.weight.normal_(std=.1)
        model.cooperative_geometry.end_to_head_factors.weight.normal_(std=.1)
    dense=model(**data);packed=model(**data,packed_actions=True)
    rows,slots=packed['action_slot_indices'].unbind(-1)
    torch.testing.assert_close(packed['action_logits'],dense['action_logits'][rows,slots],atol=2e-5,rtol=2e-5)
    for row in range(2):
        for slot in (0,1):
            actual=model.action_slot(dense['action_context'][row],slot,data['action_targets'][row,:slot].tolist(),
                previous_heads=dense['previous_heads'][row],echo_heads=dense['echo_heads'][row],
                button_load_features=dense['button_load_features'][row])
            torch.testing.assert_close(actual,dense['action_logits'][row,slot],atol=2e-5,rtol=2e-5)
    loss=-dense['next_event_log_probs'][:,0].mean()+torch.nn.functional.cross_entropy(dense['action_logits'][:,0],torch.tensor([1,2]))
    loss.backward()
    assert model.fixed_future.output.weight.grad.abs().sum()>0
    path=tmp_path/'compact.pt'
    save(path,model,None,epoch=0,args={},manifest=tmp_path/'unused',atom_catalog=model.action_factors.atom_catalog)
    restored,payload=load(path);restored.eval()
    torch.testing.assert_close(restored(**data)['action_logits'],dense['action_logits'],atol=0,rtol=0)


def test_empty_gap_audio_is_supervision_only_and_audio_reference_is_independent():
    model,data=fixture()
    before=model(**data)
    altered=model(**dict(data,empty_gap_audio=data['empty_gap_audio']+100.))
    for key in ('candidate_logits','action_logits','audio_action_logits'):
        assert torch.equal(before[key],altered[key]),key
    assert not torch.equal(before['empty_gap_rates'],altered['empty_gap_rates'])
    model.zero_grad()
    (-before['audio_action_logits'][:,0,1].mean()).backward()
    assert model.conserved_action_head.audio_macro_head.weight.grad.abs().sum()>0
    assert model.conserved_action_head.macro_head.weight.grad is None


def test_explicit_future_off_does_not_read_probes():
    model,data=fixture()
    model.config.future_horizon_policy='off'
    for key in ('quality_future_audio','quality_future_mask','quality_future_offsets'):
        data['quality_inputs'].pop(key)
    def forbidden(*args,**kwargs):
        raise AssertionError('future off must bypass the added future branch')
    model.fixed_future.forward=forbidden
    output=model(**data)
    assert torch.isfinite(output['action_logits']).all()
