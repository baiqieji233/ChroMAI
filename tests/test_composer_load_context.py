import torch
from maichart.composer.checkpoint import migrate_audio_grounded
from test_composer_compact_integration import fixture


def test_load_context_starts_neutral_and_reaches_timing_and_later_stop():
    base,data=fixture()
    with torch.no_grad():base.button_load.output.weight.normal_(std=.1)
    model=migrate_audio_grounded(base,base.action_factors.atom_catalog,
        button_load_policy='per_button_context_v2').eval()
    # Preserve even a trained lane-load head: only the NEW summary uses a
    # time-decayed run, rather than silently changing the old feature meaning.
    before=base(**data)
    after=model(**data)
    torch.testing.assert_close(after['candidate_logits'],before['candidate_logits'],atol=0,rtol=0)
    torch.testing.assert_close(after['action_logits'],before['action_logits'],atol=0,rtol=0)
    with torch.no_grad():
        model.history_pointer.weight.normal_(std=.15)
        model.button_load_context.weight.normal_(std=.1)
        model.conserved_action_head.macro_head.weight.normal_(std=.1)
    active=model(**data)
    hidden=model(**data,symbolic_keep=torch.zeros(2,1))
    assert not hidden['button_load_features'].any()
    # Both objectives must reach the single shared summary projection.
    for loss in (-active['candidate_logits'][:,0].mean(),-active['action_logits'][:,1,0].mean()):
        model.zero_grad()
        loss.backward(retain_graph=True)
        assert model.button_load_context.weight.grad.abs().sum()>0


def test_integrated_model_uses_fixed_grid_with_masked_condition_and_no_label_inputs():
    base,data=fixture()
    model=migrate_audio_grounded(base,base.action_factors.atom_catalog,
        button_load_policy='per_button_context_v2',timing_distribution_policy='survival_integrated_v2').eval()
    b,c=data['candidate_features'].shape[:2]
    data.update(survival_audio=torch.randn(b,20,104),
        survival_edges=torch.linspace(0.,8.,21,dtype=torch.float64)[None].expand(b,-1),
        survival_candidate_seconds=data['candidate_features'][...,121].double(),
        survival_start_seconds=torch.zeros(b,dtype=torch.float64),
        survival_plan_indices=torch.zeros(b,20,2,dtype=torch.long),
        survival_plan_mix=torch.zeros(b,20))
    out=model(**data)
    torch.testing.assert_close(out['candidate_logits'].exp().sum(-1)+out['no_event_log_prob'].exp(),torch.ones(b))
    torch.testing.assert_close(out['no_event_log_prob'],out['audio_no_event_log_prob'],atol=0,rtol=0)
    assert out['survival_grid_rates'].shape==(b,20)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        result=model(**data,packed_actions=True)
        loss=-result['next_event_log_probs'][:,0].mean()-result['action_logits'][:,1].mean()
    loss.backward()
    assert torch.isfinite(model.button_load_context.weight.grad).all()
    # A confidently wrong acoustic choice must still receive a corrective
    # main-task gradient; the old +/-1 tanh shape made this almost disappear.
    data['candidate_features'][...,120]=torch.tensor([.125,.25,.5,1.,2.,4.])
    raw=torch.tensor([8.,-8.,-8.,-8.,-8.,-8.],requires_grad=True)
    model.pointer.forward=lambda context:raw[None,:,None].expand(b,-1,-1)
    scored=model(**data)
    (-scored['next_event_log_probs'][:,1].mean()).backward()
    assert raw.grad[1]<-.5
    model.config.density_feedback_policy='causal_count_debt_v1'
    data['survival_prefix_counts']=torch.zeros(b,20,2)
    normal=model(**data)
    data['survival_prefix_counts']=torch.full((b,20,2),20.)
    overloaded=model(**data)
    torch.testing.assert_close(overloaded['audio_candidate_logits'],normal['audio_candidate_logits'],atol=0,rtol=0)
    assert torch.all(overloaded['no_event_log_prob']>=normal['no_event_log_prob'])
    no_history=model(**data,symbolic_keep=torch.zeros(b,1))
    torch.testing.assert_close(no_history['no_event_log_prob'],no_history['audio_no_event_log_prob'],atol=0,rtol=0)
