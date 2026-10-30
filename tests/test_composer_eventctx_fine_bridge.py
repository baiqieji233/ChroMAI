import torch
from maichart.composer.checkpoint import migrate_audio_grounded,save,load
from test_composer_compact_integration import fixture


def test_existing_phrase_fine_target_reaches_pointer_with_neutral_migration(tmp_path):
    base,data=fixture()
    model=migrate_audio_grounded(base,base.action_factors.atom_catalog,
        fine_timing_policy='relative_phrase_v1').eval()
    assert model.fine_timing_gain.numel()==1 and model.fine_timing_gain==0
    old=base(**data)
    initial=model(**data)
    for key in ('candidate_logits','action_logits','audio_action_logits'):
        torch.testing.assert_close(initial[key],old[key],atol=0,rtol=0)
    with torch.no_grad():
        model.planner.phrase.bias[8]=2.
    data['candidate_features'][:,0,120]=.125
    output=model(**data)
    (-output['candidate_logits'][:,0].mean()).backward()
    assert torch.isfinite(model.fine_timing_gain.grad) and model.fine_timing_gain.grad.abs()>0
    path=tmp_path/'fine.pt'
    save(path,model,None,epoch=0,args={},manifest=tmp_path/'unused',atom_catalog=model.action_factors.atom_catalog)
    restored,_=load(path)
    assert hasattr(restored,'fine_timing_gain')
