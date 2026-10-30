from dataclasses import replace
import importlib.util
from pathlib import Path
import pytest
import torch
from maichart.composer.model import build_model,resolve_composer_policy
from maichart.composer.checkpoint import migrate_audio_grounded,save,load,inspect_payload
from maichart.composer.phase_echo import PhaseEchoPrior,phase_echo_features,ECHO_LAGS
from maichart.composer.echo_metrics import EchoMetrics,echo_evidence_profile
from maichart.composer.action_guidance import action_logits_with_touch_guard
from test_composer_quality_integration import _fixture
from test_composer_audio_grounded_v3 import ATOMS,config


def fixture():
    old,data=_fixture(configuration_quality_policy='cooperative_v2',action_head_policy='conserved_grouped_v2',cross_event_relation_policy='relative_heads_v1')
    data['history_beat_ages']=torch.tensor([[2.,0.],[2.,0.]])
    new=migrate_audio_grounded(old,ATOMS,echo_prior_policy='phase_echo_v1').eval()
    return old,new,data


def test_zero_initialization_is_bitwise_equivalent_and_checkpoint_contract(tmp_path):
    old,new,data=fixture()
    before=old(**data);after=new(**data)
    for key in ('candidate_logits','action_logits','audio_candidate_logits','audio_action_logits','candidate_context'):
        assert torch.equal(before[key],after[key]),key
    assert not after['echo_lane_bias'].any()
    assert EchoMetrics().report(new)['echo_prior_calibration']['table_E_scaled']==[0.]*5
    p=tmp_path/'echo.pt'
    save(p,new,None,epoch=0,args={},manifest=tmp_path/'unused',atom_catalog=ATOMS)
    restored,payload=load(p)
    assert inspect_payload(payload)['model_config']['echo_prior_policy']=='phase_echo_v1'
    restored.eval();assert torch.equal(restored(**data)['action_logits'],after['action_logits'])
    old_path=tmp_path/'old.pt'
    save(old_path,old,None,epoch=0,args={},manifest=tmp_path/'unused',atom_catalog=ATOMS)
    payload=torch.load(old_path,weights_only=False);payload['model_config'].pop('echo_prior_policy')
    torch.save(payload,old_path)
    assert load(old_path)[0].config.echo_prior_policy=='off'
    assert resolve_composer_policy('echo_prior_policy',parent_config={'echo_prior_policy':'off'})=='phase_echo_v1'
    assert resolve_composer_policy('echo_prior_policy',parent_config={},resume=True)=='off'
    with pytest.raises(ValueError,match='finetune'):
        resolve_composer_policy('echo_prior_policy','phase_echo_v1',{'echo_prior_policy':'off'},resume=True)


def test_bias_gradient_symbolic_mask_category_mass_and_packed_paths():
    _,model,data=fixture()
    with torch.no_grad():
        model.echo_prior.table_E.fill_(.2);model.echo_prior.rtable.fill_(.3)
    full=model(**data)
    packed=model(**data,packed_actions=True)
    row,slot=packed['action_slot_indices'].unbind(-1)
    torch.testing.assert_close(packed['action_logits'],full['action_logits'][row,slot])
    for b in range(2):
        for s in range(2):
            direct=model.action_slot(full['action_context'][b],s,data['action_targets'][b,:s].tolist(),
                previous_heads=full['previous_heads'][b],echo_heads=full['echo_heads'][b])
            torch.testing.assert_close(direct,full['action_logits'][b,s],atol=2e-5,rtol=1e-5)
    logits=torch.randn(2,5);bias=torch.randn(2,8)
    shifted=model._echo_logits(logits,bias)
    for kind in range(3):
        ids=getattr(model,f'echo_kind_{kind}')
        torch.testing.assert_close(shifted[:,ids].logsumexp(-1),logits[:,ids].logsumexp(-1))
    assert torch.equal(shifted[:,0],logits[:,0])
    hidden=model(**data,symbolic_keep=torch.zeros(2,1))
    assert not hidden['echo_lane_bias'].any()
    torch.testing.assert_close(hidden['candidate_logits'],hidden['audio_candidate_logits'],atol=2e-6,rtol=1e-5)
    raw=full['candidate_logits']-full['audio_candidate_logits']
    # Normalizing log probabilities adds a common shift. The actual bounded
    # historical correction keeps every pairwise odds change within 2*bound.
    assert (raw.max(-1).values-raw.min(-1).values).max()<=2*model.config.history_residual_bound+1e-5
    model.zero_grad();torch.nn.functional.cross_entropy(full['action_logits'][:,0],torch.tensor([1,2])).backward()
    assert model.echo_prior.table_E.grad.abs().sum()>0


def test_evidence_unaffected_by_history_dropout_and_zero_scale_has_gradient():
    module=PhaseEchoPrior(16)
    lanes=torch.zeros(2,20,8,dtype=torch.bool);lanes[...,0]=True
    ages=torch.arange(19,-1,-1).float()[None].expand(2,-1)*.25
    valid=torch.ones(2,20,dtype=torch.bool);gaps=torch.tensor([[.25,.5]]*2)
    audio=torch.randn(2,2,16)
    a,r=module(lanes,ages,valid,gaps,audio,torch.ones(2,1))
    assert torch.equal(a,torch.zeros_like(a)) and torch.equal(r,torch.zeros_like(r))
    (a.sum()+r.sum()).backward()
    assert module.table_E.grad.abs().sum()>0 and module.rtable.grad.abs().sum()>0
    _,model,data=fixture();model.train();model.config.history_dropout=.8
    with torch.no_grad():model.echo_prior.table_E.fill_(.1)
    first=model(**data,symbolic_keep=torch.ones(2,1))['echo_lane_bias']
    second=model(**data,symbolic_keep=torch.ones(2,1))['echo_lane_bias']
    assert torch.equal(first,second)


def test_audio_reference_receives_no_echo():
    class Recorder:
        def __init__(self):self.extra=[]
        def action_slot(self,*a,**kwargs):
            self.extra.append(kwargs);return torch.zeros(4)
    m=Recorder()
    action_logits_with_touch_guard(m,torch.zeros(2),0,[],audio_context=torch.ones(2),enabled=True,
        echo_heads=torch.ones(8),touch_mask=torch.tensor([False,False,True,False]))
    assert 'echo_heads' in m.extra[0] and 'echo_heads' not in m.extra[1]


def test_touch_probability_and_stop_are_preserved_without_cross_event_policy():
    atoms=['1','2','1h[4:1]','2h[4:1]','1-5[4:1]','2-6[4:1]','A1','E7','C','Ch[4:1]']
    model=build_model(replace(config(echo_prior_policy='phase_echo_v1'),atom_vocab_size=len(atoms)+1),atoms)
    logits=torch.randn(3,len(atoms)+1)
    changed=model._echo_logits(logits,torch.randn(3,8)*4)
    assert not torch.equal(changed[:,1:7],logits[:,1:7])
    assert torch.equal(changed[:,7:],logits[:,7:])
    torch.testing.assert_close(changed.softmax(-1)[:,7:],logits.softmax(-1)[:,7:])
    torch.testing.assert_close(changed.softmax(-1)[:,0],logits.softmax(-1)[:,0])


def test_generated_metrics_match_original_evidence_script(tmp_path):
    original=Path(__file__).resolve().parents[1]/'.scratch/echo_evidence.py'
    if not original.exists():pytest.skip('local original evidence script absent')
    spec=importlib.util.spec_from_file_location('echo_reference',original)
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    text='&wholebpm=120\n&lv_5=13\n&inote_5=(120){4}'+','.join(['1','2','3','4']*10)+',E\n'
    chart=tmp_path/'maidata.txt';chart.write_text(text,encoding='utf8')
    _,_,expected=reference.analyse([chart])
    beats,lanes=reference.load(chart)
    profile=echo_evidence_profile([(b,'/'.join(str(x) for x in sorted(lanes[b]))) for b in beats])
    for e in range(5):
        assert profile['by_E'][str(e)]['count']==expected[(e,'n')]
        if expected[(e,'n')]:assert profile['by_E'][str(e)]['copy_rate']==pytest.approx(expected[(e,'hit')]/expected[(e,'n')])
