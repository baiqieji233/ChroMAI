from fractions import Fraction as F
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from maichart.composer.event_set_codec import EventSetCodec
from maichart.composer.events import event_metadata
from maichart.composer.stream_literal import pack_literal_type_blocks
from maichart.composer.stream_model import StreamComposer,StreamComposerConfig
from maichart.composer.guard import EventGuard
from maichart.composer import guard as guard_module,stream_augmentation
from maichart.event_timing import EventPoint,EventTempo


class _TokenCounterCell(nn.Module):
    def forward(self,previous,hidden):
        value=torch.cat((previous[...,:1]+1,previous[...,1:]*0),-1)
        return value,value[:,-1][None]


def _model(atoms=('A1','A2','A3','Ch[4:1]'), *, maximum=8):
    codec=EventSetCodec.fit(atoms)
    config=StreamComposerConfig(d_model=8,nhead=2,layers=1,dim_feedforward=16,
        dropout=0.,event_family_policy='semantic_mass_v1',literal_output_policy='type_blocks_v1',
        max_event_atoms=maximum)
    return StreamComposer(config,codec).eval()


def _counter_model():
    model=_model();model.literal_cell=_TokenCounterCell()
    with torch.no_grad():
        model.atom_embedding.weight.zero_()
        for text,value in [('A1',1.),('A2',2.),('A3',3.),('Ch[4:1]',4.)]:
            model.atom_embedding.weight[model.codec._atoms[text],0]=value
        model.literal_head.weight.zero_();model.literal_head.bias.zero_()
        for text,value in [('A1',12.),('A2',11.),('A3',10.),('Ch[4:1]',12.)]:
            model.literal_head.bias[model.codec._atoms[text]]=value
        model.literal_stop_head.weight.zero_();model.literal_stop_head.bias.fill_(10.)
        model.literal_stop_head.weight[3,0]=10.;model.literal_stop_head.bias[3]=-25.
    return model


def test_type_block_packing_preserves_multiple_touch_before_touch_hold():
    model=_model();target=torch.tensor([[model.codec._atoms['A1'],model.codec._atoms['A2'],
        model.codec._atoms['Ch[4:1]'],0]])
    groups=pack_literal_type_blocks(target,model.semantic_atom_families)
    assert groups[3][0].tolist()==[0]
    assert groups[3][1].tolist()==[[model.codec._atoms['A1'],model.codec._atoms['A2'],0]]
    assert groups[3][2].tolist()==[[0]]
    assert groups[4][1].tolist()==[[model.codec._atoms['Ch[4:1]'],0]]
    assert groups[4][2].tolist()==[[model.codec._atoms['A1'],model.codec._atoms['A2']]]
    assert all(not groups[i][0].numel() for i in (0,1,2))


def test_compound_family_can_finish_touch_block_before_required_hold_type():
    model=_counter_model()
    event=model.decode_event(torch.zeros(8),shape_override=0,family_override=24,temperature=0.)
    assert event is not None and event['text']=='A1/A2/Ch[4:1]'
    assert event['family']==24
    assert event_metadata(event['text']).type_counts[3:]==(2,1)


def test_packed_and_direct_block_likelihood_match_and_stop_receives_gradient():
    model=_counter_model();target=torch.tensor([[model.codec._atoms['A1'],model.codec._atoms['A2'],
        model.codec._atoms['Ch[4:1]'],0]])
    contexts=torch.zeros(1,8,requires_grad=True);families=torch.tensor([24])
    groups=pack_literal_type_blocks(target,model.semantic_atom_families)
    packed,stop=model.literal_block_nll(contexts,target,families,groups,return_stop=True)
    direct=model.literal_block_nll(contexts,target,families)
    torch.testing.assert_close(packed,direct)
    assert torch.isfinite(packed).all() and (stop>0).all()
    packed.sum().backward()
    assert model.literal_stop_head.weight.grad[3].abs().sum()>0
    assert model.literal_head.bias.grad.abs().sum()>0


def test_detail_temperature_does_not_change_the_stop_hazard_temperature():
    model=_counter_model();calls=[]
    def choose(logits,sampling_temperature=None):
        if sampling_temperature is not None:calls.append(sampling_temperature)
        return int(logits.argmax(-1))
    first=model._decode_literal_blocks(torch.zeros(1,8),24,choose,.2)
    assert calls and set(calls)=={1.}
    calls.clear()
    second=model._decode_literal_blocks(torch.zeros(1,8),24,choose,2.)
    assert first==second and set(calls)=={1.}


def test_literal_total_budget_is_shared_across_type_blocks():
    model=_counter_model();model.config.max_event_atoms=2
    assert model.decode_event(torch.zeros(8),shape_override=0,family_override=24,temperature=0.) is None


def test_duplicate_touch_variants_are_masked_across_blocks_but_distinct_pads_remain():
    model=_model(('E3','E3h[4:1]','Ch[4:1]','C','Cf'))
    with torch.no_grad():
        model.literal_head.weight.zero_();model.literal_head.bias.fill_(-20.)
        model.literal_head.bias[model.codec._atoms['E3']]=20.
        model.literal_head.bias[model.codec._atoms['E3h[4:1]']]=20.
        model.literal_head.bias[model.codec._atoms['Ch[4:1]']]=10.
        model.literal_stop_head.weight.zero_();model.literal_stop_head.bias.fill_(20.)
    event=model.decode_event(torch.zeros(8),shape_override=0,family_override=24,temperature=0.)
    assert set(event_metadata(event['text']).atoms)=={'E3','Ch[4:1]'}
    assert model._literal_bits[model.codec._atoms['C']]==model._literal_bits[model.codec._atoms['Cf']]
    assert len(event_metadata(event['text']).heads)==2


def test_type_stop_prior_is_fitted_from_real_block_and_extra_counts():
    model=_model();model.initialize_literal_stop_prior([1,1,1,90,8],[0,0,0,10,2])
    torch.testing.assert_close(model.literal_stop_head.bias[3],torch.log(torch.tensor((90+.5)/(10+.5))))
    assert StreamComposerConfig.from_dict({}).literal_output_policy=='flat_family_v1'


def _issue(*,action='enforce',kind='miacode_Overlap',evidence='miacode_static:test'):
    return SimpleNamespace(conflict_type=kind,time=.5,severity='hard',action=action,
        evidence=evidence,notes=[SimpleNamespace(beat=F(0),raw='1h[4:2]',note_type='hold',button=1,time=0.,duration=1.),
            SimpleNamespace(beat=F(1),raw='1',note_type='tap',button=1,time=.5,duration=0.)])


def test_guard_rejects_new_static_same_pad_overlap_and_preserves_old_findings(monkeypatch):
    old=_issue();new=_issue();new.notes=[old.notes[0],
        SimpleNamespace(beat=F(1),raw='new',note_type='tap',button=1,time=.5,duration=0.)]
    def detect(points,*args):
        return SimpleNamespace(conflicts_detected=[old]+([new] if points[-1].note_text=='2' else []),coverage={})
    monkeypatch.setattr(guard_module,'detect_chart_conflicts',detect)
    guard=EventGuard([EventTempo(F(0),120.)],0.,10.,touch_layout_policy='open')
    guard.commit(EventPoint(F(0),'1h[1:1]'))
    assert not guard.accepts(EventPoint(F(1),'2'))
    assert guard.accepts(EventPoint(F(1),'3'))
    assert len(guard.prefix)==1


def test_augmentation_and_generation_use_the_same_enforced_static_overlap_set():
    guard=EventGuard([EventTempo(F(0),120.)],0.,10.,touch_layout_policy='open')
    result=SimpleNamespace(conflicts_detected=[_issue(),_issue(action='report'),
        _issue(kind='miacode_SlideHeadTap'),_issue(kind='miacode_SlideTooFast')])
    keys=stream_augmentation._enforced_hard_keys(result,guard)
    assert len(keys)==1 and next(iter(keys))[0]=='miacode_Overlap'


def test_open_touch_layout_does_not_ban_intersecting_touch_sensor_circles():
    guard=EventGuard([EventTempo(F(0),120.)],0.,10.,touch_layout_policy='open')
    assert guard.accepts(EventPoint(F(1),'B3/B4/B5/B6/E3/E4/E5'))


def test_native_near_tap_static_overlap_keeps_the_explicit_zero_cooldown_contract():
    tempo=[EventTempo(F(0),120.)]
    points=[EventPoint(F(0),'1'),EventPoint(F(1,16),'1')]
    guard=EventGuard(tempo,0.,10.,min_same_head_sec=0.,touch_layout_policy='open')
    native=guard_module.detect_chart_conflicts(points,tempo,0.,guard.config)
    assert any(issue.conflict_type=='miacode_Overlap' and issue.evidence.startswith('miacode_static:')
        for issue in native.conflicts_detected)
    guard.commit(points[0])
    assert guard.accepts(points[1])


def test_native_static_button_a_ring_reference_does_not_merge_open_physical_heads():
    tempo=[EventTempo(F(0),120.)]
    points=[EventPoint(F(0),'1'),EventPoint(F(1,16),'A1')]
    guard=EventGuard(tempo,0.,10.,touch_layout_policy='open')
    native=guard_module.detect_chart_conflicts(points,tempo,0.,guard.config)
    assert any(issue.conflict_type=='miacode_Overlap' and issue.evidence.startswith('miacode_static:')
        for issue in native.conflicts_detected)
    guard.commit(points[0])
    assert guard.accepts(points[1])


@pytest.mark.parametrize('held,tap',[('7h[16:3]','7'),('Ch[16:3]','C')])
def test_native_real_sustained_head_overlap_is_rejected(held,tap):
    from maichart.generation.miacode_native import analyze_inote
    raw=analyze_inote(f'(120){{16}}{held},{tap},',first=0.)
    assert any(reference['kind']=='Overlap' and reference['alert_level']=='Muri'
        for reference in raw['static_references'])
    guard=EventGuard([EventTempo(F(0),120.)],0.,10.,min_same_head_sec=0.,touch_layout_policy='open')
    guard.commit(EventPoint(F(0),held))
    assert not guard.accepts(EventPoint(F(1,4),tap))
