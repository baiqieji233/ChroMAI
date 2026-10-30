from fractions import Fraction
from types import SimpleNamespace
import json

import pytest
import torch

from maichart.composer.guard import EventGuard,_event_syntax_error
from maichart.composer.event_set_codec import EventSetCodec
from maichart.composer.stream_model import StreamComposer,StreamComposerConfig
from maichart.composer import stream_generate as generate
from maichart.event_timing import EventPoint,EventTempo
from maichart.generation.event_compile import compile_event_points


@pytest.mark.parametrize('text',['E5/E5','E6/E6','E5/E5f','C/Ch[4:1]','1/1h[4:1]'])
def test_export_duplicate_heads_rejected_before_commit(text):
    guard=EventGuard([EventTempo(Fraction(0),120.)],0.,10.,touch_layout_policy='open')
    point=EventPoint(Fraction(1),text)
    assert not guard.accepts(point)
    assert guard.last_reason=='duplicate_event_head' and not guard.prefix
    with pytest.raises(ValueError,match='duplicate head'):
        compile_event_points([point],tempo_changes=guard.tempo,first=0.,end_beat=Fraction(4),strict_notes=True)


@pytest.mark.parametrize('text',['E5/E6','1/A1','1-4[8:1]*-6[8:1]'])
def test_distinct_touch_button_and_shared_branch_head_remain_supported(text):
    guard=EventGuard([EventTempo(Fraction(0),120.)],0.,10.,touch_layout_policy='open')
    assert _event_syntax_error(text) is None
    if text!='1/A1':assert guard.accepts(EventPoint(Fraction(1),text))


def test_literal_decode_masks_same_head_across_modifier_variants():
    codec=EventSetCodec.fit(['E5','E5f','E6'])
    config=StreamComposerConfig(d_model=8,nhead=2,layers=1,dim_feedforward=16,max_event_atoms=3,event_family_policy='legacy_v1')
    model=StreamComposer(config,codec).eval()
    with torch.no_grad():
        model.literal_head.weight.zero_();model.literal_head.bias.fill_(-10.)
        model.literal_head.bias[codec._atoms['E5']]=20.
        model.literal_head.bias[codec._atoms['E5f']]=19.
        model.literal_head.bias[codec._atoms['E6']]=18.
        model.literal_head.bias[0]=0.
    result=model.decode_event(torch.zeros(8),shape_override=0,temperature=0)
    assert result['text']=='E5/E6'


def test_bad_panel_artifact_is_recorded_and_remaining_seeds_run(tmp_path,monkeypatch):
    monkeypatch.setattr(generate,'load',lambda *a:(object(),{}))
    condition=SimpleNamespace(level=12.,to_dict=lambda:{'level':12.})
    dataset=SimpleNamespace(base=SimpleNamespace(songs=[(object(),None,None,None,None,condition)]))
    calls=[]
    def run(args,**kwargs):
        calls.append(args.seed)
        if args.seed==0:raise ValueError('duplicate head E5 x2')
        return {'event_count':1}
    monkeypatch.setattr(generate,'generate',run)
    result=generate.evaluate_panel('unused',dataset,tmp_path,count=1,seeds=(0,1))
    assert calls==[0,1] and result['status']=='completed_with_failures' and result['failed_runs']==1
    assert json.loads((tmp_path/'song_0_seed_0/generation_failure.json').read_text())['error']=='duplicate head E5 x2'
