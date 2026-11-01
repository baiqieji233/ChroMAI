"""Aligned landing contracts. Run only with the coordinated final test batch."""
from fractions import Fraction as F
from types import SimpleNamespace

import torch

from maichart.composer.event_set_codec import EventSetCodec
from maichart.composer.stream_candidate_state import candidate_state_at
from maichart.composer.stream_data import StreamDataset,StreamStep
from maichart.composer.stream_augmentation import _rebuild_fallback_candidate_states
from maichart.composer.stream_phrase import (PhraseIndex, prepare_candidate_phrase_inputs,
    collate_phrase_sequences,collate_candidate_phrase_sequences,share_aligned_phrase_sequences)
from maichart.composer.stream_phrase_model import OrderedPhraseContext
from maichart.event_timing import EventPoint, EventTempo
from maichart.generation.sustain_state import SustainStateIndex
from maichart.generation.action_state_spec import STATE_DIM


def _bank():
    beats = torch.arange(0.,24.,.1,dtype=torch.float64)
    return torch.zeros(len(beats),238),beats/2,beats


def _note(start,end,text):
    return StreamStep(2,F(start),F(end),float(start)/2,float(end)/2,text)


def test_landing_uses_only_completed_prefix_and_new_departure_reference():
    codec=EventSetCodec.fit(['1','2'])
    steps=[_note(0,1,'1'),_note(1,2,'2')]
    index=PhraseIndex(_bank(),24.,steps,sequence_codec=codec)
    # Silence gives an eight-beat first music segment. Its first committed
    # event becomes a reference only after the landing crosses that boundary.
    before=prepare_candidate_phrase_inputs(index,[0],[8.5])
    after_first=prepare_candidate_phrase_inputs(index,[1],[8.5])
    after_both=prepare_candidate_phrase_inputs(index,[2],[8.5])
    assert (before['candidate_phrase_sequence_inverse']<0).all()
    assert (after_first['candidate_phrase_sequence_inverse']<0).all()
    assert (after_both['candidate_phrase_sequence_inverse']>=0).any()
    # At the departure cursor, there is no earlier reference segment.
    departure=index.sequence_rows([2],[2.])
    assert (departure['phrase_sequence_inverse']<0).all()


def test_future_target_configuration_cannot_enter_landing_inputs():
    codec=EventSetCodec.fit(['1','2'])
    original=PhraseIndex(_bank(),24.,[_note(0,1,'1'),_note(1,2,'1')],sequence_codec=codec)
    changed=PhraseIndex(_bank(),24.,[_note(0,1,'1'),_note(1,2,'2')],sequence_codec=codec)
    left=prepare_candidate_phrase_inputs(original,[1],[8.5])
    right=prepare_candidate_phrase_inputs(changed,[1],[8.5])
    for field in ('phrase_progress','phrase_memory_features','phrase_sequence_inverse',
                  'phrase_sequence_atoms','phrase_sequence_mask'):
        torch.testing.assert_close(left['candidate_'+field],right['candidate_'+field],rtol=0,atol=0)


def test_departure_and_landing_share_one_reference_encoding_table():
    codec=EventSetCodec.fit(['1','2'])
    index=PhraseIndex(_bank(),24.,[_note(0,1,'1'),_note(1,2,'2')],sequence_codec=codec)
    item=index.sequence_rows([2],[8.5])
    item.update(prepare_candidate_phrase_inputs(index,[2],[8.5]))
    item['history_kind']=torch.zeros(1,dtype=torch.long)
    output=collate_phrase_sequences([item])
    output.update(collate_candidate_phrase_sequences([item]))
    share_aligned_phrase_sequences([item],output)
    assert output['candidate_phrase_sequence_shared'] is True
    assert output['candidate_phrase_sequence_atoms'] is output['phrase_sequence_atoms']
    assert output['candidate_phrase_sequence_inverse'].max() < len(output['phrase_sequence_kind'])


def test_aligned_read_moves_with_relative_beat_and_has_null_option():
    module=OrderedPhraseContext(4,4,aligned=True)
    with torch.no_grad():
        module.alignment_query.weight.zero_();module.alignment_query.bias.zero_()
        module.alignment_key.weight.zero_();module.alignment_key.bias.zero_()
    states=torch.tensor([[[1.,0.,0.,0.],[0.,1.,0.,0.]]])
    batch=dict(phrase_sequence_inverse=torch.zeros(1,2,1,dtype=torch.long),
        phrase_sequence_mask=torch.ones(1,2,dtype=torch.bool),
        phrase_sequence_beats=torch.tensor([[1.,3.]],dtype=torch.float64),
        phrase_memory_start_beat=torch.zeros(1,2,1),
        phrase_memory_end_beat=torch.full((1,2,1),4.),
        phrase_start_beat=torch.full((1,2),4.),phrase_end_beat=torch.full((1,2),8.),
        step_beat=torch.tensor([[5.,7.]]))
    anchor=torch.zeros(1,2,4);style=torch.zeros_like(anchor)
    values,null,_=module.aligned_read(batch,states,anchor,style,torch.ones(1,2,1,dtype=torch.bool))
    assert values[0,0,0]>values[0,0,1]
    assert values[0,1,1]>values[0,1,0]
    assert torch.all((null>0)&(null<1))


def test_selected_state_is_native_landing_query_with_future_hidden():
    tempo=[EventTempo(F(0),120.)]
    prefix=EventPoint(F(1),'1h[4:40]')
    future=EventPoint(F(2),'7h[4:40]')
    stream=[_note(0,1,prefix.note_text),StreamStep(3,F(1),F(3,2),.5,.75,'')]
    with_future=SustainStateIndex([prefix,future],tempo,0.)
    committed=SustainStateIndex([prefix],tempo,0.)
    actual,mask=candidate_state_at(with_future,stream,1)
    expected,expected_mask=committed.at(F(3,2),include_current=True,
        visible_before_beat=F(1)+F(1,10**12),visibility_version='causal_v2',include_metadata=False)[:2]
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    torch.testing.assert_close(mask,expected_mask,rtol=0,atol=0)


def test_selected_native_state_changes_when_slide_lifetime_expires():
    tempo=[EventTempo(F(0),120.)]
    source=EventPoint(F(1),'1-5[0##1]')
    index=SustainStateIndex([source],tempo,0.)
    note=_note(0,1,source.note_text)
    early=[note,StreamStep(3,F(1),F(3,2),.5,.75,'')]
    late=[note,StreamStep(3,F(1),F(5),.5,2.5,'')]
    state_early,mask_early=candidate_state_at(index,early,1)
    state_late,mask_late=candidate_state_at(index,late,1)
    assert not torch.equal(state_early.masked_fill(mask_early[:,None],0.),
                           state_late.masked_fill(mask_late[:,None],0.))


def test_rebuilt_landing_phrase_refreshes_candidate_clock():
    codec=EventSetCodec.fit(['1','2'])
    steps=[_note(0,1,'1'),_note(1,2,'2')]
    index=PhraseIndex(_bank(),24.,steps,sequence_codec=codec)
    dataset=object.__new__(StreamDataset)
    dataset.phrase_context_policy='aligned_phrase_v3'
    dataset._phrase_index=lambda _song:index
    row=dict(step_beat=torch.tensor([0.,1.],dtype=torch.float64),
             target_beat=torch.tensor([1.,2.],dtype=torch.float64),
             candidate_step_beat=torch.tensor([99.,99.],dtype=torch.float64))
    dataset.rebuild_phrase_row(row,0,steps,0,0,2)
    torch.testing.assert_close(row['candidate_step_beat'],row['target_beat'],rtol=0,atol=0)
    assert row['candidate_step_beat'].data_ptr()!=row['target_beat'].data_ptr()


def test_fallback_candidate_state_rebuilds_full_prefix_with_strict_visibility():
    tempo=[EventTempo(F(0),120.)]
    song=SimpleNamespace(tempo=tempo,first=0.)
    dataset=SimpleNamespace(base=SimpleNamespace(songs=[(song,)]))
    modified=[_note(0,1,'1h[4:40]'),
              StreamStep(3,F(1),F(3,2),.5,.75,''),
              _note(F(3,2),2,'7h[4:40]')]
    row=dict(candidate_action_state=torch.zeros(3,1,STATE_DIM),
             candidate_action_state_mask=torch.ones(3,1,dtype=torch.bool))
    _rebuild_fallback_candidate_states(dataset,row,modified,0,0,3)
    prefix_index=SustainStateIndex([EventPoint(F(1),'1h[4:40]')],tempo,0.)
    expected,expected_mask=candidate_state_at(prefix_index,modified,2)
    valid=row['candidate_action_state_mask'][2]
    torch.testing.assert_close(row['candidate_action_state'][2,~valid],expected[~expected_mask],rtol=0,atol=0)
    assert (~valid).any()
