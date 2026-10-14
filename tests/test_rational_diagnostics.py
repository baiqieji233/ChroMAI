from fractions import Fraction as F

from maichart.event_timing import EventPoint, EventTempo
from maichart.form.rational_diagnostics import rational_diagnostics


def test_diagnostics_distinguish_early_eos_declared_rests_and_failed_search():
    records=[
        {'span':0,'onsets':0,'has_allowed_positions':True,'status':{'reason':'eos'}},
        {'span':1,'onsets':0,'has_allowed_positions':False,'status':{'reason':'eos'}},
        {'span':2,'onsets':1,'status':{'reason':'invalid_position_prefix'}},
        {'span':3,'onsets':1,'status':{'reason':'position_search_exhausted','incomplete':True}},
    ]
    result=rational_diagnostics([EventPoint(F(0),'1'),EventPoint(F(37,997),'2'),
        EventPoint(F(20),'3')],records,[EventTempo(F(0),120)],0,12)
    assert result['empty_eos_span_indices']==[0]
    assert result['incomplete_spans']==[2,3]
    assert result['large_denominator_events']==1
    assert result['max_gap_sec']>9
    assert not result['musical_quality_verified']
