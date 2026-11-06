from fractions import Fraction
import pytest
from maichart.composer_v2.stream_codec import StreamCodec
from maichart.composer_v2.stream_generate import EventSampler,GrammarDeadEnd


def prefix(codec,number):
    return [codec.at_id,*[codec.ids['NUM:'+c] for c in str(number)]]


def test_reported_242_phase_exhaustion_does_not_admit_dead_integer():
    codec=StreamCodec.fit(['1']);after=Fraction(46655,192);end=Fraction(341)
    assert codec.integer_bounds(after,end)==(243,340)
    assert codec.ids['NUM:2'] not in codec.time_choices(prefix(codec,24),after,end)
    assert codec.ids['NUM:3'] in codec.time_choices(prefix(codec,24),after,end)
    assert codec.phase_id in codec.time_choices(prefix(codec,243),after,end)


@pytest.mark.parametrize('after,end',[
    (Fraction(-1),Fraction(1,10)),(Fraction(1,2),Fraction(3,4)),
    (Fraction(191,192),Fraction(1)),(Fraction(191,192),Fraction(2)),
    (Fraction(19199,192),Fraction(103)),(Fraction(191999,192),Fraction(1002)),
    (Fraction(1,2),Fraction(1,2)+Fraction(1,1000000)),
    (Fraction(9),Fraction(9)),(Fraction(10),Fraction(9)),
])
def test_every_admitted_decimal_prefix_has_a_valid_completion(after,end):
    codec=StreamCodec.fit(['1']);todo=[[]];seen=[]
    while todo:
        path=todo.pop();allowed=codec.time_choices(path,after,end)
        assert allowed, [codec.tokens[i] for i in path]
        for token in allowed:
            if token==codec.eos_id:continue
            if token in codec.phases:
                split=path.index(codec.phase_id)
                whole=int(''.join(codec.digits[i] for i in path[1:split]))
                time=whole+codec.phases[token];assert after<time<end;seen.append(time)
            else:todo.append([*path,token])
    expected={i+p for i in range(max(0,int(after)),max(0,int(end))+1) for p in codec.phases.values() if after<i+p<end}
    assert set(seen)==expected


def test_first_future_phase_remains_available_and_long_rests_are_not_capped():
    c=StreamCodec.fit(['1']);after=Fraction(43,4);end=Fraction(350)
    assert c.phase_id in c.time_choices(prefix(c,11),after,end)
    assert c.phase_id in c.time_choices(prefix(c,112),after,end)
    assert c.ids['F:191/192'] in c.time_choices(prefix(c,242)+[c.phase_id],Fraction(242)+Fraction(127,128),end)


def test_empty_grammar_diagnostic_contains_actual_prefix_without_model_call():
    s=EventSampler.__new__(EventSampler);s.codec=StreamCodec.fit(['1']);s.max_tokens=2048;s.cursor=Fraction(240)
    s.proposal_scope={'after':'46655/192','audio_end_beat':'341','target_prefix_length':1,'fixed_beat':None}
    with pytest.raises(GrammarDeadEnd) as caught:s.next([s.codec.tgt_id,*prefix(s.codec,242)],[])
    d=caught.value.diagnostic
    assert d['proposal_tokens']==['[AT]','NUM:2','NUM:4','NUM:2']
    assert d['after']=='46655/192' and d['allowed_token_count']==0


def test_native_conflict_can_revoke_latest_plain_tap_without_hold():
    from maichart.composer_v2.stream_generate import restore_prefix
    from maichart.event_timing import EventPoint,EventTempo
    from maichart.tempo_clock import TempoClock
    tempo=[EventTempo(Fraction(0),120.)]
    points=[EventPoint(Fraction(1),'1'),EventPoint(Fraction(9),'2'),EventPoint(Fraction(17),'3')]
    kept,guard,report=restore_prefix(points,tempo,TempoClock(tempo),100.,Fraction(18),('9','17','18'))
    assert kept==points[:2] and report['source_beat']=='17'
    assert report['reason']=='latest_native_conflicting_event' and report['revoked_events']==1
    assert report['removed_points']==[dict(beat='17',note_text='3')]
    assert points[-1].beat==17  # Original committed history was not mutated.
