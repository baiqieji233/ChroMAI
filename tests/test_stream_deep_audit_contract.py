"""Root-cause contracts; execute after the complete audit repair is written."""
from fractions import Fraction as F
from types import SimpleNamespace
import json

import pytest

from maichart.event_timing import EventPoint,EventTempo
from maichart.composer.guard import EventGuard


def test_augmentation_judge_keeps_intervening_long_slide_interactions():
    from maichart.composer.stream_augmentation import _validate_window
    from maichart.composer.stream_data import StreamStep,NOTE
    from maichart.event_timing import seconds_at_beat
    tempo=[EventTempo(F(0),205.)]
    guard=EventGuard(tempo,0.,160.,min_same_head_sec=0.,physical_policy='runtime_hard_v2')
    for beat,text in ((F(63623,192),'1bx-5[16.2406##0.2256]'),(F(346),'3/8-4[8:1]')):
        guard.commit(EventPoint(beat,text))
    beat=F(699,2);second=float(seconds_at_beat(beat,tempo,0.))
    step=StreamStep(NOTE,F(346),beat,float(seconds_at_beat(F(346),tempo,0.)),second,'5')
    with pytest.raises(ValueError,match='native hard conflict'):_validate_window(guard,[step])


def test_augmentation_uses_runtime_policy_not_static_tap_on_slide_references():
    from maichart.composer.stream_augmentation import _enforced_hard_keys
    from maichart.generation.conflict_detector import ConflictReport
    issue=ConflictReport(True,'miacode_TapOnSlide',1.,[],'hard','reference',evidence='miacode_static:test')
    guard=EventGuard([EventTempo(F(0),120.)],0.,10.,intercept_tap_on_slide=True,physical_policy='runtime_hard_v2')
    assert not _enforced_hard_keys(SimpleNamespace(conflicts_detected=[issue]),guard)


def test_native_augmentation_seed_also_keeps_intervening_actions():
    from maichart.composer.stream_augmentation import _native_prefix,_validate_window
    from maichart.composer.stream_recovery import _guard
    from maichart.composer.stream_data import StreamStep,NOTE
    from maichart.generation.sustain_state import SustainStateIndex
    from maichart.event_timing import seconds_at_beat
    tempo=[EventTempo(F(0),205.)]
    points=[EventPoint(F(63623,192),'1bx-5[16.2406##0.2256]'),EventPoint(F(346),'3/8-4[8:1]'),EventPoint(F(699,2),'5')]
    native=SustainStateIndex(points,tempo,0.)
    song=SimpleNamespace(tempo=tempo,first=0.,audio=SimpleNamespace(duration_sec=160.))
    dataset=SimpleNamespace(base=SimpleNamespace(songs=[(song,None,None,None,native)]),
                            views=[SimpleNamespace(context_points=points)])
    source=[];previous=F(0)
    for point in points:
        source.append(StreamStep(NOTE,previous,point.beat,float(seconds_at_beat(previous,tempo,0.)),
                                 float(seconds_at_beat(point.beat,tempo,0.)),point.note_text))
        previous=point.beat
    _,prefix=_native_prefix(dataset,0,source,2)
    assert [step.text for step in prefix]==[point.note_text for point in points[:2]]
    guard=_guard(song,prefix,len(prefix),physical_policy='runtime_hard_v2')
    with pytest.raises(ValueError,match='native hard conflict'):_validate_window(guard,[source[2]])


def test_final_report_preserves_findings_but_uses_actual_guard_actions():
    from maichart.composer.stream_generate import physical_policy_report
    from maichart.generation.conflict_detector import ConflictReport
    issues=[ConflictReport(True,'miacode_SlideHeadTap',1.,[],'hard','runtime finding',
                           evidence='miacode_runtime:test',action='report'),
            ConflictReport(True,'miacode_TapOnSlide',2.,[],'hard','static reference',
                           evidence='miacode_static:test',action='enforce'),
            ConflictReport(True,'miacode_MultiTouch',3.,[],'hard','runtime hard',
                           evidence='miacode_runtime:test',action='enforce')]
    result=SimpleNamespace(conflicts_detected=issues,remaining_conflicts=issues,
                           to_dict=lambda:dict(hard=3,remaining_hard=3,remaining_actionable_hard=2))
    guard=EventGuard([EventTempo(F(0),120.)],0.,10.,intercept_tap_on_slide=True,physical_policy='runtime_hard_v2')
    report=physical_policy_report(result,guard)
    assert report['hard']==3 and report['remaining_hard']==3
    assert report['remaining_actionable_hard']==1 and report['remaining_report_only_hard']==2
    assert [row['action'] for row in report['details']]==['report','report','enforce']
    assert report['details'][1]['engine_action']=='enforce'
    assert [row['source'] for row in report['details']]==['native_runtime','static_reference','native_runtime']


def test_sampling_observation_records_distribution_changes_with_bounded_examples():
    from maichart.composer.stream_generate import record_sampling_trace
    observation={}
    first=dict(family=1,text='1',accepted=False,rejection_reason='occupied')
    final=dict(family=2,text='2h[4:1]',accepted=True)
    trace=dict(first_proposal=first,final_proposal=final,changed_time=False,configuration_changed=True,
               onsets=[dict(trials=[first,final])])
    for _ in range(20):record_sampling_trace(observation,trace)
    assert observation['counts']['changed_family']==20
    assert observation['counts']['changed_time']==0
    assert observation['counts']['rejected_proposals']==20
    assert observation['accepted_family_counts']=={'2':20}
    assert len(observation['examples'])==12


def test_validation_panel_forwards_real_generation_settings_and_distinct_songs(tmp_path,monkeypatch):
    from maichart.composer import stream_generate as generate
    from maichart.generation.conditions import GenerationCondition
    condition=GenerationCondition(level=13.4)
    def record(source):return (SimpleNamespace(audio=SimpleNamespace(source=source)),None,None,None,None,condition)
    dataset=SimpleNamespace(base=SimpleNamespace(songs=[record('one'),record('one'),record('two')]))
    monkeypatch.setattr(generate,'load',lambda *args:(object(),{}))
    calls=[]
    def run(args,**kwargs):calls.append(args);return dict(stop_reason='audio_end')
    monkeypatch.setattr(generate,'generate',run)
    report=generate.evaluate_panel('unused',dataset,tmp_path,count=2,seeds=(73,),temperature=.6,
        generation_settings=dict(phrase_candidates=2,phrase_write_beats=6.,phrase_commit_beats=3.,
                                 pointer_temperature=0.,action_temperature=None,audio_tracking_strength=.3))
    assert len(calls)==2 and report['selected_song_indices']==[0,2]
    for args in calls:
        assert args.pointer_temperature==0. and args.action_temperature==.6
        assert args.phrase_candidates==2 and args.phrase_write_beats==6. and args.phrase_commit_beats==3.
        assert args.audio_tracking_strength==.3
    assert report['pointer_temperature']==0. and report['action_temperature']==.6


def test_explicit_validation_cannot_claim_unseen_copied_audio(tmp_path):
    from maichart.composer.evaluation import assert_unseen
    audio=tmp_path/'original.bin';duplicate=tmp_path/'copy.bin'
    audio.write_bytes(b'same music');duplicate.write_bytes(audio.read_bytes())
    manifests=[]
    for i,path in enumerate((audio,duplicate)):
        manifest=tmp_path/f'{i}.json'
        manifest.write_text(json.dumps(dict(songs=[dict(song_id=str(i),audio_path=str(path))])),encoding='utf-8')
        manifests.append(manifest)
    with pytest.raises(ValueError,match='overlap'):assert_unseen(*manifests,cache_dir=tmp_path/'cache')


def test_cpu_zero_resolves_threads_before_attempting_data(tmp_path,monkeypatch):
    from maichart.composer import stream_train as train
    monkeypatch.setattr(train.torch,'get_num_threads',lambda:8)
    threads=[];monkeypatch.setattr(train.torch,'set_num_threads',threads.append)
    monkeypatch.setattr(train,'_dataset',lambda *a,**kw:[])
    args=train.parser().parse_args(['--manifest','unused','--checkpoint-dir',str(tmp_path/'run'),
                                  '--device','cpu','--cpu-threads','0'])
    with pytest.raises(ValueError,match='No supervised'):train._train(args)
    assert threads==[8] and args.effective_cpu_threads==8


def test_completed_resume_requires_a_larger_total_epoch_target(tmp_path,monkeypatch):
    from maichart.composer import stream_train as train
    monkeypatch.setattr(train.checkpoint,'load',lambda *a,**kw:(None,dict(epoch=10,training_state={'epoch_complete':True})))
    args=train.parser().parse_args(['--manifest','unused','--checkpoint-dir',str(tmp_path/'run'),
                                  '--device','cpu','--resume','unused','--epochs','10'])
    with pytest.raises(ValueError,match='total target'):train._train(args)


def test_limited_validation_does_not_only_visit_the_first_song_windows(monkeypatch):
    from maichart.composer import stream_train as train
    class Dataset:
        def __len__(self):return 100
        def __getitem__(self,index):return index
    monkeypatch.setattr(train,'collate_stream',lambda rows:rows)
    args=SimpleNamespace(seed=0,validation_batch_size=2,validation_max_batches=3)
    loader=train._loader(Dataset(),args,0,validation=True)
    selected=[index for batch in loader for index in batch]
    assert len(selected)==6 and selected[0]==0 and selected[-1]==99
    assert any(30<=index<70 for index in selected)
