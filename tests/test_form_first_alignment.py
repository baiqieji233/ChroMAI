from fractions import Fraction as F
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from maichart.event_timing import EventTempo, EventPoint, seconds_at_beat, parse_event_chart_text
from maichart.form.data import organize_song, form_audio
from maichart.form import generate, curve
from maichart.form.clock import AUDIO_CURVE_AXIS, CHART_CURVE_AXIS, profile_seconds
from maichart.generation.conditions import GenerationCondition
from maichart.generation.event_audio import EventAudio, ACOUSTIC_DIM
from maichart.generation.event_compile import compile_event_points


TEMPO = [EventTempo(F(0),120)]


def pulse_audio():
    times = np.arange(200,dtype=np.float32)*.02
    features = np.zeros((200,ACOUSTIC_DIM),dtype=np.float32)
    features[50,0] = 1.
    features[:,95] = 1.
    return EventAudio(features,times,4.,120.)


@pytest.mark.parametrize('first,expected_beat', [(0.,F(2)),(.5,F(1)),(-1.,F(4))])
def test_offset_changes_the_sampled_chart_clock_and_export_without_moving_audio(first,expected_beat,monkeypatch):
    audio = pulse_audio()
    song = organize_song(audio,TEMPO,first,max_seconds=4)
    values = form_audio(song,song.spans[0].start,song.spans[-1].end)
    expected_peak = round((1-first)*50)
    assert int(values[:,0].argmax()) == expected_peak
    if first < 0:
        assert not values[:round(-first*50),:ACOUSTIC_DIM].any()
    from maichart.generation import intensity
    monkeypatch.setattr(intensity,'music_activity',lambda _:SimpleNamespace(times=audio.times,
                                               active=lambda:np.ones(200,dtype=bool)))
    class Planner:
        def __call__(self,audio_values,*_):
            peaks = audio_values[:,:,0]
            return {'onset_logits':torch.where(peaks>.5,8.,-8.),
                    'type_logits':torch.zeros(*peaks.shape,4),'sustain_seconds':torch.zeros_like(peaks)}
    plan = generate.plan_onsets(Planner(),song,GenerationCondition(),onset_merge_ms=0)
    assert len(plan['onsets']) == 1
    assert F(plan['onsets'][0]['beat']) == expected_beat
    assert plan['onsets'][0]['second'] == pytest.approx(1)
    text,_ = compile_event_points([EventPoint(expected_beat,'1/5')],tempo_changes=TEMPO,
                                  first=first,end_beat=song.spans[-1].end)
    parsed = parse_event_chart_text(text,5)
    assert parsed.first == first
    assert seconds_at_beat(parsed.points[0].beat,parsed.tempo_changes,parsed.first) == pytest.approx(1)


def test_positive_offset_does_not_fold_earlier_music_onto_chart_beat_zero(monkeypatch):
    audio = pulse_audio()
    song = organize_song(audio,TEMPO,1.5,max_seconds=4)
    values = form_audio(song,song.spans[0].start,song.spans[-1].end)
    assert not values[:,0].any()


def test_user_negative_offset_has_exact_silent_preroll_and_piecewise_tempo_clock():
    first = -1.82
    audio = pulse_audio()
    tempo = [EventTempo(F(0),120),EventTempo(F(2),240)]
    song = organize_song(audio,tempo,first,max_seconds=6)
    values = form_audio(song,F(0),song.spans[-1].end)
    assert values.shape[0] == 291
    assert not values[:91,:ACOUSTIC_DIM].any()
    assert int(values[:,0].argmax()) == 141
    assert seconds_at_beat(F(232,25),tempo,first) == pytest.approx(1.)
    text,_ = compile_event_points([EventPoint(F(232,25),'1')],tempo_changes=tempo,
                                  first=first,end_beat=F(14))
    parsed = parse_event_chart_text(text,5)
    assert parsed.first == -1.82
    assert seconds_at_beat(parsed.points[0].beat,parsed.tempo_changes,parsed.first) == pytest.approx(1.)


def test_training_dataset_uses_difficulty_first_and_records_the_alignment_contract(tmp_path):
    from maichart.form.data import FormDataset
    import json
    folder = tmp_path/'song'; folder.mkdir()
    (folder/'maidata.txt').write_text('&title=first fixture\n&wholebpm=120\n&first=-1.82\n&first_5=0.5\n'
                                    '&lv_5=13\n&inote_5=(120){4},1,2,3,E\n',encoding='utf-8')
    (folder/'track.wav').write_bytes(b'fixture')
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'songs':[{'song_id':'first','maidata_path':'song/maidata.txt',
        'audio_path':'song/track.wav','difficulties':[{'difficulty_index':5,'level_raw':'13'}]}]}),encoding='utf-8')
    dataset = FormDataset(manifest,cache_dir=tmp_path/'cache',audio_loader=lambda *_a,**_kw:pulse_audio())
    assert dataset.songs[0][0].first == .5
    assert dataset.audit['charts'][0]['first_override']['effective_first'] == .5
    example = dataset[0]
    assert int(example['audio'][:,0].argmax()) == 25


@pytest.mark.parametrize('first', [.5,-1.82])
def test_absolute_profile_is_converted_to_legacy_decoder_bin_clock(first):
    song = SimpleNamespace(audio=SimpleNamespace(duration_sec=5),tempo=TEMPO,first=first)
    span = SimpleNamespace(start=F(0),end=F(10))
    profile = {'time_axis':AUDIO_CURVE_AXIS,'curve_resolution_sec':1,
               'points':[{'time_sec':0,'intensity':0},{'time_sec':5,'intensity':1}]}
    values,mask,times = generate._curve_condition(song,span,profile,strength=0)
    absolute = times.numpy()+first
    active = (absolute>=0)&(absolute<5)
    np.testing.assert_allclose(values[:,0].numpy()[active],absolute[active]/5,atol=1e-6)
    assert not values[:,0].numpy()[~active].any()
    assert mask[:,0].all()


def test_saved_legacy_curve_and_canonical_curve_give_the_same_gate_coordinates():
    legacy = {'time_axis':CHART_CURVE_AXIS}
    canonical = {'time_axis':AUDIO_CURVE_AXIS}
    assert profile_seconds(legacy,np.array([1,2]),.5).tolist() == [.5,1.5]
    assert profile_seconds(canonical,np.array([1,2]),.5).tolist() == [1,2]


@pytest.mark.parametrize('axis,rest', [(AUDIO_CURVE_AXIS, (.9,1.1)),(CHART_CURVE_AXIS,(.4,.6))])
def test_rest_gate_uses_the_saved_curve_axis_with_positive_first(monkeypatch,axis,rest):
    audio = pulse_audio()
    song = organize_song(audio,TEMPO,.5,max_seconds=4)
    from maichart.generation import intensity
    monkeypatch.setattr(intensity,'music_activity',lambda _:SimpleNamespace(times=audio.times,
                                               active=lambda:np.ones(200,dtype=bool)))
    def planner(values,*_):
        peaks = values[:,:,0]
        return {'onset_logits':torch.where(peaks>.5,8.,-8.),'type_logits':torch.zeros(*peaks.shape,4),
                'sustain_seconds':torch.zeros_like(peaks)}
    profile = {'time_axis':axis,'points':[{'time_sec':0,'intensity':.8},{'time_sec':4,'intensity':.8}],
               'rest_regions':[{'start_sec':rest[0],'end_sec':rest[1]}]}
    planned = generate.plan_onsets(planner,song,GenerationCondition(),intensity_curve=profile,
                                   rhythm_snap_enabled=True,onset_merge_ms=0)
    assert not planned['onsets']


@pytest.mark.parametrize('first', [.5,-1.82])
def test_learned_curve_uses_actual_planner_frame_times_for_nonzero_first(tmp_path,monkeypatch,first):
    from maichart.form import checkpoint
    from maichart.form.codec import RationalCodec
    from maichart.generation import event_audio, music_intensity
    checkpoint_path = tmp_path/'model.pt'; checkpoint_path.write_bytes(b'fixture')
    audio = pulse_audio()
    monkeypatch.setattr(event_audio,'extract_event_audio',lambda *_a,**_kw:audio)
    monkeypatch.setattr(music_intensity,'audio_sha256',lambda _: 'test-audio')
    captured = {}
    class Planner:
        def eval(self): pass
        def __call__(self,audio_values,*_):
            from maichart.generation.event_audio import segment_frame_times
            stamps = segment_frame_times(first,audio.duration_sec)
            assert len(stamps) == audio_values.shape[1]
            captured['stamps'] = stamps
            target = torch.as_tensor(np.clip((stamps+2)/10,.01,.99),dtype=torch.float32)
            return {'load_curve':torch.logit(target)[None,:,None].expand(1,len(target),4)}
    payload = {'codec':RationalCodec().to_dict(),'capabilities':{'load_curve':{'steps':1,'contract':'load_curve_v1'}},
               'optimizer_steps':1,'blueprint_steps':1}
    monkeypatch.setattr(checkpoint,'load',lambda *_a,**_kw:(None,Planner(),None,payload))
    result = curve.generate_learned_profile(tmp_path/'song.wav',checkpoint_path,
                                          bpm=120,first=first,resolution_sec=.5)
    assert result['time_axis'] == AUDIO_CURVE_AXIS
    times = np.asarray([point['time_sec'] for point in result['points']])
    predicted = np.asarray(result['components']['event_density'])
    interior = (times>=max(first,0)+.25)&(times<audio.duration_sec-.25)
    np.testing.assert_allclose(predicted[interior],(times[interior]+2)/10,atol=1e-5)
    assert not predicted[times<max(first,0)].any()
