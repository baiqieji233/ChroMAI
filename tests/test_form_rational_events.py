from fractions import Fraction as F
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from maichart.event_timing import EventPoint, EventTempo, seconds_at_beat
from maichart.form.codec import RationalCodec
from maichart.form.data import FormDataset, organize_song, input_example
from maichart.form.rational_audio import PROFILE, extract_rational_audio, audio_queries, beat_at_second, _tempo_hint
from maichart.form.rational_generate import PositionGrammar, compose_span
from maichart.form.rational_planner import RationalPlanner, RationalPlannerConfig, rational_planner_loss


@pytest.fixture
def source(tmp_path):
    import soundfile as sf
    sr=22050
    signal=np.zeros(sr*3,dtype=np.float32)
    for time in (.117,.343,.81,1.271,2.081):
        start=round(time*sr)
        length=round(sr*.06)
        t=np.arange(length)/sr
        signal[start:start+length]=np.cos(2*np.pi*880*t)*np.exp(-t/.01)
    audio=tmp_path/'track.wav'
    sf.write(audio,signal,sr)
    chart=tmp_path/'maidata.txt'
    chart.write_text('&title=rational\n&wholebpm=190\n&lv_5=13\n&inote_5=(190){32}1,2,3,4,5,6,7,8,{20}1,2,3,E\n',encoding='utf-8')
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'songs':[dict(song_id='rational',maidata_path=str(chart),audio_path=str(audio),
        difficulties=[dict(difficulty_index=5,level_raw='13')])]}),encoding='utf-8')
    return audio,manifest


def test_native_audio_queries_are_nonuniform_and_do_not_use_legacy_frames(source,tmp_path,monkeypatch):
    from maichart.generation import event_audio
    monkeypatch.setattr(event_audio,'extract_event_audio',lambda *_a,**_k:pytest.fail('Legacy 50Hz extraction used'))
    audio=extract_rational_audio(source[0],tmp_path/'cache')
    assert audio.metadata['feature_rate_hz'] is None
    song=organize_song(audio,[EventTempo(F(0),190)],-.137,max_seconds=3)
    values,times,beats=audio_queries(song,song.spans[0].start,song.spans[-1].end)
    assert values.shape==(len(beats),120)
    assert not np.allclose(np.diff(times),.02)
    assert any(beat.denominator not in (1,2,3,4,6,8,12,16,24,32,48) for beat in beats)
    assert all(seconds_at_beat(beat,song.tempo,song.first)==pytest.approx(second) for beat,second in zip(beats,times))
    np.testing.assert_array_equal(values.numpy()[times<0],0)
    again=extract_rational_audio(source[0],tmp_path/'cache')
    assert again.metadata==audio.metadata
    local=audio_queries(song,song.spans[0].start,song.spans[0].end,max_nodes=1024)[0]
    assert input_example(song,0,[])['audio'].shape==local.shape


def test_tempo_hint_uses_global_detector_instead_of_attack_spacing(monkeypatch):
    import librosa
    import maichart.generation.tempo as tempo
    calls = {}
    monkeypatch.setattr(librosa.onset, 'onset_strength',
                        lambda **kwargs: np.ones(32, dtype=np.float32))
    def fake_detect(envelope, *, fps, **kwargs):
        calls['fps'] = fps
        calls['length'] = len(envelope)
        return 187.5, 0.4
    monkeypatch.setattr(tempo, 'detect_global_bpm', fake_detect)
    assert _tempo_hint(np.zeros(4000, dtype=np.float32), 20000) == pytest.approx(187.5)
    assert calls == {'fps': pytest.approx(200.0), 'length': 32}


def test_exact_clock_preserves_unusual_fractions_across_tempo_changes():
    tempo=[EventTempo(F(0),190),EventTempo(F(3,2),127.5)]
    for beat in (F(1,8),F(37,997),F(23,7),F(7,20)):
        # An exact clock computation with decimal BPM and first, not float snapping.
        first=F(-91,50)
        second=first+min(beat,F(3,2))*60/F(190)
        if beat>F(3,2):second+=(beat-F(3,2))*60/F('127.5')
        assert beat_at_second(second,tempo,float(first))==beat


def test_fraction_grammar_is_not_a_division_whitelist():
    grammar=PositionGrammar(-1,4)
    codec=RationalCodec()
    for text in ('0','1/8','37/997','23/7','7/20','13/41'):
        for i in range(1,len(text)+1): assert grammar.prefix(text[:i])
        assert grammar.complete(text)
        assert codec.note_id in grammar.allowed(text,codec)
    assert grammar.complete('2/4')
    for text in ('1/0','01','0/3','4','-1'):
        assert not grammar.complete(text)
    grammar.lower=F(1,8)
    assert not grammar.complete('1/8')


def test_position_prefix_has_a_reachable_later_value_and_respects_rest_holes():
    codec=RationalCodec()
    grammar=PositionGrammar(1,4)
    assert grammar.prefix('1')  # e.g. 10/3 can still be reached.
    assert not grammar.prefix('1/')
    assert codec.ids['P:/'] not in grammar.allowed('1',codec)
    hole=PositionGrammar(-1,4,allowed_ranges=[(F(2),F(4))])
    assert not hole.prefix('1/')
    assert not hole.complete('1/2')
    assert hole.complete('7/3')
    short=PositionGrammar(10,11,characters=4)
    assert not short.prefix('999')


def test_equal_fraction_normalization_rewinds_the_actual_decoder_prefix(source,tmp_path):
    codec=RationalCodec()
    canonical=codec.encode([EventPoint(F(1,2),'1')])
    raw=[codec.bos_id,codec.at_id,codec.ids['P:2'],codec.ids['P:/'],codec.ids['P:4'],
         codec.note_id,codec.ids['N:1'],codec.end_id,codec.eos_id]
    class ForcedModel:
        config=SimpleNamespace(max_tokens=512)
        sequence=raw
        rewinds=[]
        def eval(self):return self
        def encode(self,_):return torch.zeros(1,1,1),None
        def decode_step(self,token,position,*_):
            assert int(token[0,0])==self.sequence[position]
            logits=torch.full((1,1,len(codec)),-100.)
            logits[0,0,self.sequence[position+1]]=100
            return logits,None
        def rewind_decode_cache(self,cache,position):
            self.rewinds.append(position);self.sequence=canonical
            return None
    audio=extract_rational_audio(source[0],tmp_path/'cache')
    song=organize_song(audio,[EventTempo(F(0),190)],0,max_seconds=3)
    model=ForcedModel()
    points,status=compose_span(model,codec,input_example(song,0,[],codec=codec),['1'],origin=F(0),end=F(4),temperature=0,position_temperature=0)
    assert points==[EventPoint(F(1,2),'1')]
    assert model.rewinds==[1]
    assert status['equivalent_fraction_normalizations']==1


def test_all_rest_emits_an_explicit_empty_chart_without_position_dead_ends(source,tmp_path):
    from maichart.form.rational_generate import generate_rational_chart
    from maichart.form.clock import AUDIO_CURVE_AXIS
    from maichart.generation.conditions import GenerationCondition
    codec=RationalCodec()
    class Model:
        config=SimpleNamespace(max_tokens=512,max_memory_tokens=64)
        def eval(self):return self
        def encode(self,batch):return torch.zeros(1,1,1),None
        def decode_step(self,*_):
            logits=torch.zeros(1,1,len(codec));logits[0,0,codec.at_id]=100
            return logits,None
    def planner(audio,*_):return {'load_curve':torch.zeros(1,audio.shape[1],4)}
    audio=extract_rational_audio(source[0],tmp_path/'cache')
    song=organize_song(audio,[EventTempo(F(0),190)],-.137,max_seconds=3)
    curve=dict(time_axis=AUDIO_CURVE_AXIS,method='learned_song_form_v2',
        points=[dict(time_sec=0,intensity=0),dict(time_sec=3,intensity=0)],
        rest_regions=[dict(start_sec=0,end_sec=3)])
    points,report=generate_rational_chart(Model(),planner,codec,song,GenerationCondition(),['1'],
        intensity_curve=curve,style_balance_strength=0,local_chord_strength=0)
    assert points==[] and report['quality']['syntax_valid']
    assert report['blueprint']['selection']['explicit_all_rest']


def test_waveform_cache_tracks_content_when_size_and_mtime_stay_equal(tmp_path):
    import os
    import soundfile as sf
    from maichart.form.rational_audio import waveform
    path=tmp_path/'replace.wav';sr=22050
    sf.write(path,np.zeros(sr,dtype=np.float32),sr,subtype='PCM_16')
    stat=path.stat()
    before,_=waveform(path)
    sf.write(path,np.full(sr,.5,dtype=np.float32),sr,subtype='PCM_16')
    os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    assert path.stat().st_size==stat.st_size
    after,_=waveform(path)
    assert np.max(before)==0 and np.mean(after)==pytest.approx(.5)


def test_joint_decoder_emits_exact_events_without_a_planner_candidate(source,tmp_path):
    codec=RationalCodec()
    points=[EventPoint(F(37,997),'2'),EventPoint(F(1,8),'1'),EventPoint(F(7,20),'3')]
    sequence=codec.encode(points)
    class ForcedModel:
        config=SimpleNamespace(max_tokens=512)
        def eval(self):return self
        def encode(self,batch):
            assert not batch.get('skeleton',torch.zeros(1)).any()
            return torch.zeros(1,1,1),None
        def decode_step(self,token,position,*_):
            assert int(token[0,0])==sequence[position]
            logits=torch.full((1,1,len(codec)),-100.)
            logits[0,0,sequence[position+1]]=100
            return logits,None
    audio=extract_rational_audio(source[0],tmp_path/'cache')
    song=organize_song(audio,[EventTempo(F(0),190)],0,max_seconds=3)
    example=input_example(song,0,[],codec=codec)
    example['skeleton']=torch.tensor([codec.pad_id])
    generated,status=compose_span(ForcedModel(),codec,example,['1','2','3'],origin=F(0),end=F(4),temperature=0,position_temperature=0)
    assert generated==points and status['closed']


def test_rational_dataset_hides_future_onset_positions(source,tmp_path):
    dataset=FormDataset(source[1],cache_dir=tmp_path/'cache',audio_profile=PROFILE,max_region_seconds=3)
    assert len(dataset)>0
    sample=dataset[0]
    assert sample['skeleton'].tolist()==[0]
    actual=dataset.codec.decode([dataset.codec.bos_id]+sample['target'].tolist())
    assert actual[1].beat-actual[0].beat==F(1,8)
    assert any(point.beat.denominator==5 for point in actual)
    cached=FormDataset(source[1],cache_dir=tmp_path/'cache',audio_profile=PROFILE,max_region_seconds=3)
    assert cached.songs[0][0].audio_profile==PROFILE
    assert cached[0]['skeleton'].tolist()==[0]


def test_workload_planner_has_no_frame_onset_head_and_updates_conditions():
    model=RationalPlanner(RationalPlannerConfig(descriptor_dim=8,width=16,form_heads=2,form_layers=1,dropout=0))
    audio=torch.randn(1,37,120);form=torch.randn(1,3,11);condition=torch.randn(1,16)
    output=model(audio,form,condition)
    assert set(output)=={'load_curve'}
    targets=dict(load_curve_target=torch.zeros(1,37,4),load_curve_mask=torch.ones(1,37,4),frame_weight=torch.ones(1,37))
    parts=rational_planner_loss(output,targets)
    parts['loss'].backward()
    assert model.condition_projection.weight.grad.abs().sum()>0
    assert model.region_attention.in_proj_weight.grad.abs().sum()>0


def test_compact_default_retains_all_model_branches():
    from maichart.form.train import parser,fresh_model_configs
    from maichart.form.model import FormTransformer
    from maichart.form.monitoring import training_model_summary
    args=parser().parse_args(['--manifest','unused','--checkpoint-dir','unused'])
    config,planner_config=fresh_model_configs(args)
    model=FormTransformer(config)
    summary=training_model_summary(model,RationalPlanner(planner_config))
    assert config.d_model==256 and config.layers==4 and config.dim_feedforward==768
    assert summary['parameter_count']<15000000
    assert model.condition_projection is not None and model.load_curve_projection is not None
    assert model.motif_gate is not None and model.action_projection is not None


def test_real_rational_training_and_checkpoint_roundtrip(source,tmp_path):
    from maichart.form import train,checkpoint
    old_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        output=tmp_path/'model'
        assert train.main(['--manifest',str(source[1]),'--checkpoint-dir',str(output),
            '--cache-dir',str(tmp_path/'cache'),'--epochs','1','--batch-size','1','--max-steps','2',
            '--d-model','32','--planner-width','32','--layers','1','--max-tokens','256',
            '--max-memory-tokens','64','--max-region-seconds','3'])==0
        model,planner,codec,saved=checkpoint.load(output/'form_last.pt',inference=True)
        assert saved['family']=='song_form_v2'
        assert saved['input_contract']['skeleton']=='absent_joint_exact_events_v1'
        assert saved['blueprint_steps']==1 and saved['optimizer_steps']==2
        assert isinstance(planner,RationalPlanner)
        assert saved['task_training']['draft']>0 and saved['task_training']['revision']>0
        assert 'onset_head.weight' not in saved['blueprint_state_dict']
        from maichart.form.curve import generate_learned_profile
        curve=generate_learned_profile(source[0],output/'form_last.pt',bpm=190,first=-.137,device='cpu')
        assert curve['method']=='learned_song_form_v2'
        assert curve['model_family']=='song_form_v2'
        assert curve['points'][-1]['time_sec']==pytest.approx(3)
    finally:torch.set_num_threads(old_threads)
