"""Regression coverage for profile routing, causal geometry and real decoding."""
from fractions import Fraction as F
from types import SimpleNamespace
import json

import pytest
import torch

from maichart.event_timing import EventPoint, EventTempo
from maichart.form.codec import RationalCodec
from maichart.form.data import FormDataset, organize_song, input_example, collate_form
from maichart.form.model import FormConfig, FormTransformer
from maichart.form.rational_audio import PROFILE, extract_rational_audio
from maichart.form.rhythm_geometry import causal_prefix_geometry
from test_form_rational_events import source


def test_threaded_preparation_matches_serial_native_profile_and_cache(source,tmp_path):
    content=json.loads(source[1].read_text(encoding='utf-8'))
    content['songs'].append({**content['songs'][0], 'song_id':'same-audio-second-chart'})
    source[1].write_text(json.dumps(content),encoding='utf-8')
    options=dict(cache_dir=tmp_path/'cache',audio_profile=PROFILE,
                 rhythm_geometry_profile='exact_fraction_v1',max_tokens=512)
    serial=FormDataset(source[1],preparation_workers=1,**options)
    # Separate cache forces the threaded loader to execute, instead of making
    # a false-positive comparison of two reads from the same prepared result.
    threaded=FormDataset(source[1],**{**options,'cache_dir':tmp_path/'parallel'},preparation_workers=2)
    warm=FormDataset(source[1],**{**options,'cache_dir':tmp_path/'parallel'},preparation_workers=2)
    assert warm.preparation_cache_stats['hits']==2
    for dataset in (serial,threaded,warm):
        assert all(song[0].audio_profile==PROFILE for song in dataset.songs)
        assert all(song[0].audio.feature_policy=='native_measured_v2' for song in dataset.songs)
        assert dataset.songs[0][0].rhythm_geometry_profile=='exact_fraction_v1'
        example=dataset[0]
        assert example['audio_geometry'].shape==(len(example['audio']),8)
        assert example['timing_mask'].sum()==example['target'].eq(dataset.codec.at_id).sum()
        assert len(collate_form([example])['decoder_geometry'][0])==len(example['target'])
    assert serial.items==threaded.items==warm.items
    for key,value in serial[0].items():
        if isinstance(value,torch.Tensor):
            torch.testing.assert_close(value,threaded[0][key],rtol=0,atol=0)
            torch.testing.assert_close(value,warm[0][key],rtol=0,atol=0)


def test_duplicate_source_rows_are_not_lost_under_parallel_preparation(source,tmp_path,monkeypatch):
    from maichart.form.preparation_cache import PreparationCache
    monkeypatch.setattr(PreparationCache,'_legacy_roots',lambda self:[self.root])
    content=json.loads(source[1].read_text(encoding='utf-8'))
    content['songs'].extend([{**content['songs'][0], 'song_id':f'alias-{i}'} for i in range(5)])
    source[1].write_text(json.dumps(content),encoding='utf-8')
    dataset=FormDataset(source[1],cache_dir=tmp_path/'cache',audio_profile=PROFILE,
                        rhythm_geometry_profile='exact_fraction_v1',preparation_workers=4,
                        max_tokens=512)
    assert len(dataset.songs)==6
    assert len(dataset.items)==12
    assert dataset.preparation_cache_stats['rebuilt']==1
    assert dataset.preparation_cache_stats['hits']==5


@pytest.mark.parametrize('profile',['none','exact_fraction_v1'])
def test_cached_geometry_decoding_matches_full_causal_forward(source,tmp_path,profile):
    codec=RationalCodec()
    audio=extract_rational_audio(source[0],tmp_path/'cache')
    song=organize_song(audio,[EventTempo(F(0),190)],-.137,max_seconds=3,rhythm_geometry_profile=profile)
    example=input_example(song,0,[],codec=codec,max_memory_tokens=64)
    model=FormTransformer(FormConfig(len(codec),d_model=16,nhead=2,layers=1,dim_feedforward=32,
        dropout=0,max_tokens=128,max_memory_tokens=64,rhythm_geometry_profile=profile)).eval()
    ids=torch.tensor([codec.encode([EventPoint(F(37,997),'1'),EventPoint(F(1,2),'2')])[:-1]])
    if profile!='none':
        torch.nn.init.normal_(model.time_geometry_projection.weight,std=.1)
        torch.nn.init.normal_(model.decoder_geometry_projection.weight,std=.1)
    geometry=causal_prefix_geometry(ids[0],codec,origin=F(0))[None]
    with torch.inference_mode():
        memory,mask=model.encode(collate_form([example]))
        full=model.decode(ids,memory,mask,**({'timing_features':geometry} if profile!='none' else {}))
        cached=[];cache=None
        for index in range(ids.shape[1]):
            logits,cache=model.decode_step(ids[:,index:index+1],index,memory,mask,cache,
                **({'timing_features':geometry[:,index:index+1]} if profile!='none' else {}))
            cached.append(logits)
    torch.testing.assert_close(full,torch.cat(cached,1),atol=2e-5,rtol=2e-5)


@pytest.mark.parametrize('beam',[1,4])
def test_new_profile_compose_supplies_only_consumed_geometry_and_preserves_exact_events(beam):
    from maichart.form.rational_generate import compose_span
    codec=RationalCodec()
    expected=[EventPoint(F(37,997),'1'),EventPoint(F(1,2),'2')]
    sequence=codec.encode(expected)
    class Model:
        config=SimpleNamespace(max_tokens=512,rhythm_geometry_profile='exact_fraction_v1')
        rewind_decode_cache=staticmethod(FormTransformer.rewind_decode_cache)
        def eval(self):return self
        def encode(self,_):return torch.zeros(1,1,1),None
        def decode_step(self,tokens,position,memory,mask,caches=None,*,timing_features):
            previous=caches[0][0] if caches else torch.zeros(len(tokens),0,1,dtype=torch.long)
            assert previous.shape[1]==position
            present=torch.cat([previous,tokens[...,None]],dim=1)
            logits=torch.full((len(tokens),1,len(codec)),-100.)
            for row in range(len(tokens)):
                prefix=present[row,:,0].tolist()
                expected_geometry=causal_prefix_geometry(prefix,codec)[-1]
                torch.testing.assert_close(timing_features[row,0],expected_geometry,rtol=0,atol=0)
                wanted=sequence[len(prefix)] if prefix==sequence[:len(prefix)] else codec.note_id
                logits[row,0,wanted]=100.
            return logits,[(present,present,None,None)]
    example={'meta':{},'action_state_padding_mask':torch.ones(1,dtype=torch.bool)}
    points,status=compose_span(Model(),codec,example,['1','2'],origin=F(0),end=F(4),
        temperature=0,position_temperature=0,position_beam_size=beam)
    assert points==expected and status['closed']


def test_unusual_previous_fraction_does_not_create_a_hidden_interval_denominator_hole():
    from maichart.form.rational_generate import PositionGrammar
    grammar=PositionGrammar(F(37,997),4)
    text='1000/4091'
    assert grammar.complete(text)
    assert all(grammar.prefix(text[:length]) for length in range(1,len(text)+1))


def test_native_query_context_uses_full_song_extent_and_validates_cache_clock(source,tmp_path):
    from dataclasses import replace
    import numpy as np
    from maichart.form.rational_audio import audio_queries
    from maichart.form.timing import MetricSpan
    audio=extract_rational_audio(source[0],tmp_path/'cache')
    song=organize_song(audio,[EventTempo(F(0),190)],0,max_seconds=3,
        rhythm_geometry_profile='exact_fraction_v1')
    start=F(0);end=F(1)
    expected,times,beats=audio_queries(song,start,end,max_nodes=32)
    # Editing changes the computational span, not the full-song coordinate.
    edited=replace(song,spans=[replace(song.spans[0],end=end)])
    audio._views.clear()
    actual,_,_=audio_queries(edited,start,end,max_nodes=32)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    from pathlib import Path
    files=list((tmp_path/'cache'/PROFILE).glob('*.queries.npz'))
    assert len(files)==1
    with np.load(files[0],allow_pickle=False) as saved:
        payload={name:saved[name].copy() for name in saved.files}
    payload['times'][0]=np.nan
    np.savez_compressed(files[0],**payload)
    audio._views.clear()
    rebuilt,recovered,again=audio_queries(song,start,end,max_nodes=32)
    assert np.isfinite(recovered).all() and len(again)<=32
    torch.testing.assert_close(rebuilt,expected,rtol=0,atol=0)


def test_capability_detector_accepts_both_rational_contracts_and_rejects_cross_profile():
    from maichart.form.checkpoint import RATIONAL_INPUT_CONTRACT,RATIONAL_GEOMETRY_INPUT_CONTRACT
    from maichart.generation.model_compatibility import checkpoint_capabilities
    for profile,contract in (('none',RATIONAL_INPUT_CONTRACT),('exact_fraction_v1',RATIONAL_GEOMETRY_INPUT_CONTRACT)):
        payload=dict(schema='maichart-song-form-checkpoint-v2',family='song_form_v2',
            model_config={'rhythm_geometry_profile':profile},input_contract=contract)
        caps=checkpoint_capabilities(payload)
        assert caps['valid'] and caps['generation_entry']=='/'
        if profile=='exact_fraction_v1':
            payload['input_contract']=RATIONAL_INPUT_CONTRACT
            assert not checkpoint_capabilities(payload)['valid']


def test_new_geometry_paths_receive_gradients_and_clock_changes_affect_logits(source,tmp_path):
    audio=extract_rational_audio(source[0],tmp_path/'cache')
    song=organize_song(audio,[EventTempo(F(0),190)],0,max_seconds=3,
        rhythm_geometry_profile='exact_fraction_v1')
    song.load_curve_times_sec=__import__('numpy').array([.2,.5],dtype='float32')
    song.load_curve=__import__('numpy').ones((2,4),dtype='float32')*.5
    codec=RationalCodec()
    example=input_example(song,0,[],codec=codec,max_memory_tokens=64)
    ids=torch.tensor(codec.encode([EventPoint(F(1,4),'1')])[:-1])
    example['decoder_input']=ids
    example['decoder_geometry']=causal_prefix_geometry(ids,codec)
    model=FormTransformer(FormConfig(len(codec),d_model=16,nhead=2,layers=1,dim_feedforward=32,
        dropout=0,max_tokens=128,max_memory_tokens=64,rhythm_geometry_profile='exact_fraction_v1'))
    batch=collate_form([example])
    result=model(batch)
    (result['logits'].square().mean()+result['timing_prediction'].square().mean()).backward()
    assert model.time_geometry_projection.weight.grad.abs().sum()>0
    assert model.decoder_geometry_projection.weight.grad.abs().sum()>0
    assert model.interval_head.weight.grad.abs().sum()>0
    # Once trained, a changed curve timestamp must be distinguishable even if
    # its four channel values are the same. Zero migration initialization is
    # intentionally neutral until optimization updates these projections.
    with torch.no_grad():
        model.time_geometry_projection.weight.add_(model.time_geometry_projection.weight.grad)
    model.eval()
    other={**batch,'load_curve_geometry':batch['load_curve_geometry']+1}
    with torch.inference_mode():
        assert not torch.equal(model(batch)['logits'],model(other)['logits'])
