"""CPU regressions for stream data, masks, weighting and writer contracts."""
from collections import OrderedDict
from fractions import Fraction as F
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from maichart.event_timing import EventPoint, EventTempo
from maichart.form.codec import RationalCodec
from maichart.form.output_guard import training_output_guard
from maichart.stream import data, checkpoint, train
from maichart.stream.model import ChartStreamTransformer, StreamConfig
from maichart.stream.quality import event_weight, interval_weight, curve_weights
from maichart.stream.reference_guard import ReferenceCandidateGuard
from test_form_preparation_cache import cached_source


@pytest.fixture(autouse=True)
def one_thread():
    previous=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_model():
    return ChartStreamTransformer(StreamConfig(55,d_model=16,nhead=2,layers=1,
        dim_feedforward=32,dropout=0,max_tokens=64,max_history_tokens=12,max_stream_memory=3)).eval()


@pytest.mark.parametrize('length',[0,2,7])
def test_history_logits_do_not_depend_on_neighbor_padding(length):
    torch.manual_seed(4)
    model=small_model()
    history=torch.arange(1,length+1)[None]
    batch={'audio':torch.randn(1,5,120),'condition':torch.randn(1,16),
           'decoder_input':torch.tensor([[1,3,8]]),'history':history}
    with torch.inference_mode():
        alone=model(batch)['logits']
        other={key: value.repeat(2,*([1]*(value.ndim-1))) for key,value in batch.items() if key!='history'}
        other['history']=torch.stack((torch.nn.functional.pad(history[0],(0,12-length)),torch.arange(1,13)))
        other['history_padding_mask']=other['history']==0
        together=model(other)['logits'][:1]
    torch.testing.assert_close(alone,together,rtol=1e-5,atol=1e-5)


def test_pooled_history_preserves_single_row_means_and_gradients():
    model=small_model()
    tokens=torch.tensor([[1,2,3,4,5,6,7,0,0],[0]*9])
    summary,mask=model.summarize_tokens(tokens,return_padding_mask=True)
    values=model.token_embedding(tokens[0,:7])
    expected=torch.stack((values[:3].mean(0),values[3:6].mean(0),values[6]))
    torch.testing.assert_close(summary[0],expected)
    assert mask[1].all() and not mask[0].any()
    summary.sum().backward()
    assert model.token_embedding.weight.grad[0].abs().sum()==0
    assert model.token_embedding.weight.grad[1].abs().sum()>0


def loss_inputs():
    codec=RationalCodec()
    output={'logits':torch.randn(1,3,len(codec),requires_grad=True),
        'plan':torch.randn(1,3,8,requires_grad=True),'curve':torch.randn(1,2,4,requires_grad=True)}
    batch={'decoder_input':torch.tensor([[codec.bos_id,codec.at_id,codec.end_id]]),
        'target':torch.tensor([[codec.at_id,codec.ids['P:0'],codec.eos_id]]),
        'target_padding_mask':torch.zeros(1,3,dtype=torch.bool),
        'plan_mask':torch.tensor([[True,False,False]]),'plan_mask_padding_mask':torch.zeros(1,3,dtype=torch.bool),
        'plan_labels':torch.zeros(1,3,8),'curve_target':torch.zeros(1,2,4),
        'curve_target_padding_mask':torch.zeros(1,2,dtype=torch.bool),'audio_padding_mask':torch.zeros(1,2,dtype=torch.bool)}
    return output,batch


def test_quality_scales_all_gradients_even_at_batch_size_one():
    output,batch=loss_inputs()
    plain,_=train._loss(output,batch)
    scaled,_=train._loss(output,{**batch,'quality_weight':torch.tensor([.25])})
    for a,b in zip(torch.autograd.grad(plain,tuple(output.values()),retain_graph=True),
                   torch.autograd.grad(scaled,tuple(output.values()))):
        torch.testing.assert_close(b,a*.25)


def test_excluded_targets_have_zero_gradients_in_all_objectives():
    output,batch=loss_inputs()
    batch.update(target_quality=torch.tensor([[0.,1.,0.]]),curve_quality=torch.tensor([[0.,1.]]))
    loss,_=train._loss(output,batch)
    grads=torch.autograd.grad(loss,tuple(output.values()))
    assert grads[0][:,[0,2]].abs().sum()==0
    assert grads[1].abs().sum()==0
    assert grads[2][:,0].abs().sum()==0 and grads[2][:,1].abs().sum()>0
    zero,_=train._loss(output,{**batch,'target_quality':torch.zeros(1,3),'curve_quality':torch.zeros(1,2)})
    assert zero.item()==0 and torch.isfinite(zero)


def test_excluded_region_masks_jumps_eos_and_interpolation_support():
    annotation={'regions':[{'start_sec':2.,'end_sec':3.,'weight':0.}]}
    assert event_weight(0.,1.,annotation)==1
    assert event_weight(1.,2.,annotation)==0
    assert event_weight(2.5,3.5,annotation)==0
    assert event_weight(3.5,4.,annotation)==1
    assert interval_weight(1.,4.,annotation)==0  # EOS may not assert silence across the exclusion.
    song=SimpleNamespace(first=1.,curve_bin_seconds=1.,load_curve_times_sec=np.array([.5,1.5,2.5]),
                         audio=SimpleNamespace(duration_sec=4.))
    weights=curve_weights(song,np.array([1.5,1.75,2.5,3.25,3.5,3.9]),annotation)
    assert weights.tolist()==[1,0,0,0,1,1]


def test_preparation_preserves_ignored_events_only_for_stream_and_versions_cache(cached_source):
    build,chart,audio,_=cached_source
    from maichart.training.quality_annotations import SCHEMA,chart_sha256
    chart.parent.parent.joinpath('quality_annotations.json').write_text(json.dumps({
        'schema':SCHEMA,'records':[{'source_path':str(chart),'audio_path':str(audio),
        'difficulty_index':5,'chart_sha256':chart_sha256(chart,5),'label':'excellent','weight':2.,
        'regions':[{'start_sec':.4,'end_sec':.6,'weight':0.}]}]}),encoding='utf-8')
    old=build(max_region_seconds=2.)
    stream=build(preserve_excluded_context=True,max_region_seconds=2.)
    assert len(stream.songs[0][1])>len(old.songs[0][1])
    assert stream.preparation_cache_stats['rebuilt']==1
    again=build(preserve_excluded_context=True,max_region_seconds=2.)
    assert again.preparation_cache_stats['hits']==1
    assert again.songs[0][1]==stream.songs[0][1]
    disabled=build(quality_enabled=False,max_region_seconds=2.)
    assert disabled.songs[0][3]==1 and disabled.songs[0][7] is None
    assert len(disabled.songs[0][1])==len(stream.songs[0][1])


def test_stream_targets_keep_full_prefix_but_mask_ignored_event(monkeypatch):
    from maichart.generation.conditions import GenerationCondition
    from maichart.generation.sustain_state import SustainStateIndex
    tempo=[EventTempo(F(0),120.)]
    points=[EventPoint(F(i),'1') for i in range(5)]
    annotation={'regions':[{'start_sec':.9,'end_sec':1.1,'weight':0.}]}
    song=SimpleNamespace(first=0.,tempo=tempo,meters=[],spans=[SimpleNamespace(start=F(0),end=F(6))],
        quality_profile='musical_evidence_v1',curve_bin_seconds=1.,audio=SimpleNamespace(duration_sec=3.),
        load_curve_times_sec=np.array([.5,1.5,2.5]),load_curve=np.zeros((3,4),dtype=np.float32))
    record=(song,points,13.,2.,SustainStateIndex(points,tempo,0),GenerationCondition(),[(.9,1.1)],annotation)
    def prepared(*_args,**kwargs):
        assert kwargs['preserve_excluded_context']
        return SimpleNamespace(songs=[record],audit={'charts':[{'audio_sha256':'fixture'}]})
    monkeypatch.setattr(data,'FormDataset',prepared)
    monkeypatch.setattr(data,'planner_audio_queries',lambda *_:(torch.zeros(3,120),np.array([.5,1.5,2.5]),(F(1),F(3),F(5))))
    ds=data.StreamDataset('unused',cache_dir='unused',max_tokens=64)
    chunk=ds.chunks[0]; codec=ds.codec
    assert list(chunk.target)==codec.encode(points)[1:]
    at=[i for i,t in enumerate(chunk.target) if t==codec.at_id]
    assert [chunk.target_quality[i] for i in at]==[1,1,0,0,1]
    batch=data.collate_stream([ds[0]])
    assert batch['quality_weight'].tolist()==[2.]
    assert len(chunk.target_quality)==len(chunk.target)


@pytest.mark.parametrize('text,slide,hold,chord',[
    ('1p5[4:1]',1,0,0),('1q5[4:1]',1,0,0),('1w5[4:1]',1,0,0),
    ('1s5[4:1]',1,0,0),('1h[4:1]',0,1,0),('Ch[4:1]',0,1,0),
    ('1/2',0,0,1),('1h[4:1]/2p6[4:1]',1,1,1)])
def test_planner_uses_parsed_heads(text,slide,hold,chord):
    label=data._plan_for_event(EventPoint(F(1),text),F(0),text)
    assert (label[4],label[5],label[3],label[6])==(slide,hold,chord,chord)


def test_query_cache_reuses_song_keys_and_obeys_budget(monkeypatch):
    ds=object.__new__(data.StreamDataset)
    ds._query_keys=['same','same','other'];ds._audio_queries=OrderedDict()
    ds.query_hits=ds.query_misses=ds.query_bytes=0
    ds.query_budget=560
    calls=[]
    def query(song):
        calls.append(song);return torch.zeros(1,120),np.array([0.]),(F(0),)
    monkeypatch.setattr(data,'planner_audio_queries',query)
    first=ds._queries(0,'a')
    assert ds._queries(1,'b') is first and len(calls)==1
    ds._queries(2,'c')
    assert len(ds._audio_queries)==1 and ds.query_bytes<=ds.query_budget
    ds.query_budget=0;ds._audio_queries.clear();ds.query_bytes=0
    ds._queries(0,'d')
    assert ds.query_bytes==0 and not ds._audio_queries


@pytest.mark.parametrize('option,value',[
    ('batch-size','0'),('grad-accumulation','0'),('log-interval','0'),('epochs','0'),
    ('audio-cache-gb','nan'),('lr','nan'),('max-region-seconds','0'),('dropout','1'),
    ('checkpoint-interval-steps','-1'),('seed',str(2**32))])
def test_invalid_training_settings_fail_before_dataset(option,value,monkeypatch,tmp_path):
    monkeypatch.setattr(train,'StreamDataset',lambda *_a,**_kw:pytest.fail('prepared invalid input'))
    with pytest.raises(ValueError):
        train.main(['--manifest','missing','--checkpoint-dir',str(tmp_path),'--'+option,value])


def test_output_guard_and_fresh_protect_parent_weights(tmp_path):
    args=train.parser().parse_args(['--manifest','unused','--checkpoint-dir',str(tmp_path)])
    saved=tmp_path/'stream_last.pt'; saved.write_bytes(b'preserve')
    with pytest.raises(ValueError,match='新输出目录'):train.validate_output_directory(args)
    args.resume=str(saved);train.validate_output_directory(args)
    with training_output_guard(tmp_path):
        with pytest.raises(ValueError,match='已有训练任务'):
            with training_output_guard(tmp_path): pass
    assert saved.read_bytes()==b'preserve'


def test_checkpoint_codec_scaler_and_effective_dataset_identity(tmp_path):
    model=small_model(); manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    path=tmp_path/'stream.pt'
    scaler=SimpleNamespace(state_dict=lambda:{'scale':1024.})
    payload=checkpoint.save(path,model,None,epoch=1,args={},manifest=manifest,
        scaler=scaler,dataset_fingerprint='dataset-a')
    _,restored=checkpoint.load(path)
    assert restored['scaler_state_dict']=={'scale':1024.}
    train._validate_resume_data(restored,SimpleNamespace(fingerprint='dataset-a'))
    with pytest.raises(ValueError,match='Resume data changed'):
        train._validate_resume_data(restored,SimpleNamespace(fingerprint='dataset-b'))
    payload['codec']['tokens']=list(reversed(payload['codec']['tokens']))
    torch.save(payload,path)
    with pytest.raises(ValueError,match='codec'):checkpoint.load(path)
    from maichart.webui.stream_api import inspect
    assert not inspect(path)['valid']


def test_reference_fast_path_does_not_accept_three_taps():
    guard=ReferenceCandidateGuard([EventTempo(F(0),120.)],0.)
    accepted,errors=guard.check(EventPoint(F(0),'1/3/6'))
    assert not accepted and any(item['kind']=='MultiTouch' for item in errors)
    assert guard.fast_accepts==0


def test_reference_attribution_includes_only_new_or_current_diagnostics(monkeypatch):
    import maichart.stream.reference_guard as module
    guard=ReferenceCandidateGuard([EventTempo(F(0),120.)],0.)
    old={'kind':'Overlap','alert_level':'Muri','marker_key':'a','second':0.}
    new={'kind':'SlideTooFast','alert_level':'Muri','marker_key':'b','second':1.}
    report={'markers':[{'key':'a','type':'tap','second':0.,'lane':1,'raw':'1'},
                       {'key':'b','type':'tap','second':1.,'lane':3,'raw':'3'}],
            'diagnostics':[old,new]}
    monkeypatch.setattr(module,'analyze_maidata',lambda *_:report)
    monkeypatch.setattr(guard,'_markers',lambda *_:(2.,[{'type':'slide'}]))
    monkeypatch.setattr(guard,'_baseline',lambda *_:guard._signatures({**report,'diagnostics':[old]}))
    _,errors=guard.check(EventPoint(F(4),'1-5[4:1]'))
    assert errors==[new]


def test_curve_blank_bpm_uses_generation_audio_clock_and_native_evidence(monkeypatch):
    from maichart.stream import curve
    audio=SimpleNamespace(tempo_bpm=156.,duration_sec=3.,feature_policy='legacy_sparse_v1',
                          evidence_profile='balanced_evidence_v1')
    captured={}
    def organize(value,tempo,first,**kwargs):
        captured.update(bpm=tempo[0].bpm,first=first,policy=value.feature_policy)
        return SimpleNamespace()
    monkeypatch.setattr(curve,'load',lambda *_a,**_kw:(small_model(),{}))
    monkeypatch.setattr(curve,'extract_rational_audio',lambda *_a,**_kw:audio)
    monkeypatch.setattr(curve,'organize_song',organize)
    monkeypatch.setattr(curve,'planner_audio_queries',lambda *_:(torch.zeros(3,120),np.array([.5,1.5,2.5]),()))
    monkeypatch.setattr(curve,'audio_sha256',lambda *_:'fixture')
    result=curve.generate_learned_profile('unused','unused.pt',bpm=None,first=-1.)
    assert captured=={'bpm':156.,'first':-1.,'policy':'native_measured_v2'}
    assert result['bpm']==156. and result['points'][0]['time_sec']==0.
    assert result['points'][-1]['time_sec']==3.


def test_real_task_log_routes_stream_metrics_and_resume_progress():
    from maichart.webui.server import TaskState
    task=TaskState('stream-metrics','chart_stream_train')
    task.log('stream_runtime '+json.dumps({'start_epoch':5,'end_epoch':6,'parameters':100}))
    progress={'epoch':5,'chunk':25,'total':100,'loss':.27,'token_accuracy':.9,
              'token_loss':.2,'event_decision_loss':.1,'batches_per_sec':2.,'optimizer_steps':7}
    task.log('stream_progress '+json.dumps(progress))
    assert task.progress==pytest.approx(.125)
    assert task.result['stream_metrics']==progress and '90.0%' in task.message
    task.log('stream_checkpoint '+json.dumps({'epoch':5,'optimizer_steps':7,'path':'stream_step.pt'}))
    assert task.result['last_checkpoint']['path']=='stream_step.pt'
    task.log('stream_epoch '+json.dumps(progress))
    assert task.progress==.5 and len(task.result['stream_metric_history'])==1
