"""Causal, train/inference and migration contracts for grounded streams."""
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace
import json,random

import numpy as np
import pytest
import torch

from maichart.form.codec import RationalCodec
from maichart.event_timing import EventPoint,EventTempo,seconds_at_beat
from maichart.stream.grounding import CausalClock,clock_inputs,features_at_beats,masked_note_inputs,time_cell_weights
from maichart.stream.model import StreamConfig,ChartStreamTransformer
from maichart.stream.data import StreamDataset,collate_stream
from maichart.stream.checkpoint import upgrade_grounding,save,load
from maichart.stream.grounded_inference import GroundedInference,IncrementalState
from maichart.generation.sustain_state import SustainStateIndex
from maichart.stream.train import _loss,_mean_metrics


@pytest.fixture(autouse=True)
def cpu_threads():
    count=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(count)


def config():
    return StreamConfig(55,d_model=16,nhead=2,layers=1,dim_feedforward=32,dropout=0,
        max_tokens=32,max_history_tokens=12,max_stream_memory=3,grounding_profile='causal_audio_v1',
        detail_audio_nodes=256,local_audio_neighbors=8,archive_events=64,archive_retrievals=4)


@pytest.fixture(scope='module')
def source(tmp_path_factory):
    import soundfile as sf
    root=tmp_path_factory.mktemp('grounded_stream')
    t=np.arange(22050*6)/22050
    signal=(.1*np.sin(2*np.pi*440*t)*(1+.2*np.sin(t*30))).astype(np.float32)
    sf.write(root/'track.wav',signal,22050)
    (root/'maidata.txt').write_text('&wholebpm=120\n&lv_5=13\n&inote_5=(120){8}1,2,3h[8:1],4,5,6,7,8,1-5[8:1],2,3,4,E\n',encoding='utf8')
    manifest=root/'manifest.json';manifest.write_text(json.dumps({'songs':[{'song_id':'fixture',
        'maidata_path':str(root/'maidata.txt'),'audio_path':str(root/'track.wav'),
        'difficulties':[{'difficulty_index':5,'level_raw':'13'}]}]}),encoding='utf8')
    return root,manifest


@pytest.fixture(scope='module')
def dataset(source):
    root,manifest=source
    return StreamDataset(manifest,cache_dir=root/'cache',max_tokens=32,max_history_tokens=12,
        preparation_workers=1,audio_cache_gb=.1,grounding_config=config())


def activate_residuals(model):
    for layer in (model.grounding.clock_projection,model.grounding.local_input,
                  model.grounding.fusion,model.grounding.intent_projection):
        torch.nn.init.normal_(layer.weight,std=.03)
    return model.eval()


def test_clock_does_not_reveal_position_until_note_and_uses_actual_seconds():
    codec=RationalCodec();tempo=[EventTempo(F(0),120.),EventTempo(F(4),60.)]
    def rows(text):
        c=CausalClock(codec,tempo,-.5,8.,previous=F(2),committed=3)
        ids=[codec.bos_id,codec.at_id,*[codec.ids['P:'+x] for x in text],codec.note_id]
        return [c.consume(i).copy() for i in ids]
    a,b=rows('7/3'),rows('8/3')
    np.testing.assert_array_equal(a[:-1],b[:-1])
    assert not np.array_equal(a[-1],b[-1])
    assert a[-1][13]*120==pytest.approx(seconds_at_beat(F(7,3),tempo,-.5),abs=1e-6)


def test_masked_prefix_preserves_numeric_positions_and_masks_only_its_own_targets():
    codec=RationalCodec();ids=codec.encode([EventPoint(F(1,17),'1/5'),EventPoint(F(3,7),'2h[4:1]')])
    targets=ids[1:];decoder=ids[:-1]
    corrupted,quality=masked_note_inputs(decoder,codec,random.Random(1),1.,targets=targets,quality=[1.]*len(targets))
    assert [t for t in corrupted if codec.tokens[t].startswith('P:')]==[t for t in decoder if codec.tokens[t].startswith('P:')]
    assert codec.mask_id in corrupted and quality[-1]==1. and sum(quality)==1.
    assert targets[-1]==codec.eos_id


def test_native_query_at_unusual_fraction_is_not_snapped(dataset):
    song=dataset.base.songs[0][0]
    values,times=features_at_beats(song,[F(37,997),F(11,17)])
    assert values.shape==(2,120)
    assert times[0]==pytest.approx(seconds_at_beat(F(37,997),song.tempo,song.first),abs=1e-12)


def test_grounded_item_never_archives_current_or_future_events(dataset):
    for i,chunk in enumerate(dataset.chunks):
        item=dataset[i]
        assert all(x<chunk.start_event for x in item['archive_event_index'].tolist())
        assert item['decoder_clock'].shape==(len(chunk.decoder_input),20)
        assert item['detail_audio'].shape[-1]==120
        assert item['query_audio'].shape[0]==len(item['query_clock'])
        assert item['curve_time_weight'].sum()==pytest.approx(1.,abs=1e-6)


def test_future_decoder_and_query_changes_cannot_change_earlier_predictions(dataset):
    torch.manual_seed(7);model=activate_residuals(ChartStreamTransformer(config()))
    batch=collate_stream([dataset[0]])
    cut=7;changed={k:v.clone() if torch.is_tensor(v) else v for k,v in batch.items()}
    protected=int(batch['decoder_query_index'][0,:cut].max())
    changed['query_audio'][:,protected+1:]=torch.randn_like(changed['query_audio'][:,protected+1:])*20
    changed['decoder_input'][:,cut:]=torch.randint(7,55,changed['decoder_input'][:,cut:].shape)
    with torch.inference_mode():
        a=model(batch)['logits'][:,:cut];b=model(changed)['logits'][:,:cut]
    torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)


def test_full_forward_equals_native_cached_queries_archive_and_intent(dataset):
    torch.manual_seed(13);model=activate_residuals(ChartStreamTransformer(config()))
    index=len(dataset.chunks)-1;item=dataset[index];chunk=dataset.chunks[index]
    song,points,*_=dataset.base.songs[chunk.chart_index]
    batch=collate_stream([item])
    with torch.inference_mode():
        expected=model(batch)['logits']
        encoded=model.encode_audio(batch['audio'],batch['condition'],audio_padding_mask=batch['audio_padding_mask'])
        session=GroundedInference(model,dataset.codec,song,encoded)
        for point in points[:chunk.start_event]:session.commit(point)
        session.begin_context(points[:chunk.start_event],list(chunk.history))
        summary,summary_mask=model.summarize_tokens(batch['history'],batch['history_padding_mask'],return_padding_mask=True)
        memory,mask=model.encode(batch['audio'],batch['condition'],encoded_audio=encoded,
            stream_memory=summary,stream_memory_padding_mask=summary_mask,
            action_state=batch['action_state'],action_state_padding_mask=batch['action_state_padding_mask'])
        memory,mask=model.prepare_decode_memory(memory,mask,history=batch['history'],
            history_padding_mask=batch['history_padding_mask'],history_clock=batch['history_clock'])
        session.set_prefix(list(chunk.decoder_input));cache=None;rows=[];committed=chunk.start_event
        for pos,token in enumerate(chunk.decoder_input):
            if token==dataset.codec.end_id:
                session.commit(points[committed]);committed+=1
            out,cache=session.decode_step(torch.tensor([[token]]),pos,memory,mask,cache)
            rows.append(out['logits'])
        actual=torch.cat(rows,1)
    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)


def test_migration_starts_at_legacy_logits_and_preserves_old_checkpoint(source,dataset):
    torch.manual_seed(4);old=ChartStreamTransformer(replace(config(),grounding_profile='none')).eval()
    new=upgrade_grounding(old,grounding_profile='causal_audio_v1').eval()
    batch=collate_stream([dataset[-1]])
    with torch.inference_mode():torch.testing.assert_close(old(batch)['logits'],new(batch)['logits'],rtol=1e-5,atol=1e-5)
    root,manifest=source;path=root/'legacy.pt'
    save(path,old,None,epoch=1,args={},manifest=manifest)
    restored,_=load(path)
    assert restored.config.grounding_profile=='none'


def test_grounded_losses_have_finite_independent_gradients(dataset):
    model=activate_residuals(ChartStreamTransformer(config())).train()
    batch=collate_stream([dataset[0],dataset[-1]])
    output=model(batch);loss,metrics=_loss(output,batch)
    assert torch.isfinite(loss)
    loss.backward()
    for name in ['alignment_query.weight','local_value.0.weight','clock_projection.weight','router.2.weight']:
        grad=dict(model.grounding.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all()
    values=_mean_metrics(metrics,1)
    assert 'whole_configuration_accuracy' in values and 'alignment_accuracy' in values


def test_incremental_action_state_matches_causal_full_reference():
    tempo=[EventTempo(F(0),120.)]
    points=[EventPoint(F(0),'1h[4:2]'),EventPoint(F(1),'3-7[8:1]'),EventPoint(F(2),'4')]
    incremental=IncrementalState(tempo,0.)
    for i,point in enumerate(points):
        incremental.append(point)
        full=SustainStateIndex(points[:i+1],tempo,0.)
        for beat in [point.beat,point.beat+F(1,8),point.beat+1]:
            a,am,_=incremental.at(beat,include_current=True)
            b,bm,_=full.at(beat,include_current=True)
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)
            assert torch.equal(am,bm)


def test_initial_hold_at_beat_zero_cannot_leak_before_first_note_is_committed():
    from maichart.stream.grounding import action_summaries
    state=SustainStateIndex([EventPoint(F(0),'1h[4:2]')],[EventTempo(F(0),120.)],0.)
    unseen=action_summaries(state,[F(0),F(0)],[False,True],[0,0])
    assert unseen.abs().sum()==0
    seen=action_summaries(state,[F(0)],[False],[1])
    assert seen[0,-1]>0


def test_time_quadrature_conserves_duration_and_excludes_outside_nodes():
    weights=time_cell_weights([-1.,0.,.1,.9,1.,2.],0.,1.)
    torch.testing.assert_close(weights,torch.tensor([0.,.05,.45,.5,0.,0.]))
    assert float((weights*3).sum())==pytest.approx(3.)


def test_local_lookup_keeps_negative_first_nodes_sorted_and_ignores_batch_padding():
    from maichart.stream.grounded_model import GroundedContext
    module=GroundedContext(replace(config(),local_audio_neighbors=1))
    module.local_value=torch.nn.Linear(120,16,bias=False)
    torch.nn.init.zeros_(module.local_value.weight);torch.nn.init.zeros_(module.local_query.weight)
    bank=module.prepare_bank(torch.zeros(1,4,120),torch.tensor([[-1.,0.,.5,float('inf')]],dtype=torch.float64),
                             torch.tensor([[True,False,False,True]]))
    bank['projected']=torch.tensor([0.,3.,6.,0.])[None,:,None].expand(1,4,16)
    output=module.local(torch.zeros(1,1,20),torch.zeros(1,1,dtype=torch.float64),torch.zeros(1,1,120),bank)
    torch.testing.assert_close(output,torch.ones(1,1,16))


def test_selection_supervision_trains_the_acoustic_values_used_by_continuation(dataset):
    model=ChartStreamTransformer(config()).train()
    output=model(collate_stream([dataset[0]]))
    output['position_alignment_logits'].square().mean().backward()
    gradient=model.grounding.local_value[0].weight.grad
    assert gradient is not None and gradient.abs().sum()>0
