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
from maichart.stream.grounding import CausalClock,clock_inputs,features_at_beats,masked_note_inputs,time_cell_weights,action_summaries
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
                  model.grounding.fusion,model.grounding.intent_projection,
                  model.grounding.kinematic_projection):
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
    assert codec.mask_id in corrupted and quality[-1]==1.
    assert [i for i,w in enumerate(quality) if w==0.]==[i for i,t in enumerate(corrupted) if t==codec.mask_id]
    # Position and <at> targets of fully masked events are still supervised.
    assert all(quality[i]==1. for i,t in enumerate(targets) if t==codec.at_id or codec.tokens[t].startswith('P:'))
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


def test_grounded_proposal_microbatches_preserve_scores_and_causal_clock(dataset):
    from maichart.form.rational_position_search import _score_proposals
    torch.manual_seed(23)
    model=activate_residuals(ChartStreamTransformer(config()))
    codec=dataset.codec
    item=collate_stream([dataset[0]])
    song=dataset.base.songs[0][0]
    with torch.inference_mode():
        encoded=model.encode_audio(item['audio'],item['condition'],audio_padding_mask=item['audio_padding_mask'])
        session=GroundedInference(model,codec,song,encoded)
        session.begin_context([],[])
        memory,mask=model.encode(item['audio'],item['condition'],encoded_audio=encoded)
        session.set_prefix([codec.bos_id,codec.at_id])
        cache=None
        for position,token in enumerate(session.prefix):
            root,cache=session.decode_step(torch.tensor([[token]]),position,memory,mask,cache)
        texts=['1/2','1','37/997']
        before=(session.clock.beat,session.clock.committed,session.clock_position)
        parallel,_=_score_proposals(session,codec,texts,root['logits'][0,-1],cache,memory,mask,
            consumed=2,timing_features=session.timing_features)
        serial,_=_score_proposals(session,codec,texts,root['logits'][0,-1],cache,memory,mask,
            consumed=2,timing_features=session.timing_features,max_batch_size=1)
        assert serial==pytest.approx(parallel,abs=1e-5)
        assert (session.clock.beat,session.clock.committed,session.clock_position)==before


def test_migration_starts_at_legacy_logits_and_preserves_old_checkpoint(source,dataset):
    torch.manual_seed(4);old=ChartStreamTransformer(replace(config(),grounding_profile='none')).eval()
    new=upgrade_grounding(old,grounding_profile='causal_audio_v1').eval()
    batch=collate_stream([dataset[-1]])
    with torch.inference_mode():torch.testing.assert_close(old(batch)['logits'],new(batch)['logits'],rtol=1e-5,atol=1e-5)
    root,manifest=source;path=root/'legacy.pt'
    save(path,old,None,epoch=1,args={},manifest=manifest)
    restored,_=load(path)
    assert restored.config.grounding_profile=='none'


def test_older_width_checkpoint_loads_with_zero_padded_kinematics(source,dataset):
    torch.manual_seed(5);model=ChartStreamTransformer(config()).eval()
    with torch.no_grad():
        model.grounding.kinematic_projection.weight[:,:44].normal_(std=.05)
    root,manifest=source;path=root/'v10_width.pt'
    save(path,model,None,epoch=1,args={},manifest=manifest)
    payload=torch.load(path,map_location='cpu',weights_only=False)
    state=payload['model_state_dict']
    state['grounding.kinematic_projection.weight']=state['grounding.kinematic_projection.weight'][:,:44].clone()
    state['grounding.blueprint_head.weight']=torch.zeros(23,16)
    torch.save(payload,path)
    restored,_=load(path)
    assert restored.grounding.kinematic_projection.weight.shape[1]>44
    assert not restored.grounding.kinematic_projection.weight[:,44:].any()
    batch=collate_stream([dataset[-1]])
    batch['query_kinematics'][...,44:]=0.
    with torch.inference_mode():
        torch.testing.assert_close(model(batch)['logits'],restored.eval()(batch)['logits'],rtol=1e-5,atol=1e-5)


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


def test_in_flight_kinematics_and_effective_lanes_preserve_chained_and_masked_slides():
    from maichart.stream.symmetry import _effective_lanes,kinematic_rows,SLIDE
    chained=((SLIDE,1,5,.1,.5),(SLIDE,5,2,.1,.5))
    assert _effective_lanes((1,5),chained)==(2,5)
    # Event 0: Slide 1-5 starting at 0.0, tracing at 0.5, ending at 1.5.
    # Event 1: Tap 5 at 1.5 (one-stroke hand-off).
    lanes=[(1,),(5,)]
    trajs=[((SLIDE,1,5,.5,1.5),),()]
    # Query at t=0.25 (star wait) and t=1.0 (sliding) after committing event 0;
    # and at t=1.6 after committing event 1.
    rows=kinematic_rows(lanes,[0.,1.5],[1,1,2],[.25,1.0,1.6],trajectories=trajs)
    assert rows.shape==(3,65)
    # While in flight, destination lane 5 (index 44+4=48) is active and grows as arrival nears.
    assert .5 < float(rows[0,48]) < float(rows[1,48]) < 1.
    # Origin lane 1 (index 52+0=52) is 1.0 during star wait and 0.5 while sliding.
    assert float(rows[0,52])==pytest.approx(1.) and float(rows[1,52])==pytest.approx(.5)
    assert float(rows[0,60])==pytest.approx(.25) and float(rows[0,61])>float(rows[1,61])>0.
    # After Slide finishes at 1.5 and event 1 strikes lane 5, effective step (1-5 -> 5) has 0 angle (sin=0, cos=1).
    assert float(rows[2,48])==0. and float(rows[2,36+4])>0.
    assert float(rows[2,62])==pytest.approx(0.,abs=1e-6) and float(rows[2,63])==pytest.approx(1.,abs=1e-6)
    # Masking event 0 via note dropout hides both its head lane and in-flight Slide trajectory.
    masked_rows=kinematic_rows([(),(5,)],[0.,1.5],[1,2],[1.0,1.6],trajectories=[(),()])
    assert float(masked_rows[0,:16].abs().sum())==0. and float(masked_rows[0,36:].abs().sum())==0.
    assert float(masked_rows[1,64])==0.


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


def test_overlap_dataset_preserves_single_supervision_and_cached_clock_contract(source):
    root,manifest=source
    cfg=replace(config(),continuation_profile='overlap_v1',context_replay_tokens=12)
    ds=StreamDataset(manifest,cache_dir=root/'overlap_cache',max_tokens=32,max_history_tokens=12,
        preparation_workers=1,audio_cache_gb=.1,grounding_config=cfg)
    original=ds.codec.encode(ds.base.songs[0][1])[1:]
    supervised=[t for chunk in ds.chunks for t,w in zip(chunk.target,chunk.target_quality) if w>0]
    assert supervised==original
    assert any(chunk.target_quality[0]==0 for chunk in ds.chunks[1:])
    test_full_forward_equals_native_cached_queries_archive_and_intent(ds)


def test_replayed_action_queries_recover_pruned_holds_without_future_leakage(dataset):
    model=activate_residuals(ChartStreamTransformer(config()))
    torch.nn.init.zeros_(model.grounding.kinematic_projection.weight)   # isolate the action-state path
    song=dataset.base.songs[0][0];codec=dataset.codec
    points=[EventPoint(F(0),'1h[4:2]'),EventPoint(F(1),'3'),EventPoint(F(4),'5h[4:2]')]
    batch=collate_stream([dataset[0]])
    with torch.inference_mode():
        encoded=model.encode_audio(batch['audio'],batch['condition'],audio_padding_mask=batch['audio_padding_mask'])
        session=GroundedInference(model,codec,song,encoded)
        for point in points:session.commit(point)
        session.begin_context([],[])
        assert session.replay_state is not None
        assert session._context(torch.zeros(1,1,20))['action'].abs().sum()==0
        for token in codec.cell(points[0]):session.clock.consume(token)
        features=torch.as_tensor(session.clock._features)[None,None]
        actual=session._context(features)['action']
        expected=action_summaries(SustainStateIndex(points[:1],song.tempo,song.first),[F(0)],[False],[1])[None]
        torch.testing.assert_close(actual,model.grounding.action_value(expected)*expected[...,-1:].gt(0))


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
