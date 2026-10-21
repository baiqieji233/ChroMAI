"""Musical patterns, exact causal masking, migration and bounded hand memory."""
from dataclasses import replace
from fractions import Fraction as F
from types import SimpleNamespace
import json,random,time

import numpy as np
import pytest
import torch

from maichart.event_timing import EventPoint,EventTempo
from maichart.form.codec import RationalCodec
from maichart.form.type_guidance import TypeGuidance
from maichart.generation.conditions import GenerationCondition,GENERATION_DEFAULTS
from maichart.stream.hand_tracking import HandHistory,HAND_DIM,Assignment,advance
from maichart.stream.symmetry import kinematic_rows
from maichart.stream.grounding import CausalClock,clock_inputs,masked_clock_inputs
from maichart.stream.timing import interval_reliability,interval_consistency_cost,rhythm_cycle_cost
from maichart.stream.checkpoint import save,load,upgrade_grounding,TRAINING_OBJECTIVE_CONTRACT
from maichart.stream.data import StreamDataset,collate_stream,TRAINING_SEMANTICS
from maichart.stream.model import ChartStreamTransformer
from test_stream_grounding import source,config,cpu_threads,activate_residuals


@pytest.mark.parametrize('pattern',[['1','2','3','4'],['1','8','1','8'],['1','8','2','7']])
def test_sixteenth_sweeps_and_trills_do_not_force_new_lanes_or_rhythm_changes(pattern):
    atoms=[str(i) for i in range(1,9)]+['1h[8:1]','1-5[16:1]']
    condition=GenerationCondition(level=14.6,radar=GENERATION_DEFAULTS,style=GENERATION_DEFAULTS)
    guide=TypeGuidance(atoms,condition,[],planner_strength=0,style_strength=1.)
    points=[]
    for i,lane in enumerate(pattern*20):
        guide.commit(lane,F(i,4));points.append(EventPoint(F(i,4),lane))
    event=guide.event(F(80,4))
    assert event.token_bias('',list('12345678'))==[0.]*8
    assert rhythm_cycle_cost(F(80,4),points,points[-1].beat,.6)==0.


def test_identical_durations_do_not_bias_digits_or_closing_brackets_even_across_types():
    atoms=['1','8','1h[16:3]','8h[16:3]','1-5[16:3]','1-5[16:1]']
    condition=GenerationCondition(level=14,radar=GENERATION_DEFAULTS,style=GENERATION_DEFAULTS)
    guide=TypeGuidance(atoms,condition,[],planner_strength=0,style_strength=1.)
    for i in range(30):guide.commit('1h[16:3]' if i%2 else '1-5[16:3]',F(i))
    bias=guide.event(F(31))
    for prefix,choices in [('1-5[',['1','8']),('1-5[16:',['1','3']),('1-5[16:3',[']','0'])]:
        assert bias.token_bias(prefix,choices)==[0.]*len(choices)


def test_shared_interval_mask_hides_current_and_prior_echo_at_the_right_event():
    codec=RationalCodec();points=[EventPoint(F(0),'1'),EventPoint(F(1,4),'2'),EventPoint(F(1,2),'3')]
    ids=codec.encode(points);decoder=ids[:-1];targets=ids[1:]
    song=SimpleNamespace(tempo=[EventTempo(F(0),120.)],first=0.,audio=SimpleNamespace(duration_sec=10.),meters=None)
    raw=clock_inputs(decoder,codec,song)
    clock,query,chosen=masked_clock_inputs(raw['decoder_clock'],decoder,codec,random.Random(7),(),
        dropout=1.,query_clock=raw['query_clock'],query_index=raw['decoder_query_index'],targets=targets,
        return_interval_events=True)
    assert chosen==frozenset(range(3))
    pending=raw['query_clock'][:,18].numpy()>0
    counts=raw['query_committed'].numpy()
    rows=kinematic_rows([(1,),(2,),(3,)],[0.,.125,.25],counts,raw['query_seconds'].numpy(),
        interval_hidden=chosen,pending=pending,profile='causal_hands_v1')
    assert not rows[:,16:18].any()
    assert not query[:,15:17].any()
    torch.testing.assert_close(clock[:,:15],raw['decoder_clock'][:,:15],atol=0,rtol=0)
    torch.testing.assert_close(clock[:,17:],raw['decoder_clock'][:,17:],atol=0,rtol=0)
    # Current event isn't committed at NOTE: current age is owned by count,
    # while completed-interval echo is owned by count-1.
    unmasked=kinematic_rows([(1,),(2,)],[0.,.125],[1],[.125],pending=[True],profile='causal_hands_v1')
    hidden=kinematic_rows([(1,),(2,)],[0.,.125],[1],[.125],pending=[True],interval_hidden={1},profile='causal_hands_v1')
    assert unmasked[0,16]>0 and hidden[0,16]==0


def test_known_touch_indicator_is_not_decay_or_unknown_mask_in_the_new_profile():
    rows=kinematic_rows([(),(),()],[0.,.1,.2],[1,2,3],[0.,.1,.2],pending=[False]*3,profile='causal_hands_v1')
    assert rows[:,23].tolist()==[1.,1.,1.]
    unknown=kinematic_rows([(),(),()],[0.,.1,.2],[1,2,3],[0.,.1,.2],pending=[False]*3,
                           masked_events={0,1,2},profile='causal_hands_v1')
    assert not unknown[:,23].any()


def test_hand_hypotheses_keep_single_hand_sweep_and_two_hand_interaction_possible():
    for pattern in [(1,2,3,4),(1,8,1,8),(2,2,3,4)]:
        states=(Assignment(),)
        for i,lane in enumerate(pattern):states=advance(states,(lane,),i*.079)
        assert any(s.repeated for s in states)
        assert any(s.switched for s in states)
        assert len(states)<=8
        if pattern==(1,2,3,4):assert any(max(h[4] for h in s.hands)>=4 for s in states)
    timeline=HandHistory([(i,) for i in (1,8,1,8)],[0.,.08,.16,.24])
    rows=timeline.rows([1,2,3,4],[0.,.08,.16,.24])
    assert rows.shape==(4,HAND_DIM) and torch.isfinite(rows).all()
    assert torch.all(rows[:,19:21]>=0) and torch.all(rows[:,19:21]<=1)


def test_hand_history_is_prefix_causal_masked_and_retains_long_slide_uncertainty():
    lanes=[(i%8+1,) for i in range(100)]
    times=np.arange(100)*.05
    trajectories=[() for _ in times];trajectories[0]=((1,1,5,.1,10.),)
    timeline=HandHistory(lanes,times,trajectories)
    full=timeline.rows([2,20,60],[.1,1.,3.])
    for index,(count,now) in enumerate(zip([2,20,60],[.1,1.,3.])):
        prefix=HandHistory(lanes[:count],times[:count],trajectories[:count]).rows([count],[now])
        torch.testing.assert_close(full[index],prefix[0],rtol=0,atol=0)
    assert full[-1,23]==1.  # old Slide still tracing beyond 32 events
    masked=timeline.rows([1],[.05],masked_events={0})
    assert not masked[0,:16].any()
    future=lanes.copy();future[90]=(1,8)
    torch.testing.assert_close(full,HandHistory(future,times,trajectories).rows([2,20,60],[.1,1.,3.]),rtol=0,atol=0)
    assert len(timeline.recent)<=64 and len(timeline.checkpoints)<=1+len(lanes)//32


def test_interval_reliability_is_weak_for_old_or_unstable_heads_and_never_changes_allowed_fractions():
    assert interval_reliability({})['multiplier']==.25
    assert interval_reliability({'interval_log_mae':1.,'interval_targets':100})['multiplier']<.25
    assert interval_reliability({'interval_log_mae':0.,'interval_targets':100})['multiplier']==.75
    for prediction in (-.1,0.,.01,.25,.9):
        cost=interval_consistency_cost(F(1,997),prediction,.35,reliability=.25,cap=.5)
        assert 0<=cost<=.35*.25*.5
    # Separate explicit gap protection survives conservative local shaping.
    assert interval_consistency_cost(F(161),.5/8,.35,.5,reliability=.25,cap=.5)>10
    assert interval_consistency_cost(F(161),1.,.35,.5,reliability=.25,cap=.5)==0


def test_v14_migration_preserves_old_generation_and_blocks_unversioned_resume(source,tmp_path):
    model=ChartStreamTransformer(config()).eval()
    torch.nn.init.normal_(model.grounding.kinematic_projection.weight)
    old=tmp_path/'old.pt'
    manifest=source[1]
    save(old,model,None,epoch=1,args={'training_semantics':'chart_stream_training_v13_bounded_cycle_exit'},manifest=manifest)
    restored,payload=load(old)
    assert restored.config.motor_profile=='legacy_v13'
    torch.testing.assert_close(restored.grounding.kinematic_projection.weight,model.grounding.kinematic_projection.weight)
    upgraded=upgrade_grounding(restored,motor_profile='causal_hands_v1')
    assert not upgraded.grounding.hand_projection.weight.any()
    assert not upgraded.grounding.kinematic_projection.weight[:,23].any()
    with pytest.raises(ValueError,match='remove trained hand'):
        upgrade_grounding(upgraded,motor_profile='legacy_v13')
    new=tmp_path/'new.pt'
    saved=save(new,upgraded,torch.optim.AdamW(upgraded.parameters()),epoch=1,args={'training_semantics':TRAINING_SEMANTICS},
        manifest=manifest,dataset_fingerprint='same')
    assert saved['training_objective_contract']==TRAINING_OBJECTIVE_CONTRACT
    from maichart.webui.stream_api import inspect_payload
    assert not inspect_payload(payload)['can_resume'] and inspect_payload(saved)['can_resume']
    corrupt={**saved,'model_state_dict':dict(saved['model_state_dict'])}
    corrupt['model_state_dict'].pop('grounding.hand_projection.weight')
    assert not inspect_payload(corrupt)['valid']
    broken=tmp_path/'broken.pt';torch.save(corrupt,broken)
    with pytest.raises(ValueError,match='hand_projection'):load(broken)


def test_causal_hand_profile_is_used_by_training_and_incremental_inference(source,tmp_path):
    cfg=replace(config(),motor_profile='causal_hands_v1')
    dataset=StreamDataset(source[1],cache_dir=tmp_path/'cache',max_tokens=32,max_history_tokens=12,
        grounding_config=cfg,preparation_workers=1,audio_cache_gb=.1)
    index=len(dataset.chunks)-1;chunk=dataset.chunks[index];item=dataset[index]
    batch=collate_stream([item]);model=activate_residuals(ChartStreamTransformer(cfg))
    torch.nn.init.normal_(model.grounding.hand_projection.weight,std=.03)
    from maichart.stream.grounded_inference import GroundedInference
    song,points,*_=dataset.base.songs[0]
    with torch.inference_mode():
        expected=model(batch)['logits']
        encoded=model.encode_audio(batch['audio'],batch['condition'],audio_padding_mask=batch['audio_padding_mask'])
        session=GroundedInference(model,dataset.codec,song,encoded,audio_times=batch['audio_times_sec'][0].numpy())
        for point in points[:chunk.start_event]:session.commit(point)
        session.begin_context(points[:chunk.start_event],list(chunk.history))
        summary,padding=model.summarize_tokens(batch['history'],batch['history_padding_mask'],return_padding_mask=True)
        memory,mask=model.encode(batch['audio'],batch['condition'],encoded_audio=encoded,stream_memory=summary,
            stream_memory_padding_mask=padding,action_state=batch['action_state'],
            action_state_padding_mask=batch['action_state_padding_mask'])
        memory,mask=model.prepare_decode_memory(memory,mask,history=batch['history'],history_clock=batch['history_clock'])
        session.set_prefix(list(chunk.decoder_input));cache=None;committed=chunk.start_event;rows=[]
        for pos,token in enumerate(chunk.decoder_input):
            if token==dataset.codec.end_id:session.commit(points[committed]);committed+=1
            output,cache=session.decode_step(torch.tensor([[token]]),pos,memory,mask,cache)
            rows.append(output['logits'])
        actual=torch.cat(rows,1)
    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
    assert batch['query_hands'].shape[-1]==HAND_DIM
    missing=dict(batch);missing.pop('query_hands')
    with pytest.raises(ValueError,match='query_hands'):model(missing)


def test_dataset_random_interval_masks_are_shared_by_clock_kinematics_and_hands(source,tmp_path):
    cfg=replace(config(),motor_profile='causal_hands_v1')
    dataset=StreamDataset(source[1],cache_dir=tmp_path/'cache',max_tokens=32,max_history_tokens=12,
        grounding_config=cfg,preparation_workers=1,audio_cache_gb=.1,
        prefix_note_dropout=1.,history_note_dropout=1.)
    item=dataset[0]
    assert not item['query_clock'][:,15:17].any()
    assert not item['query_kinematics'][:,16:18].any()
    assert not item['query_hands'][:,[4,12]].any()
    # MASK keeps exact beat/second coordinates; this is echo-channel dropout,
    # not deletion of positions or invented negative training continuations.
    raw=dataset._structure(dataset.chunks[0],dataset.base.songs[0][0],dataset.base.songs[0][1])
    torch.testing.assert_close(item['decoder_clock'][:,:15],raw['decoder_clock'][:,:15],rtol=0,atol=0)


def test_history_interval_mask_also_covers_the_next_block_initial_query(source,tmp_path):
    cfg=replace(config(),motor_profile='causal_hands_v1')
    dataset=StreamDataset(source[1],cache_dir=tmp_path/'cache',max_tokens=32,max_history_tokens=12,
        grounding_config=cfg,preparation_workers=1,audio_cache_gb=.1,
        prefix_note_dropout=0.,history_note_dropout=1.)
    chunk=dataset.chunks[-1]
    assert chunk.start_event>=2
    item=dataset[len(dataset)-1]
    raw=dataset._structure(chunk,dataset.base.songs[0][0],dataset.base.songs[0][1])
    assert raw['query_clock'][0,15:17].abs().sum()>0
    assert not item['query_clock'][0,15:17].any()
    assert not item['decoder_clock'][0,15:17].any()
    assert item['query_kinematics'][0,17]==0.


def test_cpu_finetune_from_v13_completes_new_feature_training_and_checkpoint_contract(source,tmp_path):
    from maichart.stream.train import main
    torch.manual_seed(5)
    old=tmp_path/'parent'/'old.pt'
    model=ChartStreamTransformer(config())
    save(old,model,None,epoch=3,args={'training_semantics':'chart_stream_training_v13_bounded_cycle_exit'},manifest=source[1])
    before=old.read_bytes()
    output=tmp_path/'finetune'
    argv=['--manifest',str(source[1]),'--checkpoint-dir',str(output),'--init-checkpoint',str(old),
        '--cache-dir',str(tmp_path/'cache'),'--epochs','1','--device','cpu','--precision','fp32',
        '--batch-size','2','--cpu-threads','1','--prepare-workers','1','--audio-cache-gb','.05',
        '--detail-audio-nodes','256','--archive-events','64','--archive-retrievals','4',
        '--local-audio-neighbors','8','--symmetry-augmentation','none','--log-interval','1']
    assert main(argv)==0
    restored,payload=load(output/'stream_last.pt')
    assert restored.config.motor_profile=='causal_hands_v1'
    assert payload['training_objective_contract']==TRAINING_OBJECTIVE_CONTRACT
    assert payload['args']['training_semantics']==TRAINING_SEMANTICS
    assert payload['metrics']['interval_targets']>0
    assert restored.grounding.hand_projection.weight.abs().sum()>0
    assert old.read_bytes()==before
