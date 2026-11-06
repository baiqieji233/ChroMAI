"""Real teacher/generation readout parity for the complete new contract."""
from fractions import Fraction
from types import SimpleNamespace

import pytest
import torch

from maichart.composer.stream_data import NOTE, StreamDataset, collate_stream
from maichart.composer.stream_model import StreamComposer
from maichart.composer.event_set_codec import EventSetCodec
from maichart.event_timing import EventPoint
from maichart.composer.stream_duration import select_duration_context
from maichart.composer.stream_proposal import prepare_selected_candidate
from maichart.composer.stream_phrase_runtime import ensure_phrase_mode,execution_options,execution_metadata
from maichart.composer.stream_recovery import _context
from test_stream_draft_integration import build


def _target(ds, crossing, *, literal=False):
    choices=[]
    for sample,(_,start,end,_) in enumerate(ds.samples):
        item=ds[sample];first=int(item['step_index'][0])
        for at in range(start,end):
            position=at-first
            if ds.streams[0][at].kind!=NOTE:continue
            if bool(item['target_shape'][position]==0)!=literal:continue
            different=bool(item['candidate_phrase_start_beat'][position]!=item['phrase_start_beat'][position])
            if different==crossing:
                # Prefer a hold with a genuinely nontrivial duration catalog.
                choices.append(('h' not in ds.streams[0][at].text,sample,at,position,item))
    assert choices, 'Fixture must contain both ordinary and cross-phrase NOTE arrivals'
    _,sample,at,position,item=min(choices,key=lambda row:row[:4])
    return sample,at,position,item


def _touch_build(monkeypatch):
    old,model,song,bank=build(monkeypatch)
    texts=['1','A1/B2','3h[4:1]','C','5-1[8:1]','A4','7/3','B7/C',
        'Ch[8:1]','E3','3/7','C']*2
    codec=EventSetCodec.fit(texts)
    points=[EventPoint(Fraction(index),text) for index,text in enumerate(texts)]
    ds=StreamDataset(base=SimpleNamespace(songs=[(song,points,13.,1.,None,torch.zeros(16),(),None)]),
        codec=codec,lead_in=8,targets=4,audio_grid_seconds=.05,
        history_window_policy='block_aligned_v2',continuation_policy='pattern_runs_v2',
        phrase_context_policy='shared_ordered_v4',candidate_state_policy='selected_native_v1',
        training_window_policy='phrase_complete_v1',max_phrase_events=64,phrase_seam_events=2,
        relation_policy='future_sets_v1',draft_policy='beat_ordered_draft_v1',unified_arrival=True)
    return ds,StreamComposer(model.config,codec).eval(),song,bank


@pytest.mark.parametrize('crossing',[False,True],ids=['within_phrase','first_crossing_note'])
def test_new_contract_teacher_and_real_generation_readouts_agree(monkeypatch,crossing):
    torch.manual_seed(1729)
    ds,model,song,bank=build(monkeypatch);model.eval()
    # Zero initialization must not hide a missing draft input path.
    with torch.no_grad():
        for layer in (model.draft_action_projection,model.draft_detail_projection,model.draft_rate_projection):
            layer.weight.normal_(0,.035)
        model.phrase_context.ordered.alignment_output.weight.normal_(0,.03)
    _,at,position,item=_target(ds,crossing)
    teacher_batch=collate_stream([item]);target=ds.streams[0][at]
    with torch.no_grad():
        teacher=model(teacher_batch)
        controls=teacher['_causal_profile_condition'][0].clone()
        selected_z=int(teacher['selected_song_component'][0])
        evidence=teacher['unified_profile_evidence']
        context=_context(ds,0,bank);context.condition=controls;context.song_response_choice=selected_z
        for step in ds.streams[0][:at]:context.commit(step)
        raw=context.current_batch()
        audio=model.encode_audio(raw['song_audio'],raw['song_mask'],raw['song_seconds'],raw['song_beats'])
        audio['unified_style']=dict(prior_logits=evidence['prior_logits'].clone(),
            causal_condition=controls[None],known_mask=evidence['known_mask'].clone(),choice=selected_z)
        audio=model.prepare_plan_cache(audio,controls[None],torch.tensor([0]),raw['audio_end_sec'])
        generated=model(raw,audio_cache=audio)
        selected,prepared=prepare_selected_candidate(model,context,raw,generated,target.end,audio)
        seconds=torch.full_like(selected['step_second'],target.end_second)
        actual,load,prepared=model.candidate_action_state(prepared,selected,audio,seconds,return_button_load=True)
        torch.testing.assert_close(teacher['action_context'][0,position],actual[0,-1],rtol=1e-5,atol=2e-5)
        torch.testing.assert_close(teacher['draft_action_context'][0,position],prepared['draft_context'][0,-1],rtol=1e-5,atol=2e-5)
        expected_prior=teacher['action_execution_prior_logits'][0,position,selected_z]
        actual_prior=model.execution_prior(prepared,selected,song_choice=selected_z)[0,-1]
        torch.testing.assert_close(actual_prior.softmax(-1),expected_prior.softmax(-1),rtol=1e-5,atol=2e-6)
        generator=torch.Generator().manual_seed(19)
        ensure_phrase_mode(model,context,generated,raw,generator,temperature=0.)
        wanted=int(expected_prior.argmax())
        if crossing:
            # An already selected old mode must not own the new phrase's
            # first mark, even when the two priors happen to share argmax.
            old_start=float(raw['phrase_start_beat'][0,-1])
            context._execution_runtime['pending'][old_start]=(wanted+1)%model.config.phrase_modes
        selected_proposal=execution_options(model,context,prepared,selected,actual[0,-1],generator,
            temperature=0.,beat=target.end)
        metadata=execution_metadata(context,target.end)
        assert metadata['phrase_mode']==wanted
        assert metadata['phrase_mode_start']==float(selected['candidate_phrase_start_beat'][0,-1])
        torch.testing.assert_close(selected_proposal['execution_route_log_probs'].softmax(-1),
            teacher['execution_route_log_probs'][0,position,wanted].softmax(-1),rtol=1e-5,atol=2e-6)
        shape=int(teacher_batch['target_shape'][0,position])
        assert shape>0
        phases=teacher_batch['target_phase_mask'][0,position].nonzero(as_tuple=False)[:,0]
        phase=phases[:1]
        detail=teacher_batch['target_details'][0,position:position+1,:len(model.codec.shape_catalog[shape])]
        teacher_duration=select_duration_context(teacher['duration_audio_context'],torch.tensor([position]))
        generated_duration=model.duration_context_at(prepared,selected,seconds,audio)
        generated_duration=select_duration_context(generated_duration,torch.tensor([selected['step_second'].numel()-1]))
        teacher_detail_context=model.draft_detail_projection(teacher['draft_action_context'][0,position])[None]
        generated_detail_context=model.draft_detail_projection(prepared['draft_context'][0,-1])[None]
        detail_bank=model.shared_details.encode_catalog()
        _,teacher_details=model.regular_detail_log_probs(teacher['action_context'][0,position:position+1],
            shape,phase,detail,duration_context=teacher_duration,detail_bank=detail_bank,return_modes=True,
            mode_context=teacher_detail_context)
        _,generated_details=model.regular_detail_log_probs(actual[0,-1:],shape,phase,detail,
            duration_context=generated_duration,detail_bank=detail_bank,return_modes=True,
            mode_context=generated_detail_context)
        torch.testing.assert_close(teacher_details,generated_details,rtol=1e-5,atol=2e-5)
        for mode in range(model.config.phrase_modes):
            proposal=model.execution_proposal(actual[0,-1],prepared,selected,mode)
            expected_routes=teacher['execution_route_log_probs'][0,position,mode]
            torch.testing.assert_close(proposal['execution_route_log_probs'].softmax(-1),expected_routes.softmax(-1),rtol=1e-5,atol=2e-6)
            motion=prepared.get('motion_phase_distribution')
            if motion is not None:motion=motion[:,-1]
            base_phase=model.event_phase_logits(actual[0,-1:],torch.tensor([shape]),
                prepared['reference_phase'][:,-1],prepared['reference_phase_mask'][:,-1],
                motion_phase_distribution=motion,button_load=load[:,-1])
            phase_scores=base_phase+model.phrase_execution.phase_bias(proposal['execution_lane_bias'][None],torch.tensor([shape]))
            torch.testing.assert_close(phase_scores.softmax(-1)[0],
                teacher['execution_phase_logits'][0,position,mode].softmax(-1),rtol=1e-5,atol=2e-6)
            expected_mode=teacher_detail_context[0,mode]-teacher_detail_context[0].mean(0)
            torch.testing.assert_close(proposal['detail_mode']['context'],expected_mode,rtol=1e-5,atol=2e-5)
            # This is the exact detail-bias interface consumed by sampling.
            candidates=model.shared_details.candidate_weights(shape,0,detail_bank)
            sample_bias=model.detail_mode_control.bias(candidates,proposal['detail_mode'])
            train_bias=model.detail_mode_control.bias(candidates,teacher_detail_context)[0,mode]
            torch.testing.assert_close(sample_bias,train_bias,rtol=1e-5,atol=2e-5)


@pytest.mark.parametrize('crossing',[False,True],ids=['within_phrase','first_crossing_touch'])
def test_new_contract_literal_touch_and_stop_conditioning_agree(monkeypatch,crossing):
    torch.manual_seed(1819)
    ds,model,_,bank=_touch_build(monkeypatch)
    with torch.no_grad():
        model.draft_action_projection.weight.normal_(0,.035)
        model.draft_detail_projection.weight.normal_(0,.035)
    _,at,position,item=_target(ds,crossing,literal=True)
    batch=collate_stream([item]);target=ds.streams[0][at]
    with torch.no_grad():
        teacher=model(batch);controls=teacher['_causal_profile_condition'][0].clone()
        choice=int(teacher['selected_song_component'][0]);evidence=teacher['unified_profile_evidence']
        context=_context(ds,0,bank);context.condition=controls;context.song_response_choice=choice
        for step in ds.streams[0][:at]:context.commit(step)
        raw=context.current_batch()
        cache=model.encode_audio(raw['song_audio'],raw['song_mask'],raw['song_seconds'],raw['song_beats'])
        cache['unified_style']=dict(prior_logits=evidence['prior_logits'].clone(),
            causal_condition=controls[None],known_mask=evidence['known_mask'].clone(),choice=choice)
        cache=model.prepare_plan_cache(cache,controls[None],torch.tensor([0]),raw['audio_end_sec'])
        output=model(raw,audio_cache=cache)
        selected,prepared=prepare_selected_candidate(model,context,raw,output,target.end,cache)
        actual,prepared=model.candidate_action_state(prepared,selected,cache,
            torch.full_like(selected['step_second'],target.end_second))
        torch.testing.assert_close(actual[0,-1],teacher['action_context'][0,position],rtol=1e-5,atol=2e-5)
        tmode=model.draft_detail_projection(teacher['draft_action_context'][0,position])[None]
        gmode=model.draft_detail_projection(prepared['draft_context'][0,-1])[None]
        targets=batch['target_details'][0,position:position+1]
        # Include the real Touch/TouchHold family and variable-length STOP.
        atom_families=model.semantic_atom_families[targets.clamp_min(0)]
        bits=torch.tensor([1,2,4,8,16])
        family=((atom_families[...,None]&bits).ne(0).any(-2).long()*bits).sum(-1)
        detail_bank=model.shared_details.encode_catalog()
        tresult=model.literal_block_nll(teacher['action_context'][0,position:position+1],targets,family,
            return_stop=True,return_modes=True,detail_bank=detail_bank,mode_context=tmode)
        gresult=model.literal_block_nll(actual[0,-1:],targets,family,
            return_stop=True,return_modes=True,detail_bank=detail_bank,mode_context=gmode)
        for expected,observed in zip(tresult,gresult):
            torch.testing.assert_close(observed,expected,rtol=1e-5,atol=2e-5)
        ids=targets[targets>0]
        for mode in range(model.config.phrase_modes):
            proposal=model.execution_proposal(actual[0,-1],prepared,selected,mode)
            torch.testing.assert_close(proposal['execution_route_log_probs'].softmax(-1),
                teacher['execution_route_log_probs'][0,position,mode].softmax(-1),rtol=1e-5,atol=2e-6)
            torch.testing.assert_close(model.literal_mode_bias(ids,proposal['detail_mode'],detail_bank=detail_bank),
                model.literal_mode_bias(ids,tmode,detail_bank=detail_bank)[0,mode],rtol=1e-5,atol=2e-5)
            for type_id in (3,4):
                torch.testing.assert_close(model.detail_mode_control.stop_bias(type_id,proposal['detail_mode']),
                    model.detail_mode_control.stop_bias(type_id,tmode)[0,mode],rtol=1e-5,atol=2e-5)
