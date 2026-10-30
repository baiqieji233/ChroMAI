"""Bounded preview safety and sampling contracts; run in final acceptance."""
from copy import deepcopy
from fractions import Fraction

import numpy as np
import pytest
import torch

from maichart.composer.generate import parser, _same_intent_slot_masks, _sample_actions
from maichart.composer.guard import EventGuard
from maichart.composer.lookahead import (
    bounded_physical_lookahead, clone_generator, event_intent, short_gap_candidate,FutureAssessment,
)
from maichart.event_timing import EventPoint, EventTempo
from maichart.generation.event_cli import _NoteGrammar


def _guard():
    return EventGuard([EventTempo(Fraction(0), 180.)], 0., 20.,
                      min_same_head_sec=.05, touch_layout_policy='open')


def test_preview_switches_only_to_same_intent_without_mutating_guard_or_rng():
    guard = _guard()
    prior = EventPoint(Fraction(0), '8')
    guard.commit(prior)
    snapshot = deepcopy(guard.__dict__)
    rng = torch.Generator().manual_seed(451)
    state = rng.get_state().clone()
    global_state = torch.get_rng_state().clone()
    draws = []
    calls = []
    baseline = EventPoint(Fraction(1), '1')

    def alternative(private):
        assert private is not rng
        calls.append('alternate')
        torch.rand(4, generator=private)
        return EventPoint(Fraction(1), '2')

    def future(branch, current, beat, private):
        assert branch is not guard
        assert branch.prefix[-1][0] == current
        assert branch.last_head_seconds[current.note_text] == pytest.approx(1/3)
        calls.append(current.note_text)
        draws.append(torch.rand(3, generator=private))
        return EventPoint(beat, '1')

    result = bounded_physical_lookahead(baseline, future_beat=Fraction(9, 8),
        guard=guard, generator=rng, alternative_factory=alternative, future_factory=future)
    assert result.switched and result.event.note_text == '2'
    assert result.future_trials == 2 and result.alternate_draws == 1
    assert calls == ['1', 'alternate', '2']
    torch.testing.assert_close(draws[0], draws[1], rtol=0, atol=0)
    assert torch.equal(rng.get_state(), state)
    assert torch.equal(torch.get_rng_state(), global_state)
    assert guard.__dict__ == snapshot


@pytest.mark.parametrize('alternative_text', ['1/2', '2h[4:1]', 'A1'])
def test_preview_never_trades_away_event_quantity_or_kind(alternative_text):
    calls=[]
    result=bounded_physical_lookahead(EventPoint(Fraction(1),'1'), future_beat=Fraction(9,8),
        guard=_guard(), generator=torch.Generator().manual_seed(1),
        alternative_factory=lambda rng: EventPoint(Fraction(1),alternative_text),
        future_factory=lambda branch,current,beat,rng: (calls.append(current),EventPoint(beat,'1'))[1])
    assert not result.switched and result.reason=='different_event_intent'
    assert len(calls)==1


def test_preview_keeps_legal_original_without_drawing_alternative():
    def unexpected(_):
        raise AssertionError('A legal baseline future needs no second current proposal')
    baseline=EventPoint(Fraction(1),'1')
    result=bounded_physical_lookahead(baseline,future_beat=Fraction(9,8),guard=_guard(),
        generator=torch.Generator().manual_seed(2),alternative_factory=unexpected,
        future_factory=lambda branch,current,beat,rng: EventPoint(beat,'2'))
    assert result.event==baseline and result.future_trials==1
    assert result.alternate_draws==0


def test_both_futures_legal_still_compares_probability_destroyed_by_cooldown():
    from maichart.composer.generate import _action_legal_mask,_head_metadata
    atoms=['1','2','3'];grammar=_NoteGrammar.from_atoms(atoms)
    bits,sizes,encode,lookup,*_= _head_metadata(atoms,grammar,'open')
    prior=torch.tensor([0.,.85,.02,.13])
    observations=[]
    def future(branch,current,beat,rng):
        seconds=float(beat)*60./180.
        blocked={pad for pad,stamp in branch.last_head_seconds.items() if seconds-stamp<.05}
        mask=_action_legal_mask(bits=bits,sizes=sizes,button_sizes=sizes,encode=encode,lookup=lookup,
            used_buttons=(),blocked=blocked,selected=(),selected_head_count=0,slot=0,atoms=atoms)
        mass=float(prior[torch.from_numpy(mask)].sum())
        event=EventPoint(beat,'3')  # Both branches can still produce this legal fallback.
        assert branch.accepts(event)
        observations.append(mass)
        return FutureAssessment(event,mass)
    result=bounded_physical_lookahead(EventPoint(Fraction(1),'1'),future_beat=Fraction(9,8),
        guard=_guard(),generator=torch.Generator().manual_seed(451),
        alternative_factory=lambda _:EventPoint(Fraction(1),'2'),future_factory=future)
    assert observations==pytest.approx([.15,.98])
    assert result.switched and result.event.note_text=='2'
    assert result.reason=='alternative_preserves_more_mass'
    assert result.baseline_retained_probability==pytest.approx(.15)
    assert result.alternative_retained_probability==pytest.approx(.98)


@pytest.mark.parametrize('alternative_mass',[.4,.6])
def test_mass_preview_preserves_baseline_on_tie_or_worse(alternative_mass):
    def future(branch,current,beat,rng):
        return FutureAssessment(EventPoint(beat,'3'),.6 if current.note_text=='1' else alternative_mass)
    baseline=EventPoint(Fraction(1),'1')
    result=bounded_physical_lookahead(baseline,future_beat=Fraction(9,8),guard=_guard(),
        generator=torch.Generator().manual_seed(1),alternative_factory=lambda _:EventPoint(Fraction(1),'2'),
        future_factory=future)
    assert result.event==baseline and not result.switched


@pytest.mark.parametrize('current_log_ratio,alt_mass',[(float('-inf'),.99),(-6.9,.99),(0.,.51)])
def test_preview_never_buys_a_tiny_future_gain_with_unsupported_current_action(current_log_ratio,alt_mass):
    def future(branch,current,beat,rng):
        return FutureAssessment(EventPoint(beat,'3'),.5 if current.note_text=='1' else alt_mass)
    baseline=EventPoint(Fraction(1),'1')
    result=bounded_physical_lookahead(baseline,future_beat=Fraction(5,4),guard=_guard(),
        generator=torch.Generator().manual_seed(9),alternative_factory=lambda _:EventPoint(Fraction(1),'2'),
        future_factory=future,current_probability_factory=lambda event:0. if event.note_text=='1' else current_log_ratio)
    assert not result.switched and result.event==baseline


def test_preview_can_choose_supported_current_action_with_substantial_future_gain():
    result=bounded_physical_lookahead(EventPoint(Fraction(1),'1'),future_beat=Fraction(5,4),guard=_guard(),
        generator=torch.Generator().manual_seed(9),alternative_factory=lambda _:EventPoint(Fraction(1),'2'),
        future_factory=lambda branch,current,beat,rng:FutureAssessment(EventPoint(beat,'3'),.15 if current.note_text=='1' else .98),
        current_probability_factory=lambda event:float(torch.tensor(.8 if event.note_text=='1' else .6).log()))
    assert result.switched and result.event.note_text=='2'


def test_current_sequence_support_scores_original_distribution_and_consumes_no_rng():
    from maichart.composer.generate import _action_sequence_log_probability,_head_metadata
    atoms=['1','2'];grammar=_NoteGrammar.from_atoms(atoms)
    metadata=_head_metadata(atoms,grammar,'open')
    logits=torch.tensor([[-30.,float(np.log(.75)),float(np.log(.25))],[30.,-30.,-30.]])
    before=torch.get_rng_state().clone()
    kwargs=dict(slot_mask_decoder=lambda slot,prefix,legal:logits[slot],head_metadata=metadata)
    a=_action_sequence_log_probability(['1'],logits,atoms,grammar,1.,**kwargs)
    b=_action_sequence_log_probability(['2'],logits,atoms,grammar,1.,**kwargs)
    assert a==pytest.approx(np.log(.75),abs=1e-6)
    assert b==pytest.approx(np.log(.25),abs=1e-6)
    assert torch.equal(before,torch.get_rng_state())


@pytest.mark.parametrize('strength',[0.,.5,1.])
def test_strength_and_failed_previews_preserve_main_random_stream(strength):
    rng=torch.Generator().manual_seed(13)
    reference=clone_generator(rng)
    baseline=EventPoint(Fraction(1),'1')
    result=bounded_physical_lookahead(baseline,future_beat=Fraction(9,8),guard=_guard(),generator=rng,
        alternative_factory=lambda private: None,future_factory=lambda *args: None,strength=strength)
    assert result.event==baseline
    torch.testing.assert_close(torch.rand(8,generator=rng),torch.rand(8,generator=reference),rtol=0,atol=0)


def test_preview_cannot_broaden_future_position():
    with pytest.raises(ValueError,match='position'):
        bounded_physical_lookahead(EventPoint(Fraction(1),'1'),future_beat=Fraction(9,8),
            guard=_guard(),generator=torch.Generator().manual_seed(2),
            alternative_factory=lambda rng: None,
            future_factory=lambda branch,current,beat,rng: EventPoint(Fraction(2),'2'))


def test_short_gap_prediction_only_proposes_a_single_optional_nearby_candidate():
    candidates=[Fraction(1),Fraction(9,8),Fraction(5,4),Fraction(3,2)]
    assert short_gap_candidate(candidates,0,torch.tensor([8.,0.,0.,0.,0.,0.,0.]))==1
    assert short_gap_candidate(candidates,0,torch.tensor([0.,8.,0.,0.,0.,0.,0.]))==2
    assert short_gap_candidate(candidates,0,torch.zeros(7)) is None
    assert short_gap_candidate(candidates,0,torch.tensor([0.,0.,8.,0.,0.,0.,0.])) is None
    assert short_gap_candidate(candidates,0,torch.tensor([8.,0.,0.,0.,0.,0.,0.]),
        allowed=[True,False,False,True]) is None
    assert short_gap_candidate(candidates,3,torch.tensor([8.,0.,0.,0.,0.,0.,0.])) is None


def test_same_intent_masks_keep_slide_and_chord_size_in_actual_sampler():
    atoms=['1','2','3','4','1-5[8:1]','2-6[8:1]','A1','B2']
    selected=['1-5[8:1]','3']
    masks=_same_intent_slot_masks(atoms,selected,4)
    assert np.flatnonzero(masks[0]).tolist()==[5,6]
    assert np.flatnonzero(masks[1]).tolist()==[1,2,3,4]
    assert np.flatnonzero(masks[2]).tolist()==[0]
    grammar=_NoteGrammar.from_atoms(atoms)
    logits=torch.zeros(4,len(atoms)+1)
    logits[0,6]=10;logits[1,4]=10
    proposal=_sample_actions(logits,atoms,grammar,0.,torch.Generator().manual_seed(3),
        slot_atom_masks=masks,canonical_order=False)
    assert proposal==['2-6[8:1]','4']
    assert event_intent(EventPoint(Fraction(1),'/'.join(proposal)))==event_intent(
        EventPoint(Fraction(1),'/'.join(selected)))


def test_generation_quality_defaults_are_explicit_and_preview_off():
    args=parser().parse_args(['--audio','a.wav','--checkpoint','c.pt','--output','o.txt'])
    assert args.quality_context_strength==1.
    assert args.quality_plan_strength==1.
    assert args.quality_lookahead_strength==0.


@pytest.mark.parametrize('quality_policy',['cooperative_v1','cooperative_v2'])
def test_real_generate_runs_quality_preview_callbacks_and_preserves_rng(tmp_path, monkeypatch,quality_policy):
    """Run actual generation/guard; inject one rejected future to cover both branches."""
    import soundfile as sf
    from maichart.composer import generate as generation
    from maichart.composer import lookahead
    from maichart.composer.checkpoint import save
    from maichart.composer.grounded_model import AudioGroundedComposer
    from maichart.composer.model import ComposerConfig, GROUNDED_PROFILE, build_model

    sample_rate=8000
    times=np.arange(sample_rate*2)/sample_rate
    audio=tmp_path/'preview.wav'
    sf.write(audio,(.1*np.sin(2*np.pi*220*times)).astype(np.float32),sample_rate)
    atoms=['1','2','3','4','5','6','7','8']
    torch.manual_seed(457)
    model=build_model(ComposerConfig(atom_vocab_size=9,d_model=16,nhead=4,layers=1,
        dim_feedforward=32,dropout=0.,history_events=8,action_slots=2,candidate_count=16,
        architecture_profile=GROUNDED_PROFILE,head_timing_dim=99,lens_width=8,lens_chunk_size=8,
        action_state_policy='selected_prefix_v3',configuration_quality_policy=quality_policy),atoms)
    with torch.no_grad():
        model.eos_head.weight.zero_();model.eos_head.bias.fill_(-50.)
        model.action_head.weight.zero_()
        model.action_head.bias.copy_(torch.tensor([30.,10.,9.8,9.6,9.4,9.2,9.,8.8,8.6]))
        for embedding in model.action_factors.factors.values():embedding.weight.zero_()
        model.quality_context.gap_head[-1].weight.zero_()
        model.quality_context.gap_head[-1].bias.copy_(torch.tensor([20.,-20.,-20.,-20.,-20.,-20.,-20.]))
    checkpoint=tmp_path/'cooperative.pt'
    save(checkpoint,model,None,epoch=1,args={},manifest=tmp_path/'manifest.json',atom_catalog=atoms)
    native_forward=AudioGroundedComposer.forward

    def nearest(self,*args,**kwargs):
        result=native_forward(self,*args,**kwargs)
        scores=-torch.arange(result['candidate_logits'].size(-1),device=result['candidate_logits'].device).float()[None]
        scores=scores.masked_fill(kwargs['candidate_mask'],float('-inf'))
        result['candidate_logits']=result['audio_candidate_logits']=scores
        return result

    monkeypatch.setattr(AudioGroundedComposer,'forward',nearest)
    native_preview=lookahead.bounded_physical_lookahead
    native_sample=generation._sample_actions
    callbacks={'future':0,'alternative':0,'private_rng_checks':0}
    first_main_states={}
    current_run=[None]

    def record_sample(*args,**kwargs):
        result=native_sample(*args,**kwargs)
        if kwargs.get('diagnostics') is not None:
            first_main_states.setdefault(current_run[0],args[4].get_state().clone())
        return result

    def record_preview(baseline,**kwargs):
        main_rng=kwargs['generator'];before=main_rng.get_state().clone()
        real_future=kwargs['future_factory'];real_alternative=kwargs['alternative_factory']
        attempts=[0]

        def future(branch,current,beat,private):
            callbacks['future']+=1
            assert private is not main_rng
            proposed=real_future(branch,current,beat,private)
            assert isinstance(proposed,FutureAssessment)
            assert 0.<=proposed.retained_probability<=1.
            attempts[0]+=1
            # Exercise the actual alternate callback even when the randomly
            # initialized model happened to propose an easy legal continuation.
            # At 240 BPM an eighth-beat same-head repeat is below the real 50 ms guard.
            return EventPoint(beat,current.note_text) if attempts[0]==1 else proposed

        def alternative(private):
            callbacks['alternative']+=1
            assert private is not main_rng
            return real_alternative(private)

        result=native_preview(baseline,**dict(kwargs,future_factory=future,alternative_factory=alternative))
        assert torch.equal(main_rng.get_state(),before)
        callbacks['private_rng_checks']+=1
        return result

    monkeypatch.setattr(generation,'_sample_actions',record_sample)
    monkeypatch.setattr(lookahead,'bounded_physical_lookahead',record_preview)
    reports=[]
    previous_threads=torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for enabled in (0,1):
            current_run[0]=enabled
            args=parser().parse_args(['--audio',str(audio),'--checkpoint',str(checkpoint),
                '--output',str(tmp_path/f'preview-{enabled}.txt'),'--device','cpu','--bpm','240',
                '--max-events','6','--seed','457','--pointer-temperature','0','--action-temperature','.8',
                '--quality-lookahead-strength',str(enabled),'--no-touch-history-guard'])
            reports.append(generation.generate(args))
    finally:
        torch.set_num_threads(previous_threads)
    assert reports[0]['quality_lookahead']['proposals']==0
    assert reports[1]['configuration_quality_policy']==quality_policy
    assert reports[1]['first_slot_min_legal_mass']==(.01 if quality_policy=='cooperative_v2' else 0.)
    assert reports[1]['event_count']>0
    assert reports[1]['quality_lookahead']['proposals']>0
    assert reports[1]['quality_lookahead']['future_trials']>0
    assert callbacks['future']>0 and callbacks['alternative']>0
    assert callbacks['private_rng_checks']==reports[1]['quality_lookahead']['proposals']
    assert torch.equal(first_main_states[0],first_main_states[1])
