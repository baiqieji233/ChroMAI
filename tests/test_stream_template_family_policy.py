from types import SimpleNamespace

import torch

from maichart.composer.stream_actions import family_preserving_template_log_probs
from maichart.composer.stream_model import StreamComposer, StreamComposerConfig
from maichart.composer.stream_timing import grouped_logsumexp
from maichart.composer.stream_transition import residual_keep_logit, two_group_log_probs
from test_stream_music_exit_model import model_batch


def test_template_residual_preserves_every_family_mass_and_reallocates_within_family():
    base=torch.tensor([[.1, .1, .8]],requires_grad=True).log()
    family=torch.tensor([1,1,2])
    keep=torch.tensor([[True,False,False]])
    correction=torch.tensor([torch.log(torch.tensor(9.))/8],requires_grad=True)
    changed=family_preserving_template_log_probs(base,keep,family,correction)
    torch.testing.assert_close(changed.exp().sum(-1),torch.ones(1))
    torch.testing.assert_close(changed.exp()[0,2],torch.tensor(.8))
    torch.testing.assert_close(changed.exp()[0,:2],torch.tensor([.18,.02]))
    original_family=grouped_logsumexp(base,family-1,groups=31)
    changed_family=grouped_logsumexp(changed,family-1,groups=31)
    torch.testing.assert_close(changed_family,original_family)
    (-changed[0,0]).backward()
    assert correction.grad is not None and correction.grad.abs().sum()>0


def test_legacy_cross_family_template_path_still_shifts_family_mass():
    base=torch.tensor([[.1,.1,.8]]).log()
    keep=torch.tensor([[True,False,False]])
    correction=torch.tensor([torch.log(torch.tensor(9.))/8])
    old=two_group_log_probs(base,keep,residual_keep_logit(base,keep,correction,gain=8.))
    assert old.exp()[0,:2].sum()>.5


def test_config_migrates_old_checkpoints_without_changing_fresh_policy():
    assert StreamComposerConfig().template_policy=='family_preserving_v1'
    assert StreamComposerConfig.from_dict({}).template_policy=='cross_family_v1'
    assert StreamComposerConfig.from_dict({'template_policy':'family_preserving_v1'}).template_policy=='family_preserving_v1'


def test_teacher_likelihood_uses_the_preserved_family_marginal():
    model,batch=model_batch('music_exit_v3')
    out=model(batch)
    plain=model.event_route_log_probs(out['action_context'],family_logits=out['event_family_logits'])
    expected=grouped_logsumexp(plain,model.semantic_route_families-1,groups=31)
    actual=grouped_logsumexp(out['event_route_log_probs'],model.semantic_route_families-1,groups=31)
    torch.testing.assert_close(actual.softmax(-1),expected.softmax(-1),atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(out['event_family_logits'],actual)
    loss,metrics=model.loss(out,batch)
    assert torch.isfinite(loss)
    torch.testing.assert_close(metrics['event_family_loss'],
        .75*metrics['event_family_full_loss']+.25*metrics['event_family_audio_loss'])


def test_family_first_intent_decision_frontier_uses_family_before_template():
    scores=torch.tensor([.1,.1,.8]).log()
    supported=torch.zeros(31,dtype=torch.bool);supported[:2]=True
    stub=SimpleNamespace(config=SimpleNamespace(template_policy='family_preserving_v1'),
        semantic_route_families=torch.tensor([1,1,2]),semantic_family_supported=supported,
        event_route_log_probs=lambda *a,**kw:scores[None],
        template_keep_mask=lambda *a,**kw:torch.tensor([True,False,False]))
    called=[]
    def choose(options):
        called.append(options.numel())
        return int(options.argmax())
    family,keep=StreamComposer.sample_event_intent(stub,torch.zeros(1),family_logits=torch.zeros(31),
        template_keep_logit=torch.zeros(1),template_reference_shape=1,
        temperature=0.,decision_chooser=choose)
    assert (family,keep)==(2,False)
    assert called==[31]
