import torch

from maichart.composer.event_set_codec import EventSetCodec
from maichart.composer.events import event_metadata
from maichart.composer.stream_actions import grouped_event_log_probs, family_masks_from_atoms
from maichart.composer.stream_model import StreamComposer, StreamComposerConfig


def semantic_model():
    codec=EventSetCodec.fit(['1','1h[4:1]','1-5[8:1]','A1','C','Ch[4:1]','1/A1'])
    config=StreamComposerConfig(d_model=16,nhead=4,layers=1,dim_feedforward=32,
        audio_dim=6,condition_dim=3,dropout=0.,max_event_atoms=4,
        event_family_policy='semantic_mass_v1')
    return StreamComposer(config,codec).eval()


def test_semantic_mass_is_invariant_to_detail_temperature_and_route_count():
    raw=torch.tensor([[0.,3.,-1.,2.,5.]])
    route_families=torch.tensor([1,1,8,8,8])
    family=torch.full((1,31),-10.)
    family[0,0]=torch.log(torch.tensor(.7));family[0,7]=torch.log(torch.tensor(.3))
    supported=torch.zeros(31,dtype=torch.bool);supported[[0,7]]=True
    for temperature in (.25,.65,1.,2.):
        log_probs=grouped_event_log_probs(raw,family,route_families,supported,temperature=temperature)
        torch.testing.assert_close(log_probs.exp().sum(-1),torch.ones(1))
        torch.testing.assert_close(log_probs.exp()[:,:2].sum(-1),torch.tensor([.7]))
        torch.testing.assert_close(log_probs.exp()[:,2:].sum(-1),torch.tensor([.3]))


def test_other_family_details_do_not_receive_macro_denominator_gradient():
    raw=torch.tensor([[1.,2.,3.,4.]],requires_grad=True)
    family=torch.zeros(1,31,requires_grad=True)
    supported=torch.zeros(31,dtype=torch.bool);supported[[0,7]]=True
    lp=grouped_event_log_probs(raw,family,torch.tensor([1,1,8,8]),supported)
    (-lp[0,0]).backward()
    torch.testing.assert_close(raw.grad[0,2:],torch.zeros(2),atol=0,rtol=0)
    assert raw.grad[0,:2].abs().sum()>0
    assert family.grad.abs().sum()>0


def test_empty_blocked_support_returns_no_distribution_instead_of_nan():
    probs=grouped_event_log_probs(torch.full((1,2),-torch.inf),torch.zeros(1,31),
        torch.tensor([1,8]),torch.ones(31,dtype=torch.bool))
    assert torch.isneginf(probs).all()


def test_semantic_family_is_the_complete_event_type_union():
    model=semantic_model()
    for text,expected in [('1',1),('1-5[8:1]',2),('1h[4:1]',4),('A1',8),('Ch[4:1]',16),('1/A1',9)]:
        encoded=model.codec.encode(text)
        atoms=torch.tensor([encoded['atoms']])
        assert int(family_masks_from_atoms(atoms,model.semantic_atom_families))==expected
        batch={'target_shape':torch.tensor([[encoded['shape']]]),
               'target_details':torch.tensor([[encoded['details']]])}
        assert int(model.event_family_targets(batch))==expected


def test_literal_training_support_requires_the_complete_chosen_family():
    model=semantic_model()
    event=model.codec.encode('1/A1')
    target=torch.tensor([event['details']])
    logits=model.literal_logits(torch.zeros(1,16),target,torch.tensor([9]))
    # STOP cannot replace either required type. It becomes legal only after
    # the Tap and Touch members have both been teacher-forced.
    assert float(logits[0,0,0])==-1e4
    assert float(logits[0,1,0])==-1e4
    assert float(logits[0,2,0])>-1e4
    forbidden=model.semantic_atom_families.bitwise_and(~9).ne(0)
    assert (logits[0,:,forbidden]==-1e4).all()


def test_literal_decode_obeys_semantic_family_without_touch_count_caps():
    model=semantic_model()
    with torch.no_grad():
        model.event_family_head.weight.zero_();model.event_family_head.bias.fill_(-100.)
        model.event_family_head.bias[8]=100.  # Tap | Touch
        model.literal_head.weight.zero_();model.literal_head.bias.fill_(-100.)
        model.literal_head.bias[model.codec._atoms['A1']]=100.
        model.literal_head.bias[0]=90.
        model.literal_head.bias[model.codec._atoms['1']]=80.
    event=model.decode_event(torch.zeros(16),shape_override=0,temperature=0.)
    assert event['family']==9
    assert event['text']=='1/A1'
    assert event_metadata(event['text']).touch


def test_greedy_family_choice_is_independent_of_its_number_of_routes():
    codec=EventSetCodec.fit(['1','1/3','1/5','A1'])
    model=StreamComposer(StreamComposerConfig(d_model=16,nhead=4,layers=1,
        dim_feedforward=32,audio_dim=6,condition_dim=3,dropout=0.,event_family_policy='semantic_mass_v1'),codec).eval()
    with torch.no_grad():
        model.shape_head.weight.zero_();model.shape_head.bias.zero_()
        model.event_family_head.weight.zero_();model.event_family_head.bias.fill_(-100.)
        model.event_family_head.bias[0]=torch.tensor(.7).log()
        model.event_family_head.bias[7]=torch.tensor(.3).log()
    event=model.decode_event(torch.zeros(16),temperature=0.,reference_phase_mask=torch.tensor([True,False,False,False,False,False,False,False]))
    assert event['family']==1


def test_old_checkpoint_config_keeps_the_original_event_likelihood():
    assert StreamComposerConfig.from_dict({}).event_family_policy=='legacy_v1'
    assert StreamComposerConfig().event_family_policy=='audio_state_phrase_v4'
