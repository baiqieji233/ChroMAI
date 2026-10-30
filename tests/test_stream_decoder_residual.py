"""The streaming backbone must retain prefix differences through its depth."""
import torch

from maichart.composer.event_set_codec import EventSetCodec
from maichart.composer.stream_model import StreamComposer, StreamComposerConfig


def model(policy='prenorm_independent_v2'):
    codec=EventSetCodec.fit(['1','1/5'])
    return StreamComposer(StreamComposerConfig(d_model=16,nhead=4,layers=4,
        dim_feedforward=32,dropout=0.,audio_dim=6,condition_dim=3,
        decoder_policy=policy),codec)


def test_fresh_decoder_layers_have_independent_matrices_and_bounded_interface():
    torch.manual_seed(218)
    net=model()
    assert net.decoder.norm is not None
    torch.testing.assert_close(net.decoder.norm.weight,torch.ones(16))
    torch.testing.assert_close(net.decoder.norm.bias,torch.zeros(16))
    for layer in net.decoder.layers:
        assert layer.norm_first
    for name in ('self_attn.in_proj_weight','multihead_attn.in_proj_weight','linear1.weight','linear2.weight'):
        matrices=[dict(layer.named_parameters())[name] for layer in net.decoder.layers]
        assert all(not torch.equal(matrices[0],other) for other in matrices[1:])
    for layer in net.decoder.layers:
        for name,parameter in layer.named_parameters():
            if name.endswith('bias'):assert not parameter.count_nonzero()
            elif 'norm' in name:torch.testing.assert_close(parameter,torch.ones_like(parameter))


def test_zero_sublayers_keep_the_residual_input_until_final_normalization():
    net=model().eval()
    with torch.no_grad():
        for layer in net.decoder.layers:
            for name,parameter in layer.named_parameters():
                if 'norm' not in name:parameter.zero_()
    x=torch.randn(2,7,16)
    memory=torch.randn(2,5,16)
    causal=torch.ones(7,7,dtype=torch.bool).triu(1)
    with torch.no_grad():
        out=net.decoder(x,memory,tgt_mask=causal)
        expected=net.decoder.norm(x)
    torch.testing.assert_close(out,expected)
    # Different prefixes survive four residual layers; fixed RMS is not used
    # as a substitute for testing differences between representations.
    assert (out[0]-out[1]).square().mean()>.1


def test_multilayer_prenorm_is_causal_and_has_finite_prefix_gradients():
    net=model().eval()
    x=torch.randn(2,7,16,requires_grad=True)
    memory=torch.randn(2,5,16,requires_grad=True)
    causal=torch.ones(7,7,dtype=torch.bool).triu(1)
    out=net.decoder(x,memory,tgt_mask=causal)
    changed=x.detach().clone();changed[:,4:]=torch.randn_like(changed[:,4:])*20
    with torch.no_grad():future=net.decoder(changed,memory.detach(),tgt_mask=causal)
    torch.testing.assert_close(out[:,:4],future[:,:4],atol=2e-5,rtol=2e-5)
    loss=(out*torch.randn_like(out)).sum()
    loss.backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    assert torch.isfinite(memory.grad).all() and memory.grad.abs().sum()>0
    gradients=[p.grad for p in net.decoder.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)


def test_old_checkpoint_retains_postnorm_and_exact_original_parameter_layout():
    assert StreamComposerConfig.from_dict({}).decoder_policy=='postnorm_v1'
    old=model('postnorm_v1')
    assert old.decoder.norm is None
    assert not any(layer.norm_first for layer in old.decoder.layers)
    assert not any(key.startswith('decoder.norm.') for key in old.state_dict())
    # This is the historical initialization, not shared parameter storage.
    assert torch.equal(old.decoder.layers[0].self_attn.in_proj_weight,
                       old.decoder.layers[1].self_attn.in_proj_weight)
    assert old.decoder.layers[0].self_attn.in_proj_weight.data_ptr()!=old.decoder.layers[1].self_attn.in_proj_weight.data_ptr()
    restored=StreamComposerConfig.from_dict(model().config.to_dict())
    assert restored.decoder_policy=='prenorm_independent_v2'
