from types import SimpleNamespace
import torch
from maichart.composer.structure_backend import _stable_attend
from maichart.composer import structure_backend
import pytest


def test_query_chunking_preserves_all_key_attention_across_chunk_boundaries():
    torch.manual_seed(14)
    q=torch.randn(2,3,277,16);k=torch.randn(2,3,313,16);v=torch.randn(2,3,313,12)
    actual=_stable_attend(SimpleNamespace(training=False,scale=None),q,k,v)
    expected=torch.nn.functional.scaled_dot_product_attention(q,k,v)
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    custom=_stable_attend(SimpleNamespace(training=False,scale=.5),q,k,v)
    reference=torch.nn.functional.scaled_dot_product_attention(q,k,v,scale=.5)
    torch.testing.assert_close(custom,reference,rtol=1e-5,atol=1e-6)


def test_raw_logits_cache_identifies_implementation_and_weights():
    identity={'checkpoint_sha256':'weights','package_version':'1','attention_policy':'math','waveform_policy':'native'}
    original=structure_backend.raw_cache_key('audio',identity)
    for key in identity:
        assert structure_backend.raw_cache_key('audio',{**identity,key:'changed'})!=original


def test_generation_rejects_analyzer_identity_drift(monkeypatch):
    expected={'name':'beat_this','checkpoint_sha256':'trained_weights'}
    config=SimpleNamespace(structure_policy='metrical_context_v1',structure_backend='beat_this',
        structure_backend_identity=expected)
    monkeypatch.setattr(structure_backend,'backend_identity',lambda _,**kwargs:dict(expected))
    structure_backend.validate_backend_identity(config)
    monkeypatch.setattr(structure_backend,'backend_identity',lambda _,**kwargs:{**expected,'checkpoint_sha256':'new_weights'})
    with pytest.raises(ValueError,match='identity differs'):
        structure_backend.validate_backend_identity(config)
