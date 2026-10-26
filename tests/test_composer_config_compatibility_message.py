import pytest
import torch
from maichart.composer import checkpoint
from maichart.composer.model import ComposerConfig,build_model


def test_stale_constructor_is_reported_without_stripping_checkpoint_fields(tmp_path,monkeypatch):
    model=build_model(ComposerConfig(atom_vocab_size=3,d_model=8,nhead=2,layers=1,
        dim_feedforward=16,architecture_profile='audio_grounded_v3',head_timing_dim=99,lens_width=8),['1','2'])
    payload=checkpoint.save(tmp_path/'cp.pt',model,None,epoch=1,args={},manifest='unused',atom_catalog=['1','2'])
    expected=dict(payload['model_config'])
    def old_config(**kwargs):
        raise TypeError("ComposerConfig.__init__() got an unexpected keyword argument 'history_anchor_policy'")
    monkeypatch.setattr(checkpoint,'ComposerConfig',old_config)
    result=checkpoint.inspect_payload(payload)
    assert not result['valid'] and not result['can_generate']
    assert 'history_anchor_policy' in result['error'] and '重启' in result['error']
    assert payload['model_config']==expected
