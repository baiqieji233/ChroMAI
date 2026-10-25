import pytest
import torch

from maichart.composer import MOTIF_TRAINING_POLICY
from maichart.composer.checkpoint import save,load,inspect_payload
from maichart.composer.model import ComposerConfig,build_model


def model(motif):
    return build_model(ComposerConfig(atom_vocab_size=3,d_model=8,nhead=2,layers=1,
        dim_feedforward=16,lens_width=8,head_timing_dim=99,architecture_profile='audio_grounded_v3',
        motif_memory_policy='audio_recurrence_archive_v1' if motif else 'off'),['1','2'])


@pytest.mark.parametrize('motif',[False,True])
def test_old_training_contract_only_requires_motif_migration(tmp_path,monkeypatch,motif):
    m=model(motif);path=tmp_path/'model.pt'
    payload=save(path,m,torch.optim.AdamW(m.parameters()),epoch=1,args={},manifest='unused',atom_catalog=['1','2'])
    assert inspect_payload(payload)['can_resume']
    if motif:assert payload['motif_training_policy']==MOTIF_TRAINING_POLICY
    payload.pop('motif_training_policy',None)
    torch.save(payload,path)
    restored,_=load(path)
    for key,value in m.state_dict().items():
        if torch.is_tensor(value):torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)
    info=inspect_payload(payload)
    assert info['valid'] and info['can_generate'] and info['can_init']
    assert info['can_resume']==(not motif)
    if motif:
        from maichart.composer import train
        monkeypatch.setattr(train,'ComposerDataset',lambda *a,**kw:pytest.fail('reached dataset'))
        with pytest.raises(ValueError,match='Motif symbolic dropout contract changed'):
            train.main(['--resume',str(path),'--manifest','unused','--checkpoint-dir',str(tmp_path/'dest'),'--device','cpu'])
