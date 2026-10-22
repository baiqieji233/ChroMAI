from fractions import Fraction
from pathlib import Path

import json
import numpy as np
import soundfile as sf
import torch

from maichart.composer.candidates import candidate_beats, candidate_index
from maichart.composer.checkpoint import inspect_payload, save, load
from maichart.composer.model import ChartComposer, ComposerConfig
from maichart.composer.data import ComposerDataset


def test_candidate_set_uses_bounded_musical_grid_and_long_rest_choices():
    tempo=[type('Tempo',(),{'beat':Fraction(0),'bpm':120.})()]
    target=Fraction(37,997)
    values=candidate_beats(Fraction(0),tempo=tempo,first=0.,audio_end_sec=12.,candidate_count=16,target=target)
    assert candidate_index(target,values)==-1
    assert any(value >= 2 for value in values)
    assert all(value.denominator <= 96 for value in values)
    assert all(value.denominator in {1,2,3,4,6,8,12,16} for value in values)
    first_values=candidate_beats(None,tempo=tempo,first=0.,audio_end_sec=12.,candidate_count=96)
    assert Fraction(0) in first_values
    assert Fraction(1,2) in first_values
    assert Fraction(9,4) in first_values
    assert Fraction(17,8) in first_values


def test_active_heads_only_block_live_hold_sources(tmp_path):
    from maichart.composer.generate import _active_heads
    tempo=[type('Tempo',(),{'beat':Fraction(0),'bpm':120.})()]
    state=type('State',(),{'markers':[
        {'type':'hold','second':0.,'end_second':1.,'lane':1},
        {'type':'slide','second':0.,'end_second':1.,'lane':2},
    ]})()
    assert _active_heads(state,Fraction(2),tempo,0.) == set()
    assert _active_heads(state,Fraction(1,2),tempo,0.) == {'A1'}


def test_composer_forward_shapes_and_checkpoint_roundtrip(tmp_path):
    config=ComposerConfig(atom_vocab_size=3,candidate_feature_dim=125,d_model=24,nhead=4,
                          layers=1,dim_feedforward=48,candidate_count=8,action_slots=3)
    model=ChartComposer(config)
    features=torch.randn(2,8,125)
    history=torch.randint(0,3,(2,4,3))
    history_features=torch.randn(2,4,8)
    condition=torch.randn(2,16)
    action=torch.randn(2,50)
    output=model(features,history,condition,action,history_features=history_features,
                 candidate_mask=torch.zeros(2,8,dtype=torch.bool),
                 history_padding_mask=torch.zeros(2,4,dtype=torch.bool))
    assert output['candidate_logits'].shape==(2,8)
    assert output['action_logits'].shape==(2,8,3,3)
    assert output['future'].shape==(2,8,8)
    assert output['eos_logits'].shape==(2,8)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    path=tmp_path/'composer_last.pt'
    save(path,model,optimizer,epoch=1,args={},
         manifest=tmp_path/'manifest.json',atom_catalog=['1','2'],optimizer_steps=1,
         dataset_fingerprint='fixture')
    restored,payload=load(path)
    assert inspect_payload(payload)['valid']
    assert restored.config.to_dict()==config.to_dict()


def test_composer_dataset_builds_candidate_features_from_audio(tmp_path):
    sr=8000;t=np.arange(sr*2)/sr
    sf.write(tmp_path/'track.wav',(0.1*np.sin(2*np.pi*440*t)).astype(np.float32),sr)
    (tmp_path/'maidata.txt').write_text(
        '&wholebpm=120\n&lv_5=12\n&inote_5=(120){4}1,2,3,4,5,6,7,8,E\n',encoding='utf-8')
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'songs':[{'song_id':'composer-fixture',
        'maidata_path':str(tmp_path/'maidata.txt'),'audio_path':str(tmp_path/'track.wav'),
        'difficulties':[{'difficulty_index':5,'level_raw':'12'}]}]}),encoding='utf-8')
    dataset=ComposerDataset(manifest,cache_dir=tmp_path/'cache',candidate_count=16,
        history_events=8,action_slots=4,preparation_workers=1,audio_cache_gb=.05)
    item=dataset[0]
    assert len(dataset)>0 and item['candidate_features'].shape==(16,125)
    assert int(item['candidate_target'])>=0
    assert item['future_target'].shape==(8,) and item['history_features'].shape[-1]==8


def test_composer_cpu_training_writes_checkpoint(tmp_path):
    sr=8000;t=np.arange(sr*2)/sr
    sf.write(tmp_path/'track.wav',(0.1*np.sin(2*np.pi*330*t)).astype(np.float32),sr)
    (tmp_path/'maidata.txt').write_text(
        '&wholebpm=120\n&lv_5=12\n&inote_5=(120){4}1,2,3,4,5,6,7,8,E\n',encoding='utf-8')
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'songs':[{'song_id':'composer-train-fixture',
        'maidata_path':str(tmp_path/'maidata.txt'),'audio_path':str(tmp_path/'track.wav'),
        'difficulties':[{'difficulty_index':5,'level_raw':'12'}]}]}),encoding='utf-8')
    from maichart.composer.train import main
    assert main(['--manifest',str(manifest),'--checkpoint-dir',str(tmp_path/'out'),
        '--cache-dir',str(tmp_path/'cache'),'--epochs','1','--batch-size','2',
        '--d-model','24','--layers','1','--nhead','4','--dim-feedforward','48',
        '--candidate-count','16','--action-slots','4','--device','cpu','--precision','fp32',
        '--prepare-workers','1','--audio-cache-gb','.05','--checkpoint-interval-steps','1'])==0
    assert (tmp_path/'out'/'composer_last.pt').is_file()


def test_composer_generation_writes_chart_and_report(tmp_path):
    sr=8000;t=np.arange(sr*2)/sr
    audio_path=tmp_path/'track.wav';sf.write(audio_path,(0.1*np.sin(2*np.pi*220*t)).astype(np.float32),sr)
    (tmp_path/'maidata.txt').write_text(
        '&wholebpm=120\n&lv_5=12\n&inote_5=(120){4}1,2,3,4,5,6,7,8,E\n',encoding='utf-8')
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'songs':[{'song_id':'composer-generate-fixture',
        'maidata_path':str(tmp_path/'maidata.txt'),'audio_path':str(audio_path),
        'difficulties':[{'difficulty_index':5,'level_raw':'12'}]}]}),encoding='utf-8')
    dataset=ComposerDataset(manifest,cache_dir=tmp_path/'cache',candidate_count=16,
        history_events=8,action_slots=4,preparation_workers=1,audio_cache_gb=.05)
    model=ChartComposer(ComposerConfig(atom_vocab_size=len(dataset.codec_atoms)+1,d_model=24,nhead=4,
        layers=1,dim_feedforward=48,candidate_count=16,action_slots=4))
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    checkpoint=tmp_path/'composer_last.pt'
    save(checkpoint,model,optimizer,epoch=1,args={},
        manifest=manifest,atom_catalog=dataset.codec_atoms)
    from maichart.composer.generate import generate,parser
    output=tmp_path/'generated.txt'
    args=parser().parse_args(['--audio',str(audio_path),'--checkpoint',str(checkpoint),
        '--output',str(output),'--bpm','120','--device','cpu','--max-events','2'])
    report=generate(args)
    assert output.is_file() and output.with_suffix('.composer.json').is_file()
    assert report['family']=='chart_composer_v1'
