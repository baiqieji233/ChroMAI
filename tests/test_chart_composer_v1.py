from fractions import Fraction
from pathlib import Path

import json
import numpy as np
import soundfile as sf
import torch
import copy

from maichart.composer.candidates import candidate_beats, candidate_index
from maichart.composer.checkpoint import inspect_payload, save, load
from maichart.composer.model import ChartComposer, ComposerConfig
from maichart.composer.data import ComposerDataset


def test_selected_candidate_projection_preserves_loss_and_gradients():
    from maichart.composer.train import _loss
    torch.manual_seed(10)
    config=ComposerConfig(atom_vocab_size=7,d_model=16,nhead=4,layers=1,dim_feedforward=32,
        dropout=0.,history_events=8,action_slots=3)
    full=ChartComposer(config).eval(); selected=copy.deepcopy(full)
    features=torch.randn(2,6,141);history=torch.randint(0,7,(2,3,3))
    control=torch.randn(2,16);targets=torch.tensor([1,4])
    batch={'candidate_target':targets,'action_target':torch.tensor([[1,2,0],[3,0,0]]),
        'future_target':torch.rand(2,8),'eos_target':torch.tensor([0.,1.])}
    full_out=full(features,history,control)
    chosen_out=selected(features,history,control,candidate_indices=targets)
    assert chosen_out['action_logits'].shape==(2,3,7)
    loss_full,_=_loss(full_out,batch,7);loss_chosen,_=_loss(chosen_out,batch,7)
    torch.testing.assert_close(loss_full,loss_chosen)
    loss_full.backward();loss_chosen.backward()
    for (_,left),(_,right) in zip(full.named_parameters(),selected.named_parameters()):
        if left.grad is not None:
            torch.testing.assert_close(left.grad,right.grad,rtol=1e-4,atol=1e-6)
    hidden=full(features,history,control,decode_actions=False)
    assert 'action_logits' not in hidden
    decoded=full.decode_candidate(hidden['candidate_context'],targets)
    torch.testing.assert_close(decoded['action_logits'],chosen_out['action_logits'])


def test_generation_history_gaps_use_true_window_predecessor_and_audio_first():
    from maichart.composer.generate import _history_features
    from maichart.event_timing import EventTempo
    tempo=[EventTempo(Fraction(0),120.)]
    full=[((1,0),Fraction(10)),((2,0),Fraction(11)),((1,0),Fraction(12))]
    full_rows=_history_features(full,['1','2'],tempo,2.,2)
    tail_rows=_history_features(full[1:],['1','2'],tempo,2.,2,preceding_beat=Fraction(10))
    np.testing.assert_allclose(full_rows[1:],tail_rows)
    zero=_history_features([((1,0),Fraction(0))],['1','2'],tempo,2.,2)
    assert zero[0][0]==zero[0][1]==0.


def test_inference_export_is_lossless_and_cannot_resume(tmp_path):
    from maichart.composer.checkpoint import export_inference,read_payload
    config=ComposerConfig(atom_vocab_size=3,d_model=16,nhead=4,layers=1,dim_feedforward=32,
                          architecture_profile='music_context_v2',head_timing_dim=99)
    model=ChartComposer(config);optimizer=torch.optim.AdamW(model.parameters())
    for value in model.parameters():value.grad=torch.ones_like(value)
    optimizer.step()
    full=tmp_path/'composer_last.pt';inference=tmp_path/'composer_inference.pt'
    save(full,model,optimizer,epoch=1,args={},manifest=tmp_path/'manifest.json',atom_catalog=['1','2'])
    export_inference(full,inference)
    original=read_payload(full);slim=read_payload(inference)
    assert inspect_payload(original)['can_resume']
    assert not inspect_payload(slim)['can_resume'] and inspect_payload(slim)['can_generate']
    for key,value in original['model_state_dict'].items():
        assert torch.equal(value,slim['model_state_dict'][key])
    assert inference.stat().st_size<full.stat().st_size*.6
    assert 'optimizer_state_dict' not in load(full)[1]


def test_candidate_set_uses_bounded_musical_grid_and_long_rest_choices():
    tempo=[type('Tempo',(),{'beat':Fraction(0),'bpm':120.})()]
    target=Fraction(37,997)
    values=candidate_beats(Fraction(0),tempo=tempo,first=0.,audio_end_sec=12.,candidate_count=16,target=target)
    assert candidate_index(target,values)==-1
    assert any(value >= 2 for value in values)
    offsets={value for value in values}
    assert all(value > 0 for value in offsets)
    first_values=candidate_beats(None,tempo=tempo,first=0.,audio_end_sec=12.,candidate_count=96)
    assert Fraction(0) in first_values
    assert Fraction(1,2) in first_values
    assert Fraction(9,4) in first_values
    assert Fraction(17,8) in first_values
    fine={Fraction(index, denominator) for denominator in (1,2,3,4,6,8,12,16)
          for index in range(1, denominator*2+1)}
    assert fine.issubset({value for value in first_values if value > 0})
    for count in range(1, 9):
        small=candidate_beats(None,tempo=tempo,first=0.,audio_end_sec=100.,candidate_count=count)
        assert len(small) <= count and small == sorted(set(small)) and small[0] == 0


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
    config=ComposerConfig(atom_vocab_size=3,candidate_feature_dim=141,d_model=24,nhead=4,
                          layers=1,dim_feedforward=48,candidate_count=8,action_slots=3)
    model=ChartComposer(config)
    features=torch.randn(2,8,141)
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
    assert len(dataset)>0 and item['candidate_features'].shape==(16,141)
    assert item['workload_audio'].shape==(32,104)
    assert item['workload_target'].shape==(32,5)
    assert item['ledger'].shape==(16,6)
    assert int(item['candidate_target'])>=0
    assert item['future_target'].shape==(8,) and item['history_features'].shape[-1]==8


def test_composer_cpu_training_writes_checkpoint(tmp_path):
    sr=8000;t=np.arange(sr*2)/sr
    sf.write(tmp_path/'track.wav',(0.1*np.sin(2*np.pi*330*t)).astype(np.float32),sr)
    (tmp_path/'maidata.txt').write_text(
        '&wholebpm=120\n&lv_5=12\n&inote_5=(120){4}1/2/3/4/5/6/7/8/C,2,3,4,5,6,7,8,E\n',encoding='utf-8')
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
    assert (tmp_path/'out'/'composer_inference.pt').is_file()
    payload=torch.load(tmp_path/'out'/'composer_last.pt',map_location='cpu',weights_only=False)
    assert payload['training_state']['epoch_complete']
    assert payload['model_config']['action_slots'] == payload['args']['action_slots'] == 9
    assert payload['args']['action_slots_requested'] == 4
    restored, _ = load(tmp_path/'out'/'composer_inference.pt')
    assert restored.config.action_slots == 9
    # A minimal explicit finetune must inherit the real parent dimensions,
    # including automatically expanded slots, rather than CLI shape defaults.
    assert main(['--manifest',str(manifest),'--init-checkpoint',str(tmp_path/'out'/'composer_last.pt'),
        '--checkpoint-dir',str(tmp_path/'finetune'),'--cache-dir',str(tmp_path/'cache'),
        '--epochs','1','--batch-size','2','--max-steps','1','--lr','.0001',
        '--device','cpu','--prepare-workers','1','--audio-cache-gb','.05'])==0
    fine, payload = load(tmp_path/'finetune'/'composer_last.pt', include_optimizer=True)
    assert fine.config.to_dict() == restored.config.to_dict()
    assert payload['optimizer_state_dict']['param_groups'][0]['lr'] == .0001


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
        layers=1,dim_feedforward=48,candidate_count=16,action_slots=4,architecture_profile='music_context_v2',head_timing_dim=99))
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
    assert report['min_same_head_ms']==50.
    assert 'density_observation' in report


def test_interp_candidate_audio_and_candidate_spec_match_reference():
    from maichart.composer.candidates import candidate_spec, interp_candidate_audio
    from maichart.tempo_clock import TempoClock
    from maichart.event_timing import EventTempo

    tempo=[EventTempo(Fraction(0),150.), EventTempo(Fraction(4),180.)]
    clock=TempoClock(tempo,0.1)
    for prev in (None, Fraction(0), Fraction(3,4), Fraction(15,2)):
        for count in (8, 16, 65, 96):
            beats=candidate_beats(prev,tempo=tempo,first=0.1,audio_end_sec=10.,candidate_count=count,clock=clock)
            first_ev,prev_beat,hi,offsets,lookup,_=candidate_spec(
                prev,tempo=tempo,first=0.1,audio_end_sec=10.,candidate_count=count,clock=clock)
            reconstructed=[(Fraction(0) if first_ev else prev_beat)+off for off in offsets]
            assert beats == reconstructed
            for idx, b in enumerate(beats):
                off = b if first_ev else b - prev_beat
                assert lookup[off] == idx

    rng=np.random.default_rng(42)
    times=np.sort(rng.uniform(0.1, 10.0, size=256)).astype(np.float64)
    values=rng.normal(size=(256, 120)).astype(np.float32)
    queries=np.array([0.0, times[0], 0.5*(times[10]+times[11]), times[-1], 12.0], dtype=np.float64)
    ref=np.column_stack([np.interp(queries, times, values[:, c], left=0., right=0.) for c in range(120)]).astype(np.float32)
    got=interp_candidate_audio(values, times, queries)
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)


def test_absolute_recovery_preserves_triplets_and_provides_a_way_out():
    from maichart.event_timing import EventTempo
    tempo=[EventTempo(Fraction(0),120.)]
    for previous in (Fraction(2,3),Fraction(1,12),Fraction(11,48)):
        candidates=candidate_beats(previous,tempo=tempo,first=0.,audio_end_sec=10.,candidate_count=96)
        assert 0<len(candidates)<=96
        assert candidates==sorted(set(candidates))
        assert all(p>previous for p in candidates)
        if previous==Fraction(2,3): assert Fraction(5,6) in candidates
        if previous==Fraction(1,12): assert Fraction(1,6) in candidates
        if previous==Fraction(11,48): assert Fraction(1,3) in candidates


def test_attack_confidence_suppression_only_changes_derived_columns():
    from maichart.composer.candidates import interp_candidate_audio
    values = np.zeros((3, 120), dtype=np.float32)
    values[:, :92] = 1.0
    values[:, 92] = 1.0
    values[:, 93] = .5
    times = np.asarray([0., .02, .04], dtype=np.float64)
    queries = np.asarray([.01, .03], dtype=np.float64)
    plain = interp_candidate_audio(values, times, queries)
    adjusted = interp_candidate_audio(values, times, queries, suppress_attack_bridges=True)
    np.testing.assert_allclose(adjusted[:, :92], plain[:, :92])
    np.testing.assert_allclose(adjusted[:, 93], plain[:, 93])
    assert np.all(adjusted[:, 92] < plain[:, 92])
    assert adjusted.shape == plain.shape == (2, 120)


def test_composer_generation_masks_candidates_during_double_hold(tmp_path,monkeypatch):
    sr=8000;t=np.arange(sr*4)/sr
    audio_path=tmp_path/'track.wav';sf.write(audio_path,(0.1*np.sin(2*np.pi*220*t)).astype(np.float32),sr)
    atoms=['1h[2:1]','2h[2:1]','3']
    model=ChartComposer(ComposerConfig(atom_vocab_size=len(atoms)+1,d_model=24,nhead=4,
        layers=1,dim_feedforward=48,candidate_count=16,action_slots=4))
    with torch.no_grad():
        model.eos_head.weight.zero_()
        model.eos_head.bias.fill_(-50.)
        model.action_head.weight.zero_()
        model.action_head.bias.fill_(-50.)
        bias = model.action_head.bias.view(4, len(atoms)+1)
        bias[0, 1] = 50.  # slot 0 -> '1h[2:1]'
        bias[1, 2] = 50.  # slot 1 -> '2h[2:1]'
        bias[2, 0] = 50.  # slot 2 -> stop
        bias[3, 0] = 50.  # slot 3 -> stop
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    checkpoint=tmp_path/'composer_last.pt'
    save(checkpoint,model,optimizer,epoch=1,args={},manifest=tmp_path/'manifest.json',atom_catalog=atoms)
    from maichart.composer.generate import generate,parser
    # Always choose the first legal time, so the regression cannot be hidden
    # by a random pointer jumping to the audio end after its first event.
    original_forward=ChartComposer.forward
    def nearest(self,*args,**kw):
        result=original_forward(self,*args,**kw)
        logits=-torch.arange(result['candidate_logits'].size(-1),device=result['candidate_logits'].device).float()[None]
        result['candidate_logits']=logits.expand_as(result['candidate_logits']).masked_fill(kw['candidate_mask'],float('-inf'))
        return result
    monkeypatch.setattr(ChartComposer,'forward',nearest)
    output=tmp_path/'generated.txt'
    args=parser().parse_args(['--audio',str(audio_path),'--checkpoint',str(checkpoint),
        '--output',str(output),'--bpm','120','--temperature','0','--device','cpu','--max-events','3'])
    report=generate(args)
    assert report['event_count'] == 3
    assert report['playability']['remaining_hard'] == 0


def test_webui_composer_progress_handles_prepare_and_scan_events():
    from maichart.webui.server import TaskState, _update_composer_progress
    task = TaskState(task_id='t1', kind='composer_train')
    _update_composer_progress(task, 'form_prepare {"stage": "ready", "charts": 1787}')
    assert '1787' in task.message
    _update_composer_progress(task, 'composer_prepare {"stage": "scanning", "completed": 500, "total": 1787, "samples": 275000}')
    assert '500/1787' in task.message
    _update_composer_progress(task, 'composer_prepare {"stage": "ready", "samples": 947683}')
    assert task.result.get('samples') == 947683


def test_composer_progress_keeps_metrics_and_resume_epoch_range():
    from maichart.webui.server import TaskState,_update_composer_progress
    task=TaskState(task_id='progress',kind='composer_train')
    _update_composer_progress(task,'composer_runtime '+json.dumps({'start_epoch':8,'end_epoch':9,
        'samples':100,'parameters':1000000,'candidate_recall':.99}))
    payload={'epoch':8,'batch':50,'total':100,'loss':1.5,'pointer_accuracy':.7,
        'samples_per_sec':123,'eta_sec':12,'timing':{'data_sec':1.}}
    _update_composer_progress(task,'composer_progress '+json.dumps(payload))
    assert task.progress==.25
    assert task.result['composer_metrics']==payload
    assert task.result['composer_metric_history'][-1]==payload
    assert task.result['composer_runtime']['candidate_recall']==.99


def test_mid_epoch_resume_restores_updates_without_reloading_consumed_examples(tmp_path,monkeypatch):
    import pytest
    from maichart.composer import train
    class TinyDataset:
        codec_atoms=['1','2'];fingerprint='tiny'
        candidate_misses=0;candidate_total=4;candidate_miss_examples=[]
        def __init__(self,*args,**kwargs):
            from types import SimpleNamespace
            self.accessed=[]
            self.action_slots = self.action_slots_requested = kwargs['action_slots']
            self.max_required_action_slots = 1
            self.samples=[SimpleNamespace(song_index=0) for _ in range(4)]
        def __len__(self):return 4
        def set_epoch(self,epoch):pass
        def cache_stats(self):return {}
        def __getitem__(self,index):
            self.accessed.append(index)
            return {'candidate_features':torch.full((16,141),float(index)/10),
                'candidate_target':torch.tensor(index),'candidate_mask':torch.zeros(16,dtype=torch.bool),
                'action_target':torch.tensor([1,0]),'history':torch.zeros((0,2),dtype=torch.long),
                'history_features':torch.zeros((0,8)),'action_state':torch.zeros(50),
                'head_timing':torch.zeros((16,99)),
                'condition':torch.zeros(16),'future_target':torch.zeros(8),
                'eos_target':torch.tensor(float(index==3)),'song_index':torch.tensor(0),
                'event_index':torch.tensor(index)}
    monkeypatch.setattr(train,'ComposerDataset',TinyDataset)
    args=['--manifest','tiny.json','--epochs','1','--batch-size','2','--d-model','16',
        '--layers','1','--nhead','4','--dim-feedforward','32','--action-slots','2',
        '--device','cpu','--precision','fp32','--prefetch-batches','0','--checkpoint-interval-steps','1',
        '--architecture-profile','music_context_v2']
    train.main(args+['--checkpoint-dir',str(tmp_path/'full')])
    original_save=train.save
    def interrupting_save(*a,**kw):
        result=original_save(*a,**kw)
        if kw.get('training_state',{}).get('batch_in_epoch')==1:raise InterruptedError('stopped')
        return result
    monkeypatch.setattr(train,'save',interrupting_save)
    with pytest.raises(InterruptedError):train.main(args+['--checkpoint-dir',str(tmp_path/'partial')])
    monkeypatch.setattr(train,'save',original_save)
    train.main(args+['--checkpoint-dir',str(tmp_path/'resumed'),
        '--resume',str(tmp_path/'partial'/'composer_last.pt')])
    complete=torch.load(tmp_path/'full'/'composer_last.pt',weights_only=False,map_location='cpu')
    resumed=torch.load(tmp_path/'resumed'/'composer_last.pt',weights_only=False,map_location='cpu')
    assert resumed['optimizer_steps']==complete['optimizer_steps']==2
    for key,value in complete['model_state_dict'].items():
        torch.testing.assert_close(value,resumed['model_state_dict'][key],rtol=0,atol=0)
