from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import torch

from maichart.composer.evaluation import assert_unseen, evaluate_teacher_forced, split_manifest


def _manifest(tmp_path):
    songs = []
    for i in range(9):
        audio = tmp_path / f'audio{i}.bin'
        audio.write_bytes(f'audio {i}'.encode())
        songs.append({'song_id': str(i), 'audio_path': str(audio), 'maidata_path': str(tmp_path / f'{i}.txt')})
    # A second song id points to duplicate content at a different path.
    duplicate = tmp_path / 'duplicate.bin'; duplicate.write_bytes((tmp_path / 'audio0.bin').read_bytes())
    songs.append({'song_id': 'duplicate', 'audio_path': str(duplicate), 'maidata_path': str(tmp_path / 'duplicate.txt')})
    path = tmp_path / 'source.json'; path.write_text(json.dumps({'songs': songs}), encoding='utf-8')
    return path


def test_split_is_deterministic_and_keeps_duplicate_audio_together(tmp_path):
    source = _manifest(tmp_path)
    a = split_manifest(source, tmp_path / 'one', fraction=.1, seed=4)
    b = split_manifest(source, tmp_path / 'two', fraction=.1, seed=4)
    assert a['group_count'] == 9
    assert a['requested_total_holdout_fraction'] == .1
    assert a['actual_total_holdout_group_fraction'] == pytest.approx(2 / 9)
    for split in ('train', 'dev', 'final'):
        assert a['splits'][split]['sha256'] == b['splits'][split]['sha256']
        songs = json.loads(Path(a['splits'][split]['path']).read_text(encoding='utf-8'))['songs']
        ids = {s['song_id'] for s in songs}
        assert ('0' in ids) == ('duplicate' in ids)
    assert_unseen(a['splits']['train']['path'], a['splits']['dev']['path'])
    assert_unseen(a['splits']['train']['path'], a['splits']['final']['path'])
    with pytest.raises(ValueError, match='overlap'):
        assert_unseen(source, a['splits']['final']['path'])


def test_split_requires_audio_to_verify_identity(tmp_path):
    source = _manifest(tmp_path)
    (tmp_path / 'audio0.bin').unlink()
    with pytest.raises(FileNotFoundError, match='audio identity'):
        split_manifest(source, tmp_path / 'output')


def test_evaluation_pools_weights_and_restores_mode_and_rng(monkeypatch):
    import maichart.composer.data as data
    import maichart.composer.train as train
    class Dataset:
        codec_atoms = ['a']
        def __len__(self): return 3
        def __getitem__(self, i): return {'x': i}
    def collate(items):
        x = torch.tensor([item['x'] for item in items], dtype=torch.float32)
        return {'eos_target': x, 'sample_weight': x + 1, 'candidate_target': x.long()}
    def forward(model, tensors, *, packed_actions=False):
        assert not model.training
        torch.rand(1)
        return {'candidate_logits': tensors['eos_target']}
    def loss(output, batch, atom_vocab_size, **kwargs):
        weights = batch['sample_weight']
        value = ((batch['eos_target'] + 1) * weights).sum() / weights.sum()
        return value, {'pointer_loss': value, 'action_loss': value}
    monkeypatch.setattr(data, 'collate_composer', collate)
    monkeypatch.setattr(train, '_forward_batch', forward)
    monkeypatch.setattr(train, '_loss', loss)
    model = torch.nn.Sequential(torch.nn.Dropout(), torch.nn.Linear(1, 1))
    model.train(); model[0].eval()
    state = torch.random.get_rng_state().clone()
    result = evaluate_teacher_forced(model, Dataset(), device='cpu', batch_size=2)
    assert result['metrics']['pointer_loss'] == pytest.approx(14 / 6)
    assert result['samples'] == 3
    assert model.training and not model[0].training
    assert torch.equal(state, torch.random.get_rng_state())


def test_fraction_is_total_holdout_not_per_split(tmp_path):
    songs = []
    for i in range(40):
        audio = tmp_path / f"{i}.bin"
        audio.write_bytes(str(i).encode())
        songs.append({'song_id': str(i), 'audio_path': str(audio)})
    manifest = tmp_path / 'all.json'
    manifest.write_text(json.dumps({'songs': songs}), encoding='utf-8')
    result = split_manifest(manifest, tmp_path / 'split', fraction=.05)
    assert result['splits']['train']['groups'] == 38
    assert result['splits']['dev']['groups'] == 1
    assert result['splits']['final']['groups'] == 1
    assert result['actual_total_holdout_group_fraction'] == pytest.approx(.05)
    assert result['actual_total_holdout_song_fraction'] == pytest.approx(.05)


def test_generated_chart_metrics_distinguish_event_and_head_denominators(tmp_path):
    import importlib.util
    path = Path(__file__).resolve().parents[1] / 'tools' / 'evaluate_composer.py'
    spec = importlib.util.spec_from_file_location('evaluate_composer_for_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    chart = tmp_path / 'maidata.txt'
    chart.write_text('&wholebpm=120\n&inote_4=(120){4}1b/2x,A1/2,1/2/3,E\n', encoding='utf-8')
    report = module.chart_report(chart, 4)
    assert report['counts']['events'] == 3
    assert report['rates']['touch_events'] == {'numerator': 1, 'denominator': 3, 'rate': pytest.approx(1/3)}
    assert report['rates']['touch_heads'] == {'numerator': 1, 'denominator': 7, 'rate': pytest.approx(1/7)}
    assert report['counts']['break_heads'] == 1
    assert report['counts']['ex_heads'] == 1
    assert report['counts']['events_3plus_heads'] == 1


def test_generation_panel_fixed_conditions_deduplicates_and_records_skip(tmp_path, monkeypatch):
    import subprocess
    from types import SimpleNamespace
    from maichart.composer.evaluation import run_generation_panel
    checkpoint = tmp_path / 'model.pt'; checkpoint.write_bytes(b'weights')
    import maichart.composer.checkpoint as checkpoint_module
    monkeypatch.setattr(checkpoint_module, 'read_payload', lambda path: {'checkpoint_kind': 'inference'})
    audio1 = tmp_path / 'one.mp3'; audio1.write_bytes(b'one')
    duplicate = tmp_path / 'copy.mp3'; duplicate.write_bytes(b'one')
    audio2 = tmp_path / 'two.mp3'; audio2.write_bytes(b'two')
    reference = tmp_path / 'maidata.txt'
    reference.write_text('&wholebpm=120\n&first=0.25\n&lv_4=13\n&inote_4=(120){4}1,E\n', encoding='utf-8')
    good = {'song_id': 'one', 'audio_path': str(audio1), 'maidata_path': str(reference),
            'difficulties': [{'difficulty_index': 4, 'level': 13, 'initial_bpm': 120}]}
    songs = [good, {**good, 'audio_path': str(duplicate)},
             {'audio_path': str(audio2), 'maidata_path': str(tmp_path / 'missing.txt')}]
    manifest = tmp_path / 'holdout.json'; manifest.write_text(json.dumps({'songs': songs}), encoding='utf-8')
    calls = []
    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        directory = Path(cmd[cmd.index('--output-dir') + 1]); directory.mkdir(parents=True)
        (directory / 'evaluation.json').write_text(json.dumps({'runs': [{'seed': 0}, {'seed': 1}]}), encoding='utf-8')
        return SimpleNamespace(returncode=0, stderr='', stdout='')
    monkeypatch.setattr(subprocess, 'run', fake_run)
    report = run_generation_panel(checkpoint, manifest, tmp_path / 'panel')
    assert report['selected_distinct_audio_count'] == 2
    assert report['completed_songs'] == 1 and len(report['skipped']) == 1
    assert report['unseen_verified'] is False
    assert report['skipped'][0]['reason'] == 'reference chart file missing'
    assert len(calls) == 1
    frozen = Path(calls[0][calls[0].index('--checkpoint') + 1])
    assert frozen != checkpoint and frozen.read_bytes() == b'weights'
    assert report['source_checkpoint_sha256'] == report['checkpoint_sha256']
    assert calls[0][calls[0].index('--pointer-temperature') + 1] == '0.5'
    assert calls[0][calls[0].index('--action-temperature') + 1] == '0.75'
    assert calls[0][calls[0].index('--max-events') + 1] == '0'
    assert calls[0][calls[0].index('--first') + 1] == '0.25'


def test_generation_panel_reports_process_failure_without_metrics(tmp_path, monkeypatch):
    import subprocess
    from types import SimpleNamespace
    from maichart.composer.evaluation import run_generation_panel
    checkpoint = tmp_path / 'model.pt'; checkpoint.write_bytes(b'weights')
    import maichart.composer.checkpoint as checkpoint_module
    monkeypatch.setattr(checkpoint_module, 'read_payload', lambda path: {'checkpoint_kind': 'inference'})
    audio = tmp_path / 'audio.mp3'; audio.write_bytes(b'audio')
    ref = tmp_path / 'maidata.txt'
    ref.write_text('&wholebpm=120\n&first=0\n&lv_4=13\n&inote_4=(120){4}1,E\n', encoding='utf-8')
    manifest = tmp_path / 'holdout.json'
    manifest.write_text(json.dumps({'songs': [{'audio_path': str(audio), 'maidata_path': str(ref),
        'difficulties': [{'difficulty_index': 4, 'level': 13, 'initial_bpm': 120}]}]}), encoding='utf-8')
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr='failure', stdout=''))
    report = run_generation_panel(checkpoint, manifest, tmp_path / 'panel')
    assert report['status'] == 'failed'
    assert not report['songs'] and report['errors'][0]['stderr'] == 'failure'


def test_generation_panel_rejects_checkpoint_changed_during_export(tmp_path, monkeypatch):
    import maichart.composer.checkpoint as checkpoint_module
    from maichart.composer.evaluation import run_generation_panel
    checkpoint = tmp_path / 'model.pt'; checkpoint.write_bytes(b'before')
    manifest = tmp_path / 'holdout.json'; manifest.write_text(json.dumps({'songs': []}), encoding='utf-8')
    monkeypatch.setattr(checkpoint_module, 'read_payload', lambda path: {'checkpoint_kind': 'training'})
    def export(source, target):
        Path(target).write_bytes(b'frozen')
        Path(source).write_bytes(b'after')
    monkeypatch.setattr(checkpoint_module, 'export_inference', export)
    with pytest.raises(RuntimeError, match='changed while creating'):
        run_generation_panel(checkpoint, manifest, tmp_path / 'panel')
    assert not list((tmp_path / 'panel').glob('run_*/evaluation_checkpoint.pt'))


def test_evaluate_composer_passes_reference_tempo_map_and_explicit_override(tmp_path, monkeypatch):
    import importlib.util
    import subprocess
    from types import SimpleNamespace
    path = Path(__file__).resolve().parents[1] / 'tools' / 'evaluate_composer.py'
    spec = importlib.util.spec_from_file_location('evaluate_composer_tempo_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    reference = tmp_path / 'reference.txt'
    reference.write_text('&wholebpm=120\n&first=0\n&inote_4=(120){4}1,(180)2,E\n', encoding='utf-8')
    checkpoint = tmp_path / 'model.pt'; checkpoint.write_bytes(b'weights')
    audio = tmp_path / 'audio.mp3'; audio.write_bytes(b'audio')
    calls = []
    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        generated = Path(cmd[cmd.index('--output') + 1])
        generated.write_text('&wholebpm=120\n&first=0\n&inote_4=(120){4}1,E\n', encoding='utf-8')
        generated.with_suffix('.composer.json').write_text('{}', encoding='utf-8')
        return SimpleNamespace(returncode=0, stderr='', stdout='')
    monkeypatch.setattr(subprocess, 'run', fake_run)
    base = ['--checkpoint', str(checkpoint), '--audio', str(audio), '--reference', str(reference),
            '--difficulty', '4', '--bpm', '120', '--seeds', '0']
    out = tmp_path / 'default'
    module.main([*base, '--output-dir', str(out)])
    report = json.loads((out / 'evaluation.json').read_text(encoding='utf-8'))
    tempo_path = Path(calls[-1][calls[-1].index('--tempo-map') + 1])
    assert json.loads(tempo_path.read_text(encoding='utf-8')) == [
        {'beat': '0', 'bpm': 120.0}, {'beat': '1', 'bpm': 180.0}]
    assert report['generation_tempo_map']['source'] == 'reference'
    override = tmp_path / 'override.json'; override.write_text('[{"beat":"0","bpm":99}]', encoding='utf-8')
    out2 = tmp_path / 'explicit'
    module.main([*base, '--output-dir', str(out2), '--tempo-map', str(override)])
    report2 = json.loads((out2 / 'evaluation.json').read_text(encoding='utf-8'))
    assert calls[-1][calls[-1].index('--tempo-map') + 1] == str(override)
    assert report2['generation_tempo_map']['source'] == 'explicit'


def test_form_eligible_split_excludes_untrainable_songs_before_audio_hashing(tmp_path,monkeypatch):
    from maichart.composer.evaluation import form_chart_eligibility
    from maichart.form import cache_support
    songs=[]
    for i in range(30):
        difficulties=[dict(difficulty_index=4,level_raw='10')] if i%2 else [dict(difficulty_index=2,level_raw='13')]
        songs.append(dict(song_id=f'excluded-{i}',audio_path='missing.wav',difficulties=difficulties))
    songs.append(dict(song_id='broken',difficulties=[dict(difficulty_index=4,level_raw='13\ufffd\ufffd')]))
    for i in range(3):
        audio=tmp_path/f'{i}.wav';audio.write_bytes(bytes([i]))
        songs.append(dict(song_id=f'eligible-{i}',audio_path=str(audio),difficulties=[
            dict(difficulty_index=4,level_raw='13\ufffd\ufffd'),dict(difficulty_index=5,level_raw='13+')]))
    source=tmp_path/'source.json';original=json.dumps(dict(songs=songs));source.write_text(original)
    hashed=[]
    class Digests:
        def digest(self,path):
            hashed.append(str(path));return Path(path).read_bytes().hex()
    monkeypatch.setattr(cache_support,'source_digests',lambda directory:Digests())
    report=split_manifest(source,tmp_path/'split',fraction=.05,seed=3,cache_dir=tmp_path/'cache',
        chart_eligibility=form_chart_eligibility)
    evidence=report['chart_eligibility']
    assert evidence['source_song_count']==34 and evidence['eligible_song_count']==3
    assert evidence['excluded_song_count']==31 and evidence['eligible_chart_count']==3
    assert evidence['excluded_difficulty_reason_counts']['invalid_level']==4
    assert len(hashed)==3
    for role in ('train','dev','final'):
        output=json.loads(Path(report['splits'][role]['path']).read_text())
        assert len(output['songs'])==1
        assert output['songs'][0]['song_id'].startswith('eligible-')
        assert output['songs'][0]['difficulties']==[dict(difficulty_index=5,level_raw='13+')]
    assert source.read_text()==original


def test_form_eligible_split_still_rejects_missing_audio_for_eligible_chart(tmp_path):
    from maichart.composer.evaluation import form_chart_eligibility
    source=tmp_path/'source.json';source.write_text(json.dumps(dict(songs=[dict(
        song_id='eligible',audio_path='missing.wav',difficulties=[dict(difficulty_index=4,level_raw='12')])])) )
    with pytest.raises(FileNotFoundError,match='audio identity'):
        split_manifest(source,tmp_path/'split',chart_eligibility=form_chart_eligibility)


def test_form_chart_eligibility_matches_default_numeric_boundaries_and_ignores_bad_siblings():
    from maichart.composer.evaluation import form_chart_eligibility
    for slot,level in [(4,'12'),(5,'13+'),(6,'15')]:
        assert form_chart_eligibility(dict(difficulty_index=slot,level_raw=level)) is None
    for difficulty,reason in [(dict(index=3,level='13'),'slot_out_of_range'),
        (dict(index=4,level='15+'),'level_out_of_range'),
        (dict(index=4,level_raw='13\ufffd\ufffd',level=13),'invalid_level'),
        (dict(index='broken',level=13),'invalid_slot')]:
        assert form_chart_eligibility(difficulty)==reason
