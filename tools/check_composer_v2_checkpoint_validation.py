"""Exercise real fresh/resume training and fixed checkpoint validation on CPU."""
import contextlib
import gc
import io
import json
from pathlib import Path
import tempfile
import wave
import numpy as np
import torch
from maichart.composer_v2.train import parser, train
from maichart.composer_v2.checkpoint import inspect_payload


def run(root, output, maximum, resume=False):
    arguments = ['--manifest', str(root/'train.json'), '--validation-manifest', str(root/'dev.json'),
        '--checkpoint-dir', str(output), '--cache-dir', str(root/'cache'),
        '--device', 'cpu', '--prepare-device', 'cpu', '--precision', 'fp32', '--audio-backend', 'native',
        '--difficulty-source', 'declared', '--sequence-policy', 'continuous_events_v1',
        '--d-model', '32', '--layers', '1', '--nhead', '4', '--dim-feedforward', '64',
        '--audio-layers', '1', '--song-layers', '1', '--music-context', 'off',
        '--history-beats', '4', '--window-beats', '4', '--future-beats', '1', '--max-tokens', '128',
        '--refresh-events', '4', '--batch-size', '1', '--grad-accumulation', '1', '--cpu-threads', '1',
        '--timing-recovery', 'off', '--rhythm-structure', 'off', '--intensity-curve', 'off',
        '--onset-readout', 'off', '--auto-quality-policy', 'off', '--condition-policy', 'level_only_v1',
        '--epochs', '2', '--lr', '.00015', '--warmup-steps', '0', '--dropout', '.1',
        '--checkpoint-interval-steps', '2', '--validation-max-batches', '1',
        '--log-interval', '1', '--max-steps', str(maximum), '--save-every', '0', '--early-stopping-patience', '0']
    if resume: arguments += ['--resume', str(output/'composer_v2_last.pt')]
    stream = io.StringIO()
    try:
        with contextlib.redirect_stdout(stream): train(parser().parse_args(arguments))
    except Exception:
        print(stream.getvalue())
        raise
    records = []
    for line in stream.getvalue().splitlines():
        if line.startswith('composer_v2_progress '):
            row = json.loads(line.split(' ', 1)[1])
            if row.get('stage') == 'checkpoint_validation': records.append(row)
    payload = torch.load(output/'composer_v2_last.pt', weights_only=True, map_location='cpu')
    best = torch.load(output/'composer_v2_best.pt', weights_only=True, map_location='cpu')
    assert inspect_payload(payload)['can_resume']
    assert not best['best_provisional']
    assert payload['model_config']['attention_normalization'] == 'qk_rms_v1'
    assert payload['training_contract']['attention_normalization'] == 'qk_rms_v1'
    return payload, records


@contextlib.contextmanager
def fixture_directory():
    with tempfile.TemporaryDirectory(prefix='composer_validation_') as name:
        root = Path(name)
        try:
            yield root
        finally:
            # Process-wide caches hold Windows mmap/SQLite handles past train().
            from maichart.form.waveform_cache import _mapped
            from maichart.form.cache_support import source_digests
            _mapped.cache_clear()
            source_digests(str((root/'cache').resolve())).close()
            source_digests.cache_clear()
            gc.collect()


def main():
    with fixture_directory() as root:
        for kind, frequency in [('train', 220), ('dev', 330)]:
            audio = root/(kind+'.wav')
            waveform = (np.sin(2*np.pi*frequency*np.arange(20*24000)/24000)*1000).astype('<i2')
            with wave.open(str(audio), 'wb') as stream:
                stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(24000)
                stream.writeframes(waveform.tobytes())
            chart = root/(kind+'.txt')
            notes = [str(index%8+1) for index in range(32)]
            if kind == 'dev': notes.reverse()
            chart.write_text('&title='+kind+'\n&first=0\n&lv_5=12\n&inote_5=(120){4}'+','.join(notes)+',E\n', encoding='utf-8')
            document = dict(songs=[dict(song_id=kind, title=kind, audio_path=str(audio), maidata_path=str(chart),
                difficulties=[dict(difficulty_index=5, level_raw='12', has_chart=True)])])
            (root/(kind+'.json')).write_text(json.dumps(document), encoding='utf-8')
        full, full_checks = run(root, root/'full', 6)
        _, first_checks = run(root, root/'split', 3)
        resumed, last_checks = run(root, root/'split', 6, resume=True)
        assert [row['optimizer_steps'] for row in full_checks] == [2, 4, 6]
        assert [row['optimizer_steps'] for row in first_checks+last_checks] == [2, 4, 6]
        scores = [row['validation_loss'] for row in full_checks]
        assert full['training_state']['best_loss'] == min(scores)
        assert full['training_state'] == resumed['training_state']
        for key, actual in full['model_state_dict'].items():
            torch.testing.assert_close(actual, resumed['model_state_dict'][key], rtol=0, atol=0)
        for expected, actual in zip(full_checks, first_checks+last_checks):
            for key in ('optimizer_steps', 'validation_loss', 'validation_phase_loss', 'validation_nonphase_time_loss'):
                assert expected[key] == actual[key], (key, expected[key], actual[key])
        print('PASS: real checkpoint validation at steps 2/4/6; best uses fixed CE; fresh + resume weights/metrics match exactly.')


if __name__ == '__main__':
    main()
