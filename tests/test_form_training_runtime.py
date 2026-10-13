from copy import deepcopy
import json

import pytest
import torch

from maichart.form.training_runtime import EpochBatches, Progress, UpdateEngine, resolve_precision
from maichart.form.train import layout_loss


@pytest.mark.parametrize('partitions', [[2, 2], [2, 1], [1, 1, 1, 1], [1]])
def test_accumulation_matches_a_single_batch_including_short_tail(partitions):
    torch.manual_seed(42)
    size = sum(partitions)
    x = torch.randn(size, 3)
    y = torch.randn(size, 2)
    base = torch.nn.Linear(3, 2)
    accumulated = deepcopy(base)
    single_optimizer = torch.optim.SGD(base.parameters(), lr=.03)
    multi_optimizer = torch.optim.SGD(accumulated.parameters(), lr=.03)
    single = UpdateEngine(single_optimizer, base.parameters(), torch.device('cpu'))
    multi = UpdateEngine(multi_optimizer, accumulated.parameters(), torch.device('cpu'))
    single.backward((base(x)-y).square().mean(1).sum(), size)
    single.step()
    offset = 0
    for count in partitions:
        multi.backward((accumulated(x[offset:offset+count])-y[offset:offset+count]).square().mean(1).sum(), count)
        offset += count
    update = multi.step()
    assert update['samples'] == size and update['microbatches'] == len(partitions)
    for left, right in zip(base.parameters(), accumulated.parameters()):
        torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)
    assert multi.samples == 0 and all(p.grad is None for p in accumulated.parameters())


def test_loss_normalization_is_independent_of_microbatch_partition_and_zero_weight():
    torch.manual_seed(43)
    output = {'logits':torch.randn(3, 5, 7, requires_grad=True),
              'relation_logits':torch.randn(3, 2, requires_grad=True)}
    batch = {'target':torch.tensor([[1,2,3,0,0],[2,1,4,5,0],[1,0,0,0,0]]),
             'weight':torch.tensor([1.,.4,0.]),'relation_valid':torch.tensor([1,0,1]),
             'relation_target':torch.rand(3,2)}
    expected, _ = layout_loss(output, batch, .1)
    micro = []
    for span in (slice(0,2), slice(2,3)):
        loss, _ = layout_loss({k:v[span] for k,v in output.items()}, {k:v[span] for k,v in batch.items()}, .1)
        micro.append(loss)
    torch.testing.assert_close(sum(micro), expected)
    assert float(micro[-1]) == 0


def test_sampler_resume_skips_exactly_consumed_batches_and_retains_tail():
    whole = list(EpochBatches(19, 4, 42, 3))
    assert len(whole) == 5 and len(whole[-1]) == 3
    assert sorted(sum(whole, [])) == list(range(19))
    assert list(EpochBatches(19,4,42,3,start=2)) == whole[2:]
    assert list(EpochBatches(19,4,42,3,start=2,limit=4)) == whole[2:4]
    assert list(EpochBatches(19,4,42,4)) != whole


def test_progress_separates_batch_count_and_optimizer_updates(capsys):
    progress = Progress(1, 2175, 19086, 10., torch.device('cpu'), 'fp32')
    progress.begin('layout_revision', 1863)
    progress.add({'loss':torch.tensor(2.)})
    progress.log(1864, 19086, optimizer_steps=466, blueprint_steps=2175)
    record = json.loads(capsys.readouterr().out.split(' ',1)[1])
    assert record['unit'] == 'batches'
    assert record['optimizer_steps'] == 466
    assert record['overall_fraction'] == pytest.approx((2175+1864)/(2175+19086))
    assert record['phase_fraction'] == pytest.approx(1864/19086)
    assert record['eta_phase_sec'] >= 0 and record['losses']['loss'] == 2


def test_auto_precision_cpu_and_unsupported_modes():
    assert resolve_precision('auto', torch.device('cpu')) == 'fp32'
    with pytest.raises(ValueError):
        resolve_precision('fp16', torch.device('cpu'))


def test_scaler_skips_overflowed_norm_without_corrupting_optimizer():
    parameter = torch.nn.Parameter(torch.ones(4))
    optimizer = torch.optim.AdamW([parameter], lr=.01)
    engine = UpdateEngine(optimizer, [parameter], torch.device('cpu'))
    # CPU scaling exercises the same overflow lifecycle without requiring CUDA.
    engine.scaler = torch.amp.GradScaler('cpu')
    engine.backward(parameter.sum(), 1)
    parameter.grad.fill_(1e38)  # finite entries, infinite FP32 L2 norm
    scale = engine.scaler.get_scale()
    assert not engine.step()['applied']
    assert engine.scaler.get_scale() == scale / 2
    assert not optimizer.state
    torch.testing.assert_close(parameter, torch.ones(4), rtol=0, atol=0)
    engine.backward(parameter.sum(), 1)
    assert engine.step()['applied']
    assert torch.isfinite(parameter).all() and (parameter < 1).all()


def test_mid_epoch_checkpoint_resume_matches_uninterrupted_updates(tmp_path, monkeypatch):
    from maichart.form import train, data
    from test_form_pipeline import fixture_audio
    audio = fixture_audio(5)
    folder = tmp_path/'song'; folder.mkdir()
    (folder/'track.wav').write_bytes(b'fixture')
    (folder/'maidata.txt').write_text('&title=fixture\n&wholebpm=120\n&lv_5=13\n&inote_5=(120){4}1,2,3,4,5,6,7,8,E\n',encoding='utf-8')
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'songs':[{'song_id':'one','maidata_path':'song/maidata.txt',
        'audio_path':'song/track.wav','difficulties':[{'difficulty_index':5,'level_raw':'13'}]}]}))
    original_dataset = data.FormDataset
    monkeypatch.setattr(train, 'FormDataset', lambda *a, **kw: original_dataset(*a, **kw, audio_loader=lambda *a, **kw: audio))
    common = ['--manifest', str(manifest), '--cache-dir', str(tmp_path/'cache'), '--epochs','1',
              '--batch-size','1','--grad-accumulation','2','--d-model','16','--nhead','2','--layers','1',
              '--max-tokens','256','--max-memory-tokens','64','--max-region-seconds','2',
              '--checkpoint-interval-steps','1','--precision','fp32','--seed','33','--dropout','.1',
              '--max-steps','4']
    threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        train.main(common + ['--checkpoint-dir', str(tmp_path/'full')])
        uninterrupted = torch.load(tmp_path/'full/form_last.pt', weights_only=False)
        original_save = train._atomic_save
        def interrupt_after_safe_update(value, path):
            original_save(value, path)
            position = value['training_position']
            if position['phase'] == 'layout_revision' and position['completed'] == 2:
                raise InterruptedError('test interruption after durable checkpoint')
        monkeypatch.setattr(train, '_atomic_save', interrupt_after_safe_update)
        with pytest.raises(InterruptedError):
            train.main(common + ['--checkpoint-dir', str(tmp_path/'interrupted')])
        saved_path = tmp_path/'interrupted/form_last.pt'
        partial = torch.load(saved_path, weights_only=False)
        assert partial['epoch'] == 0 and partial['training_position']['completed'] == 2
        monkeypatch.setattr(train, '_atomic_save', original_save)
        train.main(common + ['--checkpoint-dir', str(tmp_path/'resumed'), '--resume', str(saved_path),
                             '--max-steps','0','--precision','auto'])
        resumed = torch.load(tmp_path/'resumed/form_last.pt', weights_only=False)
        assert resumed['optimizer_steps'] == uninterrupted['optimizer_steps']
        assert resumed['task_training'] == uninterrupted['task_training']
        assert resumed['precision'] == 'fp32' and resumed['args']['max_steps'] == 4
        for key in ('model_state_dict','blueprint_state_dict'):
            for name in uninterrupted[key]:
                torch.testing.assert_close(resumed[key][name], uninterrupted[key][name], rtol=0, atol=0)
    finally:
        torch.set_num_threads(threads)
