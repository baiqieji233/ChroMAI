"""Generation regressions covering the actual stream loop and API transport."""
from fractions import Fraction as F
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from maichart.event_timing import EventPoint, EventTempo
from maichart.form.codec import RationalCodec
from maichart.generation.conditions import GenerationCondition, GENERATION_DEFAULTS
from maichart.stream import checkpoint
from maichart.stream.decoding import compose_stream
from maichart.stream.generate import parser
from maichart.stream.memory import StreamMemoryBank
from maichart.stream.reference_guard import ReferenceCandidateGuard
from maichart.stream.model import ChartStreamTransformer, StreamConfig
from maichart.stream.runtime import resolve_device
from test_webui import webui_api
from maichart.webui import server as webui


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def options(**kwargs):
    args = parser().parse_args(['--audio', 'unused.wav', '--checkpoint', 'unused.pt',
                                '--output', 'unused.txt', '--device', 'cpu',
                                '--temperature', '0', '--position-temperature', '0'])
    args.position_beam_size = 1  # Exercise the character-compatible loop here.
    args.position_interval_strength = 0
    args.max_button_heads = 0
    args.hold_protection = 'off'
    args.reference_check_attempts = 0
    for key, value in kwargs.items():
        setattr(args, key, value)
    return args


def compose(monkeypatch, points, *, max_tokens=32, options_override=None, targets=None, atom_catalog=None):
    codec = RationalCodec()
    model = ChartStreamTransformer(StreamConfig(len(codec), d_model=16, nhead=2, layers=1,
        dim_feedforward=32, dropout=0, max_tokens=max_tokens, max_history_tokens=8)).eval()
    cells = [codec.cell(point) for point in points]
    calls, encoder_calls = [], []
    committed = []
    original = model.audio_encoder.forward
    original_append = StreamMemoryBank.append

    def append(bank, beat, text, vector):
        committed.append((beat, text))
        return original_append(bank, beat, text, vector)

    def encoder(*args, **kwargs):
        encoder_calls.append(1)
        return original(*args, **kwargs)

    def step(token, position, memory, mask, caches=None):
        index = len(calls)
        prefix = [] if caches is None else list(caches['prefix'])
        prefix.append(int(token[0, 0]))
        if caches is not None:
            assert caches['position'] + 1 == position
            assert torch.equal(caches['mask'], mask), 'Cross-cache and history mask diverged'
            assert torch.equal(caches['memory'], memory), 'Cross-cache and history memory diverged'
        calls.append((int(token[0, 0]), position))
        if targets is not None:
            assert index < len(targets), 'Unexpected extra decoding after completion'
            selected = targets[index]
        elif codec.at_id not in prefix or prefix[-1] == codec.end_id:
            selected = codec.eos_id if len(committed) == len(cells) else codec.at_id
        else:
            current = prefix[max(i for i, value in enumerate(prefix) if value == codec.at_id):]
            # A replay consumes earlier cells too. Their intermediate logits
            # are irrelevant; the final current-cell prefix determines the
            # next scripted token, independently of the number of replays.
            ordered = cells[len(committed):] + cells[:len(committed)]
            matching = next((cell for cell in ordered if cell[:len(current)] == current), None)
            selected = matching[len(current)] if matching is not None and len(current) < len(matching) else codec.at_id
        logits = torch.full((1, 1, len(codec)), -100.)
        logits[0, 0, selected] = 100.
        return {'logits': logits, 'hidden': torch.zeros(1, 1, 16)}, {
            'position': position, 'mask': mask.clone(), 'memory': memory.clone(), 'prefix': prefix}

    monkeypatch.setattr(model.audio_encoder, 'forward', encoder)
    monkeypatch.setattr(model, 'decode_step', step)
    monkeypatch.setattr(StreamMemoryBank, 'append', append)
    song = SimpleNamespace(audio=SimpleNamespace(duration_sec=1.),
                           tempo=[EventTempo(F(0), 120.)], first=0.)
    values = torch.randn(8, 120)
    times = np.linspace(0., .99, 8)
    condition = GenerationCondition(level=13., radar=GENERATION_DEFAULTS, style=GENERATION_DEFAULTS)
    atoms = atom_catalog or ['1', '2', '1h[4:1]']
    result, status = compose_stream(model, codec, song, values, times, condition, atoms,
                                    options(**(options_override or {})))
    assert len(encoder_calls) == 1
    return result, status, calls


def test_actual_stream_loop_never_offers_a_third_button_head(monkeypatch):
    proposed = [EventPoint(F(0), '1/2/3')]
    result, status, _ = compose(monkeypatch, proposed, atom_catalog=['1', '2', '3'],
        options_override={'max_button_heads': 2, 'max_events': 1})
    assert result == [EventPoint(F(0), '1/2')]
    assert status['basic_constraints']['blocked_token_choices'] > 0


def test_rejected_reference_candidate_replays_only_the_current_configuration(monkeypatch):
    checked = []
    def check(_guard,point):
        checked.append(point)
        return (len(checked) != 2, [])
    monkeypatch.setattr(ReferenceCandidateGuard,'check',check)
    expected = [EventPoint(F(0),'1'),EventPoint(F(1),'2/3')]
    result,status,_ = compose(monkeypatch,expected,max_tokens=32,
        atom_catalog=['1','2','3'],options_override={'reference_check_attempts':8})
    assert len(checked) >= 3 and checked[1] == expected[1]
    assert result[0] == expected[0] and result[1].beat == F(1)
    assert result[1].note_text != checked[1].note_text
    assert status['reason'] == 'eos'


def test_zero_temperature_and_first_beat_zero_commit_a_whole_cell(monkeypatch):
    expected = [EventPoint(F(0), '1h[4:1]')]
    actual, status, _ = compose(monkeypatch, expected, options_override={'max_events': 1})
    assert actual == expected
    assert status['reason'] == 'event_budget' and status['incomplete']
    assert actual[0].note_text.endswith(']')


def test_continuous_generation_survives_multiple_context_rotations(monkeypatch):
    expected = [EventPoint(F(i, 8), '1' if i % 2 else '2') for i in range(10)]
    actual, status, calls = compose(monkeypatch, expected, max_tokens=16)
    assert actual == expected
    assert status['resource_rotations'] >= 2
    assert status['reason'] == 'eos' and not status['incomplete']
    assert max(position for _, position in calls) <= 16
    assert status['audio_encoder_calls'] == 1


def test_file_tail_with_no_space_for_another_event_is_reported(monkeypatch):
    expected = [EventPoint(F(63, 32), '1')]
    actual, status, _ = compose(monkeypatch, expected)
    assert actual == expected
    assert status['reason'] == 'audio_range_exhausted' and not status['incomplete']


def test_configuration_crossing_context_boundary_is_replayed_without_losing_decisions(monkeypatch):
    expected = [EventPoint(F(0), '1'), EventPoint(F(1), '1h[4:1]')]
    actual, status, _ = compose(monkeypatch, expected, max_tokens=13)
    assert actual == expected
    assert status['resource_rotations'] >= 1 and status['reason'] == 'eos'


def test_single_configuration_exceeding_context_has_clear_error(monkeypatch):
    with pytest.raises(ValueError, match='流上下文 token'):
        compose(monkeypatch, [EventPoint(F(0), '1h[4:1]')], max_tokens=4)


def test_fraction_canonicalization_replays_only_actual_committed_spelling(monkeypatch):
    codec = RationalCodec()
    # Sample 2/4, normalize to 1/2, then replay the canonical BOS+header before
    # predicting the body. Intermediate replay predictions are ignored.
    sampled = [codec.at_id, *(codec.ids['P:'+c] for c in '2/4'), codec.note_id]
    replay = [codec.pad_id] * len(codec.position(F(1, 2)))
    schedule = sampled + replay + [codec.ids['N:1'], codec.end_id]
    expected = [EventPoint(F(1, 2), '1')]
    actual, status, calls = compose(monkeypatch, expected,
        options_override={'max_events': 1}, targets=schedule)
    assert actual == expected
    assert status['equivalent_fraction_normalizations'] == 1
    replayed = [token for token, _ in calls[len(sampled):]]
    assert replayed[:1+len(codec.position(F(1, 2)))] == [codec.bos_id, *codec.position(F(1, 2))]


def test_physical_guidance_receives_the_actual_song_clock(monkeypatch):
    expected = [EventPoint(F(0), '1h[4:1]'), EventPoint(F(1, 2), '1')]
    actual, status, _ = compose(monkeypatch, expected,
        options_override={'physical_feasibility_strength': 1.})
    assert actual == expected
    assert status['guidance']['physical_feasibility']['committed_collisions'] == 1


def test_full_and_incremental_decoding_agree_with_history_and_padding():
    torch.manual_seed(26)
    codec = RationalCodec()
    model = ChartStreamTransformer(StreamConfig(len(codec), d_model=16, nhead=2, layers=2,
        dim_feedforward=32, dropout=0, max_history_tokens=8)).eval()
    batch = torch.randn(2, 4, 120)
    controls = torch.randn(2, 16)
    history = torch.tensor([[1, 3, 8, 0], [1, 8, 9, 5]])
    tokens = torch.randint(1, len(codec), (2, 7))
    with torch.inference_mode():
        memory, mask = model.encode(batch, controls)
        expected = model.decode(tokens, memory, mask, history=history)['logits']
        memory, mask = model.prepare_decode_memory(memory, mask, history=history)
        cache, rows = None, []
        for index in range(tokens.size(1)):
            output, cache = model.decode_step(tokens[:, index:index+1], index, memory, mask, cache)
            rows.append(output['logits'])
    torch.testing.assert_close(torch.cat(rows, 1), expected, rtol=1e-5, atol=1e-5)


def test_auto_cpu_resolution_and_explicit_cpu_never_initialize_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: pytest.fail('CPU path accessed CUDA'))
    assert str(resolve_device('auto')) == 'cpu'
    assert str(resolve_device('cpu')) == 'cpu'
    with pytest.raises(ValueError, match='CUDA 不可用'):
        resolve_device('cuda')
    with pytest.raises(ValueError, match='生成设备'):
        resolve_device('mps')


def test_checkpoint_uses_cpu_storage_and_rejects_unknown_extra_weights(tmp_path, monkeypatch):
    model = ChartStreamTransformer(StreamConfig(55, d_model=16, nhead=2, layers=1, dim_feedforward=32))
    manifest = tmp_path/'manifest.json'; manifest.write_text('{}')
    path = tmp_path/'stream.pt'
    saved = checkpoint.save(path, model, None, epoch=1, args={}, manifest=manifest)
    actual_load, maps = torch.load, []

    def read(*args, **kwargs):
        maps.append(kwargs.get('map_location'))
        return actual_load(*args, **kwargs)

    monkeypatch.setattr(torch, 'load', read)
    checkpoint.load(path, 'cpu')
    assert maps == ['cpu']
    saved['model_state_dict']['event_gate.weight'] = torch.zeros(1, 16)
    torch.save(saved, path)
    checkpoint.load(path, 'cpu')  # Known unused heads are backward compatible.
    saved['model_state_dict']['decoder.unknown.weight'] = torch.zeros(1)
    torch.save(saved, path)
    with pytest.raises(ValueError, match='参数不兼容'):
        checkpoint.load(path, 'cpu')
    saved['input_contract'] = 'unsupported_contract'
    torch.save(saved, path)
    with pytest.raises(ValueError, match='输入合同不兼容'):
        checkpoint.load(path, 'cpu')


@pytest.mark.parametrize('device', ['auto', 'cpu', 'cuda'])
def test_http_generation_preserves_device_seed_and_zero_temperature(webui_api, monkeypatch, tmp_path, device):
    audio = tmp_path/'track.wav'; audio.write_bytes(b'fixture')
    path = tmp_path/'stream.pt'; path.write_bytes(b'fixture')
    calls = []
    monkeypatch.setattr(webui, '_run_subprocess_task', lambda task, args, **kwargs: calls.append((task, args)))
    request = {'model_family': 'chart_stream_v1', 'audio': str(audio), 'checkpoint': str(path),
               'device': device, 'temperature': 0., 'position_temperature': 0., 'seed': 123,
               'position_beam_size': 8, 'position_complexity': .7, 'early_eos_penalty': 0.,
               'max_button_heads': 2, 'hold_protection': 'on',
               'output': str(tmp_path/'maidata.txt'), 'condition': {'level': 13.6}}
    status, reply = webui_api('/api/chart-stream/generate', request)
    assert status == 200, reply
    task, args = calls[0]
    parsed = parser().parse_args(args[2:])
    assert (parsed.device, parsed.seed, parsed.temperature, parsed.position_temperature) == (device, 123, 0., 0.)
    assert (parsed.position_beam_size, parsed.position_complexity, parsed.early_eos_penalty) == (8, .7, 0.)
    assert (parsed.max_button_heads, parsed.hold_protection) == (2, 'on')
    assert task.result['seed'] == 123
    assert task.result['effective_settings']['condition_file'] == parsed.condition_file
    assert json.loads(__import__('pathlib').Path(parsed.condition_file).read_text()) == request['condition']
    status, second = webui_api('/api/chart-stream/generate', {**request, 'condition': {'level': 14.}})
    assert status == 200, second
    again = parser().parse_args(calls[1][1][2:])
    assert parsed.output != again.output and parsed.condition_file != again.condition_file
    assert json.loads(__import__('pathlib').Path(parsed.condition_file).read_text()) == request['condition']


def test_invalid_device_and_nan_fail_in_api_before_starting_job(webui_api, monkeypatch, tmp_path):
    path = tmp_path/'file'; path.touch()
    monkeypatch.setattr(webui, '_run_subprocess_task', lambda *_a, **_kw: pytest.fail('Started invalid job'))
    for bad in ({'device': 'auto-bad'}, {'temperature': -1}, {'first': 'nan'}, {'seed': -1}):
        status, _ = webui_api('/api/chart-stream/generate', {'audio': str(path), 'checkpoint': str(path), **bad})
        assert status == 400


def test_stream_task_collects_incomplete_status_and_gap_diagnostics(tmp_path):
    report = {'generation_status': 'incomplete', 'status': {'reason': 'position_search_exhausted'},
              'timing_diagnostics': {'max_gap_sec': 12.}, 'event_count': 3}
    path = tmp_path/'maidata.stream.json'
    path.write_text(json.dumps(report))
    task = webui.TaskState('timing-test', 'chart_stream_generate')
    webui._collect_task_results(task, [path])
    assert task.result['generation_status'] == 'incomplete'
    assert task.result['stream_report'] == report
    assert task.result['timing_diagnostics']['max_gap_sec'] == 12.


def test_http_old_training_resume_is_rejected_before_any_job_starts(webui_api, tmp_path, monkeypatch):
    manifest = tmp_path/'manifest.json';manifest.write_text('{}')
    path = tmp_path/'old.pt'
    model = ChartStreamTransformer(StreamConfig(55,d_model=16,nhead=2,layers=1,dim_feedforward=32))
    checkpoint.save(path,model,None,epoch=1,args={},manifest=manifest)
    status, inspected = webui_api('/api/checkpoint/inspect',{'checkpoint_path':str(path),'mode':'resume'})
    assert status == 200 and not inspected['can_resume'] and inspected['can_init']
    monkeypatch.setattr(webui,'_run_subprocess_task',lambda *_a,**_kw:pytest.fail('Started incompatible resume'))
    status, reply = webui_api('/api/chart-stream/train',{
        'model_family':'chart_stream_v1','manifest':str(manifest),'mode':'resume','init_checkpoint':str(path)})
    assert status == 400 and 'finetune' in reply['error']
