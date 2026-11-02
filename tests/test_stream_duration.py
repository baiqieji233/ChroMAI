import torch
import pytest

from maichart.composer.stream_duration import _duration_coefficients, _read_intervals, DurationAudioScorer


def test_native_duration_fit_retains_slide_wait_and_hold_beats():
    assert _duration_coefficients('1h[4:1]') == [0., 0., 1., 0.]
    assert _duration_coefficients('1-3[4:1]') == [1., 0., 2., 0.]
    assert _duration_coefficients('1') is None


def test_interval_reads_do_not_cross_padded_song_boundaries():
    cache = dict(seconds=torch.tensor([[0., 1., 2.], [0., 1., 0.]]),
        mask=torch.tensor([[True, True, True], [True, True, False]]),
        local=torch.tensor([[[0.], [1.], [2.]], [[10.], [11.], [999.]]]))
    queries=torch.tensor([[[.5, 3.]], [[.5, 3.]], [[1., 1.5]]])
    actual=_read_intervals(cache, queries, torch.tensor([0, 1, 0]))
    expected=torch.tensor([[[[.5], [2.]]], [[[10.5], [11.]]], [[[1.], [1.5]]]])
    torch.testing.assert_close(actual, expected)
    from maichart.composer.stream_duration import prepare_interval_clock
    prepared=dict(cache,_duration_clock=prepare_interval_clock(cache))
    torch.testing.assert_close(_read_intervals(prepared,queries,torch.tensor([0,1,0])),actual,rtol=0,atol=0)


@pytest.mark.parametrize('atom', ['1h[#0.12345]', '1h[180#4:1]',
    '1-3[0.12345##0.6789]', '1-3[180#4:1]', '1w5[4:1]'])
def test_explicit_seconds_waits_and_tempo_fit_match_native(atom):
    from maichart.generation.muri.parser import parse_markers
    coefficients = _duration_coefficients(atom)
    assert coefficients is not None, atom
    markers, errors, _ = parse_markers(f'(137){{4}}{atom},E')
    assert not errors
    marker = next(m for m in markers if m['type'] in ('hold', 'slide', 'wifi'))
    begin = marker['slide_trace_second'] if marker['type'] != 'hold' else marker['second']
    end = marker['end_second']
    torch.testing.assert_close(torch.tensor([coefficients[0]*60/137+coefficients[1],
        coefficients[2]*60/137+coefficients[3]]), torch.tensor([begin, end]))


def _scorer_fixture():
    class Codec:
        shape_catalog = [(('literal', 0),), (('hold', 0),)]
        detail_catalog = [[], [['h[#0.12345]', 'bh[#0.12345]', 'h[#0.6789]', 'not_valid']]]
    scorer = DurationAudioScorer(32, Codec())
    seconds = torch.linspace(0, 4, 101)[None]
    keys = (seconds[..., None] * torch.linspace(.1, 1., 32)).requires_grad_()
    cache = dict(seconds=seconds, mask=torch.ones_like(seconds, dtype=torch.bool), delta_keys=keys)
    context = dict(cache=cache, inverse=torch.zeros(2, dtype=torch.long),
        second=torch.tensor([.31, .97]), seconds_per_beat=torch.tensor([.5, .6]),
        attention=torch.ones(2, 32), valid=torch.ones(2, dtype=torch.bool), end_second=torch.full((2,), 4.))
    return scorer, context, keys


def test_duration_candidate_dedup_backward_and_bf16_clock_precision():
    scorer, context, keys = _scorer_fixture()
    table = getattr(scorer, scorer._tables[1, 0])
    assert len(table) == 3  # Two actual intervals and one unsupported sentinel.
    logits = torch.randn(2, 4, requires_grad=True)
    assert torch.equal(scorer(None, logits, 1, 0, context), logits)
    with torch.no_grad(): scorer.score.weight.fill_(.01)
    fp32 = scorer(None, torch.zeros_like(logits), 1, 0, context)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        bf16 = scorer(None, torch.zeros_like(logits, dtype=torch.bfloat16), 1, 0, context)
    torch.testing.assert_close(fp32, bf16.float(), atol=1e-3, rtol=.01)
    assert torch.equal(fp32[:, 0], fp32[:, 1])
    assert torch.equal(fp32[:, 3], torch.zeros(2))
    fp32.sum().backward()
    assert torch.isfinite(scorer.score.weight.grad).all()
    assert scorer.score.weight.grad.abs().sum() > 0
    assert torch.isfinite(keys.grad).all() and keys.grad.abs().sum() > 0
    assert scorer(None, logits, 0, 0, context) is logits
    assert scorer(None, logits, 1, 0, None) is logits


def test_log_time_policy_preserves_interval_evidence_with_smooth_scale():
    scorer, context, _ = _scorer_fixture()
    scorer.feature_policy = 'log_time_v2'
    with torch.no_grad():
        scorer.score.weight.zero_()
        scorer.score.weight[0, -1] = 1.
    actual = scorer(None, torch.zeros(2, 4), 1, 0, context)
    expected = torch.log1p(torch.tensor([.12345, .12345, .6789, 0.])).expand(2, -1)
    torch.testing.assert_close(actual, expected)
    assert actual[0, 2] > actual[0, 0] > 0


def test_real_regular_details_phase_marginalization_uses_same_intervals():
    from maichart.composer.event_set_codec import EventSetCodec
    from maichart.composer.stream_model import StreamComposer, StreamComposerConfig
    from maichart.composer.stream_duration import select_duration_context
    codec = EventSetCodec.fit(['1h[4:1]', '1h[8:1]', '1bh[4:1]'])
    model = StreamComposer(StreamComposerConfig(d_model=16, nhead=4, layers=1,
        dim_feedforward=32, dropout=0., duration_audio_policy='interval_v1'), codec)
    _, duration, _ = _scorer_fixture()
    encoded = codec.encode('1h[4:1]')
    ctx = torch.randn(2, 16, requires_grad=True)
    phases = torch.arange(8).repeat(2)
    rows = torch.arange(2).repeat_interleave(8)
    targets = torch.tensor(encoded['details']).expand(16, -1)
    expanded = select_duration_context(duration, rows)
    scores = model.regular_detail_log_probs(ctx.index_select(0, rows), encoded['shape'],
        phases, targets, duration_context=expanded).reshape(2, 8)
    explicit = torch.stack([model.regular_detail_log_probs(ctx, encoded['shape'],
        torch.full((2,), phase), targets[:2], duration_context=duration) for phase in range(8)], -1)
    torch.testing.assert_close(scores, explicit)
    loss = -(scores-torch.log(torch.tensor(8.))).logsumexp(-1).mean()
    loss.backward()
    assert torch.isfinite(ctx.grad).all()
    assert model.duration_audio.score.weight.grad.abs().sum() > 0
