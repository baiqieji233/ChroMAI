import torch
from maichart.composer.song_context import SongContext


def test_song_context_global_and_padding():
    torch.manual_seed(1)
    model = SongContext(3).eval()
    audio = torch.randn(1, 12, 104)
    clock = torch.arange(12.).view(1, 12, 1).expand(-1, -1, 2)
    padding = torch.zeros(1, 12, dtype=torch.bool)
    baseline = model.encode(audio, padding, clock)
    changed = audio.clone()
    changed[:, -1] += 4
    assert not torch.allclose(baseline[:, 0], model.encode(changed, padding, clock)[:, 0])
    padding[:, -1] = True
    assert torch.allclose(model.encode(audio, padding, clock), model.encode(changed, padding, clock))


def test_unique_encoding_matches_repeated_outputs_and_gradients():
    torch.manual_seed(2)
    model = SongContext(3).eval()
    audio = torch.randn(2, 8, 104)
    clock = torch.randn(2, 8, 2)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    inverse = torch.tensor([0, 1, 0])
    unique = model.encode(audio, mask, clock)[inverse]
    repeated = model.encode(audio[inverse], mask[inverse], clock[inverse])
    assert torch.allclose(unique, repeated, atol=1e-6)
    p = model.input.weight
    a = torch.autograd.grad(unique.square().sum(), p, retain_graph=True)[0]
    b = torch.autograd.grad(repeated.square().sum(), p)[0]
    assert torch.allclose(a, b, atol=2e-5, rtol=2e-5)


def test_whole_song_summary_resolution_does_not_depend_on_batch_padding():
    torch.manual_seed(23)
    model = SongContext(3).eval()
    for length in (50, 300):
        audio = torch.randn(1, length, 104)
        clock = torch.arange(length).float()[None, :, None].expand(-1, -1, 2)
        exact = model.encode(audio, torch.zeros(1,length,dtype=torch.bool), clock)
        padded = torch.nn.functional.pad(audio, (0,0,0,520-length))
        padded_clock = torch.nn.functional.pad(clock, (0,0,0,520-length))
        mask = torch.arange(520)[None] >= length
        result = model.encode(padded, mask, padded_clock)
        torch.testing.assert_close(exact, result[:,:length], atol=2e-6, rtol=2e-5)


def test_paired_history_mask_and_audio_only_isolation():
    from maichart.composer.quality_context import QualityContext
    torch.manual_seed(3)
    model = QualityContext(32, 3, paired_v3=True).eval()
    # Make the zero-initialized residual observable in this test.
    torch.nn.init.normal_(model.output.weight)
    context = torch.randn(1, 2, 32)
    condition = torch.randn(1, 3)
    inputs = {
        'quality_phrase_indices': torch.zeros(1, 2, dtype=torch.long),
        'quality_candidate_offsets': torch.tensor([[0.25, 0.5]]),
        'quality_history_audio': torch.randn(1, 3, 104),
        'quality_history_clock': torch.randn(1, 3, 2),
        'quality_lifecycle': torch.randn(1, 2, 88),
    }
    kwargs = dict(anchor_embeddings=None, song_state=torch.randn(1, 2, 96),
                  phrase_prediction=torch.rand(1, 1, 9), history_memory=torch.randn(1, 3, 32),
                  history_padding=torch.tensor([[False, True, True]]), history_song=torch.randn(1, 3, 96),
                  valid_mask=torch.ones(1, 2, dtype=torch.bool))
    original = model(context, condition, inputs, **kwargs)
    changed_inputs = dict(inputs, quality_history_audio=inputs['quality_history_audio'].clone())
    changed_inputs['quality_history_audio'][:, 1:] += 100
    changed_memory = kwargs['history_memory'].clone()
    changed_memory[:, 1:] += 100
    padded = model(context, condition, changed_inputs, **dict(kwargs, history_memory=changed_memory))
    assert torch.allclose(original['quality_residual'], padded['quality_residual'], atol=1e-6)
    changed_inputs['quality_history_audio'][:, 0] += 3
    changed = model(context, condition, changed_inputs, **kwargs)
    assert not torch.allclose(original['quality_residual'], changed['quality_residual'])
    assert torch.allclose(original['quality_audio_residual'], changed['quality_audio_residual'])
    hidden = model(context, condition, changed_inputs, **dict(kwargs, symbolic_keep=torch.zeros(1, 1)))
    assert torch.allclose(hidden['quality_residual'], hidden['quality_audio_residual'])


def test_legacy_attention_remains_in_pointer_but_not_v3_action():
    from maichart.composer.grounded_model import _action_history_context
    torch.manual_seed(4)
    other = torch.randn(2, 3, 4)
    attended = torch.randn_like(other)
    keep = torch.tensor([[1.], [0.]])
    pointer = (other + attended) * keep[:, None]
    v3 = _action_history_context(pointer, attended, keep, True)
    torch.testing.assert_close(v3, other * keep[:, None])
    torch.testing.assert_close(_action_history_context(pointer, attended, keep, False), pointer)


def test_long_song_attention_uses_all_unmasked_summary_regions():
    torch.manual_seed(5)
    model = SongContext(3).eval()
    audio = torch.randn(1, 300, 104)
    clock = torch.arange(300.).view(1, 300, 1).expand(-1, -1, 2)
    padding = torch.zeros(1, 300, dtype=torch.bool)
    before = model.encode(audio, padding, clock)
    changed = audio.clone()
    changed[:, -1] += 5
    assert not torch.allclose(before[:, 0], model.encode(changed, padding, clock)[:, 0])
    padding[:, -1] = True
    torch.testing.assert_close(model.encode(audio, padding, clock), model.encode(changed, padding, clock))


def test_song_heads_include_fine_rhythm_and_empty_width():
    model = SongContext(3, width=96).eval()
    state = model.encode(torch.randn(2, 4, 104), torch.zeros(2, 4, dtype=torch.bool), torch.randn(2, 4, 2))
    plan, phrase = model.heads(state, torch.randn(2, 3))
    assert plan.shape == (2, 4, 6)
    assert phrase.shape == (2, 4, 9)
    other = SongContext(3, width=32)
    empty = other.encode(torch.zeros(2, 0, 104), torch.zeros(2, 0, dtype=torch.bool), torch.zeros(2, 0, 2))
    assert empty.shape == (2, 0, 32)


def test_current_fine_audio_changes_history_read():
    from maichart.composer.quality_context import QualityContext
    torch.manual_seed(9)
    model = QualityContext(32, 3, paired_v3=True).eval()
    torch.nn.init.normal_(model.output.weight)
    context = torch.randn(1, 2, 32)
    inputs = {'quality_phrase_indices': torch.zeros(1, 2, dtype=torch.long),
              'quality_candidate_offsets': torch.tensor([[.25, .5]]),
              'quality_history_audio': torch.randn(1, 3, 104),
              'quality_history_clock': torch.randn(1, 3, 2),
              'quality_lifecycle': torch.zeros(1, 2, 88)}
    kwargs = dict(anchor_embeddings=None, song_state=torch.randn(1, 2, 96),
                  phrase_prediction=torch.rand(1, 1, 9), history_memory=torch.randn(1, 3, 32),
                  history_padding=torch.zeros(1, 3, dtype=torch.bool), history_song=torch.randn(1, 3, 96))
    condition = torch.randn(1, 3)
    before = model(context, condition, inputs, **kwargs)
    after = model(context + torch.randn_like(context), condition, inputs, **kwargs)
    assert not torch.allclose(before['quality_residual'], after['quality_residual'])


def test_dropped_tail_does_not_move_committed_beat_reference():
    from maichart.composer.quality_context import QualityContext
    torch.manual_seed(12)
    model = QualityContext(32, 3, paired_v3=True).eval()
    torch.nn.init.normal_(model.output.weight)
    inputs = {'quality_phrase_indices': torch.zeros(1, 2, dtype=torch.long),
              'quality_candidate_offsets': torch.tensor([[.25, .5]]),
              'quality_history_audio': torch.randn(1, 3, 104),
              'quality_history_clock': torch.tensor([[[1., 1.], [2., 2.], [3., 3.]]]),
              'quality_lifecycle': torch.zeros(1, 2, 88)}
    kwargs = dict(anchor_embeddings=None, song_state=torch.randn(1, 2, 96),
                  phrase_prediction=torch.rand(1, 1, 9), history_memory=torch.randn(1, 3, 32),
                  history_padding=torch.tensor([[False, False, True]]), history_song=torch.randn(1, 3, 96),
                  history_reference_beat=torch.tensor([3.]))
    context, condition = torch.randn(1, 2, 32), torch.randn(1, 3)
    before = model(context, condition, inputs, **kwargs)
    changed = dict(inputs, quality_history_clock=inputs['quality_history_clock'].clone())
    changed['quality_history_clock'][:, 2] = 1000
    after = model(context, condition, changed, **kwargs)
    torch.testing.assert_close(before['quality_residual'], after['quality_residual'])
    shifted = model(context, condition, inputs, **dict(kwargs, history_reference_beat=torch.tensor([2.])))
    assert not torch.allclose(before['quality_residual'], shifted['quality_residual'])
