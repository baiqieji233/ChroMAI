"""CPU regressions for bounded attention scores and legacy checkpoint behavior."""
import math
import json
import random
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
from maichart.composer_v2.model import Attention, ComposerV2, ComposerV2Config
from maichart.composer_v2.train import _validate_checkpoint, rng_state, restore_rng


class AttentionStability(unittest.TestCase):
    def test_projected_scale_cannot_amplify_attention_scores(self):
        torch.manual_seed(12)
        module = Attention(128, 2, local=True, normalization='qk_rms_v1')
        query = torch.randn(1, 2, 7, 64)
        key = torch.randn(1, 2, 19, 64)
        value = torch.randn(1, 2, 19, 64)
        captured = []
        original = F.scaled_dot_product_attention
        def checked(q, k, v, **kwargs):
            logits = (q @ k.transpose(-1, -2)) / math.sqrt(64)
            self.assertLessEqual(float(logits.detach().abs().max()), 8.00001)
            captured.append(logits.detach())
            return original(q, k, v, **kwargs)
        outputs = []
        for scale in (1., 1000.):
            q = (query * scale).requires_grad_()
            k = (key * scale).requires_grad_()
            with patch.object(F, 'scaled_dot_product_attention', checked):
                out = module(torch.zeros(1, 7, 128), projected_query=q, kv=(k, value),
                    query_beats=torch.arange(7)[None].float(), key_beats=torch.arange(19)[None].float())
            gradients = torch.autograd.grad(out.square().mean(), (q, k, module.past, module.future))
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            outputs.append(out.detach())
        torch.testing.assert_close(outputs[0], outputs[1], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(captured[0], captured[1], atol=1e-5, rtol=1e-4)

    def test_incremental_and_full_attention_agree(self):
        torch.manual_seed(7)
        module = Attention(32, 4, normalization='qk_rms_v1').eval()
        value = torch.randn(1, 9, 32)
        with torch.no_grad():
            full = module(value, value, causal=True)
            cached = None
            outputs = []
            for index in range(value.size(1)):
                query = value[:, index:index+1]
                keys = module.memory(query)
                cached = keys if cached is None else tuple(torch.cat((a, b), -2) for a, b in zip(cached, keys))
                outputs.append(module(query, kv=cached, causal=True, past_length=index))
            torch.testing.assert_close(full, torch.cat(outputs, 1), atol=1e-6, rtol=1e-5)
            module.train(); module.pack_projections = True
            torch.testing.assert_close(full, module(value, value, causal=True), atol=1e-6, rtol=1e-5)

    def test_legacy_configuration_preserves_unnormalized_attention(self):
        fresh = ComposerV2Config()
        self.assertEqual(fresh.attention_normalization, 'qk_rms_v1')
        legacy = fresh.to_dict(); legacy.pop('attention_normalization')
        self.assertEqual(ComposerV2Config.from_dict(legacy).attention_normalization, 'off')
        self.assertEqual(ComposerV2Config.from_dict(fresh.to_dict()), fresh)
        with self.assertRaises(ValueError):
            ComposerV2Config(attention_normalization='unknown')


class ValidationAccounting(unittest.TestCase):
    class Model(torch.nn.Module):
        loss = ComposerV2.loss
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(condition_policy='level_only_v1')
            self.register_buffer('time_tokens', torch.tensor([False, True, True, False]))
            self.register_buffer('phase_targets', torch.tensor([False, False, True, False]))
        def forward(self, batch):
            random.random(); np.random.random(); torch.rand(1)
            return {'logits': batch['logits']}

    def test_fixed_validation_weights_and_rng_are_preserved(self):
        # A sub-unit phase budget catches double weighting/clamping errors.
        batches = [dict(labels=torch.tensor([[1, 2, 3, -100]]),
                        logits=torch.tensor([[[1., 2., 0., 0.], [0., 1., 3., 0.],
                                               [0., 2., 0., 1.], [0., 0., 0., 0.]]]),
                        token_weights=torch.tensor([[1., .1, 2., 100.]]),
                        sample_weight=torch.tensor([.5])),
                   dict(labels=torch.tensor([[2, 1]]),
                        logits=torch.tensor([[[0., 0., 2., 1.], [1., 3., 0., 0.]]]),
                        token_weights=torch.tensor([[.2, 1.]]), sample_weight=torch.tensor([2.]))]
        model = self.Model().train()
        totals = {name: [0., 0.] for name in ('all', 'time', 'phase', 'nonphase')}
        for batch in batches:
            labels = batch['labels']
            cost = F.cross_entropy(batch['logits'].transpose(1, 2), labels, reduction='none', ignore_index=-100)
            weights = batch['token_weights'] * batch['sample_weight'][:, None] * (labels != -100)
            for name, mask in [('all', labels != -100), ('time', (labels == 1) | (labels == 2)),
                               ('phase', labels == 2), ('nonphase', labels == 1)]:
                totals[name][0] += float((cost * weights * mask).sum())
                totals[name][1] += float((weights * mask).sum())
        saved = rng_state()
        expected_draw = (random.random(), float(np.random.random()), torch.rand(3))
        restore_rng(saved)
        pending = Future(); pending.set_result(None)
        score = _validate_checkpoint(model, batches, torch.device('cpu'), 'fp32', 0, pending=[pending])
        actual_draw = (random.random(), float(np.random.random()), torch.rand(3))
        self.assertEqual(expected_draw[:2], actual_draw[:2])
        torch.testing.assert_close(expected_draw[2], actual_draw[2], rtol=0, atol=0)
        self.assertTrue(model.training)
        for name, actual in [('all', score), ('time', model.validation_time_loss),
                             ('phase', model.validation_phase_loss), ('nonphase', model.validation_nonphase_time_loss)]:
            num, mass = totals[name]
            self.assertAlmostEqual(actual, num / mass, places=6)

    def test_validation_failure_restores_mode_and_rng(self):
        model = self.Model().eval()
        saved = rng_state()
        expected = torch.rand(3); restore_rng(saved)
        def fail(batch):
            torch.rand(10)
            raise RuntimeError('validation failed')
        batch = dict(labels=torch.tensor([[1]]))
        with patch.object(model, 'forward', fail), self.assertRaisesRegex(RuntimeError, 'validation failed'):
            _validate_checkpoint(model, [batch], torch.device('cpu'), 'fp32', 0)
        self.assertFalse(model.training)
        torch.testing.assert_close(expected, torch.rand(3), rtol=0, atol=0)

    def test_prefetch_failure_is_propagated_before_validation(self):
        pending = Future(); pending.set_exception(RuntimeError('data fetch failed'))
        model = self.Model().train()
        with self.assertRaisesRegex(RuntimeError, 'data fetch failed'):
            _validate_checkpoint(model, [], torch.device('cpu'), 'fp32', 0, pending=[pending])
        self.assertTrue(model.training)


class ValidationProgress(unittest.TestCase):
    def test_server_retains_validation_history_after_training_resumes(self):
        from maichart.webui.server import TaskState, _update_composer_progress
        task = TaskState('validation-test', 'composer_v2_train')
        for step in range(205):
            row = dict(stage='checkpoint_validation', epoch=1, optimizer_steps=step,
                       validation_loss=.8, validation_nonphase_time_loss=.6)
            _update_composer_progress(task, 'composer_v2_progress '+json.dumps(row))
        _update_composer_progress(task, 'composer_v2_progress '+json.dumps(row))
        _update_composer_progress(task, 'composer_v2_progress '+json.dumps(dict(
            stage='training', epoch=1, optimizer_steps=206, batch=824, total=1000)))
        history = task.result['composer_v2_validation_history']
        self.assertEqual(len(history), 200)
        self.assertEqual(history[0]['optimizer_steps'], 5)
        self.assertEqual(history[-1]['optimizer_steps'], 204)
        self.assertEqual(task.result['composer_v2_checkpoint_validation'], row)
        self.assertEqual(task.result['composer_v2_stage'], 'training')


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
