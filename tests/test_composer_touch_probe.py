from __future__ import annotations

import importlib.util
from pathlib import Path
import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / 'tools' / 'probe_composer_touch.py'
spec = importlib.util.spec_from_file_location('composer_touch_probe', PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class Config:
    def __init__(self, name='same'): self.name = name
    def to_dict(self): return {'name': self.name}


class Model:
    def __init__(self, name='same'): self.config = Config(name)
    def action_slot(self, context, slot, selected_ids, **kwargs): return context


def test_catalog_and_configuration_are_required_to_match():
    catalog = ['A1', '1']
    assert probe.require_compatible([(Model(), {'atom_catalog': catalog}),
                                    (Model(), {'atom_catalog': list(catalog)})]) == catalog
    with pytest.raises(ValueError, match='catalogs differ'):
        probe.require_compatible([(Model(), {'atom_catalog': catalog}),
                                  (Model(), {'atom_catalog': list(reversed(catalog))})])
    with pytest.raises(ValueError, match='configurations differ'):
        probe.require_compatible([(Model(), {'atom_catalog': catalog}),
                                  (Model('different'), {'atom_catalog': catalog})])


def test_probability_pooling_masks_padding_and_guidance():
    main = torch.tensor([0., 3., 0.])
    audio = torch.tensor([0., -3., 0.])
    output = {'action_context': main.repeat(2, 1), 'audio_action_context': audio.repeat(2, 1),
              'action_logits': main.repeat(2, 3, 1), 'audio_action_logits': audio.repeat(2, 3, 1)}
    batch = {'action_target': torch.tensor([[1, 0, 0], [0, 0, 0]]),
             'sample_weight': torch.tensor([1., 2.])}
    mask = probe.touch_mask(['A1', '1'])
    total = {}
    probe.combine(total, probe.accumulate(output, batch, mask, Model()))
    result = probe.finalize(total)
    first, later = result['first'], result['later']
    assert first['weight_sum'] == 3 and first['valid_decisions'] == 2
    assert later['weight_sum'] == 1 and later['valid_decisions'] == 1
    assert first['main_mean_probability'] == pytest.approx(main.softmax(-1)[1].item())
    assert first['audio_mean_probability'] == pytest.approx(audio.softmax(-1)[1].item())
    assert first['audio_mean_probability'] < first['guided_mean_probability'] < first['main_mean_probability']
    assert first['guidance_changed_valid_decisions'] == 2
    assert 'guided_mean_probability' not in later
