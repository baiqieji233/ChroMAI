"""Condition vectors for controllable chart generation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any

import torch

RADAR_KEYS = ("note_density", "peak", "sweep", "slide_tricky", "slide_sequence", "jack", "chord")
STYLE_KEYS = ("tap_bias", "slide_bias", "hold_bias", "touch_bias", "chord_bias", "melody_following", "pattern_reuse", "directionality")
CONDITION_KEYS = ("level",) + RADAR_KEYS + STYLE_KEYS
GENERATION_DEFAULTS = {
    "level": 13.3, "note_density": 0.305, "peak": 0.375, "sweep": 0.109,
    "slide_tricky": 0.114, "slide_sequence": 0.066, "jack": 0.188, "chord": 0.280,
    "tap_bias": 0.720, "slide_bias": 0.140, "hold_bias": 0.077, "touch_bias": 0.039,
    "chord_bias": 0.280, "melody_following": 0.90, "pattern_reuse": 0.028,
    "directionality": 0.043,
}
GENERATION_SAMPLING_DEFAULTS = {
    "temperature": 0.65, "top_p": 0.9, "repetition_penalty": 1.05,
    # Decode budget is distinct from the training target cap.  Keep this in
    # sync with the generation panel and CLI; event_ctx checkpoints may still
    # benefit from an explicitly larger per-window budget.
    "repetition_window": 56, "max_tokens": 768, "candidates": 1,
    "guidance_scale": 1.0, "sample": True, "tempo_mode": "constant", "division": 16,
    "intensity_follow": True, "intensity_follow_span": 0.5, "tail_damping": False,
    "intensity_mode": "auto",
    "tail_damping_windows": 4, "tail_damping_strength": 0.6, "audio_bounds_trim": True,
    "audio_bounds_lead": 0.0, "silence_threshold": 0.15, "silence_window_share": 0.05,
}

def default_condition_values() -> dict[str, float]:
    return dict(GENERATION_DEFAULTS)

def scale_intensity(condition: "GenerationCondition", factor: float, *, keys=None) -> "GenerationCondition":
    radar = dict(condition.radar)
    for key in ("note_density", "peak") if keys is None else keys:
        if key in radar:
            radar[key] = _clip(radar[key] * float(factor))
    return replace(condition, radar=radar)

@dataclass(slots=True)
class GenerationCondition:
    level: float = 13.0
    radar: dict[str, float] = field(default_factory=dict)
    style: dict[str, float] = field(default_factory=dict)
    difficulty: int = 5
    def __post_init__(self):
        self.level = max(1.0, min(15.0, float(self.level)))
        self.radar = _complete_values(self.radar, RADAR_KEYS, 0.5)
        self.style = _complete_values(self.style, STYLE_KEYS, 0.5)
        self.difficulty = int(self.difficulty)
    def to_vector(self, *, device=None):
        return torch.tensor([self.level / 15.0] + [self.radar[k] for k in RADAR_KEYS] + [self.style[k] for k in STYLE_KEYS], dtype=torch.float32, device=device)
    def to_dict(self):
        return {"level": self.level, "difficulty": self.difficulty, "radar": dict(self.radar), "style": dict(self.style)}
    @classmethod
    def from_dict(cls, data):
        unwrap = lambda values: {key: (value.get("target", value.get("value", 0.5)) if isinstance(value, dict) else value) for key, value in dict(values or {}).items()}
        return cls(level=data.get("level", GENERATION_DEFAULTS["level"]), difficulty=data.get("difficulty", 5), radar=unwrap(data.get("radar")), style=unwrap(data.get("style")))
    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

def derive_radar_from_targets(targets, note_type_vocab):
    presence = targets["note_presence"].detach().float().reshape(-1); buttons = targets["buttons"].detach().float(); note_type = targets["note_type"].detach().long().reshape(-1)
    active = presence > .5; n = max(1, int(active.sum().item())); density = targets["density"].detach().float().reshape(-1)
    ratios = {name: float((note_type == i).float().sum()) / n for i, name in enumerate(note_type_vocab) if name != "none"}
    chord = ((buttons.sum(-1) >= 2) & active).float().mean().item()
    jack = _jack_mask(buttons, active).float().mean().item()
    slide = ratios.get("slide", 0.0)
    return {"note_density": _clip(float(density.mean()) / 2), "peak": _clip(float(density.max()) / 8 if density.numel() else 0), "sweep": _clip(slide * 1.25), "slide_tricky": _clip(slide), "slide_sequence": _clip(slide * 1.1), "jack": _clip(jack * 2), "chord": _clip(chord * 1.5)}

def derive_style_from_targets(targets, note_type_vocab):
    presence = targets["note_presence"].detach().float().reshape(-1); buttons = targets["buttons"].detach().float(); note_type = targets["note_type"].detach().long().reshape(-1); active = presence > .5; n = max(1, int(active.sum().item()))
    ratios = {name: float((note_type == i).float().sum()) / n for i, name in enumerate(note_type_vocab) if name != "none"}
    slide, hold, touch = ratios.get("slide", 0), ratios.get("hold", 0) + ratios.get("touch_hold", 0), ratios.get("touch", 0) + ratios.get("touch_hold", 0)
    chord = float(((buttons.sum(-1) >= 2) & active).float().mean())
    return {"tap_bias": _clip(ratios.get("tap", 0) + ratios.get("break", 0)), "slide_bias": _clip(slide), "hold_bias": _clip(hold), "touch_bias": _clip(touch), "chord_bias": _clip(chord), "melody_following": 0.5, "pattern_reuse": 0.5, "directionality": _clip(slide + chord * .25)}

def condition_from_targets(targets, note_type_vocab, *, level=None):
    return GenerationCondition(level=13.0 if level is None else level, radar=derive_radar_from_targets(targets, note_type_vocab), style=derive_style_from_targets(targets, note_type_vocab))

def _jack_mask(buttons, active):
    if buttons.size(0) < 2: return torch.zeros_like(active)
    single = buttons.sum(-1) == 1; same = (buttons[1:] * buttons[:-1]).sum(-1) > .5; result = torch.zeros_like(active); result[1:] = active[1:] & active[:-1] & single[1:] & single[:-1] & same; return result

def _complete_values(values, keys, default): return {key: _clip(values.get(key, default)) for key in keys}
def _clip(value):
    try: return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError): return 0.5
