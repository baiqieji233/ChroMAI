"""Small, explicit metadata for the training monitor; no quality estimates."""
from dataclasses import asdict
import json
from pathlib import Path


def training_model_summary(model, planner):
    layout = sum(parameter.numel() for parameter in model.parameters())
    blueprint = sum(parameter.numel() for parameter in planner.parameters())
    return {
        'schema': 'song_form_training_model_summary_v1',
        'parameter_count': layout + blueprint,
        'layout_parameter_count': layout,
        'planner_parameter_count': blueprint,
        'model_config': asdict(model.config),
        'blueprint_config': asdict(planner.config),
        'parameter_scope': 'unique_parameters_per_module_not_optimizer_state',
    }


def write_training_model_summary(directory, summary):
    path = Path(directory) / 'training_model_summary.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)
    return path
