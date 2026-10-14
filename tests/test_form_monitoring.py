import json
from dataclasses import dataclass
from types import SimpleNamespace

from maichart.form.monitoring import training_model_summary, write_training_model_summary
from maichart.webui.form_api import update_form_progress


@dataclass
class Config:
    d_model: int = 512


def test_summary_records_parameter_elements_without_counting_optimizer_storage(tmp_path):
    module=lambda count: SimpleNamespace(config=Config(),parameters=lambda:[SimpleNamespace(numel=lambda:count)])
    summary=training_model_summary(module(45),module(3))
    assert summary['parameter_count']==48
    assert summary['layout_parameter_count']==45 and summary['planner_parameter_count']==3
    path=write_training_model_summary(tmp_path,summary)
    assert json.loads(path.read_text(encoding='utf-8'))==summary
    task=SimpleNamespace(kind='form_train',result={})
    update_form_progress(task,'form_model_summary '+json.dumps(summary))
    assert task.result['model_summary']==summary
