"""Saved automatic recipes preserve explicit overrides and resume contracts."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

APP = (Path(__file__).resolve().parents[1] / "src/maichart/webui/static/app.js").read_text(encoding="utf-8")


def test_automatic_recipe_migration_keeps_resume_and_explicit_run_values():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed for browser settings behavior")
    start = APP.index("function migrateEventSetLearningDefaults(")
    function = APP[start:APP.index("\nfunction restoreSettings", start)]
    fixture = dict(modelFamily="chart_composer_v1", composerArchitectureProfile="event_set_stream_v1",
                   trainingMode="fresh", composerLearningPolicy="auto",
                   eventSetPolicy_plan_kind_policy="unified_first_arrival_v5",
                   composerDraftObjectivePolicy="balanced_mean_v1", composerDurationObjectivePolicy="", batchSize="24",
                   checkpointDir="checkpoints/keep")
    script = function + "\nconst base=" + json.dumps(fixture) + ";\n" + """
const fresh = {...base};
const changed = migrateEventSetLearningDefaults(fresh);
const resume = {...base, trainingMode:'resume'};
const legacy = {...base, composerLearningPolicy:'legacy'};
const alreadyCurrent = {...base, __eventSetLearningDefaults:2};
console.log(JSON.stringify({changed, fresh,
  resumeChanged:migrateEventSetLearningDefaults(resume), resume,
  legacyChanged:migrateEventSetLearningDefaults(legacy),
  currentChanged:migrateEventSetLearningDefaults(alreadyCurrent)}));
"""
    result = json.loads(subprocess.check_output([node, "-e", script], text=True))
    assert result["changed"] is True
    assert result["fresh"]["eventSetPolicy_plan_kind_policy"] == "unified_first_arrival_v5"
    assert result["fresh"]["composerDraftObjectivePolicy"] == "balanced_mean_v1"
    assert result["fresh"]["composerDurationObjectivePolicy"] == "auto"
    assert result["fresh"]["batchSize"] == "24"
    assert result["fresh"]["checkpointDir"] == "checkpoints/keep"
    assert result["resumeChanged"] is False
    assert result["resume"]["eventSetPolicy_plan_kind_policy"] == "unified_first_arrival_v5"
    assert result["legacyChanged"] is False
    assert result["currentChanged"] is False
