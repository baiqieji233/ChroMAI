from types import SimpleNamespace

from maichart.composer.model import ComposerConfig, resolve_composer_policy
from maichart.composer.train import parser
from maichart.composer.training_defaults import TRAINING_DEFAULTS
from maichart.webui import composer_api


def test_omitted_api_fields_match_cli_training_defaults(tmp_path):
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    calls=[]
    runtime=SimpleNamespace(_create_task=lambda kind:SimpleNamespace(task_id='defaults',result={}),
        _run_subprocess_task=lambda task,args,**kwargs:calls.append(args))
    composer_api.start_train(SimpleNamespace(_json_response=lambda result:None),
        dict(manifest=str(manifest),cache_dir=str(tmp_path/'cache'),checkpoint_dir=str(tmp_path/'next')),
        runtime=runtime)
    submitted=parser().parse_args(calls[0][2:])
    cli=parser().parse_args(['--manifest',str(manifest),'--checkpoint-dir',str(tmp_path/'cli')])
    for key,value in TRAINING_DEFAULTS.items():
        assert getattr(cli,key)==value
        assert getattr(submitted,key)==value
    assert submitted.motif_memory_policy=='audio_recurrence_archive_v1'


def test_fresh_motif_recipe_preserves_old_and_non_grounded_interpretation():
    assert resolve_composer_policy('motif_memory_policy')=='audio_recurrence_archive_v1'
    assert resolve_composer_policy('motif_memory_policy',architecture_profile='music_context_v2')=='off'
    assert resolve_composer_policy('motif_memory_policy',parent_config={'architecture_profile':'audio_grounded_v3'})=='off'
    assert ComposerConfig(atom_vocab_size=2).motif_memory_policy=='off'
