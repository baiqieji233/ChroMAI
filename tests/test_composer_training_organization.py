from html.parser import HTMLParser
from pathlib import Path
import pytest
from test_composer_policy_webui import harness,value
from maichart.webui import composer_api as api


class Structure(HTMLParser):
    def __init__(self):super().__init__();self.stack=[];self.nodes={}
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if 'id' in attrs:
            assert attrs['id'] not in self.nodes,attrs['id']
            self.nodes[attrs['id']]=(attrs,list(self.stack))
        if tag not in {'input','img','meta','link','br','hr','source','wbr'}:self.stack.append((tag,attrs))
    def handle_endtag(self,tag):
        for index in range(len(self.stack)-1,-1,-1):
            if self.stack[index][0]==tag:self.stack=self.stack[:index];break


def test_composer_basics_visible_without_global_advanced_and_advanced_collapsed():
    tree=Structure();tree.feed(Path('src/maichart/webui/static/index.html').read_text(encoding='utf-8'))
    for ident in ('composerConfigurationQualityPolicy','composerValidationManifest','composerGenerationEvalSongs'):
        _,ancestors=tree.nodes[ident]
        assert not any('data-advanced' in attrs for _,attrs in ancestors)
        assert not any(tag=='details' and 'open' not in attrs for tag,attrs in ancestors)
    attrs,_=tree.nodes['composerTrainingAdvanced'];assert 'open' not in attrs
    for ident in ('composerPlacementWeight','composerDModel','composerProfileSteps','composerArchiveCheckpoints'):
        _,ancestors=tree.nodes[ident]
        assert any(attrs.get('id')=='composerTrainingAdvanced' for _,attrs in ancestors)


def test_new_evaluation_and_archive_fields_reach_real_cli(harness,tmp_path):
    payload,parent,calls,handler,runtime=harness
    validation=tmp_path/'validation.json';validation.write_text('{}')
    settings=dict(validation_manifest=str(validation),holdout_fraction=0.,validation_max_batches=0,
        validation_batch_size=4,generation_eval_seeds='0,17',archive_checkpoints='off')
    api.start_train(handler,dict(payload,mode='fresh',**settings),runtime=runtime)
    args=calls[0][1]
    for key,want in settings.items():assert value(args,key)==str(want)


@pytest.mark.parametrize('seeds',['','0,','0,-1','1.5,2','hello'])
def test_bad_evaluation_seeds_fail_before_launch(harness,seeds):
    payload,parent,calls,handler,runtime=harness
    with pytest.raises(ValueError,match='随机种子'):
        api.start_train(handler,dict(payload,mode='fresh',generation_eval_seeds=seeds),runtime=runtime)
    assert not calls
