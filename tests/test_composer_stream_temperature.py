import pytest

from maichart.composer.stream_generate import parser,resolve_temperatures


@pytest.mark.parametrize('flags,expected',[
    ([],(.65,.65,.65)),
    (['--temperature','0'],(0.,0.,0.)),
    (['--temperature','.3'],(.3,.3,.3)),
    (['--temperature','.3','--pointer-temperature','0'],(.3,0.,.3)),
    (['--temperature','0','--action-temperature','.7'],(0.,0.,.7)),
])
def test_cli_stage_temperatures_inherit_total_unless_explicit(flags,expected):
    args=resolve_temperatures(parser().parse_args(['--audio','a','--checkpoint','c','--output','o',*flags]))
    assert (args.temperature,args.pointer_temperature,args.action_temperature)==expected


@pytest.mark.parametrize('flags',[['--temperature','-1'],['--pointer-temperature','nan'],['--action-temperature','inf']])
def test_invalid_temperatures_fail_before_loading_audio_or_model(flags):
    with pytest.raises(ValueError):
        resolve_temperatures(parser().parse_args(['--audio','a','--checkpoint','c','--output','o',*flags]))
