import pytest

from maichart.composer.stream_rhythm_metrics import conditioned_rhythm_summary


def test_integer_landing_retains_preceding_fine_gap_and_condition_boundary():
    result = conditioned_rhythm_summary(['0', '1/4', '1/2', '3/4', '1'],
                                       ['run', 'run', 'run', 'run', 'landing'])
    assert result['overall']['gap_at_most_quarter_beat_fraction'] == 1.
    assert result['overall']['phase_denominator_counts'] == {'1': 2, '2': 1, '4': 2}
    assert result['conditions']['landing']['phase_denominator_counts'] == {'1': 1}
    assert result['conditions']['landing']['gap_denominator_counts'] == {'4': 1}
    assert result['conditions']['landing']['mean_gap_beats'] == .25


def test_triplets_and_sparse_strata_are_measured_without_quality_judgment():
    result = conditioned_rhythm_summary(['0', '1/3', '2/3', '4'], ['a', 'b', 'a', 'b'])
    assert result['overall']['gap_denominator_counts'] == {'3': 3}
    assert result['overall']['gap_at_most_half_beat_fraction'] == 2/3
    # 'a' has the real 1/3 interval, not the fictitious gap from 0 to 2/3.
    assert result['conditions']['a']['mean_gap_beats'] == 1/3
    assert result['conditions']['b']['mean_gap_beats'] == 11/6


def test_empty_and_single_onset_have_no_invented_interval():
    assert conditioned_rhythm_summary([])['overall']['mean_gap_beats'] is None
    assert conditioned_rhythm_summary(['1'])['overall']['gaps'] == 0


@pytest.mark.parametrize('beats,labels', [(['0', '0'], None), (['1', '0'], None), (['0'], [])])
def test_rejects_duplicate_reversed_or_misaligned_input(beats, labels):
    with pytest.raises(ValueError):
        conditioned_rhythm_summary(beats, labels)
