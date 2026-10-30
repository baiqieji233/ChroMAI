import torch
import pytest
from maichart.composer.stream_sampling import delta_sampling_logits


@pytest.mark.parametrize('count',[2,8,100])
def test_temperature_preserves_continue_mass_independent_of_alternative_count(count):
    scores=torch.log(torch.tensor([.8]+[.2/count]*count))
    actual=delta_sampling_logits(scores,.65,0).softmax(-1)
    expected=torch.tensor([.8,.2]).log().div(.65).softmax(-1)[0]
    torch.testing.assert_close(actual[0],expected)
    torch.testing.assert_close(actual.sum(),torch.tensor(1.))
    torch.testing.assert_close(delta_sampling_logits(scores,1.,0).softmax(-1),scores.softmax(-1))


def test_blocked_copy_and_deterministic_sampling_follow_legal_scores():
    scores=torch.tensor([-torch.inf,1.,2.])
    torch.testing.assert_close(delta_sampling_logits(scores,.65,0),scores/.65)
    torch.testing.assert_close(delta_sampling_logits(scores,0.,1),scores)
    single=torch.tensor([-torch.inf,0.,-torch.inf])
    assert delta_sampling_logits(single,.5,1).argmax()==1
