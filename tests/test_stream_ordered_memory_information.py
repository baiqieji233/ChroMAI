from fractions import Fraction as F
from types import SimpleNamespace
import torch

from maichart.composer.stream_phrase import describe_phrase
from maichart.composer.stream_phrase_model import CausalPhraseContext
from test_stream_phrase_model import ordered_fixture


def test_ordered_memory_distinguishes_real_sequences_with_same_coarse_descriptor():
    def events(keys):
        return [SimpleNamespace(kind=2,start=F(i,4),end=F(i+1,4),text=str(key))
                for i,key in enumerate(keys)]
    # Same set of heads, same gaps and same three clockwise quarter-circle
    # moves, all within the same coarse bin; start/end placement still differs.
    first,second=events([1,3,5,7]),events([3,5,7,1])
    torch.testing.assert_close(describe_phrase(first,0,8),describe_phrase(second,0,8),atol=1e-6,rtol=0)
    model,batch,cache=ordered_fixture()
    context=CausalPhraseContext(16,3,ordered=True).eval()
    batch['phrase_sequence_atoms'][0,:4,0]=torch.tensor([1,3,5,7])
    one=context.ordered.encode_sequences(model,batch)
    other=dict(batch,phrase_sequence_atoms=batch['phrase_sequence_atoms'].clone())
    other['phrase_sequence_atoms'][0,:4,0]=torch.tensor([3,5,7,1])
    two=context.ordered.encode_sequences(model,other)
    assert not torch.allclose(one,two)
