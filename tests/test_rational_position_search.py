from fractions import Fraction as F
from types import SimpleNamespace

import torch

from maichart.form.codec import RationalCodec
from maichart.form.rational_generate import PositionGrammar
from maichart.form.rational_position_search import search_position


def test_whole_position_search_keeps_exact_rare_fraction_and_uses_batched_cache():
    codec=RationalCodec()
    desired='37/997'

    class Model:
        config=SimpleNamespace(max_tokens=128)
        batch_sizes=[]
        def decode_step(self, tokens, position, memory, mask, caches=None):
            self.batch_sizes.append(len(tokens))
            previous=(caches[0][0] if caches is not None else torch.zeros((len(tokens),0,1),dtype=torch.long))
            assert previous.shape[1]==position
            present=torch.cat([previous,tokens[...,None]],dim=1)
            logits=torch.full((len(tokens),1,len(codec)),-30.)
            for row in range(len(tokens)):
                prefix=''.join(codec.tokens[int(token)][2:] for token in present[row,:,0])
                wanted=codec.note_id if prefix==desired else codec.ids['P:'+desired[len(prefix)]] if desired.startswith(prefix) else codec.note_id
                logits[row,0,wanted]=30.
            return logits,[(present,present,None,None)]

    model=Model()
    root=torch.full((len(codec),),-30.);root[codec.ids['P:3']]=30.
    value,status=search_position(model,codec,PositionGrammar(-1,4),root,None,
        torch.zeros(1,1,1),torch.zeros(1,1,dtype=torch.bool),consumed=0,width=4,complexity=.2)
    assert F(value)==F(37,997)
    assert any(size>1 for size in model.batch_sizes)
    assert status['expanded_hypotheses']>0


def test_search_compares_full_probability_instead_of_only_the_first_digit():
    codec=RationalCodec()
    class Model:
        config=SimpleNamespace(max_tokens=16)
        def decode_step(self,tokens,position,memory,mask,caches=None):
            logits=torch.full((len(tokens),1,len(codec)),-20.)
            for row,token in enumerate(tokens[:,0]):
                if int(token)==codec.ids['P:1']:
                    logits[row,0,codec.ids['N:1']]=5.
                    logits[row,0,codec.note_id]=0.
                else:
                    logits[row,0,codec.note_id]=5.
            return logits,None
    logits=torch.full((len(codec),),-20.)
    logits[codec.ids['P:1']]=2.;logits[codec.ids['P:2']]=1.9
    result,_=search_position(Model(),codec,PositionGrammar(0,4),logits,None,
        torch.zeros(1,1,1),None,consumed=0,width=2,complexity=0)
    assert result=='2'


def test_compose_commits_selected_beam_fraction_with_a_consistent_causal_cache():
    from maichart.event_timing import EventPoint
    from maichart.form.model import FormTransformer
    from maichart.form.rational_generate import compose_span
    codec=RationalCodec()
    expected=[EventPoint(F(37,997),'1')]
    sequence=codec.encode(expected)
    class Model:
        config=SimpleNamespace(max_tokens=512)
        rewind_decode_cache=staticmethod(FormTransformer.rewind_decode_cache)
        def eval(self): return self
        def encode(self,_): return torch.zeros(1,1,1),None
        def decode_step(self,tokens,position,memory,mask,caches=None):
            previous=caches[0][0] if caches is not None else torch.zeros((len(tokens),0,1),dtype=torch.long)
            assert previous.shape[1]==position
            present=torch.cat([previous,tokens[...,None]],dim=1)
            logits=torch.full((len(tokens),1,len(codec)),-30.)
            for row in range(len(tokens)):
                history=present[row,:,0].tolist()
                wanted=sequence[len(history)] if history==sequence[:len(history)] else codec.note_id
                logits[row,0,wanted]=30.
            return logits,[(present,present,None,None)]
    example={'meta':{},'action_state_padding_mask':torch.ones(1,dtype=torch.bool)}
    points,status=compose_span(Model(),codec,example,['1'],origin=F(0),end=F(4),
        temperature=0,position_temperature=0,position_beam_size=4)
    assert points==expected
    assert status['reason']=='eos'
