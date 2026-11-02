import math
from fractions import Fraction as F
import pytest
import torch
from maichart.composer.stream_continuation import typed_event
from maichart.composer.stream_data import StreamStep,NOTE,WAIT,EOS
from maichart.composer.stream_phrase_training import (future_relation_index,future_relation_rows,RELATION_HORIZONS,RELATION_DIM,_head_cross_correlation)


def legacy_torch_future_relation_rows(stream, unknown, first, end, *, index=None):
    """O(requested rows) with a cached index; CPU labels, no model inputs."""
    count=len(stream)
    next_notes,wait_prefix,unknown_prefix,previous=(future_relation_index(stream,unknown) if index is None else index)
    values = torch.zeros(end-first, RELATION_HORIZONS, RELATION_DIM)
    masks = torch.zeros_like(values, dtype=torch.bool)
    for local, i in enumerate(range(first, end)):
        p = previous[i]
        old = typed_event(stream[p].text)[1] if p >= 0 and not unknown[p] else frozenset()
        for h, j in enumerate(next_notes[i]):
            stop = int(j)+1 if j >= 0 else count
            valid = bool(unknown_prefix[stop] == unknown_prefix[i])
            masks[local,h,19] = valid
            values[local,h,19] = float(j < 0)
            if j < 0 or not valid: continue
            target = stream[j]
            types, buttons, _ = typed_event(target.text)
            if not types: continue
            masks[local,h,:8] = True
            for button in buttons: values[local,h,button-1] = 1.
            if old and buttons:
                masks[local,h,8:16] = True
                for shift in range(8):
                    values[local,h,8+shift] = sum(((b-1+shift)%8+1) in buttons for b in old)/max(len(old),len(buttons))
            values[local,h,16] = math.log1p(max(0.,float(target.end-stream[i].start)))
            values[local,h,17] = math.log1p(max(0.,target.end_second-stream[i].start_second))
            values[local,h,18] = float(wait_prefix[j] > wait_prefix[i])
            values[local,h,20] = float('hold' in types)
            values[local,h,21] = float('slide' in types)
            masks[local,h,16:19] = True
            masks[local,h,20:22] = True
    return values, masks


@pytest.mark.parametrize('cached',[False,True])
@pytest.mark.parametrize('sliced',[False,True])
@pytest.mark.parametrize('unknown_mode',['none','alternating','all'])
def test_numpy_future_relations_exactly_match_original_torch_writes(cached,sliced,unknown_mode):
    texts=['1','1/5','2h[4:1]','A1','C','3-7[8:1]','A2h[4:1]','4/8','1/5']*3
    stream=[]
    cursor=F(0)
    for i,text in enumerate(texts):
        end=cursor+F((i%4)+1,3)
        stream.append(StreamStep(NOTE,cursor,end,float(cursor)/2,float(end)/2,text,weight=0. if i%7==0 else 1.))
        cursor=end
        if i%5==0:
            stream.append(StreamStep(WAIT,cursor,cursor+F(1,2),float(cursor)/2,float(cursor+F(1,2))/2,''));cursor+=F(1,2)
    stream.append(StreamStep(EOS,cursor,cursor,float(cursor)/2,float(cursor)/2,''))
    unknown=[unknown_mode=='all' or (unknown_mode=='alternating' and i%3==0) for i in range(len(stream))]
    first,end=(3,len(stream)-2) if sliced else (0,len(stream))
    index=future_relation_index(stream,unknown) if cached else None
    expected=legacy_torch_future_relation_rows(stream,unknown,first,end,index=index)
    actual=future_relation_rows(stream,unknown,first,end,index=index)
    assert actual[0].dtype==torch.float32 and actual[1].dtype==torch.bool
    for left,right in zip(expected,actual):assert torch.equal(left,right)


def test_numpy_future_relations_empty_and_rotation_cache_bound():
    for left,right in zip(legacy_torch_future_relation_rows([],[],0,0),future_relation_rows([],[],0,0)):
        assert torch.equal(left,right)
    _head_cross_correlation.cache_clear()
    old=frozenset((1,3,5));new=frozenset((2,4))
    expected=tuple(sum(((b-1+shift)%8+1) in new for b in old)/max(len(old),len(new)) for shift in range(8))
    assert _head_cross_correlation(old,new)==expected
    assert _head_cross_correlation(old,new)==expected
    assert _head_cross_correlation.cache_info().hits==1
    assert _head_cross_correlation.cache_info().maxsize==4096
