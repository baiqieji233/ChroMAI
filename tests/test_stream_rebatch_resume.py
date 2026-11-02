"""CPU cursor and sampling contracts for explicit microbatch changes."""
from types import SimpleNamespace
import pytest
from maichart.composer.stream_train import _LocalBatches,_resume_batch_mapping


def dataset(anchor=32):
    return SimpleNamespace(samples=[(song,i,i+(i*7)%70+1) for song in range(3) for i in range(201)],
        training_window_policy='phrase_complete_v1',_sample_order_batch_size=anchor,
        __len__=None)


class Rows:
    training_window_policy='phrase_complete_v1'
    def __init__(self,anchor):
        self._sample_order_batch_size=anchor
        self.samples=[(song,i,i+(i*7)%70+1) for song in range(3) for i in range(201)]
    def __len__(self):return len(self.samples)


def flatten(sampler):return [index for batch in sampler for index in batch]


def test_rebatch_preserves_full_order_suffix_and_second_resume():
    ds=Rows(32)
    for epoch in (1,2,17):
        original=flatten(_LocalBatches(ds,32,19,epoch))
        assert original==flatten(_LocalBatches(ds,16,19,epoch))
        for old_cursor in (0,2,8,16):
            state=dict(batch_in_epoch=old_cursor,epoch_complete=False)
            args=SimpleNamespace(batch_size=16,grad_accumulation=4,max_steps=0,rebatch_resume=True)
            cursor,change=_resume_batch_mapping(dict(batch_size=32,grad_accumulation=2,max_steps=0),state,args,len(ds))
            assert original[old_cursor*32:]==flatten(_LocalBatches(ds,16,19,epoch,cursor))
            assert change['sample_order_batch_size']==32
            second,change2=_resume_batch_mapping(dict(batch_size=16,grad_accumulation=4,max_steps=0,sample_order_batch_size=32),
                dict(batch_in_epoch=cursor,epoch_complete=False),args,len(ds))
            assert second==cursor and change2 is None


def test_rebatch_checks_original_optimizer_boundary_and_explicit_flag():
    old=dict(batch_size=32,grad_accumulation=2,max_steps=0)
    args=SimpleNamespace(batch_size=16,grad_accumulation=4,max_steps=0,rebatch_resume=True)
    with pytest.raises(ValueError,match='complete optimizer'):
        _resume_batch_mapping(old,dict(batch_in_epoch=3,epoch_complete=False),args,603)
    args.rebatch_resume=False
    with pytest.raises(ValueError,match='rebatch-resume'):
        _resume_batch_mapping(old,dict(batch_in_epoch=4,epoch_complete=False),args,603)
    args.rebatch_resume=True;args.grad_accumulation=2
    with pytest.raises(ValueError,match='effective batch'):
        _resume_batch_mapping(old,dict(batch_in_epoch=4,epoch_complete=False),args,603)
    args.grad_accumulation=4;args.max_steps=100
    with pytest.raises(ValueError,match='max_steps'):
        _resume_batch_mapping(old,dict(batch_in_epoch=4,epoch_complete=False),args,603)


def test_epoch_complete_and_final_short_batch_are_mapped_safely():
    old=dict(batch_size=32,grad_accumulation=2,max_steps=0)
    args=SimpleNamespace(batch_size=16,grad_accumulation=4,max_steps=0,rebatch_resume=True)
    assert _resume_batch_mapping(old,dict(batch_in_epoch=19,epoch_complete=True),args,603)[0]==0
    assert _resume_batch_mapping(old,dict(batch_in_epoch=19,epoch_complete=False),args,603)[0]==38
