"""Execution-only speed fixes must preserve every causal feature."""
from dataclasses import replace
from fractions import Fraction
import importlib.util
import pathlib
import random
import sys

import numpy as np
import pytest
import torch

from maichart.stream import hand_tracking as hands
from maichart.stream.data import StreamDataset,collate_stream
from maichart.stream.input_cache import LazyBeats,CompactBeatText
from test_stream_grounding import source,config,cpu_threads


def reference_advance(states,lanes,second,trajectories=(),*,hidden=False,beam=8):
    """Frozen pre-optimization implementation of the actual hand law."""
    import math
    if hidden:return (hands.Assignment(),)
    lanes=tuple(sorted(set(lanes)))
    if not lanes:return states
    if len(lanes)>4:return (hands.Assignment(),)
    holds={head:end for kind,head,_dest,_trace,end in trajectories if kind==0}
    choices={}
    for state in states:
        for groups in hands._splits(lanes):
            values=[];extra=0.
            active=[i for i,g in enumerate(groups) if g]
            actor=active[0] if len(active)==1 else 2
            repeated=actor in (0,1) and actor==state.actor
            switched=actor in (0,1) and state.actor in (0,1) and actor!=state.actor
            for old,group in zip(state.hands,groups):
                if not group:values.append(old);continue
                lane,when,dx,dy,run,hold_end=old
                target=min(group,key=lambda x:sum(math.dist(hands._XY[x],hands._XY[y]) for y in group))
                span=max((math.dist(hands._XY[a],hands._XY[b]) for a in group for b in group),default=0.)
                extra+=max(0.,span-.9)
                if lane:
                    nx,ny=hands._XY[target][0]-hands._XY[lane][0],hands._XY[target][1]-hands._XY[lane][1]
                    elapsed=max(.04,second-when)
                    extra+=.02*math.hypot(nx,ny)/elapsed
                    if dx or dy:extra+=.03*math.hypot(nx-dx,ny-dy)
                    if hold_end>second+1e-9 and target!=lane:extra+=2.
                else:nx=ny=0.
                values.append((target,second,nx,ny,min(8,run+1) if repeated else 1,
                               max((holds.get(g,0.) for g in group),default=0.)))
            values,actor=hands._canonical(tuple(values),actor)
            candidate=hands.Assignment(values,state.cost+extra,actor,repeated,switched)
            key=(values,actor,repeated,switched)
            if key not in choices or candidate.cost<choices[key].cost:choices[key]=candidate
    ordered=sorted(choices.values(),key=lambda x:x.cost)
    if not ordered:return (hands.Assignment(),)
    chosen=[]
    for predicate in (lambda s:s.repeated,lambda s:s.switched):
        candidate=next((s for s in ordered if predicate(s)),None)
        if candidate is not None and candidate not in chosen:chosen.append(candidate)
    continuous=max(ordered,key=lambda s:(max(h[4] for h in s.hands),-s.cost))
    if continuous not in chosen:chosen.append(continuous)
    chosen.extend(s for s in ordered if s not in chosen)
    minimum=ordered[0].cost
    return tuple(hands.Assignment(s.hands,s.cost-minimum,s.actor,s.repeated,s.switched) for s in chosen[:beam])


def test_hand_geometry_cache_and_identity_selection_match_original_exactly():
    rng=random.Random(73);actual=expected=(hands.Assignment(),)
    for event in range(120):
        lanes=tuple(sorted(rng.sample(range(1,9),rng.randrange(5))))
        trajectories=((0,lanes[0],lanes[0],event*.1,event*.1+.4),) if lanes and event%7==0 else ()
        hidden=event%29==0
        expected=reference_advance(expected,lanes,event*.1,trajectories,hidden=hidden)
        actual=hands.advance(actual,lanes,event*.1,trajectories,hidden=hidden)
        assert actual==expected


def test_masked_prefix_reset_skips_irrelevant_assignment_work_but_preserves_long_actions(monkeypatch):
    lanes=[(i%8+1,) for i in range(1000)];seconds=np.arange(1000)*.05
    trajectories=[() for _ in lanes];trajectories[0]=((1,1,5,.1,100.),)
    calls=[];original=hands.advance
    def counted(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(hands,'advance',counted)
    history=hands.HandHistory(lanes,seconds,trajectories,stride=128,keep_recent=False)
    result=history.rows([801,802,805],[40.1,40.2,40.3],masked_events={790,799})
    assert len(calls)==5  # starts just after the last reset, not event 0
    assert result[:,23].tolist()==[1.,1.,1.]
    baseline=(hands.Assignment(),)
    for event in range(805):
        baseline=reference_advance(baseline,lanes[event],seconds[event],trajectories[event],hidden=event in {790,799})
    np.testing.assert_array_equal(result[-1].numpy(),hands.features(baseline,40.3,1))


def test_resident_query_alias_does_not_consume_second_cache_budget(source,tmp_path,monkeypatch):
    dataset=StreamDataset(source[1],cache_dir=tmp_path/'cache',max_tokens=32,max_history_tokens=12,
        grounding_config=replace(config(),motor_profile='causal_hands_v1'),preparation_workers=1,audio_cache_gb=.02)
    expected=[dataset[i] for i in range(len(dataset))]
    fingerprint=dataset.fingerprint
    dataset.prepare_residency(policy='resident',budget_bytes=10*1024**2,reserve_bytes=1)
    monkeypatch.setattr(dataset.input_cache,'_read',lambda *_:pytest.fail('reopened resident artifact'))
    for index,item in enumerate(expected):
        result=dataset[index]
        for key,value in item.items():
            if torch.is_tensor(value):torch.testing.assert_close(result[key],value,rtol=0,atol=0,msg=key)
            else:assert result[key]==value
    assert dataset.query_bytes==dataset.structure_bytes==0
    assert dataset.fingerprint==fingerprint
    assert all(not query[2] for query,_size in dataset._audio_queries.values())
    owned=dataset.input_cache.resident_bytes
    budget=dataset.query_budget
    dataset.prepare_residency(policy='resident',budget_bytes=10*1024**2,reserve_bytes=1)
    assert dataset.input_cache.resident_bytes==owned and dataset.query_budget==budget


def test_mapped_training_query_drops_only_unused_text_and_preserves_disk_fraction_metadata(source,tmp_path):
    dataset=StreamDataset(source[1],cache_dir=tmp_path/'cache',max_tokens=32,max_history_tokens=12,
        grounding_config=config(),preparation_workers=1,audio_cache_gb=.02)
    values,times,unused=dataset._queries(0,dataset.base.songs[0][0])
    assert unused==()
    identity=dataset._query_identity(0,'planner',1024)
    saved,extra=dataset.input_cache.get(identity,lambda:pytest.fail('rebuilt'))
    assert len(extra['beats'])==len(times)
    assert list(LazyBeats(extra['beats']))==[Fraction(value) for value in extra['beats']]
    torch.testing.assert_close(saved['values'],values,rtol=0,atol=0)


def test_native_hand_backend_matches_python_states_features_and_masked_replay_exactly():
    pytest.importorskip('numba')
    from maichart.stream.native_hand_tracking import NativeHandHistory,warmup
    warmup()
    rng=random.Random(933)
    lanes=[tuple(sorted(rng.sample(range(1,9),rng.randrange(5)))) for _ in range(180)]
    seconds=np.cumsum([rng.choice([.03,.04,.08,.125,.5]) for _ in lanes])
    trajectories=[((0,row[0],row[0],seconds[i],seconds[i]+.75),)
                  if row and i%9==0 else () for i,row in enumerate(lanes)]
    trajectories[1]=((1,1,5,seconds[1]+.25,seconds[-1]+1),)
    native=NativeHandHistory(lanes,seconds,trajectories,stride=32,keep_recent=False)
    python=hands.HandHistory(lanes,seconds,trajectories,stride=32,keep_recent=False)
    for counts,hidden in [([0,1,4,35,78,129,180],()),([120,35,155,180],{12,34,128}),
                          ([35,35,35,0,129,129],{12,34,128}),
                          ([151,156,180],{127,149}),([1,2,180],{0,6}),([0,1,2],())]:
        queries=[0. if not count else seconds[count-1]+.02 for count in counts]
        expected=python.rows(counts,queries,masked_events=hidden)
        actual=native.rows(counts,queries,masked_events=hidden)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert native.count==180
    states,number=native.checkpoints[160]
    expected=python.checkpoints[160]
    assert number==len(expected)
    for row,state in zip(states,expected):
        assert tuple(row[:6])==state.hands[0] and tuple(row[6:12])==state.hands[1]
        assert row[12]==state.cost and int(row[13])==state.actor


def test_native_dataset_batch_matches_python_including_random_masks_and_symmetry(source,tmp_path):
    pytest.importorskip('numba')
    from maichart.stream.native_hand_tracking import warmup
    warmup()
    kwargs=dict(manifest=source[1],cache_dir=tmp_path/'cache',max_tokens=32,max_history_tokens=12,
        grounding_config=replace(config(),motor_profile='causal_hands_v1'),preparation_workers=1,
        audio_cache_gb=.02,history_note_dropout=.45,prefix_note_dropout=.35,symmetry_augmentation='mirror_lr')
    reference=StreamDataset(**kwargs);actual=StreamDataset(**kwargs,hand_backend='native')
    assert actual.fingerprint==reference.fingerprint
    for epoch in (1,3):
        reference.set_epoch(epoch);actual.set_epoch(epoch)
        expected=collate_stream([reference[i] for i in range(len(reference))])
        observed=collate_stream([actual[i] for i in range(len(actual))])
        for key in expected:torch.testing.assert_close(observed[key],expected[key],rtol=0,atol=0,msg=key)


def test_native_backend_auto_fallback_is_reported_without_changing_feature_profile(monkeypatch):
    import builtins
    original=builtins.__import__
    def blocked(name,*args,**kwargs):
        if name.endswith('native_hand_tracking'):raise ImportError('test missing optional dependency')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',blocked)
    assert hands.prepare_training_backend('auto')['effective']=='python'
    with pytest.raises(ValueError,match='unavailable'):hands.prepare_training_backend('native')


def test_native_backend_auto_fallback_handles_an_empty_exception(monkeypatch):
    import builtins
    original=builtins.__import__
    def blocked(name,*args,**kwargs):
        if name.endswith('native_hand_tracking'):raise RuntimeError()
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',blocked)
    assert hands.prepare_training_backend('auto')['fallback']=='RuntimeError'


def test_residency_available_memory_respects_windows_commit_headroom(monkeypatch):
    import ctypes
    import os
    from maichart.form.audio_residency import physical_memory
    if os.name!='nt':pytest.skip('Windows commit accounting')
    def read_status(pointer):
        value=pointer._obj
        value.total_phys=16*1024**3;value.available_phys=12*1024**3
        value.available_page=512*1024**2
        return 1
    monkeypatch.setattr(ctypes.windll.kernel32,'GlobalMemoryStatusEx',read_status)
    assert physical_memory()==(16*1024**3,12*1024**3)
    assert physical_memory(include_commit=True)==(16*1024**3,512*1024**2)
