"""Exact reference arithmetic for grouped CPU acoustic queries."""
import numpy as np
import pytest
import os
import runpy
from pathlib import Path
import torch

from maichart.composer.acoustics import AcousticContext


def reference_trailing(context, seconds):
    seconds = np.asarray(seconds, dtype=np.float64).reshape(-1)
    end = context._integral(seconds)
    return np.concatenate([(end-context._integral(seconds-width))/width
                           for width in (1.,4.)],axis=-1).astype(np.float32)


def reference_interval(context, committed_second, candidate_seconds):
    seconds=np.asarray(candidate_seconds,dtype=np.float64).reshape(-1)
    start=np.clip(float(committed_second),0.,context.duration)
    stop=np.clip(seconds,start,context.duration)
    area=context._integral(stop)-context._integral(np.array([start]))
    width=np.maximum(stop-start,1e-6)
    mean=area[:,:14]/width[:,None]
    if len(context.times):
        begin=np.array([np.interp(start,context.times,context.data[:,c]) for c in range(14,26)])
        end=np.column_stack([np.interp(stop,context.times,context.data[:,c]) for c in range(14,26)])
        norm=np.linalg.norm(end,axis=-1)*np.linalg.norm(begin)
        novelty=np.where(norm>1e-9,1.-(end*begin).sum(-1)/np.maximum(norm,1e-9),0.)
    else:novelty=np.zeros(len(stop))
    return np.column_stack((mean,np.log1p(np.maximum(area[:,0],0.)),novelty)).astype(np.float32)


@pytest.mark.parametrize('count',[0,1,2,57])
@pytest.mark.parametrize('duration',[0.,.037,1.137,4.273])
@pytest.mark.parametrize('empty_query',[False,True])
def test_grouped_acoustic_queries_are_exact(count,duration,empty_query):
    rng=np.random.default_rng(482)
    times=np.linspace(0.,max(duration,.01),count)
    context=AcousticContext(rng.normal(size=(count,120)).astype(np.float32),times,duration)
    queries=[] if empty_query else [-5.,-1e-12,0.,.013,.037,1.,duration-1e-12,duration,duration+1e-12,10.]
    np.testing.assert_array_equal(context.trailing(queries),reference_trailing(context,queries))
    for start in (-1.,0.,.013,duration,duration+1.):
        np.testing.assert_array_equal(context.interval(start,queries),reference_interval(context,start,queries))


@pytest.mark.skipif(os.environ.get('MAICHART_REAL_INPUT_EQUIVALENCE')!='1',
                    reason='Requires local four-chart frozen production fixture and audio banks')
def test_four_chart_frozen_inputs_all_fields_exact():
    from maichart.composer.data import collate_composer
    directory=Path(__file__).resolve().parents[1]/'.scratch/composer_gpu_efficiency_20261026'
    fixture=torch.load(directory/'frozen_batches.pt',map_location='cpu',weights_only=False)
    make_dataset=runpy.run_path(str(directory/'prepare_frozen_batches.py'))['make_dataset']
    torch.set_num_threads(2)
    dataset,_=make_dataset()
    for indices,expected in zip(fixture['provenance']['sample_indices'],fixture['batches']):
        actual=collate_composer([dataset[index] for index in indices])
        assert actual.keys()==expected.keys()
        for key,value in expected.items():
            if torch.is_tensor(value):
                assert actual[key].dtype==value.dtype,key
                assert torch.equal(actual[key],value),key
            else:assert actual[key]==value,key
