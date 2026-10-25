from fractions import Fraction

import numpy as np
import torch

from maichart.composer.motif_memory import ComposerMotifIndex, MotifMemoryAdapter
from maichart.form.recurrence import Relation, SongRelations
from maichart.form.timing import MetricSpan


def _relations(links=True):
    spans = tuple(MetricSpan(i, Fraction(start), Fraction(start+16), 4, 4, (),
        Fraction(4), (), False, 1., 'test') for i, start in enumerate((0, 16, 32)))
    return SongRelations(spans, np.zeros((3,1)), [Relation(0,2,.99,.9,1.)] if links else [])


def test_cross_chorus_archive_excludes_local_history_and_uncommitted_suffix():
    beats = list(range(48))
    atoms = np.arange(1,49)[:,None]
    index = ComposerMotifIndex(_relations(), beats, atoms)
    actual = index.retrieve(36, committed_count=36, history_start_index=20, max_events=4)
    assert actual.mask.all()
    assert set(actual.atom_ids[:,0]).issubset(set(range(1,17)))
    assert 5 in actual.atom_ids[:,0]  # First chorus phase matches beat 36.
    changed = atoms.copy(); changed[36:] = 9999
    extended = ComposerMotifIndex(_relations(), beats + [48,49], np.concatenate((changed,[[8888],[8888]])))
    other = extended.retrieve(36, committed_count=36, history_start_index=20, max_events=4)
    np.testing.assert_array_equal(actual.atom_ids,other.atom_ids)
    np.testing.assert_array_equal(actual.relative_beats,other.relative_beats)
    # Even an earlier source is not visible before it has been committed.
    prefix = index.retrieve(36, committed_count=3, history_start_index=3, max_events=8)
    assert set(prefix.atom_ids[prefix.mask,0]) == {1,2,3}


def test_no_match_budget_and_exact_boundary():
    index = ComposerMotifIndex(_relations(False), range(48), np.ones((48,2),np.int64))
    for query in (0,16,32,48):
        actual = index.retrieve(query, committed_count=32, history_start_index=16, max_events=7)
        assert actual.atom_ids.shape == (7,2)
        assert not actual.mask.any()
        assert not actual.atom_ids.any()
    index = ComposerMotifIndex(_relations(), range(48), np.ones((48,2),np.int64))
    actual = index.retrieve(32, committed_count=32, history_start_index=16, max_events=3)
    assert actual.mask.sum() == 3
    assert actual.relative_beats.min() == 0


def test_adapter_zero_init_and_empty_mask_remain_finite_after_learning():
    torch.set_num_threads(1)
    torch.manual_seed(3)
    adapter = MotifMemoryAdapter(8)
    embedding = torch.nn.Embedding(8,8,padding_idx=0)
    query = torch.randn(2,5,8)
    ids = torch.tensor([[[1,0],[2,3]],[[0,0],[0,0]]])
    relative = torch.tensor([[0.,1.],[0.,0.]])
    mask = torch.tensor([[True,True],[False,False]])
    residual = adapter(query,embedding,ids,relative,mask)
    assert torch.equal(residual,torch.zeros_like(query))
    residual.sum().backward()
    assert adapter.output.weight.grad.abs().sum() > 0
    torch.nn.init.normal_(adapter.output.weight)
    result = adapter(query,embedding,ids,relative,mask)
    assert torch.isfinite(result).all()
    assert torch.equal(result[1],torch.zeros_like(result[1]))
    assert result[0].abs().max() <= 1
    assert adapter(query[:,0],embedding,ids,relative,mask).shape == (2,8)


def test_many_candidate_phase_and_adapter_do_not_broadcast_candidates():
    torch.set_num_threads(1)
    index = ComposerMotifIndex(_relations(), range(48), np.arange(1,49)[:,None])
    rows = index.retrieve_many([32,40], committed_count=32, history_start_index=16,max_events=1)
    assert rows.atom_ids.shape == (2,1,1)
    assert rows.atom_ids[:,0,0].tolist() == [1,9]
    adapter = MotifMemoryAdapter(8)
    torch.nn.init.normal_(adapter.output.weight)
    embedding = torch.nn.Embedding(50,8)
    ids = torch.from_numpy(rows.atom_ids)[None]
    relative = torch.from_numpy(rows.relative_beats)[None]
    mask = torch.from_numpy(rows.mask)[None]
    query = torch.randn(1,2,8)
    actual = adapter.read(query,embedding(ids),ids,relative,mask)
    for i in range(2):
        expected = adapter(query[:,i],embedding,ids[:,i],relative[:,i],mask[:,i])
        torch.testing.assert_close(actual[:,i],expected)
