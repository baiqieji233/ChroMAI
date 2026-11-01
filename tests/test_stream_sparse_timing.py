"""The training timing head may skip queries without a NOTE time label."""
import copy

import torch

from maichart.composer.stream_data import pack_timing_queries
from maichart.composer.stream_model import NOTE, WAIT
from test_composer_stream_model import fixture


def test_packing_preserves_batch_axis_with_short_and_empty_rows():
    batch = dict(target_kind=torch.tensor([[NOTE, WAIT, NOTE, WAIT],
                                           [WAIT, NOTE, WAIT, WAIT],
                                           [WAIT, WAIT, WAIT, WAIT]]),
                 loss_mask=torch.tensor([[0, 1, 1, 0], [0, 1, 0, 0], [1, 1, 1, 1]],dtype=torch.bool),
                 sample_weight=torch.tensor([[1., 1., 2., 0.], [0., 1., 0., 0.], [1., 1., 1., 1.]]))
    pack_timing_queries(batch)
    assert batch['_timing_query_positions'].tolist() == [[2], [1], [0]]
    assert batch['_timing_packed_flat'].tolist() == [0, 1]
    assert batch['_timing_original_flat'].tolist() == [2, 5]


def test_supervised_absolute_timing_matches_dense_loss_and_gradients():
    model, source = fixture(timing_output_policy='absolute_grid_v1')
    model.timing_grid.configure(('0', '1/4', '1/3', '1/2', '3/4'))
    model.train()
    batch = copy.deepcopy(source)
    batch['step_beat'][0, 2] = 3.25
    batch['target_beat'] = torch.tensor([[1., 2., 4.25, 4.]],dtype=torch.float64)
    batch['loss_mask'] = torch.tensor([[False, False, True, False]])
    batch['sample_weight'] = torch.tensor([[0., 0., 1., 0.]])
    # A second, shorter song row has no supervised NOTE. Its padded query is
    # computed but must never be scattered or affect either model's gradient.
    for key,value in list(batch.items()):
        if torch.is_tensor(value) and value.ndim and value.size(0)==1 and key not in (
                'song_audio','song_mask','song_seconds','song_beats','song_inverse'):
            batch[key] = torch.cat((value,value.clone()),dim=0)
    batch['song_inverse'] = torch.tensor([0,0])
    batch['step_mask'][1,2:] = False
    batch['target_kind'][1].fill_(WAIT)
    batch['sample_weight'][1].zero_()
    pack_timing_queries(batch)
    dense = copy.deepcopy(model)
    dense._sparse_timing_enabled = False
    sparse = copy.deepcopy(model)
    sparse._sparse_timing_enabled = True
    dense_output = dense(batch)
    sparse_output = sparse(batch)
    active = batch['target_kind'].eq(NOTE) & batch['loss_mask'] & batch['sample_weight'].gt(0)
    torch.testing.assert_close(sparse_output['delta_logits'][active], dense_output['delta_logits'][active],rtol=1e-5,atol=1e-5)
    torch.testing.assert_close(sparse_output['candidate_beats'],dense_output['candidate_beats'],rtol=0,atol=0)
    torch.testing.assert_close(sparse_output['candidate_deltas'],dense_output['candidate_deltas'],rtol=0,atol=0)
    dense_loss,_ = dense.loss(dense_output,batch)
    sparse_loss,_ = sparse.loss(sparse_output,batch)
    torch.testing.assert_close(sparse_loss,dense_loss,rtol=1e-5,atol=1e-5)
    dense_loss.backward()
    sparse_loss.backward()
    for name, parameter in dense.named_parameters():
        other = dict(sparse.named_parameters())[name]
        if parameter.grad is None:
            assert other.grad is None, name
        else:
            torch.testing.assert_close(other.grad,parameter.grad,rtol=3e-4,atol=3e-5,msg=name)
    assert dense.timing_grid.interval_candidate_projection.weight.grad is not None
    assert dense.timing_grid.interval_group_projection.weight.grad is not None
    assert sparse_output['timing_query_evaluated']==2
    assert sparse_output['timing_query_supervised']==1
    assert sparse_output['timing_query_total']==8


def test_all_wait_timing_queries_stay_finite():
    model,batch=fixture(timing_output_policy='absolute_grid_v1')
    model.train()
    batch['target_kind'].fill_(WAIT)
    batch['target_delta'].fill_(0)
    batch['target_shape'].zero_()
    batch['sample_weight'].zero_()
    pack_timing_queries(batch)
    output=model(batch)
    assert output['timing_query_supervised']==0
    assert torch.isfinite(output['delta_logits']).all()
    assert output['delta_logits'].eq(-1e4).all()
