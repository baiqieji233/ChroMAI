import torch
from maichart.composer.stream_model import to_device_batch


def test_explicit_shared_bank_survives_pinning_copies_without_merging_query_indices():
    bank=torch.tensor([[1,2,3]])
    departure=torch.tensor([[0,0]])
    landing=torch.tensor([[0,-1]])
    batch={'phrase_sequence_kind':bank,'candidate_phrase_sequence_kind':bank.clone(),
        'phrase_sequence_inverse':departure,'candidate_phrase_sequence_inverse':landing,
        'candidate_phrase_sequence_shared':True}
    result=to_device_batch(batch,'cpu')
    assert result['phrase_sequence_kind'] is result['candidate_phrase_sequence_kind']
    assert result['phrase_sequence_inverse'] is not result['candidate_phrase_sequence_inverse']
    torch.testing.assert_close(result['candidate_phrase_sequence_inverse'],landing)


def test_unshared_banks_and_nested_sparse_groups_keep_their_values():
    bank=torch.tensor([1,2]);other=torch.tensor([8,9])
    batch={'phrase_sequence_kind':bank,'candidate_phrase_sequence_kind':other,
        'candidate_phrase_sequence_shared':False,'groups':[(2,bank,[bank])]}
    result=to_device_batch(batch,'cpu')
    assert result['candidate_phrase_sequence_kind'] is not result['phrase_sequence_kind']
    assert result['groups'][0][1] is result['groups'][0][2][0]
    torch.testing.assert_close(result['candidate_phrase_sequence_kind'],other)
