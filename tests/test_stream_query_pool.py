import math
import torch

from maichart.composer.stream_timing import AbsoluteTimingHead,absolute_target_indices


def _case(policy):
    head=AbsoluteTimingHead(8,('0','1/2','1','2'),('0','1/2'),
        audio_policy=policy,group_policy='phase_gap_v2')
    cursor=torch.zeros(1,1,dtype=torch.float64)
    beats,gaps=head.candidates(cursor)
    allowed=gaps.gt(0);allowed[...,0]=False
    scores=(2-gaps.float()).requires_grad_()
    args=(torch.zeros(1,1,8),cursor,torch.zeros(1,1,4),
        torch.tensor([0.,.5,1.,2.],dtype=torch.float64),torch.zeros(1,1,32),scores,allowed)
    target=absolute_target_indices(beats,torch.tensor([[.5]],dtype=torch.float64),torch.ones(1,1,dtype=torch.bool))
    return head,args,target,scores


def test_query_pool_starts_identical_to_existing_candidate_pool():
    old,args,target,_=_case('candidate_pool_v1')
    new,_,_,_=_case('query_pool_v2')
    new.load_state_dict(old.state_dict(),strict=False)
    keys=torch.randn(*args[-1].shape,32)
    before=old(*args,candidate_audio=keys)[0]
    after=new(*args,candidate_audio=keys)[0]
    torch.testing.assert_close(after,before,rtol=0,atol=0)
    assert new.query_projection.weight.count_nonzero()==0


def test_singleton_query_score_gets_macro_learning_path():
    old,args,target,scores=_case('candidate_pool_v1')
    loss=-old(*args)[0].gather(-1,target[...,None]).sum()
    loss.backward()
    chosen=int(target.item())
    assert scores.grad[0,0,chosen].abs()<1e-7  # Old within-grid score cancels.
    new,args,target,scores=_case('query_pool_v2')
    loss=-new(*args)[0].gather(-1,target[...,None]).sum()
    loss.backward()
    assert new.query_projection.weight.grad.abs().sum()>0
    new.zero_grad()
    with torch.no_grad():new.query_projection.weight.fill_(1.)
    scores=scores.detach().requires_grad_()
    args=(*args[:-2],scores,args[-1])
    loss=-new(*args)[0].gather(-1,target[...,None]).sum()
    loss.backward()
    assert scores.grad[0,0,chosen]<0
    assert torch.isfinite(scores.grad).all()


def test_query_evidence_is_count_normalized_and_ignores_invalid_candidates():
    head=AbsoluteTimingHead(8,('0','1','2'),audio_policy='query_pool_v2')
    with torch.no_grad():head.query_projection.weight.fill_(1.)
    def evaluate(values,ids,allowed):
        values=torch.tensor([values]);ids=torch.tensor(ids);allowed=torch.tensor([allowed])
        counts=torch.zeros(1,head.num_groups,dtype=torch.long).scatter_add(-1,ids[None],allowed.long())
        return head.query_audio_logits(values,allowed,counts,ids)
    original=evaluate([2.,0.],[0,1],[True,True])
    duplicated=evaluate([2.,2.,0.,1000.],[0,0,1,1],[True,True,True,False])
    torch.testing.assert_close(original,duplicated)
    torch.testing.assert_close(original[0,:2],torch.tensor([1.,-1.]))


def test_query_pool_empty_support_is_finite_and_old_policy_has_no_new_parameter():
    head,args,_,scores=_case('query_pool_v2')
    empty=torch.zeros_like(args[-1]);args=(*args[:-1],empty)
    result=head(*args)[0]
    assert torch.isfinite(result).all() and result.eq(-1e4).all()
    result.sum().backward()
    assert torch.isfinite(head.query_projection.weight.grad).all()
    old,_,_,_=_case('candidate_pool_v1')
    assert not hasattr(old,'query_projection')


def _vector_case():
    head,args,target,scores=_case('query_pool_v3')
    query=torch.ones(1,1,32,requires_grad=True)
    keys=(scores.detach()[...,None]/math.sqrt(32)).expand(*scores.shape,32).clone().requires_grad_()
    audio_scores=(query[...,None,:]*keys).sum(-1)/math.sqrt(32)
    args=(*args[:4],query,audio_scores,args[-1])
    return head,args,target,keys


def test_vector_query_pool_zero_initialization_preserves_all_v2_probabilities():
    new,args,_,keys=_vector_case()
    old,_,_,_=_case('query_pool_v2')
    with torch.no_grad():
        old.audio_projection.weight.fill_(.03)
        old.query_projection.weight.fill_(.12)
    new.load_state_dict(old.state_dict(),strict=False)
    before=old(*args,candidate_audio=keys)[0]
    after=new(*args,candidate_audio=keys)[0]
    torch.testing.assert_close(after,before,atol=0,rtol=0)
    torch.testing.assert_close(after.softmax(-1),before.softmax(-1),atol=0,rtol=0)
    assert not new.query_interaction_projection.weight.any()
    assert not hasattr(old,'query_interaction_projection')


def test_singleton_group_learns_the_vector_path_and_then_sends_gradient_to_music():
    head,args,target,keys=_vector_case()
    loss=-head(*args,candidate_audio=keys)[0].gather(-1,target[...,None]).sum()
    loss.backward()
    gradient=head.query_interaction_projection.weight.grad
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
    head,args,target,keys2=_vector_case()
    with torch.no_grad():head.query_interaction_projection.weight.fill_(.1)
    loss=-head(*args,candidate_audio=keys2)[0].gather(-1,target[...,None]).sum()
    loss.backward()
    assert keys2.grad is not None and torch.isfinite(keys2.grad).all() and keys2.grad.abs().sum()>0
    assert args[4].grad is not None and args[4].grad.abs().sum()>0


def test_vector_interaction_has_gain_eight_and_count_normalized_masked_group_means():
    head,_,_,_=_vector_case()
    with torch.no_grad():
        head.query_interaction_projection.weight.zero_()
        head.query_interaction_projection.weight[0,0]=1.
    query=torch.ones(1,1,32)
    def evaluate(values,ids,valid):
        values=torch.tensor(values,dtype=torch.float32)[None,None]
        ids=torch.tensor(ids);allowed=torch.tensor(valid)[None,None]
        keys=torch.zeros(1,1,len(ids),32);keys[...,0]=values*math.sqrt(32)
        counts=torch.zeros(1,1,head.num_groups,dtype=torch.long).scatter_add(-1,ids[None,None],allowed.long())
        return head.query_interaction_logits(query,keys,allowed,counts,ids)
    plain=evaluate([1.,0.],[0,1],[True,True])
    repeated=evaluate([1.,1.,0.,999.],[0,0,1,2],[True,True,True,False])
    torch.testing.assert_close(plain,repeated,rtol=0,atol=0)
    torch.testing.assert_close(plain[0,0,:2],torch.tensor([4.,-4.]))
    plain[0,0,0].backward()
    torch.testing.assert_close(head.query_interaction_projection.weight.grad[0,0],torch.tensor(4.))


def test_vector_pool_empty_groups_remain_finite_and_positive_music_changes_macro_mass():
    head,args,target,keys=_vector_case()
    before=head(*args,candidate_audio=keys)[0].gather(-1,target[...,None]).exp()
    with torch.no_grad():head.query_interaction_projection.weight.fill_(.1)
    after=head(*args,candidate_audio=keys)[0].gather(-1,target[...,None]).exp()
    assert (after>before).all()
    empty=torch.zeros_like(args[-1]);empty_args=(*args[:-1],empty)
    logits=head(*empty_args,candidate_audio=keys)[0]
    assert torch.isfinite(logits).all() and logits.eq(-1e4).all()
    logits.sum().backward()
    assert torch.isfinite(head.query_interaction_projection.weight.grad).all()
