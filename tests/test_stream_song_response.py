"""Whole-song response math and chart-identity regression contracts."""
import torch
from maichart.composer.stream_song_response import SongResponse
from maichart.composer.stream_phrase_execution import whole_phrase_configuration_loss
from maichart.composer.stream_phrase_training import collate_phrase_training


def test_nested_response_prior_once_with_actual_target_weights():
    nll=torch.tensor([[[.2,2.]],[[1.5,.3]]],requires_grad=True)
    weights=torch.tensor([[.5],[2.]])
    song=torch.tensor([[.7,-.4],[.7,-.4]],requires_grad=True)
    phrase=torch.tensor([[[[2.,0.],[0.,2.]]],[[[2.,0.],[0.,2.]]]],requires_grad=True)
    batch=dict(training_phrase_group=torch.tensor([[0],[1]]),training_phrase_complete=torch.ones(2,1,dtype=torch.bool),
        training_phrase_position=torch.zeros(2,1,dtype=torch.long),training_phrase_group_weight=torch.tensor([.5,2.]),
        training_song_response_group=torch.tensor([0,0]),_song_response_group_count=1)
    actual,metrics=whole_phrase_configuration_loss(nll,torch.zeros_like(nll),batch,weights,2.5,
        song_logits=song,song_phrase_logits=phrase)
    per_phrase=-torch.logsumexp(phrase[:,0].log_softmax(-1)-nll[:,0,None,:]*weights[:,0,None,None],-1)
    expected=-torch.logsumexp(song[0].log_softmax(-1)-per_phrase.sum(0),0)/2.5
    torch.testing.assert_close(actual,expected)
    assert metrics['song_response_groups']==1
    actual.backward()
    assert torch.isfinite(nll.grad).all() and torch.isfinite(song.grad).all()


def test_song_prior_ignores_phrase_and_masked_audio():
    torch.manual_seed(8)
    model=SongResponse(12,8,3,4,2)
    audio=torch.randn(2,5,12);mask=torch.tensor([[1,1,0,0,0],[1,1,1,1,0]],dtype=torch.bool)
    condition=torch.randn(2,3);anchor=torch.randn(2,6,8)
    a=model(audio,mask,condition,anchor)
    changed=audio.clone();changed[~mask]=999
    b=model(changed,mask,condition,anchor+100)
    torch.testing.assert_close(a['song_response_logits'],b['song_response_logits'])
    assert a['song_response_phrase_logits'].shape==(2,6,4,2)


def item(chart,segment):
    return dict(_training_chart_id=chart,training_phrase_group=torch.tensor([segment]),
        training_phrase_position=torch.tensor([0]),training_phrase_complete=torch.tensor([True]),
        loss_mask=torch.tensor([True]),sample_weight=torch.tensor([1.]),_training_target_budget=torch.tensor([1.]))


def test_response_group_uses_chart_identity_and_distant_phrases():
    output={}
    collate_phrase_training([item(0,0),item(1,4),item(0,4),item(1,0)],output)
    groups=output['training_song_response_group']
    assert groups[0]==groups[2] and groups[1]==groups[3] and groups[0]!=groups[1]
    out={};collate_phrase_training([item(0,0),item(0,1)],out)
    assert out['_song_response_group_count']==0

def test_configured_group_size_and_missing_recovery_chart_id():
    rows=[item(0,i*2) for i in range(6)]
    for row in rows:row['_song_response_group_size']=2
    output={};collate_phrase_training(rows,output)
    ids=output['training_song_response_group']
    assert output['_song_response_group_count']==3
    assert all(int(ids.eq(g).sum())==2 for g in range(3))
    recovery=item(0,8);recovery.pop('_training_chart_id')
    recovery['_song_response_group_size']=2
    output={};collate_phrase_training([rows[0],recovery,rows[1]],output)
    assert output['training_chart_id'].tolist()==[0,-1,0]
    assert output['profile_chart_first'].tolist()==[True,False,False]


def test_legacy_phrase_rows_without_chart_identity_do_not_invent_song_groups():
    rows=[item(0,0),item(0,4)]
    for row in rows:row.pop('_training_chart_id')
    output={};collate_phrase_training(rows,output)
    assert 'profile_chart_rows' not in output
    assert output['_song_response_group_count']==0
