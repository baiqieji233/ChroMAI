from fractions import Fraction
from types import SimpleNamespace
import torch
import pytest
from maichart.composer import stream_train,stream_data,stream_checkpoint
from maichart.event_timing import EventPoint,EventTempo
from test_composer_stream_train import synthetic_bank


def test_explicit_lr_reset_keeps_adam_step_and_data_cursor(tmp_path,monkeypatch):
    monkeypatch.setattr(stream_data,'build_audio_bank',synthetic_bank)
    song=SimpleNamespace(audio=SimpleNamespace(duration_sec=2.),
        tempo=[EventTempo(Fraction(0),120.)],first=0.)
    points=[EventPoint(Fraction(i,2),str(i%8+1)) for i in range(8)]
    base=SimpleNamespace(songs=[(song,points,12.,1.,None,torch.zeros(16),(),None)])
    def dataset(args,*,codec=None,delta_vocab=None,validation=False):
        value=stream_data.StreamDataset(base=base,codec=codec,delta_vocab=delta_vocab,
            lead_in=4,targets=2,history_window_policy=args.history_window_policy,
            continuation_policy=args.continuation_policy,phrase_context_policy=args.phrase_context_policy)
        value.physical_policy=args.physical_policy
        value.recovery_target_policy=args.recovery_target_policy
        return value
    monkeypatch.setattr(stream_train,'_dataset',dataset)
    monkeypatch.setattr(stream_train,'_condition_defaults',lambda ds:{})
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    directory=tmp_path/'run'
    argv=['--manifest',str(manifest),'--checkpoint-dir',str(directory),'--device','cpu',
        '--precision','fp32','--epochs','3','--batch-size','1','--grad-accumulation','2',
        '--d-model','8','--nhead','2','--layers','1','--dim-feedforward','16',
        '--history-events','4','--rollout-steps','2','--symbolic-pretrain-epochs','3',
        '--checkpoint-interval-steps','0','--warmup-steps','0','--cpu-threads','1',
        '--recovery-interval','0','--generation-eval-songs','0','--stop-after-updates','1']
    stream_train._train(stream_train.parser().parse_args(argv))
    path=directory/'composer_last.pt';before=stream_checkpoint.read(path)
    assert before['optimizer_steps']==1
    assert before['training_state']['batch_in_epoch']==2
    changed=argv+['--resume',str(path),'--reset-learning-rate','--lr','.0005',
        '--lr-schedule','constant','--min-lr-ratio','1']
    emit=stream_train._emit
    def interrupted(kind,**values):
        if kind=='composer_checkpoint' and values['optimizer_steps']==1:
            raise RuntimeError('stop after saving the changed plan, before another update')
        return emit(kind,**values)
    monkeypatch.setattr(stream_train,'_emit',interrupted)
    with pytest.raises(RuntimeError,match='changed plan'):
        stream_train._train(stream_train.parser().parse_args(changed))
    durable=stream_checkpoint.read(path)
    assert durable['optimizer_steps']==1
    assert durable['training_state']['batch_in_epoch']==2
    assert durable['training_state']['lr_schedule']['base_lr']==.0005
    monkeypatch.setattr(stream_train,'_emit',emit)
    stream_train._train(stream_train.parser().parse_args([arg for arg in changed if arg!='--reset-learning-rate']))
    after=stream_checkpoint.read(path)
    assert after['optimizer_steps']==2
    assert after['training_state']['batch_in_epoch']==4
    plan=after['training_state']['lr_schedule']
    assert plan['kind']=='constant' and plan['base_lr']==.0005 and plan['warmup_steps']==0
    assert all(group['lr']==.0005 for group in after['optimizer_state_dict']['param_groups'])
    history=after['training_state']['learning_rate_changes']
    assert len(history)==1 and history[0]['optimizer_step']==1
    assert history[0]['previous']==before['training_state']['lr_schedule']
    trained=[state for state in after['optimizer_state_dict']['state'].values() if 'step' in state]
    assert max(float(state['step']) for state in trained)==2
    assert any(state['exp_avg'].abs().sum()>0 for state in trained)


def test_lr_reset_is_never_implicit_in_a_new_stage():
    args=stream_train.parser().parse_args(['--manifest','unused','--checkpoint-dir','unused','--reset-learning-rate'])
    with pytest.raises(ValueError,match='requires --resume'):stream_train._train(args)
