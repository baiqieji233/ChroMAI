from dataclasses import dataclass
from fractions import Fraction
from types import SimpleNamespace
import random

import numpy as np
import pytest
import torch

from maichart.composer import stream_train


def synthetic_bank(song, clock, audio_grid_seconds=.01):
    """Deterministic acoustic contract for the tiny 120-BPM training fixtures."""
    seconds=torch.arange(0.,song.audio.duration_sec,audio_grid_seconds,dtype=torch.float64)
    bands=torch.arange(1,239,dtype=torch.float32)
    features=torch.sin(seconds.float()[:,None]*bands[None]*.1)
    return features,seconds,seconds*2.


def ordered_parameter_initialization():
    """Exact new checkpoint interface, independent of a model's state_dict."""
    expected={}
    for name in ('audio_keys','audio_summary.0','sequence_input.0','reference_fusion.0',
                 'actor_input.0','interaction.0','predict.0','predict.3'):
        for tensor in ('weight','bias'):expected[f'phrase_context.ordered.{name}.{tensor}']='random'
    for name in ('audio_summary.2','sequence_input.2','sequence_norm','reference_fusion.2',
                 'actor_input.1','interaction.2','predict.2'):
        expected[f'phrase_context.ordered.{name}.weight']='ones'
        expected[f'phrase_context.ordered.{name}.bias']='zeros'
    for name in ('reference_output','anchor_output','interaction_output'):
        expected[f'phrase_context.ordered.{name}.weight']='zeros'
    for tensor in ('weight_ih_l0','weight_hh_l0','bias_ih_l0','bias_hh_l0'):
        expected[f'phrase_context.ordered.sequence_gru.{tensor}']='random'
    return expected


def assert_ordered_parameter_initialization(state):
    for key,initialization in ordered_parameter_initialization().items():
        value=state[key]
        assert torch.isfinite(value).all(),key
        if initialization=='zeros':assert torch.count_nonzero(value)==0,key
        elif initialization=='ones':assert torch.equal(value,torch.ones_like(value)),key
        else:assert torch.count_nonzero(value)>0,key


class Rows:
    def __init__(self):self.samples=[(i//3,i,i+1,1.) for i in range(11)]
    def __len__(self):return len(self.samples)
    def __getitem__(self,index):return index


def settings(**kwargs):
    return SimpleNamespace(batch_size=2,validation_batch_size=2,seed=19,device='cpu',
                           prefetch_batches=2,sample_workers=1,**kwargs)


def test_song_local_sampler_resume_cursor_and_dataclass_samples():
    ds=Rows()
    full=list(stream_train._LocalBatches(ds,2,19,3))
    resumed=list(stream_train._LocalBatches(ds,2,19,3,skip=2))
    assert resumed==full[2:]
    assert sorted(i for row in full for i in row)==list(range(len(ds)))
    @dataclass
    class Sample:
        song_index:int
    ds.samples=[Sample(i//3) for i in range(11)]
    assert list(stream_train._LocalBatches(ds,2,19,3))==full


def test_prefetch_preserves_order_and_does_not_consume_model_rng(monkeypatch):
    monkeypatch.setattr(stream_train,'collate_stream',lambda rows:torch.tensor(rows))
    # Layout prefetch expects dictionary batches for its byte accounting.
    monkeypatch.setattr(stream_train,'collate_stream',lambda rows:{'indices':torch.tensor(rows)})
    args=settings();ds=Rows()
    before=torch.get_rng_state().clone()
    loader=stream_train._loader(ds,args,2)
    with stream_train._prepared_batches(loader,args) as prepared:
        actual=[row['indices'].tolist() for row in prepared]
        assert prepared.stats()['depth']==2
    assert actual==list(stream_train._LocalBatches(ds,2,19,2))
    assert torch.equal(torch.get_rng_state(),before)


def test_resume_rejects_partial_gradient_windows_but_allows_short_epoch_tail():
    assert stream_train._validate_resume_cursor({'batch_in_epoch':4},9,4)==4
    assert stream_train._validate_resume_cursor({'batch_in_epoch':9},9,4)==9
    with pytest.raises(ValueError,match='accumulation'):
        stream_train._validate_resume_cursor({'batch_in_epoch':5},9,4)
    with pytest.raises(ValueError,match='outside'):
        stream_train._validate_resume_cursor({'batch_in_epoch':10},9,4)


def test_generation_evaluation_cannot_change_training_rng():
    python=random.getstate();numpy=np.random.get_state();cpu=torch.get_rng_state()
    with stream_train._preserve_rng('cpu'):
        random.random();np.random.random();torch.rand(10)
    assert random.getstate()==python
    assert np.array_equal(np.random.get_state()[1],numpy[1])
    assert torch.equal(torch.get_rng_state(),cpu)


def test_audio_grid_fresh_default_and_continuation_inheritance():
    from maichart.composer.stream_model import StreamComposerConfig
    assert stream_train._resolve_audio_grid(None)==.01
    legacy=StreamComposerConfig.from_dict({})
    assert legacy.audio_grid_seconds==.05
    assert stream_train._resolve_audio_grid(None,legacy)==.05
    assert stream_train._resolve_audio_grid(.05,legacy)==.05
    with pytest.raises(ValueError,match='inherits'):
        stream_train._resolve_audio_grid(.01,legacy)
    for invalid in (0.,-1.,float('nan'),float('inf')):
        with pytest.raises(ValueError,match='positive and finite'):
            stream_train._resolve_audio_grid(invalid)


def test_dataset_train_and_validation_receive_same_audio_grid(monkeypatch):
    captured=[]
    def capture(*args,**kwargs):
        captured.append(kwargs)
        return SimpleNamespace()
    monkeypatch.setattr(stream_train,'StreamDataset',capture)
    args=stream_train.parser().parse_args(['--manifest','train.json','--checkpoint-dir','unused',
                                         '--validation-manifest','dev.json','--audio-grid-seconds','.02'])
    stream_train._dataset(args)
    stream_train._dataset(args,validation=True)
    assert [item['audio_grid_seconds'] for item in captured]==[.02,.02]


def test_audio_grid_is_part_of_dataset_fingerprint(tmp_path):
    from maichart.composer.stream_model import StreamComposerConfig
    from maichart.composer.event_set_codec import EventSetCodec
    from maichart.composer.stream_data import DeltaVocabulary
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    args=SimpleNamespace(manifest=manifest,history_events=64,rollout_steps=16,window_seconds=8.,audio_grid_seconds=.05)
    ds=SimpleNamespace(codec=EventSetCodec.fit(['1']),delta_vocab=DeltaVocabulary([Fraction(1)]))
    old=stream_train._fingerprint(args,ds)
    legacy=stream_train._fingerprint(args,ds,legacy=True)
    args.audio_grid_seconds=.01
    assert stream_train._fingerprint(args,ds)!=old
    assert stream_train._fingerprint(args,ds,legacy=True)==legacy
    assert StreamComposerConfig().audio_grid_seconds==.01
    assert StreamComposerConfig.from_dict(StreamComposerConfig().to_dict()).audio_grid_seconds==.01


def test_interrupted_cpu_training_resumes_exact_optimizer_rng_and_cursor(tmp_path,monkeypatch):
    from maichart.composer import stream_data
    monkeypatch.setattr(stream_data,'build_audio_bank',synthetic_bank)
    from maichart.composer.stream_data import StreamDataset
    from maichart.composer import stream_checkpoint
    from maichart.event_timing import EventPoint,EventTempo
    audio=SimpleNamespace(duration_sec=2.)
    song=SimpleNamespace(audio=audio,tempo=[EventTempo(Fraction(0),120.)],first=0.)
    points=[EventPoint(Fraction(i,2),str(i%8+1)) for i in range(8)]
    base=SimpleNamespace(songs=[(song,points,12.,1.,None,torch.zeros(16),(),None)])
    def dataset(args,*,codec=None,delta_vocab=None,validation=False):
        result=StreamDataset(base=base,codec=codec,delta_vocab=delta_vocab,lead_in=4,targets=2,
            history_window_policy=args.history_window_policy,continuation_policy=args.continuation_policy,
            phrase_context_policy=args.phrase_context_policy)
        result.physical_policy=args.physical_policy
        return result
    monkeypatch.setattr(stream_train,'_dataset',dataset)
    monkeypatch.setattr(stream_train,'_condition_defaults',lambda ds:{})
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    def arguments(directory,resume=None):
        argv=['--manifest',str(manifest),'--checkpoint-dir',str(directory),'--device','cpu',
              '--precision','fp32','--epochs','1','--batch-size','1','--grad-accumulation','2',
              '--d-model','8','--nhead','2','--layers','1','--dim-feedforward','16',
              '--history-events','4','--rollout-steps','2','--symbolic-pretrain-epochs','1',
              '--checkpoint-interval-steps','1','--warmup-steps','0','--cpu-threads','1',
              '--recovery-interval','0','--generation-eval-songs','0']
        if resume:argv+=['--resume',str(resume)]
        return stream_train.parser().parse_args(argv)
    complete=tmp_path/'complete';interrupted=tmp_path/'interrupted'
    stream_train._train(arguments(complete))
    original=stream_checkpoint.save
    def interrupt(path,*args,**kwargs):
        result=original(path,*args,**kwargs)
        if str(path).startswith(str(interrupted)) and kwargs.get('optimizer_steps')==1:
            raise RuntimeError('simulated interruption after complete checkpoint')
        return result
    monkeypatch.setattr(stream_checkpoint,'save',interrupt)
    with pytest.raises(RuntimeError,match='simulated interruption'):
        stream_train._train(arguments(interrupted))
    monkeypatch.setattr(stream_checkpoint,'save',original)
    stream_train._train(arguments(interrupted,interrupted/'composer_last.pt'))
    left=stream_checkpoint.read(complete/'composer_last.pt')
    right=stream_checkpoint.read(interrupted/'composer_last.pt')
    assert left['optimizer_steps']==right['optimizer_steps']
    assert left['model_config']['phrase_context_policy']==right['model_config']['phrase_context_policy']=='shared_ordered_v4'
    assert left['model_config']['duration_audio_policy']==right['model_config']['duration_audio_policy']=='interval_v1'
    assert left['training_state']['batch_in_epoch']==right['training_state']['batch_in_epoch']
    assert torch.equal(left['training_state']['torch_rng'],right['training_state']['torch_rng'])
    assert left['training_state']['python_rng']==right['training_state']['python_rng']
    left_numpy,right_numpy=left['training_state']['numpy_rng'],right['training_state']['numpy_rng']
    assert left_numpy[0]==right_numpy[0] and np.array_equal(left_numpy[1],right_numpy[1])
    assert left_numpy[2:]==right_numpy[2:]
    assert left['optimizer_state_dict']['param_groups']==right['optimizer_state_dict']['param_groups']
    assert left['optimizer_state_dict']['state'].keys()==right['optimizer_state_dict']['state'].keys()
    for key,value in left['model_state_dict'].items():
        torch.testing.assert_close(value,right['model_state_dict'][key],atol=0,rtol=0)
    for key,state in left['optimizer_state_dict']['state'].items():
        for name,value in state.items():
            torch.testing.assert_close(value,right['optimizer_state_dict']['state'][key][name],atol=0,rtol=0)


def test_learning_stage_auto_repairs_finetune_and_resume_keeps_legacy():
    from maichart.composer.stream_model import StreamComposerConfig
    parent=StreamComposerConfig.from_dict({})
    args=SimpleNamespace()
    fresh=stream_train.resolve_learning_policies(args,parent)
    assert fresh['phrase_context_policy']=='shared_ordered_v4'
    assert fresh['history_window_policy']=='block_aligned_v2'
    assert fresh['transition_exit_weight']==0
    assert fresh['lane_load_policy']=='direct_buttons_v1'
    assert fresh['continuation_policy']=='pattern_runs_v2'
    assert fresh['motion_continuation_policy']=='context_motion_v3'
    assert fresh['recovery_policy']=='first_step_phrase_v6'
    assert fresh['song_choice_objective_policy']=='event_credit_v1'
    assert fresh['transition_policy']=='off'
    assert fresh['event_family_policy']=='shared_music_mixture_v2'
    assert fresh['literal_set_policy']=='permuted_unique_v1'
    assert fresh['duration_objective_policy']=='hold_crps_v2'
    assert fresh['curve_preview_policy']=='audio_counts_v1'
    assert fresh['recovery_target_policy']=='causal_correspondence_v1'
    assert fresh['gap_continuation_policy']=='off'
    assert fresh['prefix_augmentation_rate']==.15
    resumed=stream_train.resolve_learning_policies(args,parent,resume=True)
    assert resumed['phrase_context_policy']=='off'
    assert resumed['history_window_policy']=='rolling_legacy_v1'
    assert resumed['transition_exit_weight']==0
    assert resumed['lane_load_policy']=='off'
    assert resumed['recovery_policy']=='persistent_v1'
    assert resumed['literal_set_policy']=='canonical_v1'
    assert resumed['duration_objective_policy']=='categorical_v1'
    assert resumed['curve_preview_policy']=='off'
    assert resumed['transition_policy']=='off'
    assert resumed['prefix_augmentation_rate']==0
    with pytest.raises(ValueError,match='Resume inherits'):
        stream_train.resolve_learning_policies(SimpleNamespace(continuation_policy='typed_runs_v1'),parent,resume=True)


def test_learning_stage_migration_preserves_all_parent_tensors_and_zero_initializes_additions():
    from maichart.composer.stream_model import StreamComposer,StreamComposerConfig
    from maichart.composer.event_set_codec import EventSetCodec
    config=StreamComposerConfig.from_dict(dict(d_model=16,nhead=4,layers=1,dim_feedforward=32,motion_continuation_policy="motion_continuation_v1"))
    parent=StreamComposer(config,EventSetCodec.fit(['1','2']))
    policies=stream_train.resolve_learning_policies(SimpleNamespace(plan_kind_policy='off',training_window_policy='block_v1',phrase_execution_policy='off',detail_representation_policy='private_v1',duration_feature_policy='legacy_v1',song_profile_policy='masked_ordered_joint_v2',song_response_policy='off',phrase_context_policy='aligned_phrase_v3'),config)
    upgraded=stream_train.upgrade_learning_stage(parent,policies)
    for key,value in parent.state_dict().items():assert torch.equal(value,upgraded.state_dict()[key]),key
    additions=set(upgraded.state_dict())-set(parent.state_dict())
    expected={
        'touch_load_projection.weight','touch_load_projection.bias',
        'continuation_projection.weight','continuation_projection.bias',
        'keep_gap_audio.weight','keep_gap_audio.bias',
        'lane_load_projection.weight','action_continuation_projection.weight',
        'kind_continuation_projection.weight','gap_continuation_projection.weight',
        'motion_context_projection.weight','history_gap_projection.weight',
        'timing_grid.grid_head.weight','timing_grid.grid_head.bias',
        'timing_grid.bucket_head.weight','timing_grid.bucket_head.bias',
        'timing_grid.feature_key.weight','timing_grid.audio_projection.weight',
        'timing_grid.query_projection.weight',
        'timing_grid.query_interaction_projection.weight',
        'timing_grid.interval_candidate_projection.weight','timing_grid.interval_group_projection.weight',
        'event_family_head.weight','event_family_head.bias',
        'literal_route_head.weight','literal_route_head.bias','literal_family_embedding.weight',
        'family_audio_projection.weight','family_rhythm_projection.weight',
        'audio_event_family_head.weight','audio_event_family_head.bias',
        'family_continuity_head.weight','family_continuity_head.bias',
        'family_state_phrase.features.0.weight','family_state_phrase.features.0.bias',
        'family_state_phrase.features.2.weight','family_state_phrase.features.2.bias',
        'family_state_phrase.output.weight','family_state_phrase.output.bias',
        'literal_stop_head.weight','literal_stop_head.bias','literal_type_embedding.weight',
        'decoder.norm.weight','decoder.norm.bias',
        'music_exit_head.0.weight','music_exit_head.0.bias','music_exit_head.2.weight','music_exit_head.2.bias',
        'duration_audio.score.weight','phrase_context.null_reference',
    }
    for name in ('audio','writing','shared','intent','progress'):
        for layer in (0,2):
            for tensor in ('weight','bias'):expected.add(f'phrase_context.{name}.{layer}.{tensor}')
    for layer in (0,1):
        for tensor in ('weight','bias'):expected.add(f'phrase_context.style.{layer}.{tensor}')
    for name in ('query','key','predict','tracking','tracking_acoustic'):
        for tensor in ('weight','bias'):expected.add(f'phrase_context.{name}.{tensor}')
    for name in ('intent_output','progress_output','style_output','tracking_output'):
        expected.add(f'phrase_context.{name}.weight')
    ordered=ordered_parameter_initialization()
    expected.update(ordered)
    expected.update({'candidate_state_projection.weight','phrase_context.ordered.alignment_null',
        'phrase_context.ordered.alignment_query.weight','phrase_context.ordered.alignment_query.bias',
        'phrase_context.ordered.alignment_key.weight','phrase_context.ordered.alignment_key.bias',
        'phrase_context.ordered.alignment_output.weight'})
    expected={name for name in expected if not name.startswith((
        'keep_gap_audio.','family_continuity_head.','family_state_phrase.','family_rhythm_projection.','music_exit_head.'))}
    expected.update({'song_profile.network.0.weight','song_profile.network.0.bias',
                     'song_profile.network.2.weight','song_profile.network.2.bias'})
    expected.update({'curve_preview_head.weight','curve_preview_head.bias','curve_preview_trained_updates'})
    assert additions==expected
    assert_ordered_parameter_initialization(upgraded.state_dict())
    for key in additions:
        value=upgraded.state_dict()[key]
        if key.startswith('song_profile.'):
            assert torch.isfinite(value).all(),key
            continue  # Newly trained control distribution; does not alter the actor forward.
        if key.startswith('curve_preview_head.'):
            assert torch.isfinite(value).all() and torch.count_nonzero(value)>0,key
            continue  # Detached readout; parent actor tensors above remain exact.
        if key in ordered:
            continue  # Exact individual initialization contracts checked above.
        elif key.startswith('phrase_context.'):
            assert torch.isfinite(value).all(),key
            if key=='phrase_context.null_reference' or '_output.weight' in key:
                assert torch.count_nonzero(value)==0,key
            elif key.endswith('.2.weight') or key=='phrase_context.style.1.weight':
                assert torch.equal(value,torch.ones_like(value)),key
            elif key.endswith('.2.bias') or key=='phrase_context.style.1.bias':
                assert torch.count_nonzero(value)==0,key
            elif key.endswith('weight'):
                assert torch.count_nonzero(value)>0,key
        elif key in ('family_state_phrase.features.0.weight','family_state_phrase.features.0.bias'):
            assert torch.isfinite(value).all(),key
            assert torch.count_nonzero(value)>0,key
        elif key=='family_state_phrase.features.2.weight':
            assert torch.equal(value,torch.ones_like(value)),key
        elif key in ('family_state_phrase.features.2.bias','family_state_phrase.output.weight','family_state_phrase.output.bias'):
            assert torch.count_nonzero(value)==0,key
        elif key=='decoder.norm.weight':
            assert torch.equal(value,torch.ones_like(value)),key
        elif key=='music_exit_head.0.weight':
            assert torch.isfinite(value).all(),key
            assert torch.count_nonzero(value)>0,key
        elif key=='music_exit_head.0.bias':
            assert torch.isfinite(value).all(),key
        else:
            assert torch.count_nonzero(value)==0,key
    original=parent.state_dict
    parent.state_dict=lambda:{key:value for key,value in original().items() if key!='kind_head.weight'}
    with pytest.raises(ValueError,match='Unsafe stage migration'):
        stream_train.upgrade_learning_stage(parent,policies)


def test_causal_to_ordered_migration_preserves_complete_actor_outputs_with_real_sequence_tables(monkeypatch):
    from maichart.composer import stream_data
    from maichart.composer.stream_data import StreamDataset,collate_stream
    from maichart.composer.stream_model import StreamComposer,StreamComposerConfig
    from maichart.event_timing import EventPoint,EventTempo
    monkeypatch.setattr(stream_data,'build_audio_bank',synthetic_bank)
    song=SimpleNamespace(audio=SimpleNamespace(duration_sec=8.),tempo=[EventTempo(Fraction(0),120.)],first=0.)
    points=[EventPoint(Fraction(i,2),str(i%8+1)) for i in range(32)]
    base=SimpleNamespace(songs=[(song,points,12.,1.,None,torch.zeros(16),(),None)])
    dataset=StreamDataset(base=base,lead_in=4,targets=2,history_window_policy='block_aligned_v2',
        continuation_policy='pattern_runs_v2',phrase_context_policy='ordered_phrase_v2')
    batch=collate_stream([dataset[len(dataset)-1]])
    assert batch['phrase_sequence_kind'].size(0)>0
    assert batch['phrase_sequence_mask'].any()
    config=StreamComposerConfig(d_model=16,nhead=4,layers=1,dim_feedforward=32,dropout=0.,
        history_events=4,rollout_steps=2,delta_beats=('0',*map(str,dataset.delta_vocab.values)),
        phrase_context_policy='causal_phrase_v1')
    parent=StreamComposer(config,dataset.codec).eval()
    with torch.no_grad():parent.phrase_context.intent_output.weight.normal_(0,.03)
    before=torch.get_rng_state().clone()
    upgraded=stream_train.upgrade_learning_stage(parent,{'phrase_context_policy':'ordered_phrase_v2'}).eval()
    assert torch.equal(torch.get_rng_state(),before)
    additions=set(upgraded.state_dict())-set(parent.state_dict())
    assert additions==set(ordered_parameter_initialization())
    assert_ordered_parameter_initialization(upgraded.state_dict())
    for key,value in parent.state_dict().items():assert torch.equal(value,upgraded.state_dict()[key]),key
    with torch.no_grad():
        expected=parent(batch)
        observed=upgraded(batch)
    for key in ('kind_logits','delta_logits','shape_logits','phase_logits','hidden','action_context',
                'phrase_intent','phrase_prediction','phrase_reference_weights'):
        torch.testing.assert_close(expected[key],observed[key],atol=0,rtol=0)
    assert torch.count_nonzero(observed['phrase_ordered_actor_residual_rms'])==0


def test_cpu_finetune_inherits_parent_shape_resets_adam_and_records_parent_stage(tmp_path,monkeypatch):
    from maichart.composer import stream_data
    monkeypatch.setattr(stream_data,'build_audio_bank',synthetic_bank)
    from maichart.composer.stream_data import StreamDataset
    from maichart.composer import stream_checkpoint
    from maichart.event_timing import EventPoint,EventTempo
    song=SimpleNamespace(audio=SimpleNamespace(duration_sec=2.),tempo=[EventTempo(Fraction(0),120.)],first=0.)
    points=[EventPoint(Fraction(i,2),str(i%8+1)) for i in range(8)]
    base=SimpleNamespace(songs=[(song,points,12.,1.,None,torch.zeros(16),(),None)])
    def dataset(args,*,codec=None,delta_vocab=None,validation=False):
        result=StreamDataset(base=base,codec=codec,delta_vocab=delta_vocab,lead_in=args.history_events,targets=args.rollout_steps,
            history_window_policy=args.history_window_policy,continuation_policy=args.continuation_policy,
            phrase_context_policy=args.phrase_context_policy)
        result.physical_policy=args.physical_policy
        return result
    monkeypatch.setattr(stream_train,'_dataset',dataset)
    monkeypatch.setattr(stream_train,'_condition_defaults',lambda ds:{})
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    common=['--manifest',str(manifest),'--device','cpu','--precision','fp32','--epochs','1',
        '--batch-size','1','--grad-accumulation','1','--symbolic-pretrain-epochs','1',
        '--warmup-steps','0','--cpu-threads','1','--recovery-interval','0','--generation-eval-songs','0']
    parent_dir=tmp_path/'parent'
    baseline=[]
    for key,value in stream_train.LEGACY_LEARNING_POLICIES.items():baseline+=['--'+key.replace('_','-'),value]
    stream_train._train(stream_train.parser().parse_args(common+baseline+['--transition-exit-weight','0','--prefix-augmentation-rate','0',
        '--checkpoint-dir',str(parent_dir),'--d-model','8','--nhead','2','--layers','1','--dim-feedforward','16',
        '--history-events','4','--rollout-steps','2','--max-steps','2']))
    parent=stream_checkpoint.read(parent_dir/'composer_last.pt')
    output=tmp_path/'repaired'
    stream_train._train(stream_train.parser().parse_args(common+['--checkpoint-dir',str(output),
        '--init-checkpoint',str(parent_dir/'composer_last.pt'),'--max-steps','1']))
    repaired=stream_checkpoint.read(output/'composer_last.pt')
    assert parent['optimizer_steps']==2 and repaired['optimizer_steps']==1
    assert repaired['model_config']['d_model']==8 and repaired['model_config']['history_events']==4
    assert repaired['model_config']['continuation_policy']=='pattern_runs_v2'
    assert repaired['model_config']['lane_load_policy']=='direct_buttons_v1'
    metadata=repaired['training_state']['parent_stage']
    assert metadata['optimizer_steps']==2
    assert metadata['learning_policies']['history_window_policy']=='rolling_legacy_v1'
    assert all(state['step']==1 for state in repaired['optimizer_state_dict']['state'].values())


def test_history_and_continuation_fingerprint_changes_preserve_legacy_hash(tmp_path):
    from maichart.composer.event_set_codec import EventSetCodec
    from maichart.composer.stream_data import DeltaVocabulary
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    args=SimpleNamespace(manifest=manifest,history_events=64,rollout_steps=16,window_seconds=8.,audio_grid_seconds=.01)
    ds=SimpleNamespace(codec=EventSetCodec.fit(['1']),delta_vocab=DeltaVocabulary([Fraction(1)]))
    original=stream_train._fingerprint(args,ds)
    args.history_window_policy='rolling_legacy_v1';args.continuation_policy='off'
    assert stream_train._fingerprint(args,ds)==original
    args.history_window_policy='block_aligned_v2'
    aligned=stream_train._fingerprint(args,ds)
    assert aligned!=original
    args.continuation_policy='typed_runs_v1'
    assert stream_train._fingerprint(args,ds) not in {original,aligned}


def test_typed_continuation_stage_expands_only_weight_prefix_without_discarding_parent():
    from maichart.composer.stream_model import StreamComposer,StreamComposerConfig
    from maichart.composer.event_set_codec import EventSetCodec
    config=StreamComposerConfig.from_dict(dict(d_model=16,nhead=4,layers=1,dim_feedforward=32,
        continuation_policy='typed_runs_v1',motion_continuation_policy='motion_distinct_v2'))
    parent=StreamComposer(config,EventSetCodec.fit(['1','2']))
    with torch.no_grad():parent.continuation_projection.weight.copy_(torch.randn_like(parent.continuation_projection.weight))
    policies=stream_train.resolve_learning_policies(SimpleNamespace(plan_kind_policy='off',training_window_policy='block_v1',phrase_execution_policy='off',detail_representation_policy='private_v1',duration_feature_policy='legacy_v1',song_profile_policy='masked_ordered_joint_v2',song_response_policy='off',phrase_context_policy='aligned_phrase_v3'),config)
    upgraded=stream_train.upgrade_learning_stage(parent,policies)
    assert upgraded.continuation_projection.weight.shape==(16,16)
    assert torch.equal(upgraded.continuation_projection.weight[:,:8],parent.continuation_projection.weight)
    assert torch.count_nonzero(upgraded.continuation_projection.weight[:,8:])==0
    for key,value in parent.state_dict().items():
        if key!='continuation_projection.weight':assert torch.equal(value,upgraded.state_dict()[key]),key
    with pytest.raises(ValueError,match='Unsafe stage migration shape'):
        stream_train.upgrade_learning_stage(upgraded,{**policies,'continuation_policy':'typed_runs_v1'})


def test_prefix_augmentation_rate_validation_and_resume_contract():
    from maichart.composer.stream_model import StreamComposerConfig
    parent=StreamComposerConfig.from_dict({})
    with pytest.raises(ValueError,match='Resume inherits prefix_augmentation_rate'):
        stream_train.resolve_learning_policies(SimpleNamespace(prefix_augmentation_rate=.15),parent,resume=True)
    for invalid in (-.01,1.01,float('nan')):
        with pytest.raises(ValueError,match='Prefix augmentation rate'):
            stream_train.resolve_learning_policies(SimpleNamespace(prefix_augmentation_rate=invalid))


def test_exit_coverage_counts_note_wait_eos_without_padding_or_zero_weight():
    kinds=torch.tensor([[2,3,4,2,3]])
    batch=dict(target_kind=kinds,transition_exit_mask=torch.ones_like(kinds,dtype=torch.bool),
        transition_exit_labels=torch.ones_like(kinds,dtype=torch.bool),
        loss_mask=torch.tensor([[True,True,True,False,True]]),sample_weight=torch.tensor([[1.,1.,1.,1.,0.]]))
    assert {key:int(value) for key,value in stream_train._exit_coverage(batch).items()}==dict(exit_note_targets=1,exit_wait_targets=1,exit_eos_targets=1)


def test_training_collator_prefetch_resume_preserves_consumed_batch_seeds(monkeypatch):
    from maichart.composer import stream_augmentation
    import threading
    calls=[]
    def augment(dataset,items,*,rate,seed,input_policy):
        calls.append((seed,threading.current_thread().name))
        local=random.Random(seed)
        return [value+local.randrange(1000)*100 for value in items],{'attempted':len(items),'seed':seed}
    monkeypatch.setattr(stream_augmentation,'augment_stream_items',augment)
    monkeypatch.setattr(stream_train,'collate_stream',lambda rows:{'indices':torch.tensor(rows)})
    ds=Rows();epoch=3
    args=settings(prefix_augmentation_rate=.15)
    def consume(depth,skip=0,limit=None):
        args.prefetch_batches=depth
        with stream_train._prepared_batches(stream_train._loader(ds,args,epoch,skip),args) as prepared:
            rows=[]
            for batch in prepared:
                rows.append((batch['indices'].tolist(),dict(batch['_augmentation'])))
                if limit is not None and len(rows)==limit:break
            return rows
    before=torch.get_rng_state().clone()
    plain=consume(0)
    prefetched=consume(2)
    consumed=consume(2,limit=2)
    resumed=consume(2,skip=2)
    assert prefetched==plain
    assert consumed+resumed==plain
    assert [diag['seed'] for _,diag in resumed]==[args.seed+epoch*1000003+i for i in range(3,len(plain)+1)]
    assert torch.equal(before,torch.get_rng_state())
    assert any(name!=threading.main_thread().name for _,name in calls)


def test_validation_loader_keeps_plain_collate_without_augmentation(monkeypatch):
    from maichart.composer import stream_augmentation
    def forbidden(*args,**kwargs):raise AssertionError('Validation must not augment')
    monkeypatch.setattr(stream_augmentation,'augment_stream_items',forbidden)
    monkeypatch.setattr(stream_train,'collate_stream',lambda rows:{'indices':torch.tensor(rows)})
    args=settings(prefix_augmentation_rate=1.)
    rows=list(stream_train._loader(Rows(),args,2,validation=True))
    assert all('_augmentation' not in batch for batch in rows)
    assert [value for batch in rows for value in batch['indices'].tolist()]==list(range(11))
