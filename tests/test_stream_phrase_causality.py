"""Real stream-row and active model-path causality regression tests."""
from fractions import Fraction
from types import SimpleNamespace
import random

import pytest
import torch

from maichart.event_timing import EventPoint, EventTempo
from maichart.composer.event_set_codec import EventSetCodec
from maichart.composer.stream_data import StreamDataset, StreamInferenceContext, collate_stream
from maichart.composer.stream_data import recovery_sample
from maichart.composer.stream_augmentation import augment_stream_row
from maichart.composer.stream_model import StreamComposer, StreamComposerConfig
from maichart.composer.stream_phrase import PHRASE_ROW_FIELDS, PhraseIndex, _event_vector
from maichart.composer.stream_phrase_model import CausalPhraseContext


INPUT_FIELDS = tuple(key for key in PHRASE_ROW_FIELDS if not key.startswith('target_'))


@pytest.fixture
def paired_datasets(monkeypatch):
    torch.set_num_threads(2)
    beats = torch.arange(0.,24.,.1,dtype=torch.float64)
    values = torch.zeros(len(beats),238)
    values[:,104:184] = torch.sin(beats.float()[:,None]/3)
    values[:,196] = torch.cos(beats.float()).abs()
    values[:,197] = .5+torch.sin(beats.float()/4)*.2
    values[:,198] = torch.sin(beats.float()*2).abs()
    bank = values,beats/2,beats
    monkeypatch.setattr(StreamDataset,'_bank',lambda self,song:bank)
    song = SimpleNamespace(audio=SimpleNamespace(duration_sec=12.),tempo=[EventTempo(Fraction(0),120.)],first=0.)
    codec = EventSetCodec.fit([str(i) for i in range(1,9)])
    def make(changed):
        points = [EventPoint(Fraction(i),str((i+(3 if changed and i>=13 else 0))%8+1)) for i in range(20)]
        base = SimpleNamespace(songs=[(song,points,12.,1.,None,torch.zeros(16),(),None)])
        return StreamDataset(base=base,codec=codec,lead_in=8,targets=4,audio_grid_seconds=.05,
            history_window_policy='block_aligned_v2',continuation_policy='pattern_runs_v2',
            phrase_context_policy='causal_phrase_v1')
    return make(False),make(True),song,bank


def test_real_dataset_rows_and_incremental_inference_agree_without_future_chart(paired_datasets):
    first,second,song,bank = paired_datasets
    assert first.samples == second.samples
    compared = 0
    changed_supervision = False
    for sample_index in range(len(first)):
        a,b = first[sample_index],second[sample_index]
        for local,query in enumerate(a['step_index'].tolist()):
            if query>13:continue
            for key in INPUT_FIELDS:
                torch.testing.assert_close(a[key][local],b[key][local],rtol=0,atol=0,msg=key)
            if not torch.equal(a['target_phrase_features'][local],b['target_phrase_features'][local]):
                changed_supervision = True
            context = StreamInferenceContext(song,first.codec,first.delta_vocab,first.conditions[0],
                lead_in=8,audio_bank=bank,history_window_policy='block_aligned_v2',rollout_steps=4,
                continuation_policy='pattern_runs_v2',phrase_context_policy='causal_phrase_v1')
            for step in first.streams[0][:query]:context.commit(step)
            inference = context.current_batch()
            for key in INPUT_FIELDS:
                torch.testing.assert_close(a[key][local],inference[key][0,-1],rtol=0,atol=0,msg=key)
            assert not inference['target_phrase_mask'].any()
            compared += 1
    assert compared>13 and changed_supervision


def test_nonzero_full_model_cannot_read_phrase_supervision(paired_datasets):
    dataset,alternate,_,_ = paired_datasets
    batch = collate_stream([dataset[3]])
    config = StreamComposerConfig(d_model=16,nhead=4,layers=1,dim_feedforward=32,dropout=0.,
        audio_dim=238,condition_dim=16,audio_summary_tokens=4,rollout_steps=4,
        delta_beats=tuple(str(value) for value in dataset.delta_vocab.values),max_event_atoms=4,
        history_window_policy='block_aligned_v2',continuation_policy='pattern_runs_v2',
        phrase_context_policy='causal_phrase_v1')
    torch.manual_seed(824)
    model = StreamComposer(config,dataset.codec).eval()
    with torch.no_grad():
        # Activate every previously zero residual path to avoid a vacuous proof.
        for parameter in model.phrase_context.parameters():parameter.normal_(0.,.2)
        before = model(batch)
        changed = dict(batch,target_phrase_features=torch.full_like(batch['target_phrase_features'],float('nan')),
            target_phrase_mask=~batch['target_phrase_mask'],target_phrase_weight=batch['target_phrase_weight']*100.)
        after = model(changed)
    for key in ('hidden','kind_logits','delta_logits','shape_logits','phrase_intent','phrase_prediction'):
        torch.testing.assert_close(before[key],after[key],rtol=0,atol=0,msg=key)
    # True teacher-forced future history exists later in this same window.
    # It must not flow backwards through the phrase memory or decoder mask.
    alternate_batch = collate_stream([alternate[3]])
    with torch.no_grad():future_changed = model(alternate_batch)
    visible = batch['step_index']<=13
    for key in ('hidden','kind_logits','delta_logits','phrase_intent','phrase_prediction'):
        torch.testing.assert_close(before[key][visible],future_changed[key][visible],rtol=0,atol=0,msg=key)
    assert (~visible).any()
    assert not torch.equal(before['hidden'][~visible],future_changed['hidden'][~visible])
    # The phrase path must actually contribute to hidden states in this test.
    with torch.no_grad():
        for layer in (model.phrase_context.intent_output,model.phrase_context.progress_output,
                      model.phrase_context.style_output,model.phrase_context.tracking_output):layer.weight.zero_()
        disabled = model(batch)
    assert not torch.equal(before['hidden'],disabled['hidden'])


def test_directionless_chords_and_nonbutton_steps_break_motion():
    def step(text,kind=2):return SimpleNamespace(kind=kind,start=Fraction(0),end=Fraction(1),text=text)
    _,lane = _event_vector(step('1/5'),0.,8.,.2)
    assert lane is None
    for action in (step('',3),step('A1')):
        value,lane = _event_vector(action,0.,8.,.2)
        assert lane is None
        assert not value.reshape(4,24)[:,13:16].any()


def test_music_layout_is_shared_but_committed_prefix_is_independent(paired_datasets):
    dataset,_,_,bank = paired_datasets
    first,second = PhraseIndex(bank,24.),PhraseIndex(bank,24.)
    assert first.layout is second.layout
    assert not first.boundaries.flags.writeable and not first.audio.flags.writeable
    first.append(dataset.streams[0][0])
    assert first.count==1 and second.count==0
    assert not second.row(0,0)['phrase_progress'].any()


def test_augmentation_rebuilds_phrase_and_disables_untrusted_future_aux(paired_datasets):
    dataset,_,_,_ = paired_datasets
    original = dataset[3]
    changed,mode,_ = augment_stream_row(dataset,3,rng=random.Random(8),mode='abab')
    assert mode=='abab'
    assert not torch.equal(original['history_atoms'],changed['history_atoms'])
    assert not torch.equal(original['phrase_progress'],changed['phrase_progress'])
    assert not changed['target_phrase_mask'].any()
    assert not changed['target_phrase_weight'].any()
    torch.testing.assert_close(original['sample_weight'].sum(),changed['sample_weight'].sum(),rtol=0,atol=0)


def test_recovery_and_prefix_rewrite_use_actual_committed_events(paired_datasets):
    dataset,_,song,bank = paired_datasets
    source = dataset.streams[0]
    query = 13
    from dataclasses import replace
    prefix = list(source[:query])
    prefix[10] = replace(prefix[10],text='8')
    row = recovery_sample(dataset,0,prefix,source[query])
    fresh = PhraseIndex(bank,24.,prefix).row(query,prefix[-1].end)
    for key in INPUT_FIELDS:
        torch.testing.assert_close(row[key][-1],fresh[key],rtol=0,atol=0,msg=key)
    assert not row['target_phrase_mask'].any() and not row['target_phrase_weight'].any()
    context = StreamInferenceContext(song,dataset.codec,dataset.delta_vocab,dataset.conditions[0],
        lead_in=8,audio_bank=bank,phrase_context_policy='causal_phrase_v1')
    context.batch(source[:query],source[query].start)
    rewritten = context.batch(prefix,prefix[-1].end)
    for key in INPUT_FIELDS:
        torch.testing.assert_close(rewritten[key][0,-1],fresh[key],rtol=0,atol=0,msg=key)


@pytest.mark.parametrize('supervise',(False,True))
def test_bulk_rows_are_bitwise_equal_to_scalar_visibility_reference(paired_datasets,supervise):
    dataset,_,_,bank = paired_datasets
    steps = dataset.streams[0]
    index = PhraseIndex(bank,24.,steps)
    for first,end in ((0,len(steps)),(3,17),(12,14)):
        bulk = index.rows(steps,first,end,supervise=supervise)
        scalar = [index.row(query,steps[query].start,supervise=supervise) for query in range(first,end)]
        for key in PHRASE_ROW_FIELDS:
            torch.testing.assert_close(bulk[key],torch.stack([row[key] for row in scalar]),rtol=0,atol=0,msg=key)
    # A new commit invalidates only the affected segment's packed values.
    incremental = PhraseIndex(bank,24.)
    for query,step in enumerate(steps[:-1]):
        incremental.append(step)
        cursor = float(step.end)
        bulk = incremental.rows_at([query+1],[cursor],supervise=supervise)
        scalar = incremental.row(query+1,cursor,supervise=supervise)
        for key in PHRASE_ROW_FIELDS:
            torch.testing.assert_close(bulk[key][0],scalar[key],rtol=0,atol=0,msg=key)
