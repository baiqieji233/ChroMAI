from fractions import Fraction as F
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from maichart.event_timing import EventPoint, EventTempo
from maichart.tempo_clock import TempoClock
from maichart.composer.stream_data import (
    BOS, EOS, NOTE, WAIT, DeltaVocabulary, StreamDataset, beat_at_second,
    build_steps, collate_stream, second_at_beat, StreamInferenceContext,
)


def test_plan_does_not_wrap_pre_audio_events_into_final_cell():
    from maichart.composer.stream_data import _audio_plan
    views=SimpleNamespace(context_points=[EventPoint(F(0),'1'),EventPoint(F(1),'2'),EventPoint(F(2),'3')],
        seconds=np.array([-.1,0.,.2]),trusted=np.ones(3,dtype=bool),
        window_weight=lambda left,right:np.ones_like(left))
    target,weight,exposure,blank_weight=_audio_plan(views,.103,.01,0.)
    assert target[:,1:].sum(0).tolist()==[1.,1.]
    assert target[0,2]==1 and target[-1,2]==0
    assert float(exposure[-1])==pytest.approx(.003)


def test_steps_preserve_fraction_notes_and_cover_intro_tail_and_tempo_changes():
    clock = TempoClock([EventTempo(F(0), 120.), EventTempo(F(4), 180.)], .5)
    points = [EventPoint(F(1, 3), '1'), EventPoint(F(23, 3), '2')]
    steps = build_steps(points, clock, 9.)
    assert steps[0].start == F(0)
    assert [s.end for s in steps if s.kind == NOTE] == [F(1, 3), F(23, 3)]
    assert steps[-1].kind == EOS and steps[-1].end_second == 9.
    assert all(0 <= s.end-s.start <= 2 for s in steps)
    assert all(a.end == b.start for a,b in zip(steps,steps[1:]))
    assert abs(second_at_beat(clock, beat_at_second(clock, 7.)) - 7.) < 1e-10
    vocab = DeltaVocabulary.fit([steps])
    restored = DeltaVocabulary.from_dict(vocab.to_dict())
    for step in steps:
        if step.kind != EOS:
            assert vocab.encode(step.end-step.start) == restored.encode(step.end-step.start)
    with pytest.raises(ValueError, match='out-of-vocabulary'):
        vocab.encode(F(1, 97))


def test_dense_windows_cover_each_target_with_causal_history(monkeypatch):
    from maichart.composer import stream_data
    class EmptyState:
        def __init__(self, *args): self.points=[]
        def append(self, point): self.points.append(point)
        def at(self, *args, **kwargs):
            return torch.zeros(1, 8), torch.ones(1, dtype=torch.bool), {}
    monkeypatch.setattr(stream_data, 'SustainStateIndex', EmptyState)
    monkeypatch.setattr(stream_data, 'IncrementalState', EmptyState)
    monkeypatch.setattr(stream_data, 'STATE_DIM', 8)
    bank = (torch.zeros(160,238), torch.arange(160,dtype=torch.float64)/20,
            torch.arange(160,dtype=torch.float64)/10)
    monkeypatch.setattr(StreamDataset, '_bank', lambda self, song: bank)
    points = [EventPoint(F(i, 8), str(i % 8 + 1)) for i in range(48)]
    audio = SimpleNamespace(duration_sec=8.)
    song = SimpleNamespace(audio=audio, tempo=[EventTempo(F(0),120.)], first=0.)
    base = SimpleNamespace(songs=[(song, points, 12., 1., None, torch.zeros(16), (), None)])
    dataset = StreamDataset(base=base, lead_in=4, targets=5, window_seconds=8., audio_grid_seconds=.05)
    assert dataset.coverage['supervised_steps'] == len(dataset.streams[0])
    assert len(dataset.samples) > 1
    assert sum((end-start)*weight for _,start,end,weight in dataset.samples) == pytest.approx(dataset.targets)
    assert sum(dataset.plan_weights) == pytest.approx(1.)
    for index, (song_index, start, end, _) in enumerate(dataset.samples):
        row = dataset[index]
        assert int(row['loss_mask'].sum()) == end-start
        first = max(0, start-dataset.lead_in)
        for i in range(len(row['history_kind'])):
            absolute = first+i
            assert row['history_kind'][i] == (BOS if absolute == 0 else dataset.streams[0][absolute-1].kind)
            assert row['target_kind'][i] == dataset.streams[0][absolute].kind
        assert not row['loss_mask'][:start-first].any()
    batch = collate_stream([dataset[0], dataset[1]])
    assert batch['song_audio'].shape == (1,160,238)
    assert batch['song_inverse'].tolist() == [0,0]
    assert not batch['history_phase_mask'][batch['history_kind'] != NOTE].any()
    assert batch['target_details'].min() == -1
    assert batch['action_state'].shape[:2] == batch['step_mask'].shape
    assert batch['step_index'][1,0] == max(0,dataset.samples[1][1]-dataset.lead_in)
    assert batch['audio_plan_target'].shape == (2,160,3)
    assert batch['target_second'].shape == batch['step_second'].shape
    _, start, _, _ = dataset.samples[1]
    context = StreamInferenceContext(song,dataset.codec,dataset.delta_vocab,torch.zeros(16),lead_in=4,audio_bank=bank)
    infer = context.batch(dataset.streams[0][:start],dataset.streams[0][start].start)
    teacher = dataset[1]
    target_row = int((~teacher['loss_mask']).sum())
    for key in ('history_kind','history_delta','history_shape','history_phase','history_atoms','history_phase_mask','step_beat','step_second','step_index'):
        torch.testing.assert_close(infer[key][0,-1], teacher[key][target_row])
    assert not infer['loss_mask'].any()
    dataset.set_symbolic_only(True)
    def forbidden(*args): raise AssertionError('symbolic stage must not load an audio bank')
    monkeypatch.setattr(StreamDataset,'_bank',forbidden)
    symbolic = collate_stream([dataset[0],dataset[1]])
    assert symbolic['song_audio'].shape[1:] == (1,238)
    assert symbolic['audio_plan_weight'].sum() == 0
    assert symbolic['audio_end_beat'].tolist() == [16.,16.]


def test_unknown_spans_mask_wait_and_eos_without_deleting_history():
    from maichart.composer.event_views import EventViews
    tempo = [EventTempo(F(0),120.)]
    clock = TempoClock(tempo, 0.)
    points = [EventPoint(F(1), '1'), EventPoint(F(8), '2')]
    views = EventViews.build(points, tempo, 0., 6., excluded=[(1.,5.)])
    steps = build_steps(views.context_points, clock, 6., views=views)
    assert len([s for s in steps if s.kind == NOTE]) == 2
    assert any(s.kind == WAIT and s.weight == 0 for s in steps)
    assert [s.weight for s in steps if s.kind == NOTE][-1] == 0
    from maichart.composer.stream_data import _audio_plan
    target, weight, exposure, blank_weight = _audio_plan(views, 6., .05)
    assert target.shape == (120,3)
    assert weight[30] == 0
    assert target[:,2].sum() == 1


def test_first_note_at_audio_zero_is_supervised():
    from maichart.composer.event_views import EventViews
    tempo=[EventTempo(F(0),120.)]
    points=[EventPoint(F(0),'1')]
    views=EventViews.build(points,tempo,0.,2.)
    steps=build_steps(views.context_points,TempoClock(tempo,0.),2.,views=views)
    assert steps[0].kind == NOTE and steps[0].start == steps[0].end == 0
    assert steps[0].weight > 0


def test_validation_oov_mask_and_known_atoms_use_literal_fallback(monkeypatch):
    from maichart.composer import stream_data
    class EmptyState:
        def __init__(self,*args): pass
        def at(self,*args,**kwargs): return torch.zeros(1,8),torch.ones(1,dtype=torch.bool),{}
    monkeypatch.setattr(stream_data,'STATE_DIM',8)
    monkeypatch.setattr(stream_data,'SustainStateIndex',EmptyState)
    bank=(torch.zeros(40,238),torch.arange(40,dtype=torch.float64)/20,torch.arange(40,dtype=torch.float64)/10)
    monkeypatch.setattr(StreamDataset,'_bank',lambda self,song:bank)
    song=SimpleNamespace(audio=SimpleNamespace(duration_sec=2.),tempo=[EventTempo(F(0),120.)],first=0.)
    def base(points):
        return SimpleNamespace(songs=[(song,points,12.,1.,None,torch.zeros(16),(),None)])
    train=StreamDataset(base=base([EventPoint(F(1,4),'1'),EventPoint(F(1,2),'2')]))
    dev=StreamDataset(base=base([EventPoint(F(1,4),'1/2'),EventPoint(F(1,2),'A1')]),
                      codec=train.codec,delta_vocab=train.delta_vocab,unknown_policy='mask')
    assert dev.oov['fallback_recovered'] == 1
    assert dev.oov['action_oov'] == dev.oov['masked_steps'] == 1
    row=dev[0]
    assert row['target_shape'][0] == 0
    assert row['target_details'][0, :3].tolist() == [train.codec._atoms['1'],train.codec._atoms['2'],0]
    assert row['sample_weight'][0] > 0 and row['sample_weight'][1] == 0
    assert row['history_kind'][2] == NOTE and not row['history_atoms'][2].any()
    assert not row['history_phase_mask'][2].any()
    with pytest.raises(KeyError):
        StreamDataset(base=base([EventPoint(F(1,4),'A1')]),codec=train.codec,delta_vocab=train.delta_vocab)


def test_song_bank_cache_is_safe_for_prefetch_and_recovery(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import time
    from maichart.composer import stream_data
    audio=SimpleNamespace(duration_sec=2.,source='one-song')
    song=SimpleNamespace(audio=audio,tempo=[EventTempo(F(0),120.)],first=0.,meters=[])
    base=SimpleNamespace(songs=[(song,[EventPoint(F(0),'1')],12.,1.,None,torch.zeros(16),(),None)])
    dataset=StreamDataset(base=base)
    calls=[]
    def build(song,clock,audio_grid_seconds):
        calls.append(1);time.sleep(.01)
        return (torch.zeros(2,238),torch.zeros(2),torch.zeros(2))
    monkeypatch.setattr(stream_data,'build_audio_bank',build)
    with ThreadPoolExecutor(max_workers=4) as pool:
        banks=list(pool.map(dataset._bank,[0]*8))
    assert len(calls) == 1 and all(bank is banks[0] for bank in banks)


def test_dense_and_sparse_time_windows_have_equal_unmasked_target_mass():
    song=SimpleNamespace(audio=SimpleNamespace(duration_sec=16.),tempo=[EventTempo(F(0),120.)],first=0.)
    points=[EventPoint(F(i,8),str(i%8+1)) for i in range(64)]
    base=SimpleNamespace(songs=[(song,points,12.,1.,None,torch.zeros(16),(),None)])
    dataset=StreamDataset(base=base,targets=16,window_seconds=8.)
    masses={}
    for song_index,start,end,weight in dataset.samples:
        window=int(dataset.streams[song_index][start].start_second//8.)
        masses[window]=masses.get(window,0.)+(end-start)*weight
    assert len(masses) == 2
    assert masses[0] == pytest.approx(16.)
    assert masses[1] == pytest.approx(16.)
    assert dataset.coverage['empty_time_windows'] == 0


def test_incremental_inference_matches_rebuilt_prefix_and_full_sustain_state():
    from maichart.composer.event_set_codec import EventSetCodec
    from maichart.composer.stream_data import StreamStep
    from maichart.generation.sustain_state import SustainStateIndex
    tempo=[EventTempo(F(0),120.)]
    song=SimpleNamespace(audio=SimpleNamespace(duration_sec=6.),tempo=tempo,first=0.)
    steps=[StreamStep(NOTE,F(0),F(0),0.,0.,'1h[4:4]'),
           StreamStep(WAIT,F(0),F(1),0.,.5),
           StreamStep(NOTE,F(1),F(3,2),.5,.75,'2'),
           StreamStep(WAIT,F(3,2),F(2),.75,1.)]
    codec=EventSetCodec.fit(['1h[4:4]','2'])
    vocab=DeltaVocabulary.fit([steps])
    bank=(torch.zeros(120,238),torch.arange(120,dtype=torch.float64)/20,torch.arange(120,dtype=torch.float64)/10)
    cached=StreamInferenceContext(song,codec,vocab,torch.zeros(16),lead_in=2,audio_bank=bank)
    committed=StreamInferenceContext(song,codec,vocab,torch.zeros(16),lead_in=2,audio_bank=bank)
    full=SustainStateIndex([EventPoint(F(0),'1h[4:4]'),EventPoint(F(3,2),'2')],tempo,0.)
    for count in range(len(steps)+1):
        if count: committed.commit(steps[count-1])
        prefix=steps[:count];cursor=prefix[-1].end if prefix else F(0)
        result=cached.batch(prefix,cursor)
        rebuilt=StreamInferenceContext(song,codec,vocab,torch.zeros(16),lead_in=2,audio_bank=bank).batch(prefix,cursor)
        for key in ('history_kind','history_delta','history_shape','history_phase','history_atoms','history_phase_mask',
                    'step_beat','step_second','step_index','action_state','action_state_mask'):
            torch.testing.assert_close(result[key],rebuilt[key],rtol=0,atol=0)
            torch.testing.assert_close(result[key],committed.current_batch()[key],rtol=0,atol=0)
        if prefix:
            cutoff=cursor+F(1,10**12) if prefix[-1].kind == NOTE else cursor
            state,mask,_=full.at(cursor,include_current=True,visible_before_beat=cutoff,
                                 visibility_version='causal_v2',include_metadata=False)
            actual=result['action_state'][0,-1]
            valid=~result['action_state_mask'][0,-1]
            torch.testing.assert_close(actual[valid],state[~mask],rtol=0,atol=1e-6)
    # A shortened or changed prefix rebuilds rather than retaining future state.
    shorter=cached.batch(steps[:1],steps[0].end)
    assert shorter['history_kind'].shape[1] == 2


def test_symmetric_history_phase_mask_is_shared_with_inference_and_wait(monkeypatch):
    from maichart.composer import stream_data
    from maichart.composer.stream_data import StreamStep
    class EmptyState:
        def __init__(self,*args): self.points=[]
        def append(self,point): self.points.append(point)
        def at(self,*args,**kwargs): return torch.zeros(1,8),torch.ones(1,dtype=torch.bool),{}
    monkeypatch.setattr(stream_data,'STATE_DIM',8)
    monkeypatch.setattr(stream_data,'SustainStateIndex',EmptyState)
    monkeypatch.setattr(stream_data,'IncrementalState',EmptyState)
    bank=(torch.zeros(120,238),torch.arange(120,dtype=torch.float64)/20,torch.arange(120,dtype=torch.float64)/10)
    monkeypatch.setattr(StreamDataset,'_bank',lambda self,song:bank)
    song=SimpleNamespace(audio=SimpleNamespace(duration_sec=6.),tempo=[EventTempo(F(0),120.)],first=0.)
    base=SimpleNamespace(songs=[(song,[EventPoint(F(0),'1/5'),EventPoint(F(4),'2')],12.,1.,None,torch.zeros(16),(),None)])
    ds=StreamDataset(base=base,audio_grid_seconds=.05)
    steps=ds.streams[0]
    assert steps[0].kind==NOTE and steps[1].kind==WAIT
    expected=torch.tensor(ds.codec.encode('1/5')['phase_mask'])
    assert expected.sum()==2
    item=ds[0]
    assert torch.equal(item['history_phase_mask'][1],expected)
    assert not item['history_phase_mask'][2].any()
    context=StreamInferenceContext(song,ds.codec,ds.delta_vocab,torch.zeros(16),audio_bank=bank,audio_grid_seconds=.05)
    context.commit(steps[0])
    assert torch.equal(context.current_batch()['history_phase_mask'][0,-1],expected)
    context.commit(steps[1])
    assert not context.current_batch()['history_phase_mask'][0,-1].any()


def test_real_narrow_waveform_attack_survives_cell_pooling_and_tail_exposure(tmp_path,monkeypatch):
    import hashlib
    import soundfile as sf
    from maichart.composer import stream_data
    from maichart.composer.acoustics import native_dynamics
    from maichart.composer.event_views import EventViews
    sr=8000;duration=.103
    signal=np.zeros(round(sr*duration),dtype=np.float32)
    signal[56:64]=.8  # 7–8 ms pulse; misses the 0 and 10 ms attack samples.
    path=tmp_path/'narrow.wav'
    sf.write(path,signal,sr,subtype='FLOAT')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    audio=SimpleNamespace(source=str(path),duration_sec=duration,cache_dir=str(tmp_path),
                          metadata={'audio_sha256':digest,'normalization_scale':np.ones(98,dtype=np.float32)})
    tempo=[EventTempo(F(0),120.)]
    song=SimpleNamespace(audio=audio,tempo=tempo,first=0.)
    # The cell sampler receives no pre-existing attack nodes: this verifies
    # the actual waveform-derived 5 ms path rather than a synthetic peak row.
    nodes=np.zeros((4,120),dtype=np.float32);nodes[1,97]=5.
    monkeypatch.setattr(stream_data,'composer_audio_nodes',lambda song:
        (nodes,np.array([0.,.005,.009,duration],dtype=np.float64)))
    bank,seconds,_=stream_data.build_audio_bank(song,TempoClock(tempo,0.),.01)
    assert np.allclose(native_dynamics(audio,np.array([0.,.01]))[:,0],0.)
    assert bank[0,104+92] > 0
    assert bank[0,104+97] == 5.  # Native flux node also survives inside the cell.
    legacy,_,_=stream_data.build_audio_bank(song,TempoClock(tempo,0.),.05)
    assert legacy[0,104+92] == 0  # Old scratch checkpoints retain point sampling.
    assert len(seconds)==11 and seconds[-1] == pytest.approx(.1)
    views=EventViews.build([],tempo,0.,duration)
    target,weight,exposure,blank_weight=stream_data._audio_plan(views,duration,.01)
    assert len(target)==len(seconds)==len(exposure)
    assert exposure[-1].item() == pytest.approx(.003,abs=1e-6)
    assert float(exposure.sum()) == pytest.approx(duration,abs=1e-6)
    assert float(weight.sum()) == pytest.approx(1.)


def test_centered_blank_window_masks_edges_and_unknown_without_masking_counts():
    from maichart.composer.stream_data import _audio_plan
    from maichart.composer.event_views import EventViews
    tempo=[EventTempo(F(0),120.)]
    points=[EventPoint(F(1),'1'),EventPoint(F(3),'2')]
    views=EventViews.build(points,tempo,0.,2.,excluded=[(1.,1.1)])
    target,count_weight,exposure,blank_weight=_audio_plan(views,2.,.05,.5)
    old,old_weight,_,old_blank_weight=_audio_plan(views,2.,.05,0.)
    torch.testing.assert_close(target[:,1:],old[:,1:])
    torch.testing.assert_close(count_weight,old_weight)
    torch.testing.assert_close(old_weight,old_blank_weight)
    assert blank_weight[0] == 0  # Window extends before audio.
    assert blank_weight[17] == 0  # Window overlaps unknown interval.
    assert count_weight[17] > 0  # The cell itself remains known.
    assert target[9,0] == 0  # Its centered window contains the .5 s onset.
    assert target[29,0] == 0  # Its centered window contains the 1.5 s onset.
    assert blank_weight[29] > 0
    assert exposure.sum().item() == pytest.approx(2.)


def test_authored_state_cache_is_exact_and_byte_bounded(monkeypatch):
    from maichart.composer import stream_data
    class CountingState:
        calls=0
        constructions=0
        def __init__(self,*args): type(self).constructions+=1
        def at(self,beat,*args,**kwargs):
            type(self).calls+=1
            return torch.full((1,8),float(beat)),torch.zeros(1,dtype=torch.bool),{}
    monkeypatch.setattr(stream_data,'STATE_DIM',8)
    monkeypatch.setattr(stream_data,'SustainStateIndex',CountingState)
    bank=(torch.zeros(40,238),torch.arange(40,dtype=torch.float64)/20,torch.arange(40,dtype=torch.float64)/10)
    monkeypatch.setattr(StreamDataset,'_bank',lambda self,song:bank)
    song=SimpleNamespace(audio=SimpleNamespace(duration_sec=2.),tempo=[EventTempo(F(0),120.)],first=0.)
    points=[EventPoint(F(i,4),str(i%8+1)) for i in range(12)]
    base=SimpleNamespace(songs=[(song,points,12.,1.,None,torch.zeros(16),(),None)])
    cached=StreamDataset(base=base,lead_in=4,targets=4,audio_cache_gb=.01,audio_grid_seconds=.05)
    first=cached[0]; calls=CountingState.calls
    cached[1]
    assert CountingState.calls == calls
    stats=cached.cache_stats()['state']
    assert stats['hits'] == 1 and 0 < stats['estimated_bytes'] <= stats['budget_bytes']
    uncached=StreamDataset(base=base,lead_in=4,targets=4,audio_cache_gb=0.,audio_grid_seconds=.05)
    direct=uncached[0]
    for key,value in first.items():
        if torch.is_tensor(value):
            assert torch.equal(value,direct[key]),key
    original = cached[0]
    mutable = cached[0]
    for key in cached._symbols[0][0]:
        mutable[key].fill_(0)
    fresh = cached[0]
    for key in cached._symbols[0][0]:
        assert torch.equal(original[key],fresh[key]),key
    symbol_stats = cached.cache_stats()['symbol']
    assert 0 < symbol_stats['estimated_bytes'] <= symbol_stats['budget_bytes']
    assert uncached.cache_stats()['symbol']['entries'] == 0
    torch.testing.assert_close(first['action_state'],direct['action_state'],rtol=0,atol=0)
    assert torch.equal(first['action_state_mask'],direct['action_state_mask'])
    assert uncached.cache_stats()['state']['uncached'] == 1
    small=StreamDataset(base=base,lead_in=4,targets=4,audio_cache_gb=.01,audio_grid_seconds=.05)
    small._state_budget=1000
    small[0]
    constructions=CountingState.constructions
    small[1]
    assert CountingState.constructions == constructions
    assert small.cache_stats()['state']['hits'] == 1
    assert small.cache_stats()['state']['uncached'] == 1


def test_state_visibility_never_includes_arbitrarily_close_target():
    from maichart.composer.stream_data import StreamStep
    class Index:
        def at(self,beat,**kwargs):
            self.cutoff=kwargs['visible_before_beat']
            return torch.zeros(1,8),torch.ones(1,dtype=torch.bool),{}
    tiny=F(1,10**13)
    stream=[StreamStep(NOTE,F(0),F(0),0.,0.,'1'),StreamStep(NOTE,F(0),tiny,0.,1e-13,'2')]
    index=Index()
    StreamDataset._state_at(index,stream,1)
    assert index.cutoff == tiny
