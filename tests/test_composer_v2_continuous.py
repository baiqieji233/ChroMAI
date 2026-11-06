from fractions import Fraction
from pathlib import Path
import numpy as np
import pytest
import torch
from maichart.event_timing import EventPoint,EventTempo
from maichart.tempo_clock import TempoClock
from maichart.composer_v2.stream_codec import StreamCodec
from maichart.composer_v2.stream_data import StreamDataset,collate_stream
from maichart.composer_v2.data import ComposerV2Song
from maichart.composer_v2.model import ComposerV2,ComposerV2Config


class Audio:
    duration_sec=100.
    identity={'fixture':True}
    def sample(self,times):
        x=np.asarray(times)
        return np.tile(np.sin(x[:,None]),(1,98)).astype('float32'),np.tile(np.cos(x[:,None]),(1,1024)).astype('float32'),np.ones(len(x),bool)
    def summary(self,n):
        times=np.linspace(0,100,8);a,b,c=self.sample(times)
        return dict(mel=a,semantic=b,semantic_valid=c,times_sec=times)


def setup():
    data=StreamDataset.__new__(StreamDataset)
    data.codec=StreamCodec.fit(['1','2','3h[4:1]','1-5[8:1]'],phases=[Fraction(1,101)])
    data.max_tokens=256;data.history_beats=16;data.window_beats=16;data.future_beats=4
    points=[EventPoint(Fraction(15)+Fraction(i,7),'1' if i%2 else '2') for i in range(12)]
    song=ComposerV2Song('fixture',Path('chart'),Path('audio'),5,13.,points,
        TempoClock([EventTempo(Fraction(0),120.),EventTempo(Fraction(17),150.)],.25),Fraction(240),np.zeros(16,'float32'),audio=Audio())
    song.summary={**song.audio.summary(160),'beats':np.linspace(0,200,8)}
    return data,song


def test_exact_unusual_phases_cross_arbitrary_computational_boundaries():
    data,song=setup();codec=data.codec
    encoded=codec.encode_points(song.points)
    assert codec.decode_points(encoded)==song.points
    for cut in (1,3,7,10):
        assert codec.encode_points(song.points[:cut])+codec.encode_points(song.points[cut:])==encoded
    assert codec.encode_points(song.points,origin=0)==codec.encode_points(song.points,origin=16)
    for phase in (Fraction(1,5),Fraction(3,7),Fraction(9,13),Fraction(1,101)):
        time=Fraction(16)+phase;ids=codec._time(time)
        for i,token in enumerate(ids):assert token in codec.time_choices(ids[:i],Fraction(15),Fraction(19))


def test_codec_serializes_after_populating_note_token_cache():
    import pickle

    data, song = setup()
    expected = data.codec.encode_points(song.points)
    restored = pickle.loads(pickle.dumps(data.codec))
    assert restored.to_dict() == data.codec.to_dict()
    assert restored.encode_points(song.points) == expected


def test_only_real_song_end_has_eos_target_and_history_is_causal():
    data,song=setup()
    middle=data.build_stream(song,song.points[:4],song.points[3].beat,target=song.points[4:8],training=True)
    ending=data.build_stream(song,song.points[:8],song.points[7].beat,target=song.points[8:],ended=True,training=True)
    assert data.codec.eos_id not in middle['labels'].tolist()
    assert ending['labels'].tolist().count(data.codec.eos_id)==1
    assert int((middle['labels']==data.codec.event_end_id).sum())==4
    prefix=middle['input_ids'][:(middle['labels']!=-100).nonzero()[0].item()+1]
    assert max(data.codec.token_clocks(prefix))<=float(song.points[3].beat)


@pytest.mark.parametrize('backend', ['mert', 'native'])
def test_reference_descriptors_keep_exact_scores_and_use_bounded_audio_reads(backend):
    from maichart.composer_v2.audio import AudioFeatures

    data, song = setup()
    rng = np.random.default_rng(7)
    times = np.linspace(0, 100, 200)
    audio = AudioFeatures(rng.normal(size=(200, 98)).astype(np.float32), times,
                          rng.normal(size=(200, 1024)).astype(np.float16), times,
                          100., {'backend': backend})

    class CountingAudio:
        def __init__(self):
            self.rows = []

        def sample(self, seconds):
            self.rows.append(len(seconds))
            return audio.sample(seconds)

    song.audio = CountingAudio()
    song.point_beats = [Fraction(i, 4) + Fraction(i % 3, 7) for i in range(360)]
    song.point_beats.sort()
    song.points = [EventPoint(beat, '1') for beat in song.point_beats]
    prefix_end = 290
    cursor = song.point_beats[prefix_end - 1]
    beats = [beat for beat in song.point_beats[:prefix_end] if beat <= cursor - 32]
    candidates = [beats[index] for index in np.linspace(0, len(beats) - 1, 64, dtype=int)]
    reference = {}
    for beat in [cursor, *candidates]:
        seconds = song.clock.at_floats(float(beat) + np.arange(64) / 4)
        mel, semantic, valid = audio.sample(seconds)
        vec = (semantic if np.any(valid) else mel).mean(0)
        reference[beat] = vec / max(float(np.linalg.norm(vec)), 1e-8)
    expected = max(candidates, key=lambda beat: float(np.dot(reference[cursor], reference[beat])))
    actual = data._reference_stream(song, song.points, cursor, prefix_end=prefix_end)
    assert actual == expected
    assert set(song.descriptor_cache) == set(reference)
    for beat, descriptor in reference.items():
        np.testing.assert_array_equal(song.descriptor_cache[beat], descriptor)
    assert max(song.audio.rows) <= 1024
    assert len(song.audio.rows) < 10
    assert sum(song.audio.rows) < 65 * 64
    song.audio.rows.clear()
    assert data._reference_stream(song, song.points, cursor, prefix_end=prefix_end) == expected
    assert not song.audio.rows


def test_indexed_prefix_matches_literal_prefix_with_carry_reference_and_future_targets():
    from maichart.generation.sustain_state import SustainStateIndex

    data, song = setup()
    song.points = [EventPoint(Fraction(0), '1'), EventPoint(Fraction(1), '2'),
                   EventPoint(Fraction(10), '3h[4:40]')]
    song.points.extend(EventPoint(Fraction(i, 7), '2') for i in range(230, 379))
    song.point_beats = [point.beat for point in song.points]
    song.sustain = SustainStateIndex(song.points, [EventTempo(Fraction(0), 120.),
                                                 EventTempo(Fraction(17), 150.)], .25)
    for limit in (96, 4096):
        data.max_tokens = limit
        for begin in (0, 72, 124):
            cursor = song.points[begin - 1].beat if begin else Fraction(0)
            options = dict(target=song.points[begin:begin + 2], training=True,
                           reference_start=Fraction(0) if begin else None)
            ordinary = data.build_stream(song, song.points[:begin], cursor, **options)
            indexed = data.build_stream(song, song.points, cursor, _prefix_end=begin, **options)
            for key, value in ordinary.items():
                if isinstance(value, torch.Tensor):
                    assert torch.equal(value, indexed[key]), key
                else:
                    assert value == indexed[key], key


def test_long_rest_targets_absolute_position_without_wait_or_window_eos():
    data,song=setup();point=EventPoint(Fraction(190)+Fraction(3,7),'1')
    item=data.build_stream(song,[],0,target=[point],training=True)
    target=item['labels'][item['labels']!=-100].tolist()
    assert data.codec.decode_points(target)==[point]
    assert data.codec.eos_id not in target
    assert 190 in item['anchors']


def test_query_local_audio_has_no_future_symbolic_bank_leak_and_kv_agrees():
    data,song=setup()
    item=data.build_stream(song,[],0,target=[EventPoint(Fraction(1),'1'),EventPoint(Fraction(25),'2')],training=True)
    batch=collate_stream([item])
    model=ComposerV2(ComposerV2Config(d_model=32,nhead=4,layers=2,dim_feedforward=64,
        audio_layers=1,song_layers=1,dropout=0.),data.codec).eval()
    with torch.inference_mode():
        full=model(batch)['logits'];cache=model.encode_audio(batch);parts=[]
        for end in range(1,batch['input_ids'].size(1)+1):
            query={**batch,**{k:batch[k][:,:end] for k in ('input_ids','token_mask','token_beats')},'tile_ids_cpu':[batch['tile_ids_cpu'][0][:end]]}
            parts.append(model.decode(query,cache)[:,-1:])
        torch.testing.assert_close(torch.cat(parts,1),full,atol=3e-6,rtol=2e-5)
        changed={**batch,'bank_audio':batch['bank_audio'].clone(),'bank_semantic':batch['bank_semantic'].clone()}
        future=item['anchors'].index(25)
        frames=batch['tile_frames'][future]
        changed['bank_audio'][frames]+=5;changed['bank_semantic'][frames]*=-1
        prefix=(batch['token_beats'][0]<25).nonzero().flatten()
        torch.testing.assert_close(model(changed)['logits'][0,prefix],full[0,prefix],atol=2e-6,rtol=2e-5)


def test_grouped_audio_padding_does_not_change_valid_logits():
    data,song=setup()
    one=data.build_stream(song,[],0,target=song.points[:1],training=True)
    two=data.build_stream(song,[],0,target=song.points[:7],training=True)
    model=ComposerV2(ComposerV2Config(d_model=32,nhead=4,layers=2,dim_feedforward=64,
        audio_layers=1,song_layers=1,dropout=0.),data.codec).eval()
    with torch.inference_mode():
        solo=model(collate_stream([one]))['logits'][0]
        batched=model(collate_stream([one,two]))['logits'][0,:len(one['input_ids'])]
        torch.testing.assert_close(solo,batched,atol=3e-6,rtol=3e-5)


def test_grouped_audio_bf16_backward_is_finite():
    data,song=setup();batch=collate_stream([data.build_stream(song,[],0,target=song.points[:2],training=True)])
    model=ComposerV2(ComposerV2Config(d_model=32,nhead=4,layers=2,dim_feedforward=64,
        audio_layers=1,song_layers=1,dropout=0.),data.codec)
    with torch.autocast('cpu',dtype=torch.bfloat16):loss,_=model.loss(model(batch),batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_shared_audio_frames_equal_explicit_independent_tiles():
    data,song=setup();batch=collate_stream([data.build_stream(song,[],0,target=song.points[:8],training=True)])
    dense={k:v for k,v in batch.items() if not k.startswith('bank_') and k!='tile_frames'}
    dense.update(local_audio=batch['bank_audio'][batch['tile_frames']],
        local_semantic=batch['bank_semantic'][batch['tile_frames']],
        local_semantic_mask=batch['bank_semantic_mask'][batch['tile_frames']])
    model=ComposerV2(ComposerV2Config(d_model=32,nhead=4,layers=2,dim_feedforward=64,
        audio_layers=1,song_layers=1,dropout=0.),data.codec).eval()
    with torch.inference_mode():torch.testing.assert_close(model(batch)['logits'],model(dense)['logits'],atol=2e-6,rtol=2e-5)


def test_training_frame_packets_match_interpolated_audio_logits_and_gradients():
    from types import SimpleNamespace
    from maichart.composer_v2.audio import AudioFeatures

    data, song = setup()
    times = np.linspace(0, 100, 200)
    rng = np.random.default_rng(19)
    audio = AudioFeatures(rng.normal(size=(200, 98)).astype(np.float32), times,
        rng.normal(size=(200, 1024)).astype(np.float16), times, 100., {'backend':'mert'})
    song.audio = audio
    packed = collate_stream([data.build_stream(song, [], 0, target=song.points[:8], training=True)])
    song.audio = SimpleNamespace(sample=audio.sample, duration_sec=100., identity=audio.identity)
    dense = collate_stream([data.build_stream(song, [], 0, target=song.points[:8], training=True)])
    model = ComposerV2(ComposerV2Config(d_model=32, nhead=4, layers=2, dim_feedforward=64,
        audio_layers=1, song_layers=1, dropout=0.), data.codec).eval()
    expected = model(dense)
    model.loss(expected, dense)[0].backward()
    gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    actual = model(packed)
    torch.testing.assert_close(actual['logits'], expected['logits'], atol=0., rtol=0.)
    model.loss(actual, packed)[0].backward()
    for name, p in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(p.grad, gradients[name], atol=2e-6, rtol=1e-4)


def test_rollback_removes_exact_source_event_not_aligned_block():
    from maichart.composer_v2.stream_generate import restore_prefix
    tempo=[EventTempo(Fraction(0),120.)];clock=TempoClock(tempo)
    points=[EventPoint(Fraction(1),'2'),EventPoint(Fraction(9),'3h[1:8]'),EventPoint(Fraction(17),'4')]
    kept,guard,report=restore_prefix(points,tempo,clock,100.,Fraction(24))
    assert kept==points[:1] and report['source_beat']=='9'
    assert len(guard.prefix)==1


def test_generation_is_event_continuous_for_different_context_refreshes(tmp_path,monkeypatch):
    from maichart.composer_v2.stream_generate import generate_continuous,EventSampler,RefreshContext
    from types import SimpleNamespace
    data,song=setup()
    authored=[EventPoint(Fraction(15)+Fraction(i,7),'1' if i%2 else '2') for i in range(20)]
    authored+=[EventPoint(Fraction(150)+Fraction(1,5),'3h[4:1]')]
    class Scripted(torch.nn.Module):
        def __init__(self,refresh):
            super().__init__();self.codec=data.codec;self.config=ComposerV2Config(d_model=8,nhead=2,layers=2,refresh_events=refresh,max_tokens=256)
        def encode_audio(self,batch):
            count=batch['local_beats'].size(0)
            return {'local':torch.zeros(count,2,8),'local_mask':torch.ones(count,2,dtype=torch.bool),'local_beats':batch['local_beats'][:,:2]}
        def decode(self,batch,cache):
            ids=batch['input_ids'][0].tolist();split=ids.index(self.codec.tgt_id)
            cursor=batch['cursor'][0];has_history=self.codec.at_id in ids[:split]
            remaining=[p for p in authored if not has_history or p.beat>cursor]
            expected=self.codec.encode_points(remaining)+[self.codec.eos_id]
            emitted=ids[split+1:]
            assert expected[:len(emitted)]==emitted
            logits=torch.full((1,1,self.codec.vocab_size),-100.)
            logits[0,0,expected[len(emitted)]]=100.;return logits
        @staticmethod
        def reset_decode_cache(cache):pass
    args=SimpleNamespace(temperature=.9,top_p=1.,window_retries=2,candidates=1,rollback_windows=0,
        level=13.,seed=0,max_events=0,checkpoint='scripted-contract-test')
    original=EventSampler.propose
    for refresh in (1,3,16):
        interrupted=[False]
        def propose(self,*a,**kw):
            if not interrupted[0]:interrupted[0]=True;raise RefreshContext('unfinished first proposal')
            return original(self,*a,**kw)
        monkeypatch.setattr(EventSampler,'propose',propose)
        args.output=str(tmp_path/f'out{refresh}.txt')
        report=generate_continuous(Scripted(refresh),song.audio,song.audio_path,
            [EventTempo(Fraction(0),120.)],0.,{},song.condition,np.asarray([True]+[False]*15),{},args,torch.Generator().manual_seed(0),'cpu',Fraction(200))
        from maichart.event_timing import parse_event_chart
        parsed=parse_event_chart(Path(args.output),5)
        assert parsed.points==authored
        assert report['events']==len(authored) and 'windows' not in report
        assert report['repair_count']==1
