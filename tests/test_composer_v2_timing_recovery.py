from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import torch

from maichart.event_timing import EventPoint
from maichart.composer_v2.stream_codec import StreamCodec
from maichart.composer_v2.stream_data import StreamDataset
from maichart.composer_v2.timing_recovery import POLICY, perturb_prefix, timing_ids
from test_composer_v2_continuous import Audio


def test_corruption_preserves_committed_cursor_and_sustains():
    points=[EventPoint(Fraction(i,2),'1' if i%2 else '2') for i in range(14)]
    points[5]=EventPoint(points[5].beat,'3h[4:8]')
    codec=StreamCodec.fit([p.note_text for p in points])
    for dense in (True,False):
        changed,mode=perturb_prefix(points,set(codec.phases.values()),7,dense)
        assert changed[-1]==points[-1]
        assert points[5] in changed
        assert all(p.beat<=points[-1].beat for p in changed)
        assert len({p.beat for p in changed})==len(changed)
        assert changed!=points
        assert codec.decode_points(codec.encode_points(changed))==changed


def fixture(tmp_path):
    notes=','.join(str(i%8+1) for i in range(256))
    chart=tmp_path/'maidata.txt'
    # An authored eight-beat rest precedes the last group.
    chart.write_text(f'&lv_5=13\n&inote_5=(120){{8}}{notes},{{1}},,{{8}}1,2,3,4,E\n',encoding='utf-8')
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(songs=[dict(song_id='fixture',maidata_path=str(chart),
        audio_path=str(tmp_path/'audio.wav'),difficulties=[dict(index=5,level='13')])])),encoding='utf-8')
    audio=Audio()
    audio.identity={'audio_sha256':'fixture','backend':'mert'}
    store=SimpleNamespace(prepare=lambda _:audio,close=lambda:None)
    return manifest,store


def test_extra_views_keep_all_original_targets_and_supervise_first_recovery_time(tmp_path):
    manifest,store=fixture(tmp_path)
    original=StreamDataset(manifest,tmp_path/'cache',audio_store=store,condition_policy='level_only_v1')
    data=StreamDataset(manifest,tmp_path/'cache',audio_store=store,condition_policy='level_only_v1',
                       timing_recovery_policy=POLICY,augmentation_seed=2)
    assert data.windows[:len(original)]==original.windows
    assert data.recovery_sources
    assert not any(data.windows[i][1]==0 for i in data.recovery_sources)
    for index in range(len(original)):
        a,b=original[index],data[index]
        assert torch.equal(a['input_ids'],b['input_ids'])
        assert torch.equal(a['labels'],b['labels'])
    time_set=timing_ids(data.codec)
    for index,source in data.recovery_sources.items():
        clean=data[source];changed=data[index]
        expected=[t for t in clean['labels'].tolist() if t in time_set]
        assert changed['labels'][changed['labels']!=-100].tolist()==expected
        split=changed['input_ids'].tolist().index(data.codec.tgt_id)
        assert changed['labels'][split].item()==data.codec.at_id
        assert (changed['token_weights'][changed['labels']!=-100]==2).all()
        assert torch.equal(changed['input_ids'],data[index]['input_ids'])
    data.set_epoch(1)
    index=next(iter(data.recovery_sources))
    changed=data[index]
    data.set_epoch(0)
    assert not torch.equal(changed['input_ids'],data[index]['input_ids'])
    assert not data._reference_descriptors
    # The final original window begins after the authored long rest. A
    # corrupted past must still supervise that rest and the true song end.
    last=len(original)-1
    recovery=data._recovery_item(last)
    clean=data[last]
    assert recovery['labels'][recovery['labels']!=-100].tolist()==[
        t for t in clean['labels'].tolist() if t in time_set]
    si,begin=data.windows[last]
    assert data.songs[si].points[begin].beat-data.songs[si].points[begin-1].beat>4
    assert data.codec.eos_id in recovery['labels'].tolist()


def test_no_history_disables_recovery_views(tmp_path):
    manifest,store=fixture(tmp_path)
    data=StreamDataset(manifest,tmp_path/'cache',audio_store=store,condition_policy='level_only_v1',
        timing_recovery_policy=POLICY,history_beats=0)
    assert not data.recovery_sources
