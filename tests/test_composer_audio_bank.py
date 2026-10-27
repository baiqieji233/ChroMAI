import json
import numpy as np
import soundfile as sf

from maichart.composer.audio_bank import bank_path,load_bank,publish_bank


def test_mapped_bank_is_lossless_and_reused(tmp_path):
    rng=np.random.default_rng(17)
    arrays=(rng.normal(size=(15,120)).astype(np.float32),
            np.sort(rng.uniform(0.,10.,15)).astype(np.float64))
    path=tmp_path/'bank.npy'
    mapped=publish_bank(path,arrays)
    for actual,expected in zip(mapped,arrays):
        np.testing.assert_array_equal(actual,expected)
        assert not actual.flags.writeable
        assert isinstance(actual,np.memmap)
    for actual,expected in zip(load_bank(path),arrays):
        np.testing.assert_array_equal(actual,expected)
    assert not list(tmp_path.glob('*.tmp'))


def test_bank_identity_covers_clock_and_source_revision(tmp_path):
    source=tmp_path/'audio.npz';source.write_bytes(b'first')
    a=bank_path(tmp_path,('song',120.,'first=0'),source)
    assert a==bank_path(tmp_path,('song',120.,'first=0'),source)
    assert a!=bank_path(tmp_path,('song',120.,'first=1'),source)
    source.write_bytes(b'new source revision')
    assert a!=bank_path(tmp_path,('song',120.,'first=0'),source)


def test_corrupt_or_wrong_shape_bank_is_rebuilt(tmp_path):
    path=tmp_path/'bank.npy'
    path.write_bytes(b'truncated')
    assert load_bank(path) is None
    np.save(path,np.zeros((4,120),dtype=np.float32))
    assert load_bank(path) is None
    arrays=(np.zeros((2,120),np.float32),np.array([.1,.2],np.float64))
    assert publish_bank(path,arrays)[0].shape==(2,120)


def test_dataset_reuses_bank_when_resident_cache_is_disabled(tmp_path,monkeypatch):
    from maichart.composer import data
    sr=8000;t=np.arange(sr*2)/sr
    sf.write(tmp_path/'track.wav',(.1*np.sin(2*np.pi*440*t)).astype(np.float32),sr)
    (tmp_path/'maidata.txt').write_text(
        '&wholebpm=120\n&lv_5=12\n&inote_5=(120){4}1,2,3,4,5,6,7,8,E\n',encoding='utf8')
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'songs':[{'song_id':'mapped-fixture',
        'maidata_path':str(tmp_path/'maidata.txt'),'audio_path':str(tmp_path/'track.wav'),
        'difficulties':[{'difficulty_index':5,'level_raw':'12'}]}]}),encoding='utf8')
    dataset=data.ComposerDataset(manifest,cache_dir=tmp_path/'cache',candidate_count=16,
        history_events=8,action_slots=4,preparation_workers=1,audio_cache_gb=0.)
    first=dataset[0]
    def forbidden(*args,**kwargs):
        raise AssertionError('A mapped bank must avoid reloading/decompressing RationalAudio')
    monkeypatch.setattr(dataset,'_source_song',forbidden)
    second=dataset[0]
    for key in first:
        np.testing.assert_array_equal(first[key].numpy(),second[key].numpy())
    assert dataset.cache_stats()['mapped_bank_hits']==1
    assert dataset.cache_stats()['bytes']==0


def test_bank_path_is_resolved_again_after_lazy_source_cache_creation(tmp_path,monkeypatch):
    from collections import OrderedDict
    from types import SimpleNamespace
    import threading
    from maichart.composer import data
    source=tmp_path/'late_audio.npz'
    lazy=SimpleNamespace(path=tmp_path/'track.wav',store=SimpleNamespace(disk_cache_path=lambda path:source))
    dataset=data.ComposerDataset.__new__(data.ComposerDataset)
    dataset.base=SimpleNamespace(songs=[(SimpleNamespace(audio=lazy),)])
    dataset.cache_dir=str(tmp_path);dataset._bank_paths={}
    dataset._feature_lock=threading.RLock();dataset._feature_cache=OrderedDict()
    dataset._mapped_bank_cache=OrderedDict();dataset._mapped_bank_cache_limit=2
    dataset._feature_cache_effective_budget_bytes=0;dataset._feature_cache_bytes=0
    for name in ('_feature_cache_hits','_feature_cache_misses','_bank_hits','_bank_builds',
                 '_bank_write_errors','_bank_mmap_loads','_bank_view_hits'):
        setattr(dataset,name,0)
    monkeypatch.setattr(dataset,'_check_cache_headroom',lambda:None)
    monkeypatch.setattr(dataset,'_song_feature_key',lambda index:('late-source',))
    def resolve(index,**kwargs):
        source.write_bytes(b'created only by lazy source loading')
        return SimpleNamespace()
    monkeypatch.setattr(dataset,'_source_song',resolve)
    values=np.arange(360,dtype=np.float32).reshape(3,120);times=np.array([0.,.1,.2])
    monkeypatch.setattr(data,'composer_audio_nodes',lambda song:(values,times))
    first=dataset._song_features(0)
    second=dataset._song_features(0)
    assert dataset._bank_builds==1
    assert dataset._bank_hits==1
    assert dataset._bank_paths[0].is_file()
    for actual,expected in zip(first,second):
        np.testing.assert_array_equal(actual,expected)
        assert isinstance(actual,np.memmap)
