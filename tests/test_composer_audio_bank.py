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
