from pathlib import Path

import numpy as np
import pytest

from maichart.composer import structure


def test_structure_bank_reuses_concurrent_mapped_publication(tmp_path, monkeypatch):
    records = np.zeros((2, len(structure.PERIODS), 8), dtype=np.float32)
    monkeypatch.setattr(structure, '_build', lambda *args:
        (np.array([0., 4.]), records, np.array([0., 2., 4.]), np.array([0., .5, 0.])))
    published = []
    def concurrent_replace(source, target):
        # Reproduce the Windows outcome after another process publishes and
        # maps exactly this key before our atomic replacement.
        target = Path(target)
        target.write_bytes(Path(source).read_bytes())
        published.append(target.name)
        raise PermissionError('destination is already mapped by another process')
    monkeypatch.setattr(structure.os, 'replace', concurrent_replace)
    bank, novelty, _, _ = structure._build_bank(None, tmp_path, {'backend_effective': {}}, 'race')
    assert published == ['race.npy', 'race.novelty.npy', 'race.json']
    assert isinstance(bank, np.memmap) and isinstance(novelty, np.memmap)
    np.testing.assert_array_equal(bank[:, 0], [0., 4.])
    np.testing.assert_array_equal(novelty[:, 1], [0., .5, 0.])
    assert not list((tmp_path/'composer_structure_v4').glob('*.tmp'))


def test_publication_rejects_invalid_competing_array(tmp_path, monkeypatch):
    values = np.array([[0., 1.], [4., 2.]], dtype=np.float32)
    def invalid_replace(source, target):
        invalid = values.copy(); invalid[1, 1] = np.nan
        np.save(target, invalid, allow_pickle=False)
        raise PermissionError('mapped')
    monkeypatch.setattr(structure.os, 'replace', invalid_replace)
    with pytest.raises(ValueError, match='Invalid concurrently published'):
        structure._publish_structure_array(tmp_path/'invalid.npy', values)
    assert not list(tmp_path.glob('*.tmp'))


def test_permission_failure_without_another_publisher_is_not_hidden(tmp_path, monkeypatch):
    def denied(*args):
        raise PermissionError('write denied')
    monkeypatch.setattr(structure.os, 'replace', denied)
    with pytest.raises(PermissionError, match='write denied'):
        structure._publish_structure_array(tmp_path/'denied.npy', np.zeros((2, 2), dtype=np.float32))
    assert not list(tmp_path.glob('*.tmp'))
