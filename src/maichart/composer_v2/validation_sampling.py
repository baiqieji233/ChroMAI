"""Deterministic validation coverage without changing training data or order."""
from collections import defaultdict

POLICY = 'audio_balanced_spread_v1'
LEGACY = 'manifest_prefix_v1'


def audio_key(song):
    identity = getattr(getattr(song, 'audio', None), 'identity', {}) or {}
    return identity.get('audio_sha256') or str(getattr(song, 'audio_path', song.song_id))


def _spread(indices, count):
    if count >= len(indices):
        return list(indices)
    if count == 1:
        return [indices[len(indices) // 2]]
    return [indices[i * (len(indices) - 1) // (count - 1)] for i in range(count)]


def balanced_window_indices(dataset, budget):
    """Allocate coverage to audio identities first, then spread within each song.

    Repeated copies and different difficulties stay in the dataset, but a song
    with many copies/windows does not consume the whole limited evaluation.
    Zero budget includes all windows. No audio is read or hashed here.
    """
    if budget < 0:
        raise ValueError('Validation budget must be nonnegative')
    if not budget or budget >= len(dataset.windows):
        return list(range(len(dataset.windows)))
    groups = defaultdict(list)
    for index, (song_index, _) in enumerate(dataset.windows):
        groups[audio_key(dataset.songs[song_index])].append(index)
    keys = list(groups)
    if budget < len(keys):
        keys = _spread(keys, budget)
    allocation = dict.fromkeys(keys, 0)
    remaining = budget
    while remaining:
        for key in keys:
            if allocation[key] < len(groups[key]):
                allocation[key] += 1
                remaining -= 1
                if not remaining:
                    break
    return sorted(index for key in keys for index in _spread(groups[key], allocation[key]))


def coverage_report(dataset, indices, policy, *, maximum_batches=0):
    selected = {dataset.windows[index][0] for index in indices}
    total_audio = {audio_key(song) for song in dataset.songs}
    selected_audio = {audio_key(dataset.songs[index]) for index in selected}
    def chart_key(index):
        song = dataset.songs[index]
        return (audio_key(song), getattr(song, 'chart_sha256', None) or str(song.source_path), song.difficulty)
    excluded = len(getattr(dataset, 'audit', {}).get('vocabulary_exclusions', ()))
    return dict(policy=policy, maximum_batches=maximum_batches, vocabulary_excluded_chart_records=excluded,
        selected_windows=len(indices), total_windows=len(dataset.windows),
        selected_chart_records=len(selected), total_chart_records=len(dataset.songs),
        selected_unique_charts=len({chart_key(index) for index in selected}),
        total_unique_charts=len({chart_key(index) for index in range(len(dataset.songs))}),
        selected_audio=len(selected_audio), total_audio=len(total_audio),
        complete=len(indices) == len(dataset.windows) and not excluded,
        window_fraction=len(indices) / max(1, len(dataset.windows)))


class ValidationBatchSampler:
    """Selected windows only; no fetching of discarded validation prefixes."""
    def __init__(self, dataset, indices, batch_size):
        # Length sorting limits padding after song-balanced selection.
        lengths = getattr(dataset, 'lengths', None)
        self.indices = sorted(indices, key=lengths.__getitem__) if lengths is not None else list(indices)
        self.batch_size = batch_size

    def __iter__(self):
        for index in range(0, len(self.indices), self.batch_size):
            yield self.indices[index:index + self.batch_size]

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size
