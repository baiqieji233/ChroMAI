"""Train on sampled Tap keys while preserving two authored contacts at the join."""
import numpy as np
import torch

_PLAIN_RIM_TAPS = frozenset('12345678')

def training_item(dataset, song, configuration, begin, end, seed):
    points = song.points
    buckets = {'continue': [], 'change': [], 'gap': []}
    for seam in range(begin + 3, end):
        first = seam
        while first > max(begin, seam - 5) and points[first - 1].note_text in _PLAIN_RIM_TAPS:
            first -= 1
        if seam - first < 3:
            continue
        start, stop = configuration.seconds[[first, seam]]
        if configuration.quality.covered(start, np.nextafter(stop, np.inf)) <= 0:
            continue
        # Include the waiting period before a Slide starts as well as its trace.
        sustains = configuration._overlapping_sustains(start, stop)
        if len(sustains) and np.any(sustains[:, 2] >= start):
            continue
        before, last, current = [int(point.note_text) for point in points[seam - 3:seam]]
        step = (current - last) % 8
        if (last - before) % 8 != step:
            continue
        next_key = str((current - 1 + step) % 8 + 1)
        if points[seam].beat - points[seam - 1].beat >= 1:
            outcome = 'gap'
        else:
            outcome = 'continue' if points[seam].note_text == next_key else 'change'
        buckets[outcome].append((first, seam))

    populated = [group for group in buckets.values() if group]
    if not populated:
        return None
    rng = np.random.default_rng(seed)
    group = populated[rng.integers(len(populated))]
    first, seam = group[rng.integers(len(group))]
    stop = min(end, seam + 4)
    cursor = points[first - 1].beat if first else 0
    item = dataset.build_stream(song, points, cursor, target=points[first:stop],
                                training=True, reference_start=None, _prefix_end=first)
    codec = dataset.codec
    offset = item['input_ids'].tolist().index(codec.tgt_id) + 1
    queries = []
    for index in range(first, seam):
        if index < seam - 2:
            queries.append(offset + len(codec._time(points[index].beat)))
        offset += len(codec.encode_points([points[index]]))
    # Shifted labels: the last terminator before the seam predicts the next AT.
    item['labels'][:offset - 1] = -100
    item['token_weights'][:offset - 1] = 0
    if not item['token_weights'].any():
        return None
    item['layout_replay'] = (queries, rng.random(len(queries)).tolist())
    for name in ('condition', 'condition_known'):
        item[name] = item[name].clone()
        item[name][1:] = 0
    item['curve_teacher_allowed'] = False
    dataset._writing_targets(item, configuration, song, first, stop, offset=rng.random())
    return item


def collate(items, batch):
    rows = [index for index, item in enumerate(items) if 'layout_replay' in item]
    if not rows:
        return
    width = 1 + max(max(items[i]['layout_replay'][0]) for i in rows)
    beat_tiles = [batch['tile_ids_cpu'][i][:width] for i in rows]
    local_tiles = batch.get('local_tile_ids_cpu', batch['tile_ids_cpu'])
    local = [local_tiles[i][:width] for i in rows]
    used = sorted({tile for row in local for tile in row if tile >= 0})
    mapping = {tile: index for index, tile in enumerate(used)}
    coordinates = [(row, position, quantile)
                   for row, index in enumerate(rows)
                   for position, quantile in zip(*items[index]['layout_replay'])]
    batch['layout_replay'] = dict(
        rows=rows, width=width, tiles=beat_tiles, used=used,
        local=[[mapping.get(tile, -1) for tile in row] for row in local],
        coordinates=coordinates,
    )


@torch.no_grad()
def replace_keys(model, batch, cache):
    """One causal prefix pass on participating rows, reusing cached audio encodings."""
    replay = batch['layout_replay']
    rows, width = replay['rows'], replay['width']
    device = batch['input_ids'].device
    selected = torch.tensor(rows, device=device)
    probe = dict(batch, start_beat=[batch['start_beat'][i] for i in rows],
                 tile_ids_cpu=replay['tiles'], local_tile_ids_cpu=replay['local'])
    for name in ('input_ids', 'token_beats', 'token_mask', 'reference_evidence',
                 'paired_music_features', 'curve_left', 'curve_right', 'curve_mix', 'curve_override_queries'):
        if name in batch:
            probe[name] = batch[name][:, :width].index_select(0, selected)
    for name in ('reference_trust', 'condition', 'condition_known'):
        if name in batch:
            probe[name] = batch[name].index_select(0, selected)

    memory = dict(cache)
    for name in ('condition', 'curve_context', 'song', 'song_beats', 'song_mask'):
        if name in cache:
            memory[name] = cache[name].index_select(0, selected)
    used = torch.tensor(replay['used'], device=device)
    for name in ('local', 'local_mask', 'local_beats'):
        memory[name] = cache[name].index_select(0, used)

    rows, positions, quantiles = zip(*replay['coordinates'])
    rows = torch.tensor(rows, device=device)
    positions = torch.tensor(positions, device=device)
    quantiles = torch.tensor(quantiles, device=device)
    logits = model.decode(probe, memory, incremental=False, button_queries=(rows, positions))
    cumulative = logits.float().softmax(-1).cumsum(-1)
    choices = (cumulative <= quantiles[:, None]).sum(-1).clamp_max(logits.size(-1) - 1)
    inputs = batch['input_ids'].clone()
    revised = model.rim_button_ids[choices]
    locations = selected[rows], positions + 1
    changed = (inputs[locations] != revised).sum()
    inputs[locations] = revised
    return dict(batch, input_ids=inputs, layout_replay_changed=changed)
