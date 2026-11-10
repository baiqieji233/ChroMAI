"""Music-context versions shared by training, inspection and the workbench."""
POLICY = 'song_paired_v2'
POLICIES = ('off', 'song_detail_v1', POLICY)
CHOICES = ('auto', *POLICIES)


def resolve_policy(requested, parent=None, *, resume=False):
    previous = (parent or {}).get('music_context_policy', 'off')
    continuous = parent is None or parent.get('sequence_policy') == 'continuous_events_v1'
    latest = POLICY if continuous else previous
    value = previous if resume and requested == 'auto' else latest if requested == 'auto' else requested
    if value not in POLICIES:
        raise ValueError('Unknown music context policy')
    order = {name: index for index, name in enumerate(POLICIES)}
    if parent and value != previous and (resume or order[value] < order[previous]):
        raise ValueError('Music context changes require finetune upgrading in a new directory')
    if value == POLICY and not continuous:
        raise ValueError('Paired music context requires continuous events')
    return value, parent is not None and value != previous
