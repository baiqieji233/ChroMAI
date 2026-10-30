import json

import pytest

from maichart.composer.event_set_codec import EventSetCodec, rotate_atom


EVENTS = ['1/5', '1h[8:3]/5h[8:3]', '1h[4:1]/5h[8:3]',
          '1b-4[8:3]x/7', 'A1/Cf', '1pp5[8:1]', '1/1',
          '7xh/2h[4:1]x', '12', '1-4[8:1]*-6[8:1]']


def test_lossless_unordered_roundtrip_and_json():
    codec = EventSetCodec.fit(EVENTS)
    restored = EventSetCodec.from_dict(json.loads(json.dumps(codec.to_dict())))
    for event in EVENTS:
        encoded = codec.encode(event)
        assert codec.decode(encoded['shape'], encoded['phase'], encoded['details']) == '/'.join(sorted(event.split('/')))
        assert restored.encode(event) == encoded
        assert encoded['phase_mask'][encoded['phase']]
        assert codec.encode('/'.join(reversed(event.split('/')))) == encoded


def test_rotation_never_changes_duration_or_modifiers():
    assert rotate_atom('1b-4[8:3]x', 2) == '3b-6[8:3]x'
    assert rotate_atom('7xh', 3) == '2xh'
    assert rotate_atom('2h[4:1]x', 7) == '1h[4:1]x'
    for atom in ['1b-4[8:3]x', '7xh', '2h[4:1]x']:
        for phase in range(8):
            assert rotate_atom(rotate_atom(atom, phase), -phase) == atom


def test_orbit_shared_shape_details_and_full_event_phase_targets():
    codec = EventSetCodec.fit(EVENTS)
    for event in EVENTS[:4]:
        base = codec.encode(event)
        for phase in range(8):
            rotated = '/'.join(rotate_atom(atom, phase) for atom in event.split('/'))
            actual = codec.encode(rotated)
            assert actual['shape'] == base['shape']
            assert actual['details'] == base['details']
            assert codec.decode(actual['shape'], actual['phase'], actual['details']) == '/'.join(sorted(rotated.split('/')))
    assert sum(codec.encode('1/5')['phase_mask']) == 2
    assert sum(codec.encode('1h[8:3]/5h[8:3]')['phase_mask']) == 2
    assert sum(codec.encode('1h[4:1]/5h[8:3]')['phase_mask']) == 1


def test_detail_factorization_does_not_make_shape_duration_products():
    codec = EventSetCodec.fit(['1h[4:1]/3h[8:1]', '1h[8:1]/3h[4:1]'])
    assert len(codec.shape_catalog) == 2  # reserved fallback plus one regular shape
    assert [len(slot) for slot in codec.detail_catalog[1]] == [2, 2]
    # Independent conditional details can compose a previously unseen duration pair.
    encoded = codec.encode('1h[4:1]/3h[4:1]')
    assert codec.decode(encoded['shape'], encoded['phase'], encoded['details']) == '1h[4:1]/3h[4:1]'


def test_conservative_fallback_coverage_and_errors():
    codec = EventSetCodec.fit(EVENTS)
    assert codec.coverage['fallback_events'] == 4
    for event in ['A1/Cf', '1pp5[8:1]', '12', '1-4[8:1]*-6[8:1]']:
        encoded = codec.encode(event)
        assert encoded['fallback']
    with pytest.raises(ValueError): rotate_atom('A1', 1)
    with pytest.raises(ValueError): EventSetCodec.fit([])
    with pytest.raises(ValueError): codec.decode(-1, 0, [])
    with pytest.raises(ValueError): codec.decode(0, 8, [])
    with pytest.raises(ValueError): EventSetCodec.from_dict({'schema': 'old'})
    assert codec.num_atoms == len(codec.atom_catalog) + 1
    assert codec.num_shapes == len(codec.shape_catalog)
    ids = [codec._atoms['A1'], codec._atoms['1pp5[8:1]'], 0]
    assert codec.decode(0, 0, ids) == '1pp5[8:1]/A1'
    with pytest.raises(ValueError): codec.decode(0, 0, [codec.num_atoms])
    with pytest.raises(ValueError): codec.decode(0, 0, [0, 1])


def test_unseen_shape_of_known_atoms_uses_fallback_without_dev_fitting():
    codec = EventSetCodec.fit(['1', 'A1'])
    result = codec.encode('1/3/5')
    assert result['fallback'] and result['shape'] == codec.fallback_shape_id
    assert result['details'][-1] == 0
    assert codec.decode(result['shape'], result['phase'], result['details']) == '1/3/5'
    with pytest.raises(KeyError): codec.encode('C')


@pytest.mark.parametrize('atom',['1v2[8:3]','1p4[8:3]','1q4[8:3]',
                               '1s5[8:3]','1z5[8:3]','1w5[8:3]','1V72[8:3]'])
def test_native_geometry_proved_slide_orbits_preserve_all_timing_bytes(atom):
    codec=EventSetCodec.fit([atom])
    base=codec.encode(atom)
    assert not base['fallback']
    for shift in range(8):
        rotated=rotate_atom(atom,shift)
        assert rotated.endswith('[8:3]')
        assert rotate_atom(rotated,-shift)==atom
        event=codec.encode(rotated)
        assert event['shape']==base['shape'] and event['details']==base['details']
        assert codec.decode(event['shape'],event['phase'],event['details'])==rotated


def test_parser_accepted_but_missing_native_geometry_falls_back():
    assert EventSetCodec.fit(['1w2[8:1]']).encode('1w2[8:1]')['fallback']
