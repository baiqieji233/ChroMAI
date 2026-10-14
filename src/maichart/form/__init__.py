"""Independent song-form architecture; legacy event checkpoints remain separate."""

FAMILY = "song_form_v1"
CHECKPOINT_SCHEMA = "maichart-song-form-checkpoint-v1"
RATIONAL_FAMILY = "song_form_v2"
RATIONAL_CHECKPOINT_SCHEMA = "maichart-song-form-checkpoint-v2"
FORM_FAMILIES = frozenset({FAMILY, RATIONAL_FAMILY})
