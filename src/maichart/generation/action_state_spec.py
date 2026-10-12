"""Versioned, lightweight contract shared by sustain features and model inputs."""

PROFILE = "sustain_state_v1"
STATE_DIM = 50
MAX_STATES = 32
STATE_KEYS = {
    "action_state_projection.weight", "action_state_projection.bias",
    "action_state_type_embedding",
}
