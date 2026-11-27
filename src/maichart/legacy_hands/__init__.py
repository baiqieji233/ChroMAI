"""Experimental legacy player simulation and batch hand prediction."""

SWEEP_STYLES = ('continuous', 'relay')

# On disk: positive is left/same; magnitude 1 is a lean and 2 is supported.
HAND_FIELDS = ('first_left', 'second_left', 'same_previous', 'same_pair', 'same_return')
MISSING = -128
NO_RIVAL = -127
LINK_SECONDS = .45
