# Condition Profile Measurements

`condition_profile.json` summarizes chart-derived control medians. The profile measurement pass parses charts without loading audio, so its `melody_following` value is not measured: `condition_from_events(..., audio=None)` stores a zero placeholder. New profiles record this explicitly in `control_measurements`; generation falls back to `GENERATION_DEFAULTS["melody_following"]` for that field. Missing or explicitly unmeasured controls also use their generation defaults rather than 0.5.

Old `maichart-condition-profile-v1` files have no measurement metadata. A zero median for `melody_following` is treated as the known chart-only placeholder and falls back to 0.9. Other finite medians remain intact. A profile that explicitly marks `melody_following` as measured preserves even a measured zero.

The 0.9 value is the training-side audio onset-alignment proxy, not voice separation or true melody recognition. An explicit condition control supplied by the user takes precedence over the profile fallback; the profile supplies only omitted controls.
