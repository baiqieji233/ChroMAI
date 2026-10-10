# Condition Profile Measurements

`condition_profile.json` summarizes chart-derived control medians. The profile measurement pass parses charts without loading audio, so its `melody_following` value is not measured: `condition_from_events(..., audio=None)` stores a zero placeholder. New profiles record this explicitly in `control_measurements`; generation falls back to `GENERATION_DEFAULTS["melody_following"]` for that field. Missing or explicitly unmeasured controls also use their generation defaults rather than 0.5.

Old `maichart-condition-profile-v1` files have no measurement metadata. A zero median for `melody_following` is treated as the known chart-only placeholder and falls back to 0.9. Other finite medians remain intact. A profile that explicitly marks `melody_following` as measured preserves even a measured zero.

The 0.9 value is the training-side audio onset-alignment proxy, not voice separation or true melody recognition. An explicit condition control supplied by the user takes precedence over the profile fallback; the profile supplies only omitted controls.

The event CLI also treats a condition file as a partial request. Supplying only `level` or a few radar/style controls preserves the generation baseline for all omitted fields, with or without a profile. Explicit numeric zero and wrapped `target`/`value` entries override the corresponding field only. The checkpoint condition guard runs last and uses its recorded numeric null value and difficulty metadata.

Previously, a bare API request with an automatically selected chart-only profile resolved `melody_following` to zero, while the WebUI's full parameter payload supplied 0.9. Historical generated artifacts retain their actual conditions; comparisons across this repair must read those values and rerun matched arms where needed. Existing checkpoints remain loadable. The next event-generation subprocess loads the correction from disk; a running training job is unaffected.
