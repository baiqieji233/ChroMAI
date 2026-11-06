# Composer generation temperature controls

Composer generation now accepts `pointer_temperature` and `action_temperature` in addition to the legacy `temperature` field. If either new value is omitted, it inherits `temperature`, so older API clients and command lines keep their previous behavior. The WebUI/backend contract is `composer_generation_controls_v5_workload_sampling`; an older v4 service is rejected before launch.

The candidate beat sampler uses `pointer_temperature`; the per-event action slot sampler uses `action_temperature`. The effective values are written to the runtime log, task settings, and `.composer.json` report. The main WebUI exposes both controls with conservative defaults (0.55 for timing and 0.85 for actions); the standalone Composer page exposes the same controls. They are included in the generation parameter persistence list, so page reloads retain them. The result summary reports greedy decoding only when both effective temperatures are zero.

This only changes inference sampling. It does not alter training targets or checkpoint schemas.
