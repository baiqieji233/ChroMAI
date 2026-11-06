# Prism_Re_v2 preparation provenance

Read-only evidence collected on October 15, 2026 for `checkpoints/Prism_Re_v2/form_last.pt`.

The checkpoint contains 1,866 chart identities and is still at epoch 1, layout batch 16,000 of 32,709. Matching every `song_form_prepare/v1/*.index.json` against the checkpoint's `(source, difficulty, chart_sha256)` identities found 11,204 cache records: 7,465 use the legacy `event-audio-v4-rhythm` 50 Hz identity and 3,739 use `rational_audio_queries_v1`. Across the entire cache root the counts are 7,781 legacy and 3,739 rational records.

This proves that the cache namespace contains both feature families for chart identities included in the checkpoint. It does not prove that all legacy records were consumed by this exact run: cache files can predate the run or represent another setting. The checkpoint's `dataset_audit.json` has no per-chart audio/rhythm profile field, so the exact per-example provenance cannot be reconstructed from the saved checkpoint.

The risk is nevertheless material. The historical fixed worker path could prepare a v2 chart with legacy `EventAudio`; that changes the audio semantics and can leave the non-rational skeleton path available, while free-joint rational generation does not consume that future skeleton. This checkpoint should be treated as provenance-incomplete and not resumed across the repaired preparation contract. A repaired fresh/finetune run should save explicit per-chart `training_inputs` (`audio_profile`, `rhythm_geometry`, `skeleton`) in its dataset contract.
