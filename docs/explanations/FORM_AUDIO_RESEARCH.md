# Song Form Audio Research

This note describes the audio contract for the independent `song_form_v1`
path. It separates the currently implemented 50 Hz features from possible
future pretrained or multi-resolution branches. The references below are the
primary sources reviewed in `.scratch/form_research.json`; their findings are
research context, not evidence that this project has already implemented or
validated those systems.

## Current Audio Input

`event_audio` emits 112 values per 50 Hz frame: a 98-value acoustic block, four
beat-normalized rhythm values, and ten chart-time context values. The acoustic
block contains 80 log-mel values, 12 chroma values, onset strength, RMS, spectral
flux, relative RMS, intensity slope, and harmonic novelty. The context block
contains song progress, duration, local tempo, beat phase, beat progress, and
the legacy bar and phrase phase pairs.

`form_audio` clears the last four legacy bar/phrase phase values and appends
eight features from the explicit metrical timeline. Its final width is 120:
the existing 112 columns plus quarter-beat phase sine/cosine, bar phase
sine/cosine, within-group phase sine/cosine, meter-known, and grouping-known.
The eight new values are calculated on the same frame timestamps as the
acoustic values. Unknown meter leaves bar and group features at zero and marks
them unknown. See `src/maichart/generation/event_audio.py` and
`src/maichart/form/timing.py` for the definitions.

The old bar and phrase channels encoded fixed 4-beat and 16-beat cycles. Those
are not valid universal priors: 16 quarter-note beats happen to describe four
bars of 4/4, while four bars of 3/4 occupy 12 quarter-note beats. A `16`-beat
phrase may be useful as one observed grouping, but it must not be asserted for
every song. A Simai `{12}` directive selects a subdivision of event positions;
it does not identify 3/4 or any other meter. Exact event positions remain
quarter-note `Fraction` values, tempo maps convert those positions to seconds,
and a separate explicit meter map defines bar length and grouping. A BPM change
changes the beat-to-seconds conversion; it does not by itself change meter or
restart bar phase.

Current segmentation limits regions to 14 seconds. At 50 frames per second,
that is at most 700 frames, below `FormConfig.max_audio_tokens=1024`; the model
does not currently pool audio. It rejects sequences beyond that configured
limit. If longer regions are allowed later, pooling must be explicit and
auditable. Acoustic columns can be pooled over each time bin, but averaging
sine/cosine phases can erase phase or lower their magnitude. Recompute beat,
bar, and group features at each pooled bin's timestamp from the tempo and meter
maps; do not treat averaged phase values as an exact clock. Preserve known flags
and use an explicit unknown policy where a bin crosses a meter boundary.

The model currently receives this single 120-value stream. It does not load
MERT, MT3, madmom, or any other pretrained audio model automatically.

## Acoustic Evidence And Authored Onsets

An acoustic onset is evidence of a transient in the waveform. The existing
onset-strength and spectral-flux channels describe changes in audio; they do
not assert that a chart note belongs at that instant. A chart onset is an
authored event at an exact beat, mapped to audio time through `first` and the
tempo map. It is supervision or a generated gameplay action, not a measurement
of the waveform.

The distinction matters in both directions. A kick, cymbal, piano attack, or
vocal consonant can produce an acoustic onset without a corresponding playable
note. A chart note can anticipate, sustain through, or otherwise differ from a
detected transient. Any future onset-target objective should describe which
signal it predicts and should evaluate audio evidence separately from authored
chart timing. Do not feed target-derived onset labels into an input branch for
the same target interval.

## Staged Audio Resolution

The current 50 Hz stream is a practical baseline for section-scale energy,
harmonic change, and local timing context. It should be evaluated against
fixed-audio failures before increasing model or cache complexity. A staged
extension can keep distinct jobs separate:

1. Keep the current acoustic stream and exact metrical clock as the baseline.
2. Add a higher-time-resolution transient branch only for failures that need
   finer attacks, such as weak drums or piano notes. Preserve timestamp and
   confidence so it remains evidence, rather than a hard note-placement rule.
3. Optionally add a slower, pretrained music representation for timbre,
   harmony, and broad musical continuity. Compare frozen embeddings with the
   baseline on song-disjoint retrieval and generation; account for model
   latency, cache size, domain shift, and licensing before adopting one.
4. Treat automatic transcription as an auxiliary, fallible evidence source.
   Multi-instrument transcription may propose pitches and times, but alignment
   and instrument errors must not overwrite the original waveform features or
   chart ground truth.

The MERT paper describes music-focused self-supervised acoustic representations
trained with acoustic and CQT-derived teacher signals. This supports testing a
slower learned representation as a complement to local spectral and transient
features; it does not establish that MERT embeddings directly encode maimai
chart intent or improve this model. MT3 studies multi-task, multi-track music
transcription and highlights alignment and evaluation challenges. A
transcription branch should therefore remain a confidence-bearing proposal,
not a substitute for exact chart events.

## Section Memory And Form

The current recurrence path retrieves earlier audio-similar sections and
provides their chart tokens as a separately gated motif memory. The learned
gate and user `memory_strength` let the model ignore or attenuate that memory;
they do not encode an explicit rule that a repeated section must be copied or
varied in a particular way. Similar-section memory can support an echo,
transposition-like reuse, or local variation when examples teach those
relationships. The present relation target predicts similarity evidence, not
an intentional form label. It does not yet guarantee contrast, development, or
a meaningful chorus return. A body-similarity and intentional-contrast target
is deferred until those labels and song-level evaluations are defined.

Music Transformer motivates modeling musical self-reference over multiple
timescales, from motifs to phrases and reused sections. It is architectural
motivation for recurrence-aware experiments, not proof that retrieved
similarity alone yields coherent song form. InCoder shows code infilling with
bidirectional context and provides an analogy for an edit/revision task; it is
not evidence of chart revision performance. For both draft and revision, keep
the decoder causal and ensure right context is provided only for explicit edit
examples.

## References

- “Music Transformer,” arXiv:1809.04281. `https://arxiv.org/abs/1809.04281`
- “MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training,” arXiv:2306.00107. `https://arxiv.org/abs/2306.00107`
- “InCoder: A Generative Model for Code Infilling and Synthesis,” arXiv:2204.05999. `https://arxiv.org/abs/2204.05999`
- “MT3: Multi-Task Multitrack Music Transcription,” arXiv:2111.03017. `https://arxiv.org/abs/2111.03017`
- madmom 0.16.1 downbeat tracking documentation, including joint beat/downbeat activations and configurable beats-per-bar priors. `https://madmom.readthedocs.io/en/v0.16.1/modules/features/downbeats.html`
