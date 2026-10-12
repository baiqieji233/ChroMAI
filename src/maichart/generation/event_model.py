"""Audio-conditioned autoregressive model for event-level chart generation.

The decoder jointly predicts rational event positions and Simai note
characters. Note characters preserve combinations not seen as a whole token
in training, including simultaneous mixed note types and slide branches.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn

from maichart.generation.conditions import CONDITION_KEYS
from maichart.generation.music_plan import (
    CONTINUOUS_COUNT, LEGACY_PROFILE, MUSIC_STRUCTURE_PROFILE, PLAN_DIM,
    continuous_keys, is_music_structure, plan_values, target_to_logits,
    validate_profile,
)

from maichart.generation.action_state_spec import (
    STATE_DIM as ACTION_STATE_DIM, MAX_STATES as MAX_ACTION_STATE_TOKENS, STATE_KEYS,
)

EVENT_PAD = "<pad>"
EVENT_BOS = "<bos>"
EVENT_EOS = "<eos>"
EVENT_UNK = "<unk>"
EVENT_END = "<event_end>"
EVENT_SPECIALS = (EVENT_PAD, EVENT_BOS, EVENT_EOS, EVENT_UNK, EVENT_END)
NOTE_CHARACTERS = "1234567809ABCDEbhxf-<>^pqszvwV*/[]:#."


def fraction_text(value: Fraction) -> str:
    value = Fraction(value)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def position_token(value: Fraction) -> str:
    return f"AT:{fraction_text(value)}"


def char_token(char: str) -> str:
    return f"CHAR:{char}"


@dataclass(slots=True)
class EventSequence:
    """A relative-time event target beginning at beat zero."""

    beats: list[Fraction]
    note_texts: list[str]


class EventVocabulary:
    """Vocabulary for exact local beat positions and compositional note text."""

    def __init__(self, tokens: Iterable[str] | None = None) -> None:
        values = list(tokens or EVENT_SPECIALS)
        for token in EVENT_SPECIALS:
            if token not in values:
                values.insert(0, token)
        self.tokens = tuple(dict.fromkeys(values))
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        self.id_to_token = {index: token for index, token in enumerate(self.tokens)}

    @property
    def pad_id(self) -> int:
        return self.token_to_id[EVENT_PAD]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[EVENT_BOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EVENT_EOS]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[EVENT_UNK]

    def __len__(self) -> int:
        return len(self.tokens)

    @classmethod
    def from_event_charts(cls, charts: Iterable[Any], *, segment_beats: int = 16,
                          max_position_denominator: int = 384,
                          extra_specials: Sequence[str] = ()) -> "EventVocabulary":
        """Build the position alphabet from observed charts.

        A ``{division}`` directive sets local subdivision, which is how Simai
        spends precision only where the music needs it.  The grid below is the
        conventional layer, and every observed beat is added on top so an
        unusual but real subdivision survives.

        ``max_position_denominator`` rejects positions off any musical grid.  A
        single chart carrying a ``{999}`` directive generates denominators such
        as 7992 (= lcm(8, 999)) that are not subdivisions any author writes, and
        admitting them inflated one production vocabulary by 95 dead entries
        (~9.4%) for ~0.09% of the training targets.  Genuinely fine grids --
        including 96, 192 and 384 -- are still accepted.

        ``extra_specials`` appends family-specific control tokens after the
        shared ones.  Empty by default, so the event_v1 alphabet (and therefore
        every existing checkpoint) is unchanged; the cross-window family adds
        its context delimiters here and gets a different vocabulary signature,
        which is exactly what stops the two families from loading each other's
        weights.
        """

        values = (list(EVENT_SPECIALS) + list(extra_specials)
                  + [char_token(char) for char in dict.fromkeys(NOTE_CHARACTERS)])
        positions = {
            Fraction(numerator, denominator)
            for denominator in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
            for numerator in range(segment_beats * denominator)
        }
        rejected: set[Fraction] = set()
        for chart in charts:
            for point in chart.points:
                if not str(point.note_text).strip():
                    continue
                relative = Fraction(point.beat) % segment_beats
                if relative.denominator > max_position_denominator:
                    rejected.add(relative)
                else:
                    positions.add(relative)
                for char in "".join(point.note_text.split()):
                    if char not in NOTE_CHARACTERS:
                        raise ValueError(f"Unsupported note character {char!r}; document must be inspected.")
        vocabulary = cls(values + [position_token(beat) for beat in sorted(positions)])
        vocabulary.rejected_positions = tuple(sorted(rejected))
        return vocabulary

    def encode(self, sequence: EventSequence) -> list[int]:
        if len(sequence.beats) != len(sequence.note_texts):
            raise ValueError("EventSequence beats and note_texts must have equal length.")
        result = [self.bos_id]
        previous: Fraction | None = None
        for beat, text in zip(sequence.beats, sequence.note_texts):
            beat = Fraction(beat)
            if beat < 0 or (previous is not None and beat <= previous):
                raise ValueError("Event beats must be strictly increasing after event grouping.")
            event = [position_token(beat)] + [char_token(char) for char in "".join(text.split())] + [EVENT_END]
            missing = [token for token in event if token not in self.token_to_id]
            if missing:
                raise ValueError(f"Target is outside the checkpoint vocabulary: {missing[:5]}")
            result.extend(self.token_to_id[token] for token in event)
            previous = beat
        result.append(self.eos_id)
        return result

    def decode(self, ids: Iterable[int]) -> EventSequence:
        tokens = [self.id_to_token.get(int(index), EVENT_UNK) for index in ids]
        beats: list[Fraction] = []
        texts: list[str] = []
        pending_beat: Fraction | None = None
        characters: list[str] = []
        for token in tokens:
            if token in {EVENT_PAD, EVENT_BOS}:
                continue
            if token == EVENT_EOS:
                break
            if token.startswith("AT:"):
                if pending_beat is not None:
                    raise ValueError("Unclosed event before next position.")
                pending_beat = Fraction(token[3:])
                characters = []
                continue
            if token.startswith("CHAR:") and pending_beat is not None:
                characters.append(token[5:])
            elif token == EVENT_END and pending_beat is not None:
                if not characters or (beats and pending_beat <= beats[-1]):
                    raise ValueError("Empty event or non-monotonic event positions.")
                beats.append(pending_beat)
                texts.append("".join(characters))
                pending_beat = None
            else:
                raise ValueError(f"Invalid event token {token!r}.")
        if pending_beat is not None:
            raise ValueError("Decoder stopped inside an unfinished event.")
        return EventSequence(beats=beats, note_texts=texts)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "maichart-event-vocabulary-v1", "tokens": list(self.tokens)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EventVocabulary":
        tokens = value.get("tokens")
        if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
            raise ValueError("Event vocabulary must contain a string list 'tokens'.")
        return cls(tokens)


@dataclass(slots=True)
class EventTransformerConfig:
    # Defaults follow maichart.generation.event_audio: 112 feature columns and
    # a GLOBAL_MEMORY_FRAMES-sized full-song memory.  Training passes both
    # explicitly, so these defaults only matter for ad-hoc construction.
    input_dim: int = 112
    condition_dim: int = len(CONDITION_KEYS)
    vocab_size: int = 4
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_audio_tokens: int = 1024
    # Must equal the number of global-memory frames the dataset produces
    # (GLOBAL_MEMORY_FRAMES); the positional table is sized to hold both the
    # segment frames and these global frames, so a mismatch would index past
    # the table.  encode_audio validates it with a clear message.
    global_audio_tokens: int = 256
    max_tokens: int = 2048
    # How the 16-dim condition reaches the encoder.
    #
    # "memory_token" (current): the condition is projected once and appended to
    # the encoder memory as its own token, so cross-attention can read it.
    # "per_frame_bias" (checkpoints trained before the condition token): the
    # same projected vector was added to every audio frame.  Those checkpoints
    # have no ``condition_type_embedding`` parameter and must run with this
    # fusion, or the conditioning they were trained with vanishes.  The loader
    # picks the value from the weights (see ``load_weights_compatible``).
    condition_fusion: str = "memory_token"
    # Five continuous window controls plus four configuration-class logits.
    plan_dim: int = 9
    planner_profile: str = LEGACY_PROFILE
    onset_profile: str = "none"
    local_pooling_policy: str = "legacy_uniform"
    action_state_profile: str = "none"

    def __post_init__(self) -> None:
        validate_profile(self.planner_profile)
        if is_music_structure(self.planner_profile):
            # The head width is determined by the profile, never by a caller's
            # number: a mismatch would shift every configuration logit.
            self.plan_dim = PLAN_DIM[self.planner_profile]
        elif self.plan_dim not in {6, PLAN_DIM[LEGACY_PROFILE]}:
            raise ValueError("legacy_v1 requires plan_dim=6 or 9")
        if self.onset_profile not in {"none", "chart_onsets_v1"}:
            raise ValueError("onset_profile must be 'none' or 'chart_onsets_v1'.")
        if self.local_pooling_policy not in {"legacy_uniform", "per_sample_v2"}:
            raise ValueError("local_pooling_policy must be 'legacy_uniform' or 'per_sample_v2'.")
        if self.onset_profile == "chart_onsets_v1" and self.local_pooling_policy != "per_sample_v2":
            raise ValueError("chart_onsets_v1 requires local_pooling_policy='per_sample_v2'.")
        if self.action_state_profile not in {"none", "sustain_state_v1"}:
            raise ValueError("action_state_profile must be 'none' or 'sustain_state_v1'.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def pool_local_tracks(
    tracks: torch.Tensor,
    padding_mask: torch.Tensor | None,
    max_audio_tokens: int,
    policy: str = "legacy_uniform",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool local audio or aligned frame tracks and return their padding mask.

    ``legacy_uniform`` preserves the historical batch-level behavior: if the
    padded batch length exceeds the cap, every sample's valid frames are pooled
    or expanded to exactly ``max_audio_tokens``. ``per_sample_v2`` only pools
    samples that individually exceed the cap, so shorter samples keep their
    original frame grid when batched beside a longer song window.
    """
    if tracks.ndim != 3:
        raise ValueError("tracks must have shape [B, T, C].")
    if max_audio_tokens <= 0:
        raise ValueError("max_audio_tokens must be positive.")
    if policy not in {"legacy_uniform", "per_sample_v2"}:
        raise ValueError("unknown local pooling policy.")
    batch, length, _ = tracks.shape
    if padding_mask is None:
        padding_mask = torch.zeros((batch, length), dtype=torch.bool, device=tracks.device)
    elif padding_mask.shape != (batch, length) or padding_mask.dtype != torch.bool:
        raise ValueError("padding_mask must be bool with shape [B, T].")
    else:
        padding_mask = padding_mask.to(device=tracks.device)

    if policy == "legacy_uniform":
        if length <= max_audio_tokens:
            return tracks, padding_mask
        pooled = []
        for index in range(batch):
            valid = tracks[index][~padding_mask[index]]
            if valid.size(0) == 0:
                raise ValueError("each sample must contain at least one valid local frame.")
            pooled.append(torch.nn.functional.adaptive_avg_pool1d(
                valid.transpose(0, 1).unsqueeze(0), max_audio_tokens)[0].transpose(0, 1))
        result = torch.stack(pooled)
        return result, torch.zeros((batch, max_audio_tokens), dtype=torch.bool, device=tracks.device)

    pooled_samples = []
    lengths = []
    for index in range(batch):
        valid = tracks[index][~padding_mask[index]]
        if valid.size(0) == 0:
            raise ValueError("each sample must contain at least one valid local frame.")
        if valid.size(0) > max_audio_tokens:
            valid = torch.nn.functional.adaptive_avg_pool1d(
                valid.transpose(0, 1).unsqueeze(0), max_audio_tokens)[0].transpose(0, 1)
        pooled_samples.append(valid)
        lengths.append(valid.size(0))
    output_length = max(lengths)
    result = tracks.new_zeros((batch, output_length, tracks.size(-1)))
    result_mask = torch.ones((batch, output_length), dtype=torch.bool, device=tracks.device)
    for index, valid in enumerate(pooled_samples):
        result[index, :valid.size(0)] = valid
        result_mask[index, :valid.size(0)] = False
    return result, result_mask

def _cached_decoder_layer(
    layer: nn.TransformerDecoderLayer,
    hidden: torch.Tensor,
    memory: torch.Tensor,
    past_key: torch.Tensor | None,
    past_value: torch.Tensor | None,
    memory_key_padding_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Run one pre-norm decoder layer for a single new position.

    ``nn.TransformerDecoderLayer`` offers no incremental API, so this mirrors its
    pre-norm branch exactly while reusing cached self-attention keys/values.
    Only the newest query is projected and attended, which is what removes the
    quadratic re-computation of the uncached path.

    When ``past_key`` is ``None`` the layer falls back to a single-position
    causal self-attention, so a caller can start cold without special-casing.
    """

    size = hidden.size(1)
    if size != 1:
        raise ValueError("_cached_decoder_layer expects exactly one new position.")

    def _project(module: nn.MultiheadAttention, inputs: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(inputs, module.in_proj_weight, module.in_proj_bias)

    # Self-attention.  The cached keys/values must store the *normalized* inputs,
    # because norm_first normalizes the whole sequence before projecting; caching
    # raw hidden states while querying with norm1(hidden) would attend every new
    # position against a differently-scaled prefix.
    if layer.norm_first:
        normalized = layer.norm1(hidden)
    else:
        normalized = hidden
    projected = _project(layer.self_attn, normalized)
    query, key, value = projected.chunk(3, dim=-1)
    keys = key if past_key is None else torch.cat([past_key, key], dim=1)
    values = value if past_value is None else torch.cat([past_value, value], dim=1)
    attended = _attention(layer.self_attn, query, keys, values, mask=None)
    hidden = hidden + layer.dropout1(attended)
    if not layer.norm_first:
        hidden = layer.norm1(hidden)

    # Cross-attention: the query is normalized, the encoder memory is not.
    if layer.norm_first:
        query_source = layer.norm2(hidden)
    else:
        query_source = hidden
    query = _project(layer.multihead_attn, query_source).chunk(3, dim=-1)[0]
    cross = _project(layer.multihead_attn, memory)
    _, key, value = cross.chunk(3, dim=-1)
    cross_mask = None
    if memory_key_padding_mask is not None:
        cross_mask = torch.zeros(
            memory_key_padding_mask.shape, dtype=query.dtype, device=query.device
        ).masked_fill(memory_key_padding_mask, float("-inf"))
    attended = _attention(layer.multihead_attn, query, key, value, mask=cross_mask)
    hidden = hidden + layer.dropout2(attended)
    if not layer.norm_first:
        hidden = layer.norm2(hidden)

    source = layer.norm3(hidden) if layer.norm_first else hidden
    feedforward = layer.linear2(layer.dropout(layer.activation(layer.linear1(source))))
    hidden = hidden + layer.dropout3(feedforward)
    if not layer.norm_first:
        hidden = layer.norm3(hidden)

    return hidden, (keys, values)


def _attention(
    module: nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Scaled dot-product attention matching ``nn.MultiheadAttention`` shapes."""

    batch, length, _ = query.shape
    heads = module.num_heads
    head_dim = module.head_dim

    def split(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(batch, -1, heads, head_dim).transpose(1, 2)

    # ``mask`` here is additive over key positions; broadcast over heads.
    additive = None
    if mask is not None:
        additive = mask.view(batch, 1, 1, -1) if mask.dim() == 2 else mask
    # The scale must be 1/sqrt(head_dim) to match nn.MultiheadAttention.
    # scaled_dot_product_attention defaults to 1/sqrt(query.size(-1)), and for
    # this model head_dim (64) differs from d_model (256), so omitting ``scale``
    # would silently produce a differently-scaled attention.
    attended = torch.nn.functional.scaled_dot_product_attention(
        split(query), split(key), split(value), attn_mask=additive, dropout_p=0.0,
        scale=1.0 / math.sqrt(head_dim),
    )
    attended = attended.transpose(1, 2).reshape(batch, -1, heads * head_dim)
    return module.out_proj(attended)


class EventConditionedTransformer(nn.Module):
    """Encoder-decoder Transformer with an audio memory and causal decoder."""

    def __init__(self, config: EventTransformerConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        self.config = config or EventTransformerConfig(**kwargs)
        if self.config.d_model % self.config.nhead:
            raise ValueError("d_model must be divisible by nhead.")
        if self.config.condition_fusion not in {"memory_token", "per_frame_bias"}:
            raise ValueError(
                f"unknown condition_fusion {self.config.condition_fusion!r}; "
                "expected 'memory_token' or 'per_frame_bias'."
            )
        self.audio_projection = nn.Linear(self.config.input_dim, self.config.d_model)
        self.condition_projection = nn.Sequential(
            nn.Linear(self.config.condition_dim, self.config.d_model),
            nn.GELU(),
            nn.Linear(self.config.d_model, self.config.d_model),
        )
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.position_embedding = nn.Embedding(self.config.max_tokens, self.config.d_model)
        self.audio_position_embedding = nn.Embedding(self.config.max_audio_tokens + self.config.global_audio_tokens, self.config.d_model)
        self.global_type_embedding = nn.Parameter(torch.zeros(self.config.d_model))
        self.condition_type_embedding = nn.Parameter(torch.zeros(self.config.d_model))
        self.condition_null_embedding = nn.Parameter(torch.zeros(self.config.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.config.num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.config.num_layers)
        self.norm = nn.LayerNorm(self.config.d_model)
        self.output = nn.Linear(self.config.d_model, self.config.vocab_size)
        self.planner_profile = validate_profile(self.config.planner_profile)
        planner_input = (self.config.d_model * 3 if is_music_structure(self.planner_profile)
                         else self.config.d_model)
        self.plan_pool_projection = (nn.Sequential(
            nn.Linear(planner_input, self.config.d_model), nn.GELU(),
            nn.Linear(self.config.d_model, self.config.d_model)
        ) if is_music_structure(self.planner_profile) else nn.Identity())
        self.plan_head = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            nn.GELU(),
            nn.Linear(self.config.d_model // 2, self.config.plan_dim),
        )
        self.plan_condition = nn.Linear(self.config.plan_dim, self.config.d_model)
        self.onset_head = (nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            nn.GELU(),
            nn.Linear(self.config.d_model // 2, 1),
        ) if self.config.onset_profile == "chart_onsets_v1" else None)
        if self.config.action_state_profile == "sustain_state_v1":
            self.action_state_projection = nn.Linear(ACTION_STATE_DIM, self.config.d_model)
            self.action_state_type_embedding = nn.Parameter(torch.zeros(self.config.d_model))
        self.planner_enabled = True

    def load_weights_compatible(self, state_dict: Mapping[str, torch.Tensor], *,
                                allow_missing_onset_head: bool = False,
                                allow_missing_action_state: bool = False) -> dict[str, Any]:
        """Load weights, accepting a checkpoint from before the condition token.

        The condition-token architecture moved the condition from a per-frame
        additive bias to its own encoder memory token and added
        ``condition_type_embedding``.  A checkpoint trained before that has no
        such parameter: loading it strictly raises "Missing key(s)", and loading
        it into the new fusion path would silently drop the conditioning the
        weights were trained with.

        So the presence of that parameter decides the fusion mode for this
        instance, and only it is tolerated as a missing key.  Any other mismatch
        is still a hard error, because it means the checkpoint is not simply
        older -- it is a different model.
        """

        def present(name: str) -> bool:
            return any(key == name or key.endswith("." + name) for key in state_dict)

        action_state_keys = STATE_KEYS
        action_state_present = {key for key in action_state_keys if present(key)}
        if self.config.action_state_profile == "sustain_state_v1":
            if action_state_present and action_state_present != action_state_keys:
                raise ValueError("checkpoint has an incomplete action-state projection.")
            if not action_state_present and not allow_missing_action_state:
                raise ValueError("checkpoint enables sustain_state_v1 but has no trained action-state projection.")
        elif action_state_present:
            raise ValueError("checkpoint contains an action-state projection but action_state_profile is disabled.")

        has_condition_token = present("condition_type_embedding")
        # The recorded fusion mode wins when the checkpoint carries one.  A run
        # that loaded a pre-token checkpoint, finetuned it and saved again keeps
        # fusion="per_frame_bias" in its config while the (unused) parameter is
        # still serialized; trusting the key's presence alone would silently
        # switch it back to the new path and change its behaviour.
        recorded = self.config.condition_fusion
        if not has_condition_token or recorded == "per_frame_bias":
            self.config.condition_fusion = "per_frame_bias"
        else:
            self.config.condition_fusion = "memory_token"
        result = self.load_state_dict(state_dict, strict=False)
        planner_keys = {
            "plan_head.0.weight", "plan_head.0.bias", "plan_head.2.weight", "plan_head.2.bias",
            "plan_condition.weight", "plan_condition.bias",
        }
        has_planner = all(present(key) for key in planner_keys)
        # Old event checkpoints predate planner injection. Keep them usable for
        # generation without letting randomly initialized planner weights alter
        # their behavior; a fresh training run re-enables the head explicitly.
        self.planner_enabled = has_planner
        tolerated = planner_keys | {"condition_null_embedding"}
        if not has_condition_token:
            tolerated.add("condition_type_embedding")
        if self.config.action_state_profile == "sustain_state_v1" and not action_state_present:
            tolerated.update(action_state_keys)
        onset_keys = {"onset_head.0.weight", "onset_head.0.bias",
                      "onset_head.2.weight", "onset_head.2.bias"}
        onset_present = {key for key in onset_keys if present(key)}
        if self.config.onset_profile == "chart_onsets_v1":
            if onset_present and onset_present != onset_keys:
                raise ValueError("checkpoint has an incomplete local onset head.")
            if not onset_present:
                if not allow_missing_onset_head:
                    raise ValueError("checkpoint enables chart_onsets_v1 but has no trained local onset head.")
                tolerated.update(onset_keys)
        elif onset_present:
            raise ValueError("checkpoint contains a local onset head but onset_profile is disabled.")
        missing = [key for key in result.missing_keys if key not in tolerated]
        unexpected = list(result.unexpected_keys)
        if missing or unexpected:
            raise ValueError(
                "checkpoint 与当前模型结构不兼容，无法加载："
                f"缺少={missing[:5]}，多余={unexpected[:5]}。"
                "请确认该 checkpoint 来自同一次训练；结构不同的模型必须重新训练。"
            )
        return {"condition_fusion": self.config.condition_fusion, "legacy": not has_condition_token,
                "planner": has_planner, "onset_head": onset_present == onset_keys,
                "action_state": action_state_present == action_state_keys}

    def local_onset_logits(self, memory: torch.Tensor) -> torch.Tensor:
        """Predict chart-onset logits from the local encoder-memory region."""
        if self.onset_head is None:
            raise ValueError("local onset prediction requires onset_profile='chart_onsets_v1'.")
        layout = getattr(memory, "_planner_layout", None)
        if not layout:
            raise ValueError("encoded memory is missing its local/global layout.")
        local_len = int(layout[0])
        if local_len <= 0 or local_len > memory.size(1):
            raise ValueError("encoded memory has an invalid local region length.")
        return self.onset_head(memory[:, :local_len]).squeeze(-1)

    def append_plan_memory(
        self,
        memory: torch.Tensor,
        audio_padding_mask: torch.Tensor | None,
        plan_override: torch.Tensor | None = None,
        plan_controls: Mapping[str, float] | None = None,
        plan_target: torch.Tensor | None = None,
        plan_teacher_mix: float | torch.Tensor = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Append the window planner token used by both training and decoding."""

        if not self.planner_enabled:
            plan_pred = torch.zeros(
                (memory.size(0), self.config.plan_dim), dtype=memory.dtype, device=memory.device
            ) if plan_override is None else plan_override.to(memory.dtype)
            return memory, audio_padding_mask, plan_pred
        if plan_override is None:
            layout = getattr(memory, "_planner_layout", None)
            if is_music_structure(self.planner_profile) and layout is not None:
                local_len, global_len, condition_len = (int(value) for value in layout)
                regions = (memory[:, :local_len],
                           memory[:, local_len:local_len + global_len],
                           memory[:, local_len + global_len:local_len + global_len + condition_len])
                pooled_regions = []
                for offset, region in zip((0, local_len, local_len + global_len), regions):
                    if region.size(1) == 0:
                        pooled_regions.append(memory.new_zeros((memory.size(0), memory.size(-1))))
                        continue
                    if audio_padding_mask is None:
                        valid = torch.ones((memory.size(0), region.size(1), 1), dtype=memory.dtype, device=memory.device)
                    else:
                        valid = (~audio_padding_mask[:, offset:offset + region.size(1)]).to(memory.dtype).unsqueeze(-1)
                    if offset == 0 and getattr(memory, "_planner_core_mask", None) is not None:
                        core_mask = memory._planner_core_mask[:, :region.size(1)]
                        valid = valid * (~core_mask).to(memory.dtype).unsqueeze(-1)
                    pooled_regions.append((region * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0))
                pooled = self.plan_pool_projection(torch.cat(pooled_regions, dim=-1))
            else:
                valid = (~audio_padding_mask).to(memory.dtype).unsqueeze(-1) if audio_padding_mask is not None else torch.ones_like(memory[:, :, :1])
                pooled = (memory * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            plan_pred = self.plan_head(pooled)
            raw_plan_pred = plan_pred
            if plan_controls:
                # The first five planner channels are sigmoid-regressed
                # continuous controls (count, density, chord, slide, last).
                # Replace only explicitly requested channels in probability
                # space, preserving the four configuration logits predicted by
                # the checkpoint.  This keeps the 9-D planner contract intact.
                names = continuous_keys(self.planner_profile)
                adjusted = plan_pred.clone()
                for index, name in enumerate(names):
                    if name not in plan_controls or index >= adjusted.size(1):
                        continue
                    raw_value = plan_controls[name]
                    value = float(raw_value)
                    if not math.isfinite(value):
                        continue
                    value = min(1.0 - 1e-5, max(1e-5, value))
                    strength = plan_controls.get("__strength", 1.0)
                    strength = torch.as_tensor(strength, device=adjusted.device, dtype=adjusted.dtype).reshape(-1, 1).clamp(0.0, 1.0)
                    if strength.size(0) == 1 and adjusted.size(0) != 1:
                        strength = strength.expand(adjusted.size(0), 1)
                    desired = torch.full((adjusted.size(0),), value, device=adjusted.device, dtype=adjusted.dtype)
                    current = plan_values(raw_plan_pred, profile=self.planner_profile)[:, index]
                    adjusted[:, index] = torch.lerp(current, desired, strength[:, 0]).clamp(1e-5, 1 - 1e-5).logit()
                plan_pred = adjusted
                # Fine-note preference is relative to THIS window's prediction.
                # 0.5 is neutral; endpoints shift odds by at most exp(+/-2),
                # retaining musical variation and avoiding a fixed song-wide
                # fine-note quota. Curve __strength affects other controls only.
                if "fine_rhythm" in names and "__fine_rhythm_bias" in plan_controls:
                    bias = float(plan_controls["__fine_rhythm_bias"])
                    if not math.isfinite(bias) or not 0.0 <= bias <= 1.0:
                        raise ValueError("fine-note-strength must be finite in 0..1")
                    channel = names.index("fine_rhythm")
                    plan_pred[:, channel] = raw_plan_pred[:, channel] + 4.0 * (bias - 0.5)
        else:
            plan_pred = plan_override.to(device=memory.device, dtype=memory.dtype)
            if plan_pred.ndim == 1:
                plan_pred = plan_pred.unsqueeze(0)
            if plan_pred.size(0) != memory.size(0) or plan_pred.size(1) != self.config.plan_dim:
                raise ValueError(
                    f"plan_override must have shape [{memory.size(0)}, {self.config.plan_dim}], "
                    f"got {tuple(plan_pred.shape)}."
                )
        raw_plan = raw_plan_pred if plan_override is None else plan_pred
        if plan_target is not None:
            target_logits = target_to_logits(plan_target.to(device=memory.device), profile=self.planner_profile).to(memory.dtype)
            mix = torch.as_tensor(plan_teacher_mix, device=memory.device, dtype=memory.dtype)
            if mix.ndim == 0:
                mix = mix.reshape(1, 1)
            elif mix.ndim == 1:
                mix = mix.reshape(-1, 1)
            mix = mix.clamp(0.0, 1.0)
            if mix.size(0) == 1 and memory.size(0) != 1:
                mix = mix.expand(memory.size(0), 1)
            if mix.size(0) != memory.size(0):
                raise ValueError("plan_teacher_mix must be scalar or have one value per batch item.")
            predicted_values = plan_values(plan_pred, profile=self.planner_profile)
            target_values = plan_values(target_logits, profile=self.planner_profile)
            mixed_values = torch.lerp(predicted_values, target_values, mix)
            # Both structure profiles consume probability-space values; the
            # legacy path historically received the same tensor, so one branch
            # suffices and is kept explicit for the contract comment.
            plan_hidden_input = mixed_values
        else:
            plan_hidden_input = plan_values(plan_pred, profile=self.planner_profile)
        plan_hidden = self.plan_condition(plan_hidden_input).unsqueeze(1)
        memory = torch.cat([memory, plan_hidden], dim=1)
        if audio_padding_mask is not None:
            plan_mask = torch.zeros((plan_hidden.size(0), 1), dtype=torch.bool, device=memory.device)
            audio_padding_mask = torch.cat([audio_padding_mask, plan_mask], dim=1)
        # v2 exposes raw logits for auxiliary supervision; legacy callers relied
        # on receiving the controlled tensor and retain that behavior.
        return memory, audio_padding_mask, (
            raw_plan if is_music_structure(self.planner_profile) else plan_pred)

    def encode_audio(
        self,
        audio: torch.Tensor,
        condition: torch.Tensor,
        audio_padding_mask: torch.Tensor | None = None,
        global_audio: torch.Tensor | None = None,
        condition_null: torch.Tensor | None = None,
        plan_audio_mask: torch.Tensor | None = None,
        action_state: torch.Tensor | None = None,
        action_state_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if audio.ndim != 3 or condition.ndim != 2 or audio.size(0) != condition.size(0):
            raise ValueError("audio must be [B,T,F] and condition must be [B,C].")
        if audio.size(-1) != self.config.input_dim:
            raise ValueError(f"Expected {self.config.input_dim} audio features, got {audio.size(-1)}.")
        action_state_enabled = self.config.action_state_profile == "sustain_state_v1"
        if not action_state_enabled and (action_state is not None or action_state_padding_mask is not None):
            raise ValueError("action-state inputs require action_state_profile='sustain_state_v1'.")
        if action_state_enabled and action_state is None:
            raise ValueError("action_state is required when action_state_profile='sustain_state_v1'.")
        if action_state is not None:
            if (action_state.ndim != 3 or action_state.size(0) != audio.size(0)
                    or action_state.size(2) != ACTION_STATE_DIM
                    or not 1 <= action_state.size(1) <= MAX_ACTION_STATE_TOKENS):
                raise ValueError(
                    f"action_state must have shape [B,N,{ACTION_STATE_DIM}] with "
                    f"1 <= N <= {MAX_ACTION_STATE_TOKENS} and match the audio batch."
                )
            if action_state_padding_mask is None:
                action_state_padding_mask = torch.zeros(
                    action_state.shape[:2], dtype=torch.bool, device=action_state.device)
            elif (action_state_padding_mask.shape != action_state.shape[:2]
                  or action_state_padding_mask.dtype != torch.bool):
                raise ValueError("action_state_padding_mask must be bool with shape [B,N].")
            action_state = action_state.to(device=audio.device, dtype=audio.dtype)
            action_state_padding_mask = action_state_padding_mask.to(device=audio.device)
        if plan_audio_mask is not None:
            if plan_audio_mask.shape != audio.shape[:2] or plan_audio_mask.dtype != torch.bool:
                raise ValueError("plan_audio_mask must be bool with shape [B, local_audio_tokens].")
            plan_audio_mask = plan_audio_mask.to(device=audio.device)
        original_length = audio.size(1)
        original_padding_mask = audio_padding_mask
        planner_core_mask = plan_audio_mask
        if planner_core_mask is not None:
            tracks = torch.cat([audio, planner_core_mask.to(audio.dtype).unsqueeze(-1)], dim=-1)
        else:
            tracks = audio
        tracks, pooled_padding_mask = pool_local_tracks(
            tracks, audio_padding_mask, self.config.max_audio_tokens,
            self.config.local_pooling_policy)
        if planner_core_mask is not None:
            audio = tracks[:, :, :self.config.input_dim]
            planner_core_mask = tracks[:, :, self.config.input_dim] > 0.5
        else:
            audio = tracks
        if (self.config.local_pooling_policy == "legacy_uniform"
                and original_length > self.config.max_audio_tokens):
            # Preserve the old path, which treated all pooled bins as valid.
            audio_padding_mask = None
        elif original_padding_mask is None:
            audio_padding_mask = None
        else:
            audio_padding_mask = pooled_padding_mask
        hidden = self.audio_projection(audio)
        positions = torch.arange(hidden.size(1), device=audio.device)
        hidden = hidden + self.audio_position_embedding(positions).unsqueeze(0)
        cond = self.condition_projection(condition).unsqueeze(1)
        legacy_fusion = self.config.condition_fusion == "per_frame_bias"
        if legacy_fusion:
            # Pre-condition-token architecture: the condition was folded into
            # every encoder frame.  A checkpoint trained that way carries no
            # ``condition_type_embedding`` and only makes sense with this path.
            hidden = hidden + cond
        if global_audio is not None:
            if global_audio.size(-1) != self.config.input_dim:
                raise ValueError(f"Expected {self.config.input_dim} global audio features, got {global_audio.size(-1)}.")
            if global_audio.size(1) > self.config.global_audio_tokens:
                raise ValueError(
                    f"global audio has {global_audio.size(1)} frames but this model holds "
                    f"{self.config.global_audio_tokens}; rebuild the dataset with the matching "
                    f"global-memory frame count or train a model with a larger global_audio_tokens."
                )
            global_hidden = self.audio_projection(global_audio)
            global_positions = torch.arange(global_hidden.size(1), device=audio.device) + self.config.max_audio_tokens
            global_hidden = global_hidden + self.audio_position_embedding(global_positions).unsqueeze(0)
            global_hidden = global_hidden + self.global_type_embedding
            if legacy_fusion:
                global_hidden = global_hidden + cond
            hidden = torch.cat([hidden, global_hidden], dim=1)
            if audio_padding_mask is not None:
                global_mask = torch.zeros(global_hidden.shape[:2], dtype=torch.bool, device=audio.device)
                audio_padding_mask = torch.cat([audio_padding_mask, global_mask], dim=1)
        if legacy_fusion:
            if audio_padding_mask is not None and audio_padding_mask.size(1) != hidden.size(1):
                raise ValueError("audio_padding_mask length does not match encoded audio memory.")
            state_count = 0
            if action_state_enabled:
                if audio_padding_mask is None:
                    audio_padding_mask = torch.zeros(hidden.shape[:2], dtype=torch.bool, device=audio.device)
                state_hidden = (self.action_state_projection(action_state)
                                + self.action_state_type_embedding)
                hidden = torch.cat([hidden, state_hidden], dim=1)
                audio_padding_mask = torch.cat([audio_padding_mask, action_state_padding_mask], dim=1)
                state_count = action_state.size(1)
            memory = self.encoder(hidden, src_key_padding_mask=audio_padding_mask)
            memory._planner_layout = (audio.size(1), global_audio.size(1) if global_audio is not None else 0,
                                      state_count)
            memory._planner_core_mask = planner_core_mask
            return memory, audio_padding_mask
        condition_hidden = cond + self.condition_type_embedding
        if condition_null is not None:
            null_mask = condition_null.to(device=audio.device, dtype=condition_hidden.dtype).reshape(-1, 1, 1)
            condition_hidden = condition_hidden + null_mask * self.condition_null_embedding.reshape(1, 1, -1)
        if audio_padding_mask is None:
            audio_padding_mask = torch.zeros(hidden.shape[:2], dtype=torch.bool, device=audio.device)
        elif audio_padding_mask.size(1) != hidden.size(1):
            raise ValueError("audio_padding_mask length does not match encoded audio memory.")
        hidden = torch.cat([hidden, condition_hidden], dim=1)
        condition_mask = torch.zeros((condition_hidden.size(0), 1), dtype=torch.bool,
                                     device=audio.device)
        audio_padding_mask = torch.cat([audio_padding_mask, condition_mask], dim=1)
        state_count = 0
        if action_state_enabled:
            state_hidden = (self.action_state_projection(action_state)
                            + self.action_state_type_embedding)
            hidden = torch.cat([hidden, state_hidden], dim=1)
            audio_padding_mask = torch.cat([audio_padding_mask, action_state_padding_mask], dim=1)
            state_count = action_state.size(1)
        memory = self.encoder(hidden, src_key_padding_mask=audio_padding_mask)
        memory._planner_layout = (audio.size(1), global_audio.size(1) if global_audio is not None else 0,
                                  1 + state_count)
        memory._planner_core_mask = planner_core_mask
        return memory, audio_padding_mask

    def forward(
        self,
        audio: torch.Tensor,
        decoder_input: torch.Tensor,
        condition: torch.Tensor,
        condition_null: torch.Tensor | None = None,
        audio_padding_mask: torch.Tensor | None = None,
        decoder_padding_mask: torch.Tensor | None = None,
        global_audio: torch.Tensor | None = None,
        plan_override: torch.Tensor | None = None,
        plan_controls: Mapping[str, float] | None = None,
        plan_target: torch.Tensor | None = None,
        plan_teacher_mix: float | torch.Tensor = 0.0,
        plan_audio_mask: torch.Tensor | None = None,
        return_plan: bool = False,
        return_onset: bool = False,
        action_state: torch.Tensor | None = None,
        action_state_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        memory, audio_padding_mask = self.encode_audio(
            audio, condition, audio_padding_mask, global_audio, condition_null,
            plan_audio_mask=plan_audio_mask, action_state=action_state,
            action_state_padding_mask=action_state_padding_mask)
        onset_logits = self.local_onset_logits(memory) if return_onset else None
        # Feed the predicted window plan back into the event decoder. The plan
        # is computed from encoder memory, so this remains causal with respect
        # to the target tokens while making density/configuration intent useful
        # beyond an auxiliary training metric.
        memory, audio_padding_mask, plan_pred = self.append_plan_memory(
            memory, audio_padding_mask, plan_override=plan_override,
            plan_controls=plan_controls, plan_target=plan_target,
            plan_teacher_mix=plan_teacher_mix)
        logits = self.decode_memory(memory, decoder_input, condition, audio_padding_mask, decoder_padding_mask)
        if return_onset:
            return logits, plan_pred, onset_logits
        if not return_plan:
            return logits
        return logits, plan_pred

    def decode_memory(
        self, memory: torch.Tensor, decoder_input: torch.Tensor, condition: torch.Tensor,
        audio_padding_mask: torch.Tensor | None = None,
        decoder_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if decoder_input.ndim != 2:
            raise ValueError("decoder_input must have shape [B,T].")
        if decoder_input.size(1) > self.config.max_tokens:
            raise ValueError("decoder sequence exceeds max_tokens.")
        positions = torch.arange(decoder_input.size(1), device=decoder_input.device)
        target = self.token_embedding(decoder_input) + self.position_embedding(positions).unsqueeze(0)
        target = target + self.condition_projection(condition).unsqueeze(1)
        size = decoder_input.size(1)
        causal = torch.triu(torch.ones(size, size, device=decoder_input.device, dtype=torch.bool), diagonal=1)
        hidden = self.decoder(
            target,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=decoder_padding_mask,
            memory_key_padding_mask=audio_padding_mask,
        )
        return self.output(self.norm(hidden))

    def decode_step(
        self,
        memory: torch.Tensor,
        token: torch.Tensor,
        condition: torch.Tensor,
        caches: list[tuple[torch.Tensor, torch.Tensor]],
        position: int,
        audio_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """One cached autoregressive step, returning logits for the new token.

        The uncached path re-runs the whole prefix through every decoder layer
        for each generated token, which makes generation quadratic in sequence
        length.  Autoregressive decoding is inherently sequential in the token
        dimension, so the quadratic term is pure waste: each step only needs the
        new position attended against the cached keys and values.  This turns a
        window from O(T^2) into O(T), which is what stops generation from
        stalling the GPU for minutes per window.

        ``caches`` holds one ``(key, value)`` pair per decoder layer, each shaped
        ``[B, S, d_model]`` with ``S`` growing by one per step. Heads are split
        internally by attention, after the cache is assembled. The caller owns the
        list and passes the returned one forward; nothing here mutates in place,
        so a failed step cannot corrupt a partially built prefix.
        """

        if token.ndim != 2 or token.size(1) != 1:
            raise ValueError("decode_step expects a single new token per sequence.")
        if position < 0 or position >= self.config.max_tokens:
            raise ValueError("decoder position exceeds max_tokens.")
        positions = torch.tensor([position], device=token.device, dtype=torch.long)
        hidden = self.token_embedding(token) + self.position_embedding(positions).unsqueeze(0)
        hidden = hidden + self.condition_projection(condition).unsqueeze(1)

        new_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_index, layer in enumerate(self.decoder.layers):
            past_key, past_value = caches[layer_index] if layer_index < len(caches) else (None, None)
            hidden, present = _cached_decoder_layer(layer, hidden, memory, past_key, past_value,
                                                     audio_padding_mask)
            new_caches.append(present)
        hidden = self.norm(hidden)
        return self.output(hidden), new_caches
