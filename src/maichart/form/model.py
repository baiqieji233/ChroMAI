"""Independent ``song_form_v1`` neural model for structured chart drafting."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class FormConfig:
    vocab_size: int
    d_model: int = 256
    nhead: int = 8
    layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    audio_dim: int = 120
    descriptor_dim: int = 504
    condition_dim: int = 16
    max_tokens: int = 2048
    max_audio_tokens: int = 1024
    max_memory_tokens: int = 512

    def __post_init__(self) -> None:
        for name in (
            "vocab_size", "d_model", "nhead", "layers", "dim_feedforward",
            "audio_dim", "descriptor_dim", "condition_dim", "max_tokens",
            "max_audio_tokens", "max_memory_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.d_model % self.nhead:
            raise ValueError("d_model must be divisible by nhead.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")


class FormTransformer(nn.Module):
    """Audio-conditioned form encoder with causal event-token decoding.

    Decoder inputs are the caller-authored prefix only. For draft examples,
    supplying draft or right-context tokens is rejected to catch target leaks.
    """

    architecture = "song_form_v1"
    _TOKEN_BRANCHES = ("history", "motif", "draft", "right")

    def __init__(self, config: FormConfig) -> None:
        super().__init__()
        self.config = config
        width = config.d_model

        self.token_embedding = nn.Embedding(config.vocab_size, width, padding_idx=0)
        self.position_embedding = nn.Embedding(
            max(config.max_tokens, config.max_audio_tokens, config.max_memory_tokens, 131),
            width,
        )
        self.branch_embedding = nn.Embedding(11, width)
        self.audio_projection = nn.Linear(config.audio_dim, width)
        self.form_projection = nn.Linear(config.descriptor_dim + 3, width)
        self.action_projection = nn.Linear(50, width)
        self.load_curve_projection = nn.Linear(4, width)
        self.condition_projection = nn.Linear(config.condition_dim, width)
        self.condition_null_embedding = nn.Parameter(torch.zeros(width))
        self.relation_projection = nn.Linear(4, width)
        self.task_embedding = nn.Embedding(2, width)
        self.motif_gate = nn.Sequential(
            nn.Linear(config.condition_dim + 4, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, 1),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.memory_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=width,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.layers)
        self.memory_norm = nn.LayerNorm(width)
        self.decoder_norm = nn.LayerNorm(width)
        self.output_head = nn.Linear(width, config.vocab_size)
        self.relation_head = nn.Linear(width, 2)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        memory, memory_mask = self.encode(batch)
        decoder_input = batch.get("decoder_input")
        if decoder_input is None:
            raise ValueError("batch must include decoder_input [B, L].")
        hidden = self.decode(decoder_input, memory, memory_mask, return_hidden=True)
        return {
            "logits": self.output_head(hidden),
            "relation_logits": self.relation_head(memory[:, -1]),
        }

    def encode(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode audio and authored memories once for one or more decode steps."""

        audio = self._required(batch, "audio", 3)
        if audio.size(-1) != self.config.audio_dim:
            raise ValueError(f"audio feature dimension must be {self.config.audio_dim}.")
        batch_size = audio.size(0)
        device = audio.device
        audio_mask = self._mask(batch, "audio_padding_mask", audio.shape[:2], device)
        if audio.size(1) > self.config.max_audio_tokens:
            raise ValueError("audio exceeds max_audio_tokens.")
        self._finite(audio, "audio")

        form = batch.get("form")
        if form is None:
            form = audio.new_zeros((batch_size, 0, self.config.descriptor_dim + 3))
        if form.ndim != 3 or form.shape[0] != batch_size or form.size(-1) != self.config.descriptor_dim + 3:
            raise ValueError(
                f"form must have shape [B, S, {self.config.descriptor_dim + 3}]."
            )
        if form.size(1) > 128:
            raise ValueError("form descriptor sequence cannot exceed 128 tokens.")
        form_mask = self._mask(batch, "form_padding_mask", form.shape[:2], device)
        self._finite(form, "form")

        condition = self._required(batch, "condition", 2)
        if condition.shape != (batch_size, self.config.condition_dim):
            raise ValueError(f"condition must have shape [B, {self.config.condition_dim}].")
        self._finite(condition, "condition")
        condition_null = batch.get("condition_null")
        if condition_null is None:
            # Legacy song_form checkpoints did not serialize this field; their
            # behavior remains conditioned exactly as before.
            condition_null = torch.zeros((batch_size,), dtype=torch.bool, device=device)
        if condition_null.shape not in ((batch_size,), (batch_size, 1)):
            raise ValueError("condition_null must have shape [B] or [B, 1].")
        condition_null = condition_null.to(device=device, dtype=torch.bool).reshape(batch_size)
        task = batch.get("task")
        if task is None or task.shape != (batch_size,):
            raise ValueError("task must have shape [B], with 0=draft and 1=revise.")
        task = task.to(device=device, dtype=torch.long)
        if torch.any((task < 0) | (task > 1)):
            raise ValueError("task values must be 0 (draft) or 1 (revise).")

        relation = batch.get("relation")
        if relation is None:
            relation = audio.new_zeros((batch_size, 4))
        if relation.shape != (batch_size, 4):
            raise ValueError("relation must have shape [B, 4].")
        self._finite(relation, "relation")
        strength = batch.get("memory_strength")
        if strength is None:
            strength = audio.new_ones((batch_size,))
        if strength.shape not in ((batch_size,), (batch_size, 1)):
            raise ValueError("memory_strength must have shape [B] or [B, 1].")
        strength = strength.to(device=device, dtype=audio.dtype).reshape(batch_size, 1)
        self._finite(strength, "memory_strength")
        if torch.any((strength < 0) | (strength > 1)):
            raise ValueError("memory_strength must be between 0 and 1.")

        branches: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        authored_lengths = 0

        def add(values: torch.Tensor, mask: torch.Tensor, branch_id: int) -> None:
            nonlocal authored_lengths
            if values.size(1) == 0:
                return
            if branch_id == 0:
                if values.size(1) > self.config.max_audio_tokens:
                    raise ValueError("audio exceeds max_audio_tokens.")
            elif branch_id == 1:
                if values.size(1) > 128:
                    raise ValueError("form descriptor sequence cannot exceed 128 tokens.")
                authored_lengths += values.size(1)
            elif branch_id in (2, 3, 4, 5, 10):
                if values.size(1) > self.config.max_memory_tokens:
                    raise ValueError("A chart-token memory branch exceeds max_memory_tokens.")
                authored_lengths += values.size(1)
            elif branch_id == 6:
                if values.size(1) > 32:
                    raise ValueError("action_state cannot exceed 32 tokens.")
                authored_lengths += values.size(1)
            positions = torch.arange(values.size(1), device=device).unsqueeze(0)
            values = (
                values
                + self.position_embedding(positions)
                + self.branch_embedding.weight[branch_id]
            )
            branches.append(values)
            masks.append(mask)

        audio_values = self.audio_projection(audio)
        add(audio_values, audio_mask, 0)
        add(self.form_projection(form), form_mask, 1)

        for branch_id, name in enumerate(self._TOKEN_BRANCHES, start=2):
            ids = batch.get(name)
            if ids is None:
                ids = torch.zeros((batch_size, 0), dtype=torch.long, device=device)
            if ids.ndim != 2 or ids.size(0) != batch_size:
                raise ValueError(f"{name} must have shape [B, L].")
            ids = ids.to(device=device, dtype=torch.long)
            if torch.any((ids < 0) | (ids >= self.config.vocab_size)):
                raise ValueError(f"{name} contains token ids outside the vocabulary.")
            mask = self._mask(batch, f"{name}_padding_mask", ids.shape, device, default=ids == 0)
            self._validate_padding_ids(ids, mask, name)
            if name in ("draft", "right"):
                leaked = (task == 0).unsqueeze(1) & (ids != 0)
                if torch.any(leaked):
                    raise ValueError(f"{name} tokens cannot be supplied for draft task rows.")
                mask = mask | (task == 0).unsqueeze(1)
            values = self.token_embedding(ids)
            if name == "motif":
                gate = torch.sigmoid(self.motif_gate(torch.cat((condition, relation), dim=-1)))
                values = values * gate.unsqueeze(1) * strength.unsqueeze(1)
                mask = mask | (strength == 0)
            add(values, mask, branch_id)

        action = batch.get("action_state")
        if action is None:
            action = audio.new_zeros((batch_size, 0, 50))
        if action.ndim != 3 or action.shape[0] != batch_size or action.size(-1) != 50:
            raise ValueError("action_state must have shape [B, N, 50].")
        self._finite(action, "action_state")
        action_mask = self._mask(
            batch,
            "action_state_padding_mask",
            action.shape[:2],
            device,
            default=torch.zeros(action.shape[:2], dtype=torch.bool, device=device),
        )
        add(self.action_projection(action), action_mask, 6)

        load_curve = batch.get("load_curve")
        if load_curve is not None:
            if load_curve.ndim != 3 or load_curve.shape[0] != batch_size or load_curve.size(-1) != 4:
                raise ValueError("load_curve must have shape [B, T, 4]")
            raw_curve_mask = batch.get("load_curve_condition_mask")
            if raw_curve_mask is None:
                curve_mask = torch.zeros(load_curve.shape[:2], dtype=torch.bool, device=device)
            else:
                if raw_curve_mask.shape not in (load_curve.shape[:2], load_curve.shape):
                    raise ValueError("load_curve_condition_mask must have shape [B,T] or [B,T,4]")
                raw_curve_mask = raw_curve_mask.to(device=device, dtype=torch.bool)
                curve_mask = raw_curve_mask if raw_curve_mask.ndim == 2 else ~raw_curve_mask.any(dim=-1)
            self._finite(load_curve, "load_curve")
            add(self.load_curve_projection(load_curve), curve_mask, 10)

        # Future *planned onsets* are supplied by the whole-song blueprint,
        # not future note configurations. Arrangement can see upcoming demands
        # before choosing a long Hold/Slide. Masked-note tokens carry no layout.
        skeleton = batch.get("skeleton")
        if skeleton is not None:
            if skeleton.ndim != 2 or skeleton.size(0) != batch_size:
                raise ValueError("skeleton must have shape [B,L]")
            skeleton = skeleton.to(device=device, dtype=torch.long)
            if torch.any((skeleton < 0) | (skeleton >= self.config.vocab_size)):
                raise ValueError("skeleton token outside vocabulary")
            if skeleton.size(1) > self.config.max_memory_tokens:
                raise ValueError("skeleton exceeds max_memory_tokens")
            # Unlike a completed right context this branch is valid for draft
            # and revise tasks; the input builder validates its mask-only cells.
            add(self.token_embedding(skeleton), skeleton == 0, 10)

        condition_token = self.condition_projection(condition) + condition_null[:, None].to(audio.dtype) * self.condition_null_embedding
        global_tokens = torch.stack(
            (
                condition_token,
                self.task_embedding(task),
                self.relation_projection(relation),
            ),
            dim=1,
        )
        # A null condition is an explicit learned state, not an invented
        # vector that could be mistaken for a real low-valued target.
        global_positions = torch.arange(3, device=device).unsqueeze(0)
        global_tokens = (
            global_tokens
            + self.position_embedding(global_positions)
            + self.branch_embedding.weight[7:10].unsqueeze(0)
        )
        # The global tokens always remain visible, even when every memory branch is empty.
        branches.append(global_tokens)
        masks.append(torch.zeros((batch_size, 3), dtype=torch.bool, device=device))
        max_authored_length = 5 * self.config.max_memory_tokens + 128 + 32
        if authored_lengths > max_authored_length:
            raise ValueError("Combined authored memories exceed the configured memory budget.")

        memory = torch.cat(branches, dim=1)
        memory_mask = torch.cat(masks, dim=1)
        memory = self.memory_norm(self.memory_encoder(memory, src_key_padding_mask=memory_mask))
        return memory, memory_mask

    def decode(
        self,
        ids: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        *,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """Decode a causal event-token prefix against previously encoded memory."""

        if ids.ndim != 2 or ids.size(1) == 0:
            raise ValueError("decoder ids must have nonempty shape [B, L].")
        if ids.size(1) > self.config.max_tokens:
            raise ValueError("decoder input exceeds max_tokens.")
        if memory.ndim != 3 or memory.size(0) != ids.size(0):
            raise ValueError("memory must have shape [B, M, d_model] matching decoder ids.")
        if memory_mask.shape != memory.shape[:2]:
            raise ValueError("memory_mask must have shape [B, M].")
        ids = ids.to(device=memory.device, dtype=torch.long)
        if torch.any((ids < 0) | (ids >= self.config.vocab_size)):
            raise ValueError("decoder ids contain token ids outside the vocabulary.")
        length = ids.size(1)
        positions = torch.arange(length, device=ids.device).unsqueeze(0)
        target = self.token_embedding(ids) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=ids.device), diagonal=1
        )
        hidden = self.decoder(
            target,
            memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_mask.to(device=ids.device, dtype=torch.bool),
        )
        hidden = self.decoder_norm(hidden)
        return hidden if return_hidden else self.output_head(hidden)

    def decode_step(self, token, position, memory, memory_mask, caches=None):
        """Incremental inference; completed decoder prefixes are not recomputed."""
        if self.training:
            raise ValueError("Cached decoding requires eval mode")
        if token.ndim != 2 or token.size(1) != 1 or not 0 <= position < self.config.max_tokens:
            raise ValueError("decode_step expects one token within max_tokens")
        from maichart.generation.event_model import _cached_decoder_layer
        hidden = self.token_embedding(token) + self.position_embedding(
            torch.tensor([position],device=token.device)).unsqueeze(0)
        caches=caches or []
        updated=[]
        for i,layer in enumerate(self.decoder.layers):
            previous=caches[i] if i < len(caches) else (None,None)
            hidden,present=_cached_decoder_layer(layer,hidden,memory,*previous,memory_mask)
            updated.append(present)
        return self.output_head(self.decoder_norm(hidden)),updated

    @staticmethod
    def _required(batch: dict[str, torch.Tensor], key: str, ndim: int) -> torch.Tensor:
        value = batch.get(key)
        if value is None or value.ndim != ndim:
            raise ValueError(f"batch must include {key} with {ndim} dimensions.")
        return value

    @staticmethod
    def _mask(
        batch: dict[str, torch.Tensor],
        key: str,
        shape: torch.Size | tuple[int, ...],
        device: torch.device,
        *,
        default: torch.Tensor | None = None,
    ) -> torch.Tensor:
        value = batch.get(key, default)
        if value is None:
            value = torch.zeros(shape, dtype=torch.bool, device=device)
        if value.shape != shape:
            raise ValueError(f"{key} must have shape {tuple(shape)}.")
        return value.to(device=device, dtype=torch.bool)

    @staticmethod
    def _finite(tensor: torch.Tensor, name: str) -> None:
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain only finite values.")

    @staticmethod
    def _validate_padding_ids(ids: torch.Tensor, mask: torch.Tensor, name: str) -> None:
        if torch.any((ids == 0) & ~mask):
            raise ValueError(f"{name} padding id 0 must be masked.")
