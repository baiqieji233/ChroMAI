"""Independent macro classification with group-conserved action details.

Why this module exists
----------------------
Large legal-action catalogs contain many variants of the same starting pad
and note kind. The macro classifier chooses that pad/kind independently;
the detail scores distinguish legal variants within its bucket. This separates
the direct atom-classifier gradient from unrelated buckets without claiming
that shared context learning, speed, or musical quality is thereby solved.

Mathematical guarantee
----------------------
Every action ``a in 0..V-1`` is partitioned into a disjoint Macro Group
``g(a) = (kind_class, actual_start_pad_set, ring_class)`` (``STOP`` is group 0).
At a fixed slot context the head combines independent macro probabilities
and a conditional detail distribution:

    final_logit(a) = L_macro(g(a)) + (s_detail(a) - logsumexp_{a' in g(a)} s_detail(a'))

where:
1. ``L_macro(g) = log_softmax(macro_head(h_slot))[g]``
   governs the exact total probability mass of choosing ``(macro_kind, start_pad)``
   (e.g. "Tap on lane 1" vs. "Slide starting on lane 1" vs. "STOP").
2. ``s_detail(a)`` refines the within-group choice with two conditioned scores:
      Step 2a: trajectory shape, relative endpoint, and break/ex modifier
               conditioned on the chosen macro group ``g(a)``;
      Step 2b: hold/slide duration conditioned on BOTH ``g(a)`` and the
               trajectory shape / relative endpoint.
3. The conditional detail probabilities sum to 1 inside each group. Therefore,
   up to floating-point precision, its full group probability is the macro
   probability:

       logsumexp_{a in g} final_logit(a) == L_macro(g)

   Both detail terms are scored together, then normalized over legal atoms in
   the group; this is not a sequence of separately sampled attribute tokens.
   Base scores participate only in the conditional detail distribution, with
   no direct cross-group gradient through atom logits. Context encoders and
   shared parameters can still couple groups during learning.
   Learned relative-head adjustments, legality masks, and user type controls
   intentionally alter macro probabilities after this head; the guarantee is
   limited to changing internal detail scores at fixed macro inputs.
   The independent macro classifier is new and requires training when migrating
   a flat checkpoint; unlike the future-horizon residual it is not a neutral
   zero-step change.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

ACTION_HEAD_POLICIES = frozenset({'flat_v1', 'conserved_grouped_v2', 'compact_grouped_v3'})
AUDIO_MACRO_GROUP_POLICY = 'kind_ring_modifier_duration_no_position_v1'


def build_macro_groups(action_factors) -> tuple[torch.Tensor, int]:
    """Partition actions by parsed kind and all actual starting pads.

    The ``atom_*`` factor buffers contain category IDs, not raw bit masks.
    Equal kind/ring IDs are useful group keys, but Touch pads must come from
    the parser: button-head masks intentionally do not contain Touch positions.
    """
    from .events import event_metadata
    kinds = action_factors.atom_kind_mask.detach().cpu().tolist()
    rings = action_factors.atom_touch_ring.detach().cpu().tolist()
    vocab_size = len(kinds)
    group_map: dict[tuple[int, tuple[str, ...], int], int] = {(0, (), 0): 0}
    group_ids = [0] * vocab_size
    for idx in range(1, vocab_size):
        heads = event_metadata(action_factors.atom_catalog[idx - 1]).heads
        key = (int(kinds[idx]), heads, int(rings[idx]))
        gid = group_map.get(key)
        if gid is None:
            gid = len(group_map)
            group_map[key] = gid
        group_ids[idx] = gid
    return torch.tensor(group_ids, dtype=torch.long), len(group_map)


def build_audio_macro_groups(action_factors) -> tuple[torch.Tensor, int]:
    """Partition actions by position-invariant acoustic attributes.

    The audio-only auxiliary branch sees local acoustic features and same-beat
    prefixes, not cross-event pad history. Grouping by ``(kind_mask, touch_ring,
    break_ex, hold_duration, slide_duration)`` (with ``STOP`` as group 0) lets
    ``audio_action_loss`` supervise note type, modifier, touch ring, and
    sustain duration without penalizing rotational pad symmetries.
    """
    kinds = action_factors.atom_kind_mask.detach().cpu().tolist()
    rings = action_factors.atom_touch_ring.detach().cpu().tolist()
    breaks = action_factors.atom_break_ex.detach().cpu().tolist()
    hold_durs = action_factors.atom_hold_duration.detach().cpu().tolist()
    slide_durs = action_factors.atom_slide_duration.detach().cpu().tolist()
    vocab_size = len(kinds)
    group_map: dict[tuple[int, int, int, int, int], int] = {(0, 0, 0, 0, 0): 0}
    group_ids = [0] * vocab_size
    for idx in range(1, vocab_size):
        key = (int(kinds[idx]), int(rings[idx]), int(breaks[idx]), int(hold_durs[idx]), int(slide_durs[idx]))
        gid = group_map.get(key)
        if gid is None:
            gid = len(group_map)
            group_map[key] = gid
        group_ids[idx] = gid
    return torch.tensor(group_ids, dtype=torch.long), len(group_map)


def group_logsumexp(logits: torch.Tensor, group_ids: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Vectorized per-group ``logsumexp`` over the last dimension ``[..., V] -> [..., G]``."""
    scores = logits.float()
    idx = group_ids.to(device=scores.device)
    expanded = idx.view(*([1] * (scores.ndim - 1)), -1).expand_as(scores)
    group_max = scores.new_full((*scores.shape[:-1], num_groups), float('-inf'))
    group_max.scatter_reduce_(-1, expanded, scores, reduce='amax', include_self=True)
    safe_max = torch.where(torch.isfinite(group_max), group_max, torch.zeros_like(group_max))
    shifted = (scores - safe_max.gather(-1, expanded)).exp()
    group_sum = scores.new_zeros((*scores.shape[:-1], num_groups))
    group_sum.scatter_add_(-1, expanded, shifted)
    result = safe_max + group_sum.clamp_min(1e-30).log()
    return torch.where(group_sum > 0, result, torch.full_like(result, float('-inf')))


class ConservedGroupedActionHead(nn.Module):
    """Independent macro probabilities and group-conditioned legal-atom details."""

    def __init__(self, action_factors, width: int):
        super().__init__()
        self.width = int(width)
        group_ids, num_groups = build_macro_groups(action_factors)
        audio_group_ids, num_audio_groups = build_audio_macro_groups(action_factors)
        self.num_groups = int(num_groups)
        self.num_audio_groups = int(num_audio_groups)
        self.register_buffer('atom_group_id', group_ids, persistent=False)
        self.register_buffer('audio_macro_group_id', audio_group_ids, persistent=False)
        rep_atoms = [0] * self.num_groups
        for atom_idx, gid in enumerate(group_ids.tolist()):
            if rep_atoms[gid] == 0 and atom_idx != 0:
                rep_atoms[gid] = atom_idx
        rep_t = torch.tensor(rep_atoms, dtype=torch.long)
        self.register_buffer('group_kind_id', action_factors.atom_kind_mask[rep_t].clone(), persistent=False)
        self.register_buffer('group_lane_id', action_factors.atom_head_lane_mask[rep_t].clone(), persistent=False)
        self.register_buffer('group_ring_id', action_factors.atom_touch_ring[rep_t].clone(), persistent=False)
        object.__setattr__(self, '_shared_macro_factors', (
            action_factors.factors['kind_mask'],
            action_factors.factors['head_lane_mask'],
            action_factors.factors['touch_ring'],
        ))
        self.register_buffer('atom_shape_id', action_factors.atom_trajectory_shape.clone(), persistent=False)
        self.register_buffer('atom_endpoint_id', action_factors.atom_relative_endpoint.clone(), persistent=False)
        self.register_buffer('atom_break_id', action_factors.atom_break_ex.clone(), persistent=False)
        self.register_buffer('atom_hold_dur_id', action_factors.atom_hold_duration.clone(), persistent=False)
        self.register_buffer('atom_slide_dur_id', action_factors.atom_slide_duration.clone(), persistent=False)
        gseb = torch.stack((
            self.atom_group_id, self.atom_shape_id, self.atom_endpoint_id, self.atom_break_id
        ), dim=-1)
        u_gseb, inv_gseb = torch.unique(gseb, dim=0, return_inverse=True)
        gse = torch.stack((self.atom_group_id, self.atom_shape_id, self.atom_endpoint_id), dim=-1)
        u_gse, inv_gse = torch.unique(gse, dim=0, return_inverse=True)
        hdsd = torch.stack((self.atom_hold_dur_id, self.atom_slide_dur_id), dim=-1)
        u_hdsd, inv_hdsd = torch.unique(hdsd, dim=0, return_inverse=True)
        self.register_buffer('_u_gseb', u_gseb, persistent=False)
        self.register_buffer('_inv_gseb', inv_gseb, persistent=False)
        self.register_buffer('_u_gse', u_gse, persistent=False)
        self.register_buffer('_inv_gse', inv_gse, persistent=False)
        self.register_buffer('_u_hdsd', u_hdsd, persistent=False)
        self.register_buffer('_inv_hdsd', inv_hdsd, persistent=False)

        n_shapes = int(action_factors.factors['trajectory_shape'].num_embeddings)
        n_ends = int(action_factors.factors['relative_endpoint'].num_embeddings)
        n_breaks = int(action_factors.factors['break_ex'].num_embeddings)
        n_hold_durs = int(action_factors.factors['hold_duration'].num_embeddings)
        n_slide_durs = int(action_factors.factors['slide_duration'].num_embeddings)

        # Stage 1 predicts independent group probabilities. Uniform initial
        # group logits avoid inheriting the flat vocabulary's branch-count bias;
        # this new classifier must learn rather than promise old-model parity.
        self.macro_norm = nn.LayerNorm(self.width)
        self.macro_head = nn.Linear(self.width, self.num_groups, bias=False)
        nn.init.zeros_(self.macro_head.weight)

        # Stage 2a: Trajectory/Endpoint/Modifier conditioned on Macro Group g(a)
        # Non-zero context projection + zero-init factor tables ensure exact 0 output
        # at step 0 WITH non-zero gradients on the very first backward pass.
        self.traj_context = nn.Linear(self.width, self.width, bias=False)
        self.group_traj_gate = nn.Embedding(self.num_groups, self.width)
        self.shape_detail = nn.Embedding(n_shapes, self.width)
        self.endpoint_detail = nn.Embedding(n_ends, self.width)
        self.break_detail = nn.Embedding(n_breaks, self.width)
        nn.init.zeros_(self.group_traj_gate.weight)
        nn.init.zeros_(self.shape_detail.weight)
        nn.init.zeros_(self.endpoint_detail.weight)
        nn.init.zeros_(self.break_detail.weight)

        # Stage 2b: Duration conditioned on BOTH Macro Group g(a) AND Trajectory/Endpoint
        self.dur_context = nn.Linear(self.width, self.width, bias=False)
        self.group_dur_gate = nn.Embedding(self.num_groups, self.width)
        self.shape_dur_gate = nn.Embedding(n_shapes, self.width)
        self.end_dur_gate = nn.Embedding(n_ends, self.width)
        self.hold_dur_detail = nn.Embedding(n_hold_durs, self.width)
        self.slide_dur_detail = nn.Embedding(n_slide_durs, self.width)
        nn.init.zeros_(self.group_dur_gate.weight)
        nn.init.zeros_(self.shape_dur_gate.weight)
        nn.init.zeros_(self.end_dur_gate.weight)
        nn.init.zeros_(self.hold_dur_detail.weight)
        nn.init.zeros_(self.slide_dur_detail.weight)

        self._table_cache_key = None
        self._table_cache = None

    def _clear_table_cache(self):
        self._table_cache_key = None
        self._table_cache = None

    def train(self, mode: bool = True):
        self._clear_table_cache()
        return super().train(mode)

    def detail_tables(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(traj_table, dur_table)`` of shape ``[V, width]``, cached in eval mode."""
        if self.training or torch.is_grad_enabled():
            self._clear_table_cache()
        cache_key = None
        emb_list = (
            self.group_traj_gate, self.shape_detail, self.endpoint_detail, self.break_detail,
            self.group_dur_gate, self.shape_dur_gate, self.end_dur_gate,
            self.hold_dur_detail, self.slide_dur_detail,
        )
        if (not self.training and not torch.is_grad_enabled()
                and not any(torch.is_inference(m.weight) for m in emb_list)):
            cache_key = tuple((m.weight._version, m.weight.device, m.weight.dtype) for m in emb_list)
            if self._table_cache_key == cache_key and self._table_cache is not None:
                return self._table_cache

        u_gseb = self._u_gseb
        u_gse = self._u_gse
        u_hdsd = self._u_hdsd

        t_gate = (1.0 + torch.tanh(self.group_traj_gate.weight)).index_select(0, u_gseb[:, 0])
        t_base = (self.shape_detail(u_gseb[:, 1])
                  + self.endpoint_detail(u_gseb[:, 2])
                  + self.break_detail(u_gseb[:, 3]))
        u_traj = (t_base * t_gate) * (u_gseb[:, 0] != 0).unsqueeze(-1)
        traj_table = u_traj.index_select(0, self._inv_gseb)

        d_gate = (1.0 + torch.tanh(
            self.group_dur_gate(u_gse[:, 0])
            + self.shape_dur_gate(u_gse[:, 1])
            + self.end_dur_gate(u_gse[:, 2])
        )) * (u_gse[:, 0] != 0).unsqueeze(-1)
        u_dur_base = self.hold_dur_detail(u_hdsd[:, 0]) + self.slide_dur_detail(u_hdsd[:, 1])
        dur_table = u_dur_base.index_select(0, self._inv_hdsd) * d_gate.index_select(0, self._inv_gse)

        result = (traj_table, dur_table)
        if cache_key is not None:
            self._table_cache_key = cache_key
            self._table_cache = result
        return result

    def forward(self, base_logits: torch.Tensor, slot_context: torch.Tensor, *,
                tables: tuple[torch.Tensor, torch.Tensor] | None = None,
                macro_scale: float = 1.0) -> torch.Tensor:
        """Apply group-conserved macro + conditional detail refinement to ``base_logits``."""
        if base_logits.shape[:-1] != slot_context.shape[:-1] or slot_context.size(-1) != self.width:
            raise ValueError('base_logits and slot_context leading shapes must match')
        if tables is None:
            tables = self.detail_tables()
        traj_table, dur_table = tables

        normed = self.macro_norm(slot_context)
        macro_raw = self.macro_head(normed).float()
        shared = getattr(self, '_shared_macro_factors', None)
        if shared is not None:
            kind_emb, lane_emb, ring_emb = shared
            dev = slot_context.device
            macro_factor_weight = (
                kind_emb.weight.to(dev).index_select(0, self.group_kind_id.to(dev)).float()
                + lane_emb.weight.to(dev).index_select(0, self.group_lane_id.to(dev)).float()
                + ring_emb.weight.to(dev).index_select(0, self.group_ring_id.to(dev)).float()
            )
            macro_raw = macro_raw + F.linear(slot_context.float(), macro_factor_weight)
        macro_delta = float(macro_scale) * macro_raw
        # Keep STOP (group 0) as the gauge reference so macro shifts represent
        # log-odds relative to STOP or between groups without drifting global offset.
        macro_delta = macro_delta - macro_delta[..., :1]

        base_f = base_logits.float()
        target_group_lse = macro_delta.log_softmax(-1)

        h_traj = self.traj_context(normed)
        h_dur = self.dur_context(normed)
        detail_delta = (F.linear(h_traj, traj_table.to(h_traj.dtype)).float()
                        + F.linear(h_dur, dur_table.to(h_dur.dtype)).float())
        detail_scores = base_f + detail_delta
        detail_group_lse = group_logsumexp(detail_scores, self.atom_group_id, self.num_groups)

        idx = self.atom_group_id.to(device=base_logits.device)
        expanded = idx.view(*([1] * (base_f.ndim - 1)), -1).expand_as(base_f)
        within_log_probs = detail_scores - detail_group_lse.gather(-1, expanded)
        final = target_group_lse.gather(-1, expanded) + within_log_probs
        # Normalizers and final log-probabilities stay in float32 under BF16
        # autocast; rounding them back to BF16 changes the conserved masses.
        return final


class CompactGroupedActionHead(ConservedGroupedActionHead):
    """Compact macro classifier with one kind-conserved geometry adjustment.

    ``base_logits`` contains only shared compositional detail scores. Head bias
    is [..., 3, 8] (tap, hold, slide); endpoint/shape bias belongs in the
    separate [..., V] ``detail_bias``. STOP and Touch groups are untouched by
    the geometry adjustment. The independent audio reference predicts acoustic
    groups, with uniform atom details; it does not learn slide trajectories.
    """

    def __init__(self, action_factors, width: int):
        super().__init__(action_factors, width)
        from .events import event_metadata

        representatives = [0] * self.num_groups
        for atom, group in enumerate(self.atom_group_id.tolist()):
            representatives[group] = atom
        categories = []
        for atom in representatives:
            if atom == 0:
                categories.append(-1)
                continue
            meta = event_metadata(action_factors.atom_catalog[atom - 1])
            categories.append(-1 if meta.touch else 2 if meta.slide else 1 if meta.hold else 0)
        self.register_buffer('macro_kind_category', torch.tensor(categories, dtype=torch.long), persistent=False)
        self.register_buffer('group_head_projection', action_factors.atom_head_lanes[
            torch.tensor(representatives, dtype=torch.long)].float().clone(), persistent=False)
        # Precompute even absent kinds: forward never needs a device-to-host
        # synchronization to discover whether a class has groups.
        for kind in range(3):
            self.register_buffer(f'_kind_groups_{kind}',
                                 torch.tensor([g for g, k in enumerate(categories) if k == kind], dtype=torch.long),
                                 persistent=False)
        counts = torch.bincount(self.audio_macro_group_id, minlength=self.num_audio_groups)
        self.register_buffer('_audio_log_group_sizes', counts.float().log(), persistent=False)
        self.audio_macro_head = nn.Linear(self.width, self.num_audio_groups)

    def audio_logits(self, slot_context: torch.Tensor) -> torch.Tensor:
        """Return normalized atom log-probabilities with uniform acoustic-group details."""
        groups = self.audio_macro_head(self.macro_norm(slot_context)).float().log_softmax(-1)
        return (groups - self._audio_log_group_sizes).index_select(-1, self.audio_macro_group_id)

    def forward(self, base_logits: torch.Tensor, slot_context: torch.Tensor, *,
                tables: tuple[torch.Tensor, torch.Tensor] | None = None,
                macro_scale: float = 1.0, head_bias: torch.Tensor | None = None,
                detail_bias: torch.Tensor | None = None) -> torch.Tensor:
        if (base_logits.shape[:-1] != slot_context.shape[:-1]
                or slot_context.size(-1) != self.width
                or base_logits.size(-1) != self.atom_group_id.numel()):
            raise ValueError('base_logits and slot_context shapes must match the action catalog and width')
        if tables is None:
            tables = self.detail_tables()
        normed = self.macro_norm(slot_context)
        macro_raw = self.macro_head(normed).float()
        kind_emb, lane_emb, ring_emb = self._shared_macro_factors
        macro_weight = (kind_emb.weight.index_select(0, self.group_kind_id).float()
                        + lane_emb.weight.index_select(0, self.group_lane_id).float()
                        + ring_emb.weight.index_select(0, self.group_ring_id).float())
        macro_raw = macro_raw + F.linear(normed.float(), macro_weight)
        macro = float(macro_scale) * macro_raw
        macro = (macro - macro[..., :1]).log_softmax(-1)
        traj_table, dur_table = tables
        h_traj, h_dur = self.traj_context(normed), self.dur_context(normed)
        scores = (base_logits.float() + F.linear(h_traj, traj_table.to(h_traj.dtype)).float()
                  + F.linear(h_dur, dur_table.to(h_dur.dtype)).float())
        old_lse = group_logsumexp(scores, self.atom_group_id, self.num_groups)
        new_lse = old_lse
        gain = torch.zeros_like(macro)
        if head_bias is not None:
            if head_bias.shape[-2:] != (3, 8):
                raise ValueError('head_bias must end in [3, 8] (tap, hold, slide)')
            for kind in range(3):
                groups = getattr(self, f'_kind_groups_{kind}')
                if groups.numel():
                    delta = F.linear(head_bias[..., kind, :].float(),
                                     self.group_head_projection.index_select(0, groups)).float()
                    gain = gain.index_copy(-1, groups, delta.expand(*macro.shape[:-1], groups.numel()))
        if detail_bias is not None:
            # Differences in detail partition functions are necessary for exact
            # equivalence to applying endpoint/shape bias after atom expansion.
            scores = scores + detail_bias.float()
            new_lse = group_logsumexp(scores, self.atom_group_id, self.num_groups)
            gain = gain + new_lse - old_lse
        adjusted_macro = macro
        if head_bias is not None or detail_bias is not None:
            for kind in range(3):
                groups = getattr(self, f'_kind_groups_{kind}')
                if groups.numel():
                    original = macro.index_select(-1, groups)
                    shifted = original + gain.index_select(-1, groups)
                    conserved = shifted + original.logsumexp(-1, keepdim=True) - shifted.logsumexp(-1, keepdim=True)
                    adjusted_macro = adjusted_macro.index_copy(-1, groups, conserved)
        expanded = self.atom_group_id.view(*([1] * (scores.ndim - 1)), -1).expand_as(scores)
        # Touch and STOP keep both their original macro mass and original detail
        # probabilities, even if a caller supplies a full-V geometry bias.
        within = scores - new_lse.gather(-1, expanded)
        if detail_bias is not None:
            untouched = self.macro_kind_category.index_select(0, self.atom_group_id) < 0
            original_within = scores - detail_bias.float() - old_lse.gather(-1, expanded)
            within = torch.where(untouched, original_within, within)
        return adjusted_macro.gather(-1, expanded) + within
