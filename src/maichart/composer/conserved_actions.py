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

ACTION_HEAD_POLICIES = frozenset({'flat_v1', 'conserved_grouped_v2'})


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
        self.num_groups = int(num_groups)
        self.register_buffer('atom_group_id', group_ids, persistent=False)
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

        g = self.atom_group_id
        s = self.atom_shape_id
        e = self.atom_endpoint_id
        b = self.atom_break_id
        hd = self.atom_hold_dur_id
        sd = self.atom_slide_dur_id

        traj_gate = 1.0 + torch.tanh(self.group_traj_gate(g))
        traj_base = self.shape_detail(s) + self.endpoint_detail(e) + self.break_detail(b)
        traj_table = (traj_base * traj_gate) * (g != 0).unsqueeze(-1)

        dur_gate = 1.0 + torch.tanh(self.group_dur_gate(g) + self.shape_dur_gate(s) + self.end_dur_gate(e))
        dur_base = self.hold_dur_detail(hd) + self.slide_dur_detail(sd)
        dur_table = (dur_base * dur_gate) * (g != 0).unsqueeze(-1)

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
