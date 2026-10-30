"""Audio-grounded Composer: ordered listening, hierarchical time, shared actions."""
from dataclasses import replace
import math
import torch
from torch import nn
from torch.nn import functional as F
from .model import ChartComposer
from .audio_lens import AudioLens
from .action_factors import CompositionalActions
from .planning import AudioPlanner,read_plan
from .stability import bounded_history_memory, history_attention_context
from .events import merge_separate_head_timing

GROUNDED_PROFILE='audio_grounded_v3'
BIN_EDGES=(.5,1.,2.,4.,8.)
PLACEMENT_GRAD_SCALE=.15
LOCAL_LEAD_IN_EVENTS=8


def _action_history_context(symbolic, attended, symbolic_keep, paired):
    """Keep legacy attention in the pointer while excluding it from v3 actions."""
    return symbolic - attended * symbolic_keep[:, None] if paired else symbolic


def hierarchical_log_probs(fine,coarse,bins,valid):
    """Choose a time band first; its mass does not grow with candidate count."""
    membership=F.one_hot(bins,6).bool()&valid[...,None]
    present=membership.any(1)
    safe_coarse=coarse.float().masked_fill(~present,-1e4)
    coarse_log=safe_coarse.log_softmax(-1)
    energies=fine.float()[...,None].masked_fill(~membership,-1e4)
    normalizer=energies.logsumexp(1)
    within=fine.float()-normalizer.gather(1,bins)
    return (within+coarse_log.gather(1,bins)).masked_fill(~valid,-1e4)


def bin_pool(hidden,bins,valid):
    membership=F.one_hot(bins,6).to(hidden)*valid[...,None]
    return torch.einsum('bcg,bcd->bgd',membership,hidden)/membership.sum(1).clamp_min(1)[...,None]


class _CandidateCompetition(nn.Module):
    """Compare adjacent valid candidates' fused AudioLens + placement and distance-weighted prominence."""

    def __init__(self, width: int):
        super().__init__()
        # Local context (width) + left diff (width) + right diff (width) + beat/distance stats (28) + prominence (5)
        self.norm = nn.LayerNorm(width)
        self.in_proj = nn.Sequential(
            nn.Linear(width * 3 + 33, width),
            nn.GELU(),
        )
        self.out_proj = nn.Linear(width, width)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, audio_context, valid, *, workload_audio=None, placement_probs=None, candidate_time=None):
        batch, count, width = audio_context.shape
        if count == 0:
            return audio_context
        valid_b = valid.bool()
        mask = valid_b.unsqueeze(-1)
        safe_audio = torch.where(mask, audio_context, torch.zeros_like(audio_context))
        normed = torch.where(mask, self.norm(safe_audio), torch.zeros_like(safe_audio))

        left_valid = torch.zeros_like(valid_b)
        right_valid = torch.zeros_like(valid_b)
        if count > 1:
            left_valid[:, 1:] = valid_b[:, 1:] & valid_b[:, :-1]
            right_valid[:, :-1] = valid_b[:, :-1] & valid_b[:, 1:]
        left_mask = left_valid.unsqueeze(-1)
        right_mask = right_valid.unsqueeze(-1)

        gap_beats_left = torch.zeros((batch, count, 1), device=normed.device, dtype=torch.float32)
        gap_beats_right = torch.zeros_like(gap_beats_left)
        gap_sec_left = torch.zeros_like(gap_beats_left)
        gap_sec_right = torch.zeros_like(gap_beats_left)
        if candidate_time is not None and candidate_time.size(1) >= count and candidate_time.size(-1) >= 2:
            raw_time = candidate_time[:, :count, :2].float()
            safe_time = torch.where(mask, raw_time, torch.zeros_like(raw_time))
            beats = safe_time[..., 0:1]
            secs = safe_time[..., 1:2]
            if count > 1:
                edge_valid = (valid_b[:, 1:] & valid_b[:, :-1]).unsqueeze(-1)
                edge_beats = torch.where(edge_valid, (beats[:, 1:] - beats[:, :-1]).clamp_min(0.), torch.zeros_like(beats[:, 1:]))
                edge_secs = torch.where(edge_valid, (secs[:, 1:] - secs[:, :-1]).clamp_min(0.), torch.zeros_like(secs[:, 1:]))
                gap_beats_left[:, 1:] = edge_beats
                gap_beats_right[:, :-1] = edge_beats
                gap_sec_left[:, 1:] = edge_secs
                gap_sec_right[:, :-1] = edge_secs
            w_left = torch.where(left_mask, torch.exp(-gap_beats_left / 0.25), torch.zeros_like(gap_beats_left))
            w_right = torch.where(right_mask, torch.exp(-gap_beats_right / 0.25), torch.zeros_like(gap_beats_right))
        else:
            w_left = left_mask.to(normed.dtype)
            w_right = right_mask.to(normed.dtype)

        left = torch.zeros_like(normed)
        right = torch.zeros_like(normed)
        if count > 1:
            left[:, 1:] = torch.where(left_mask[:, 1:], normed[:, :-1], torch.zeros_like(normed[:, :-1]))
            right[:, :-1] = torch.where(right_mask[:, :-1], normed[:, 1:], torch.zeros_like(normed[:, 1:]))
        diff_left = torch.where(left_mask, (normed - left) * w_left, torch.zeros_like(normed))
        diff_right = torch.where(right_mask, (normed - right) * w_right, torch.zeros_like(normed))

        neighbor_geom = torch.cat((
            torch.log1p(gap_beats_left),
            torch.log1p(gap_beats_right),
            torch.log1p(gap_sec_left),
            torch.log1p(gap_sec_right),
            w_left,
            w_right,
            left_mask.to(normed.dtype),
            right_mask.to(normed.dtype),
        ), dim=-1)
        neighbor_geom = torch.where(mask, neighbor_geom, torch.zeros_like(neighbor_geom))

        if workload_audio is not None and workload_audio.size(1) >= count and workload_audio.size(-1) >= 20:
            raw_wl = workload_audio[:, :count].to(normed)
            safe_wl = torch.where(mask, raw_wl, torch.zeros_like(raw_wl))
            if safe_wl.size(-1) >= 40:
                acoustic_20 = torch.cat((safe_wl[..., :14] - safe_wl[..., 26:40], safe_wl[..., :6]), dim=-1)
            else:
                acoustic_20 = safe_wl[..., :20]
        else:
            acoustic_20 = normed.new_zeros((batch, count, 20))
        beat_stats = torch.cat((acoustic_20, neighbor_geom), dim=-1)

        if placement_probs is not None and placement_probs.size(1) >= count:
            raw_p = placement_probs[:, :count, :5].to(normed)
            p = torch.where(mask, raw_p, torch.zeros_like(raw_p))
            p_left = torch.zeros_like(p)
            p_right = torch.zeros_like(p)
            if count > 1:
                p_left[:, 1:] = torch.where(left_mask[:, 1:], p[:, :-1], torch.zeros_like(p[:, :-1]))
                p_right[:, :-1] = torch.where(right_mask[:, :-1], p[:, 1:], torch.zeros_like(p[:, 1:]))
            w_sum = w_left + w_right
            has_neighbor = w_sum > 1e-6
            neighbor_mean = torch.where(
                has_neighbor,
                (w_left * p_left + w_right * p_right) / w_sum.clamp_min(1e-6),
                torch.zeros_like(p),
            )
            prominence = torch.where(mask & has_neighbor, p - neighbor_mean, torch.zeros_like(p))
        else:
            prominence = normed.new_zeros((batch, count, 5))

        features = torch.cat((normed, diff_left, diff_right, beat_stats, prominence), dim=-1)
        residual = torch.where(mask, self.out_proj(self.in_proj(features.to(normed.dtype))), torch.zeros_like(normed))
        return audio_context + residual


class _LocalLeadInContext(nn.Module):
    """Multi-scale variable-horizon phrase memory, D8 symmetry relations, and anti-repetition context.

    Operates across the full history sequence (up to 128 events) rather than a
    fixed 8-event slice:
    1. Reads unclipped committed beat distances and pools
       4 variable-length musical beat horizons: Micro (0..2b), Bar (2..8b),
       Phrase (8..16b), and Section (16..64b).
    2. Computes octagonal dihedral group D8 symmetry alignments (Identity,
       Left-Right Mirror, Up-Down Mirror, 180-deg Inversion, CW/CCW Rotation)
       and note-type progression between earlier bars/phrases and the current phrase.
    3. Retrieves candidate-specific metrical phase counterparts at 4 beats (1 bar),
       8 beats (2 bars), and 16 beats (4 bars) before each candidate beat.
    4. Tracks repetition saturation (AA, ABA, ABAB, ABCABC, and 8-lane overuse
       entropy) to suppress self-reinforcing configuration loops.
    Starts as exact identity via zero-initialized out_proj and gates.
    """

    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.diff_norm = nn.LayerNorm(width)
        self.horizon_mix = nn.Parameter(torch.tensor([0.50, 0.25, 0.15, 0.10]))
        self.in_proj = nn.Sequential(
            nn.Linear(width * 2 + 134, width),
            nn.GELU(),
        )
        self.out_proj = nn.Linear(width, width)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.landing_gate = nn.Parameter(torch.zeros(()))
        self.symmetry_gate = nn.Parameter(torch.zeros(()))
        self.anti_rep_gate = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _d8_alignments(curr: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        curr_n = F.normalize(curr, p=2, dim=-1, eps=1e-6)
        prev_n = F.normalize(prev, p=2, dim=-1, eps=1e-6)
        ud_idx = [3, 2, 1, 0, 7, 6, 5, 4]
        return torch.stack(
            (
                (curr_n * prev_n).sum(-1),
                (curr_n * prev_n.flip(-1)).sum(-1),
                (curr_n * prev_n[..., ud_idx]).sum(-1),
                (curr_n * torch.roll(prev_n, 4, dims=-1)).sum(-1),
                (curr_n * torch.roll(prev_n, 1, dims=-1)).sum(-1),
                (curr_n * torch.roll(prev_n, -1, dims=-1)).sum(-1),
            ),
            dim=-1,
        )

    @staticmethod
    def _history_timing(history_features, lengths, hist_len, *, exact_ages=None):
        """Timing remains FP32 even when memory/attention uses BF16.

        Legacy stand-alone callers may omit ages, but clipped input features
        cannot recover long rests. Training, recovery and generation supply
        the real committed-prefix clock.
        """
        positions=torch.arange(hist_len,device=lengths.device)[None]
        in_bounds=positions<lengths[:,None]
        steps=(in_bounds & (positions>0)).float()*.5
        if history_features is not None:
            steps=history_features[...,0].float().clamp(0.,1.)*4.
        if exact_ages is not None:
            ages=exact_ages.float().clamp_min(0.)
            if hist_len>1:
                steps=steps.clone()
                steps[:,1:]=(ages[:,:-1]-ages[:,1:]).clamp_min(0.)
        else:
            cumulative=(steps*(in_bounds & (positions>0))).cumsum(-1)
            ages=(cumulative.gather(1,(lengths-1).clamp_min(0)[:,None])-cumulative).clamp_min(0.)
        return ages*in_bounds, steps*in_bounds

    def forward(
        self,
        memory,
        history,
        history_features,
        safe_padding,
        lengths,
        atom_head_lanes,
        atom_end_lanes,
        candidate_time,
        valid,
        history_beat_ages=None,
    ):
        batch, count = valid.shape
        width = memory.size(-1)
        dtype = memory.dtype
        device = memory.device
        hist_len = history.size(1)
        if count == 0 or hist_len == 0:
            return memory.new_zeros((batch, count, width)), memory.new_zeros((batch, 8))
        rows = torch.arange(batch, device=device)
        pos_idx = torch.arange(hist_len, device=device)[None, :]
        in_bounds = pos_idx < lengths[:, None]
        valid_hist = in_bounds & (~safe_padding)
        valid_hist_f = valid_hist.to(dtype)

        # 1. Use the committed-prefix clock, independent of stochastic masking.
        if history_features is not None and history_features.size(1) == hist_len:
            all_feats = history_features.to(dtype) * valid_hist_f[:, :, None]
        else:
            all_feats = memory.new_zeros((batch, hist_len, 8))
        age_beats,step_beats=self._history_timing(history_features,lengths,hist_len,exact_ages=history_beat_ages)

        # 2. All-history head lanes and slide/note landing lanes [batch, H, 8].
        all_heads = atom_head_lanes[history].any(-2).to(dtype) * valid_hist_f[:, :, None]
        all_ends = atom_end_lanes[history].any(-2).to(dtype) * valid_hist_f[:, :, None]

        # 3. Variable-length musical beat horizons: Micro (0..2b), Bar (2..8b), Phrase (8..16b), Section (16..64b).
        k_micro = torch.exp(-age_beats / 1.0) * valid_hist_f
        k_bar = torch.exp(-((age_beats - 4.0) / 2.5).square()) * valid_hist_f
        k_phrase = torch.exp(-((age_beats - 10.0) / 4.5).square()) * valid_hist_f
        k_section = torch.exp(-((age_beats - 28.0) / 16.0).square()) * valid_hist_f
        raw_h = torch.stack((k_micro, k_bar, k_phrase, k_section), dim=1)  # [B, 4, H]
        w_h = (raw_h / raw_h.sum(-1, keepdim=True).clamp_min(1e-6)).to(dtype)

        horizon_mem = torch.einsum("bkh,bhd->bkd", w_h, memory)
        mix = self.horizon_mix.to(dtype).softmax(0)[None, :, None]
        pooled_mem = self.norm((mix * horizon_mem).sum(1))
        bar_diff_mem = self.diff_norm(horizon_mem[:, 0] - horizon_mem[:, 1])

        heads_h = torch.einsum("bkh,bhl->bkl", w_h, all_heads)  # [B, 4, 8]
        ends_h = torch.einsum("bkh,bhl->bkl", w_h[:, :2], all_ends)  # [B, 2, 8]
        feats_h = torch.einsum("bkh,bhf->bkf", w_h[:, :3], all_feats)  # [B, 3, 8]
        h_micro, h_bar, h_phrase, _h_sec = heads_h.unbind(1)
        e_micro, _e_bar = ends_h.unbind(1)
        f_micro, f_bar, _f_phrase = feats_h.unbind(1)

        # Explicit recent 4 events for exact step-level transitions & cycle detection.
        heads_list, ends_list, feat_list, valid_list = [], [], [], []
        for k in range(4):
            idx_k = (lengths - 1 - k).clamp_min(0)
            vk = ((lengths > k) & (~safe_padding[rows, idx_k])).to(dtype)
            heads_list.append(all_heads[rows, idx_k] * vk[:, None])
            ends_list.append(all_ends[rows, idx_k] * vk[:, None])
            feat_list.append(all_feats[rows, idx_k] * vk[:, None])
            valid_list.append(vk)
        h0, h1, h2, h3 = heads_list
        e0, e1, _e2, _e3 = ends_list

        # 4. D8 Octagonal Symmetry & Relational Progression features.
        d8_micro_bar = self._d8_alignments(h_micro, h_bar)
        d8_bar_phrase = self._d8_alignments(h_bar, h_phrase)
        d8_rel = torch.cat((d8_micro_bar, d8_bar_phrase), dim=-1)  # [B, 12]
        type_prog = torch.cat((f_micro[:, 2:8], f_micro[:, 2:8] - f_bar[:, 2:8]), dim=-1)  # [B, 12]

        # 5. Repetition saturation & anti-self-reinforcement signals.
        same_01 = (h0 * h1).sum(-1, keepdim=True)
        aba_02 = (h0 * h2).sum(-1, keepdim=True)
        abab_13 = (h1 * h3).sum(-1, keepdim=True)
        abc_03 = (h0 * h3).sum(-1, keepdim=True)
        handoff_01 = (h0 * e1).sum(-1, keepdim=True)
        exact_match_h0 = ((all_heads == h0[:, None, :]).all(-1)
                          & h0.any(-1)[:,None]).to(dtype) * valid_hist_f
        exact_bar_repeats = (exact_match_h0 * w_h[:, 1]).sum(-1, keepdim=True)
        pair_mask = (h0 + h1).clamp(0.0, 1.0)
        pingpong_match = ((all_heads * pair_mask[:, None, :]).sum(-1) >= 0.9).to(dtype) * valid_hist_f
        pingpong_bar_repeats = (pingpong_match * w_h[:, 1]).sum(-1, keepdim=True)
        h_bar_prob = h_bar / h_bar.sum(-1, keepdim=True).clamp_min(1e-6)
        lane_entropy = -(h_bar_prob * h_bar_prob.clamp_min(1e-6).log()).sum(-1, keepdim=True) / math.log(8.0)
        lane_overuse = (h_bar_prob - 0.125).clamp_min(0.0)  # [B, 8]
        rep_sat = torch.cat(
            (
                same_01,
                aba_02,
                abab_13,
                abc_03,
                handoff_01,
                exact_bar_repeats,
                pingpong_bar_repeats,
                lane_entropy,
                lane_overuse,
            ),
            dim=-1,
        )  # [B, 16]

        h_bar_lr = h_bar.flip(-1)
        h_bar_r180 = torch.roll(h_bar, 4, dims=-1)
        flow_geom = torch.cat(
            (
                h0,
                e0,
                h1,
                e1,
                h_bar,
                h_phrase,
                h_bar_lr,
                h_bar_r180,
                d8_rel,
                rep_sat,
                type_prog,
            ),
            dim=-1,
        )  # [B, 104]

        # 6. Candidate-specific rhythm continuity & 4b/8b/16b metrical phase-aligned lookback.
        f0, f1 = feat_list[0], feat_list[1]
        v0, v1 = valid_list[0][:, None, None], valid_list[1][:, None, None]
        db0 = step_beats[rows,(lengths-1).clamp_min(0)][:,None,None]
        db1 = step_beats[rows,(lengths-2).clamp_min(0)][:,None,None]
        bar_db = (w_h[:,1].float()*step_beats).sum(-1)[:,None,None]
        if candidate_time is not None and candidate_time.size(1) >= count and candidate_time.size(-1) >= 2:
            cand_db = candidate_time[:, :count, 0:1].float().clamp_min(0.0)
        else:
            cand_db = memory.new_zeros((batch, count, 1))
        match_db0 = torch.exp(-(cand_db - db0).abs() / 0.08) * v0
        match_db1 = torch.exp(-(cand_db - db1).abs() / 0.08) * v1
        double_db0 = torch.exp(-(cand_db - 2.0 * db0).abs() / 0.12) * v0
        match_bar_step = torch.exp(-(cand_db - bar_db).abs() / 0.15)
        harmonics = torch.cat(
            [torch.cos(2.0 * math.pi * db0 * h) for h in (2.0, 3.0, 4.0, 6.0, 8.0)],
            dim=-1,
        ) * v0

        # Beat distance from each candidate c back to every history event h: [B, C, H]
        cand_hist_beats = age_beats[:, None, :] + cand_db
        vh_bc = valid_hist_f[:, None, :]
        w_4b = torch.exp(-((cand_hist_beats - 4.0) / 0.35).square()) * vh_bc
        w_8b = torch.exp(-((cand_hist_beats - 8.0) / 0.50).square()) * vh_bc
        w_16b = torch.exp(-((cand_hist_beats - 16.0) / 0.75).square()) * vh_bc
        hit_4b = w_4b.max(dim=-1, keepdim=True).values
        hit_8b = w_8b.max(dim=-1, keepdim=True).values
        hit_16b = w_16b.max(dim=-1, keepdim=True).values
        w_4b_n = (w_4b / w_4b.sum(-1, keepdim=True).clamp_min(1e-6)).to(dtype)
        w_8b_n = (w_8b / w_8b.sum(-1, keepdim=True).clamp_min(1e-6)).to(dtype)
        feat_4b = torch.einsum("bch,bhf->bcf", w_4b_n, all_feats[:, :, 2:7])
        feat_8b = torch.einsum("bch,bhf->bcf", w_8b_n, all_feats[:, :, 2:7])

        rhythm_cont = torch.cat(
            (
                match_db0,
                match_db1,
                double_db0,
                match_bar_step,
                harmonics.expand(-1, count, -1),
                f0[:, None, :].expand(-1, count, -1),
                hit_4b,
                hit_8b,
                hit_16b,
                feat_4b,
                feat_8b,
            ),
            dim=-1,
        )  # [B, C, 30]

        combined = torch.cat(
            (
                pooled_mem[:, None, :].expand(-1, count, -1),
                bar_diff_mem[:, None, :].expand(-1, count, -1),
                flow_geom[:, None, :].expand(-1, count, -1),
                rhythm_cont,
            ),
            dim=-1,
        )
        mask = valid.bool().unsqueeze(-1) & lengths.gt(0)[:, None, None]
        safe_combined = torch.where(mask, combined, torch.zeros_like(combined))
        residual = torch.where(
            mask,
            self.out_proj(self.in_proj(safe_combined)),
            torch.zeros((batch, count, width), dtype=dtype, device=device),
        )
        ud_idx = [3, 2, 1, 0, 7, 6, 5, 4]
        sym_template = 0.5 * h_bar_lr + 0.3 * h_bar_r180 + 0.2 * h_bar[..., ud_idx]
        loop_pressure = (0.5 * same_01 + 0.5 * aba_02 * abab_13 + pingpong_bar_repeats).clamp(0.0, 2.0)
        saturated_lanes = 0.6 * h0 + 0.4 * h1 + lane_overuse
        landing_heads = (
            self.landing_gate.tanh() * (0.7 * e0 + 0.3 * e_micro)
            + self.symmetry_gate.tanh() * sym_template
            - self.anti_rep_gate.tanh() * loop_pressure * saturated_lanes
        )
        return residual, landing_heads


class AudioGroundedComposer(ChartComposer):
    def __init__(self,config,atom_catalog):
        # Retain shared parameter names for explicit, partial finetune transfer.
        super().__init__(replace(config,architecture_profile='music_context_v2'))
        self.config=config
        width=config.d_model
        self.lens=AudioLens(config.lens_width,config.lens_chunk_size)
        self.lens_projection=nn.Linear(config.lens_width,width)
        self.placement_head=nn.Linear(config.lens_width+config.condition_dim,5)
        self.placement_projection=nn.Linear(5,width)
        self.song_paired = getattr(config,'configuration_quality_policy','off') == 'song_paired_v3'
        if self.song_paired:
            from .song_context import SongContext
            self.planner=SongContext(config.condition_dim)
            self.song_projection=nn.Linear(96,width,bias=False)
            nn.init.zeros_(self.song_projection.weight)
        else:
            self.planner=AudioPlanner(config.lens_width,config.condition_dim)
        self.plan_projection=nn.Linear(6,width)
        self.audio_fusion=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width),nn.GELU())
        self.candidate_competition=_CandidateCompetition(width)
        self.local_lead_in=_LocalLeadInContext(width)
        self.bin_embedding=nn.Embedding(6,width)
        self.bin_head=nn.Linear(width,1)
        self.register_buffer('_bin_edges',torch.tensor(BIN_EDGES,dtype=torch.float32),persistent=False)
        self.history_pointer=nn.Linear(width,1)
        self.history_bin=nn.Linear(width,1)
        nn.init.zeros_(self.history_pointer.weight);nn.init.zeros_(self.history_pointer.bias)
        nn.init.zeros_(self.history_bin.weight);nn.init.zeros_(self.history_bin.bias)
        self.action_factors=CompositionalActions(list(atom_catalog),width,config.history_anchor_policy,config.slide_factor_policy)
        self.action_relation=nn.Linear(8,width,bias=False)
        nn.init.zeros_(self.action_relation.weight)
        if config.structure_policy != 'off':
            from .structure import structure_dim_for_policy
            structure_dim=structure_dim_for_policy(config.structure_policy)
            self.structure_projection=nn.Linear(structure_dim,width,bias=False)
            self.structure_action_projection=nn.Linear(structure_dim,width,bias=False)
            self.structure_transition_projection=nn.Linear(structure_dim,width,bias=False)
            # New feature branches have their own versioned input contract.
            # Neutral initialization also supports explicit transfer studies.
            nn.init.zeros_(self.structure_projection.weight)
            nn.init.zeros_(self.structure_action_projection.weight)
            nn.init.zeros_(self.structure_transition_projection.weight)
        if config.head_timing_policy=='separate_pads_v2':
            self.head_timing_detail_projection=nn.Linear(123,width,bias=False)
            nn.init.zeros_(self.head_timing_detail_projection.weight)
        if config.cross_event_relation_policy=='relative_heads_v1':
            self.cross_event_relative_factors=nn.Embedding(8,width)
            nn.init.zeros_(self.cross_event_relative_factors.weight)
            self.register_buffer('cross_event_geometry',self.action_factors._normalized_head_lanes.clone(),persistent=False)
            kinds=[None]+[(None if atom[0] not in '12345678' else
                2 if any(c in atom for c in '-<>^pqszvwV') else 1 if 'h' in atom else 0) for atom in atom_catalog]
            for kind in range(3):
                self.register_buffer(f'cross_event_kind_{kind}',torch.tensor(
                    [i for i,value in enumerate(kinds) if value==kind],dtype=torch.long),persistent=False)
        if config.motif_memory_policy != 'off':
            from .motif_memory import MotifMemoryAdapter
            self.motif_memory=MotifMemoryAdapter(width)
        self.compact_actions=getattr(config,'action_head_policy','flat_v1')=='compact_grouped_v3'
        if config.fine_timing_policy=='relative_phrase_v1':
            if config.configuration_quality_policy!='song_paired_v3':
                raise ValueError('relative_phrase_v1 requires song_paired_v3 short-phrase predictions')
            self.fine_timing_gain=nn.Parameter(torch.zeros(()))
        if self.compact_actions:
            from .fixed_horizon import FixedFutureAudio
            self.fixed_future=FixedFutureAudio(width)
            lanes=torch.arange(8)
            self.register_buffer('_compact_offsets',(lanes[None]-lanes[:,None])%8,persistent=False)
        elif getattr(config,'future_horizon_policy','off')=='candidate_horizon_v1':
            from .future_horizon import FutureAudioHorizon
            # A valid base model may have e.g. width=30 and nhead=6. Its
            # width need not divide four, so choose a compatible small head
            # count instead of rejecting otherwise supported configurations.
            horizon_heads=min(4,config.nhead)
            while width % horizon_heads:
                horizon_heads-=1
            self.future_horizon=FutureAudioHorizon(width,nhead=horizon_heads)
        if getattr(config,'echo_prior_policy','off')=='phase_echo_v1':
            from .phase_echo import PhaseEchoPrior
            from .events import event_metadata
            self.echo_prior=PhaseEchoPrior(width)
            kinds=[None]+[None if (m:=event_metadata(atom)).touch else
                2 if m.slide else 1 if m.hold else 0 for atom in atom_catalog]
            for kind in range(3):
                self.register_buffer(f'echo_kind_{kind}',torch.tensor(
                    [i for i,value in enumerate(kinds) if value==kind],dtype=torch.long),persistent=False)
        if getattr(config,'button_load_policy','off') in {'per_button_v1','per_button_context_v2'}:
            from .button_load import PerButtonLoad,PerButtonLoadV2
            self.button_load=(PerButtonLoadV2 if self.compact_actions else PerButtonLoad)(width)
            if config.button_load_policy=='per_button_context_v2':
                if not self.compact_actions:
                    raise ValueError('per_button_context_v2 requires compact_grouped_v3')
                # Absolute mean/max load survives lane-centering and can
                # influence both timing and every slot's count/STOP decision.
                self.button_load_context=nn.Linear(46,width,bias=False)
                nn.init.zeros_(self.button_load_context.weight)
            if not hasattr(self,'echo_kind_0'):
                from .events import event_metadata
                kinds=[None]+[None if (m:=event_metadata(atom)).touch else
                    2 if m.slide else 1 if m.hold else 0 for atom in atom_catalog]
                for kind in range(3):
                    self.register_buffer(f'echo_kind_{kind}',torch.tensor(
                        [i for i,value in enumerate(kinds) if value==kind],dtype=torch.long),persistent=False)
        if getattr(config,'action_head_policy','flat_v1')=='conserved_grouped_v2':
            from .conserved_actions import ConservedGroupedActionHead
            self.conserved_action_head=ConservedGroupedActionHead(self.action_factors,width)
        elif self.compact_actions:
            from .conserved_actions import CompactGroupedActionHead
            self.conserved_action_head=CompactGroupedActionHead(self.action_factors,width)
            del self.action_head
        self.cooperative_geometry=None
        if getattr(config,'configuration_quality_policy','off') in {'cooperative_v1','cooperative_v2','song_paired_v3'}:
            from .cooperative_geometry import CooperativeGeometry
            self.cooperative_geometry=CooperativeGeometry(self.action_factors,width)
        self.quality_context=None
        if getattr(config,'configuration_quality_policy','off') in {'cooperative_v1','cooperative_v2','song_paired_v3'}:
            from .quality_context import QualityContext
            self.quality_context=QualityContext(width,config.condition_dim,audio_dim=config.audio_dim,
                future_evidence=config.configuration_quality_policy in {'cooperative_v2','song_paired_v3'} and not self.compact_actions,
                paired_v3=self.song_paired)
        # A full ledger still describes valid causal state. Its direct and
        # discrepancy routes cannot write an unbounded pointer logit anymore.

    def _embedding(self,ids,*,factor_weight=None):
        return (self.event_embedding(ids)+self.action_factors.embed(ids,composed_weight=factor_weight))*ids.ne(0)[...,None]

    def forward(self,candidate_features,history,condition,action_state=None,history_features=None,
                candidate_mask=None,history_padding_mask=None,candidate_indices=None,
                decode_actions=True,workload_audio=None,ledger=None,action_targets=None,
                curve_strength=1.,interval_audio=None,trailing_audio=None,workload_override=None,
                trailing_override=None,
                symbolic_keep=None,audio_auxiliary=True,head_timing=None,audio_lens=None,
                audio_lens_mask=None,planner_audio=None,planner_mask=None,planner_indices=None,
                planner_mix=None,planner_override=None,lens_embeddings=None,
                motif_ids=None,motif_relative_beats=None,motif_mask=None,motif_strength=1.,
                selected_action_state=None,structure_features=None,structure_transition=None,
                packed_actions=False,planner_unique_indices=None,planner_inverse=None,
                packed_slot_indices=None,packed_first_indices=None,audio_lens_inverse=None,history_beat_ages=None,history_second_ages=None,
                quality_inputs=None,quality_context_strength=1.,quality_plan_strength=1.,
                song_unique_indices=None,song_inverse=None,prepared_song_state=None,
                song_audio=None,song_mask=None,song_clock=None,survival_start=None,empty_gap_audio=None,
                survival_audio=None,survival_edges=None,survival_candidate_seconds=None,
                survival_start_seconds=None,survival_plan_indices=None,survival_plan_mix=None,
                survival_grid_rates_override=None,survival_prefix_counts=None,**unused):
        if (audio_lens is None and lens_embeddings is None) or planner_audio is None or head_timing is None:
            raise ValueError('audio_grounded_v3 requires local audio, coarse plan audio and causal head timing')
        # A full training decode already needs every atom's factor vector.
        # Reuse that differentiable table for history, teacher prefixes and
        # both output branches. Keep it local to this forward, never cached
        # across optimizer updates. Pointer-only calls retain sparse lookups.
        factor_weight=self.action_factors.composed_weight() if decode_actions else None
        batch,count=candidate_features.shape[:2]
        # Mask the same target-derived controls before EVERY internal consumer.
        # Otherwise planner/workload/placement predictions leak the hidden
        # condition back into the action context through their detached paths.
        effective_condition=condition
        if self.training and self.config.condition_mask_prob>0. and condition.size(-1)>1:
            cond_mask=(torch.rand(batch,1,device=condition.device)<self.config.condition_mask_prob)
            masked_tail=torch.where(cond_mask,condition.new_full((batch,condition.size(-1)-1),.5),condition[:,1:])
            effective_condition=torch.cat((condition[:,:1],masked_tail),dim=-1)
        structure_term=None
        if self.config.structure_policy != 'off':
            from .structure import structure_dim_for_policy
            if structure_features is None or structure_features.shape != (batch,count,structure_dim_for_policy(self.config.structure_policy)):
                raise ValueError('Structure-enabled Composer requires matching candidate structure features')
            structure_features=structure_features.to(candidate_features)
            if structure_transition is None:structure_transition=torch.zeros_like(structure_features)
            if structure_transition.shape!=structure_features.shape:
                raise ValueError('structure_transition shape must match structure_features')
            structure_transition=structure_transition.to(structure_features)
            if self.training and self.config.structure_dropout:
                # One coherent mask for all candidates and all branches in
                # this training sample, never candidate-by-candidate flicker.
                keep=(torch.rand(batch,1,1,device=candidate_features.device)>=self.config.structure_dropout)
                structure_features=structure_features*keep
                structure_transition=structure_transition*keep
            structure_term=(self.structure_action_projection(structure_features)
                            +self.structure_transition_projection(structure_transition))
        valid=~candidate_mask if candidate_mask is not None else torch.ones((batch,count),device=candidate_features.device,dtype=torch.bool)
        hidden=self.audio_projection(candidate_features[...,:self.config.audio_dim])+self.time_projection(candidate_features[...,self.config.audio_dim:])
        hidden=self.audio_encoder(hidden,src_key_padding_mask=~valid)
        if structure_term is not None:
            hidden=hidden+self.structure_projection(structure_features)
        lens=(self.lens(audio_lens,audio_lens_mask,inverse=audio_lens_inverse) if audio_lens_inverse is not None
              else self.lens(audio_lens,audio_lens_mask)) if lens_embeddings is None else lens_embeddings
        if self.training and lens_embeddings is not None:
            raise ValueError('Cached audio embeddings are inference-only; training must retain lens gradients')
        placement=self.placement_head(torch.cat((lens,effective_condition[:,None].expand(-1,lens.size(1),-1)),-1)).float()
        # Explicit external overrides remain explicit evidence (e.g. an edited
        # generation curve); masking only applies to model-computed controls.
        song_state = phrase_prediction = None
        if self.song_paired:
            if quality_inputs is None:
                raise ValueError('song_paired_v3 requires quality inputs')
            clock=quality_inputs['quality_plan_clock']
            if prepared_song_state is not None:
                if self.training or torch.is_grad_enabled():
                    raise ValueError('Prepared song state is inference-only and requires gradients disabled')
                if prepared_song_state.shape != (*planner_audio.shape[:2], self.planner.width):
                    raise ValueError('Prepared song state shape must match planner audio')
                song_state=prepared_song_state.to(planner_audio)
            elif song_audio is not None:
                if song_mask is None or song_clock is None or song_inverse is None:
                    raise ValueError('Compact song audio requires mask, clock and inverse')
                song_state=self.planner.encode(song_audio,song_mask,song_clock)[song_inverse]
            elif song_unique_indices is not None and song_inverse is not None:
                song_state=self.planner.encode(planner_audio[song_unique_indices], planner_mask[song_unique_indices], clock[song_unique_indices])[song_inverse]
            else:
                song_state=self.planner.encode(planner_audio, planner_mask, clock)
            plan, phrase_grid, plan_rest_logits=self.planner.heads(song_state,effective_condition,return_rest_logits=True)
            phrase_prediction=read_plan(phrase_grid,quality_inputs['quality_phrase_plan_indices'],quality_inputs['quality_phrase_plan_mix'])
            if planner_override is not None:
                plan=planner_override.to(hidden)
                plan_rest_logits=None
        elif planner_override is not None:
            plan=planner_override.to(hidden)
            plan_rest_logits=None
        elif planner_unique_indices is not None and planner_inverse is not None:
            # Group by the actual controls after masking, preserving the exact
            # original per-row RNG draw and distinct difficulty conditions.
            if not self.training or self.config.condition_mask_prob<=0. or condition.size(-1)<=1:
                representatives,inverse=planner_unique_indices,planner_inverse
                unique_plan,unique_rest_logits=self.planner(planner_audio[representatives],effective_condition[representatives],planner_mask[representatives],return_rest_logits=True)
                plan,plan_rest_logits=unique_plan[inverse],unique_rest_logits[inverse]
            else:
                rep_2u=planner_unique_indices.repeat_interleave(2)
                base_cond=condition[planner_unique_indices]
                masked_cond=torch.cat((base_cond[:,:1],base_cond.new_full((base_cond.size(0),base_cond.size(1)-1),.5)),dim=-1)
                cond_2u=torch.stack((base_cond,masked_cond),dim=1).reshape(-1,condition.size(-1))
                inverse=planner_inverse*2+cond_mask.squeeze(-1).to(dtype=planner_inverse.dtype)
                unique_plan,unique_rest_logits=self.planner(planner_audio[rep_2u],cond_2u,planner_mask[rep_2u],return_rest_logits=True)
                plan,plan_rest_logits=unique_plan[inverse],unique_rest_logits[inverse]
        else:
            plan,plan_rest_logits=self.planner(planner_audio,effective_condition,planner_mask,return_rest_logits=True)
        candidate_plan=read_plan(plan,planner_indices,planner_mix)
        workload=self.predict_workload(workload_audio,effective_condition)
        decision_workload=workload if workload_override is None else workload_override.to(workload)
        audio_context=hidden+self.lens_projection(lens[:,:count])+self.condition_projection(effective_condition)[:,None]
        audio_context=audio_context+self.interval_projection(interval_audio)
        if self.song_paired:
            candidate_song=read_plan(song_state,planner_indices,planner_mix)
            audio_context=audio_context+self.song_projection(candidate_song)
        audio_context=audio_context+float(curve_strength)*(self.workload_projection(decision_workload[:,:count].detach())+self.plan_projection(candidate_plan.detach()))
        # Occupancy is chart-conditional evidence, not a mandatory note rule.
        # Allow a small straight-through gradient from main pointer/action losses into placement_head.
        place_probs=placement[:,:count].sigmoid()
        place_input=(place_probs.detach()+PLACEMENT_GRAD_SCALE*(place_probs-place_probs.detach())
                     if self.training else place_probs.detach())
        audio_context=audio_context+self.placement_projection(place_input)
        audio_context=self.audio_fusion(audio_context)
        if getattr(self,'candidate_competition',None) is not None:
            audio_context=self.candidate_competition(
                audio_context,valid,workload_audio=workload_audio,placement_probs=place_input,
                candidate_time=candidate_features[...,self.config.audio_dim:])
        if symbolic_keep is None:
            symbolic_keep=((torch.rand(batch,1,device=hidden.device)>=self.config.symbolic_dropout).to(hidden)
                if self.training else hidden.new_ones(batch,1))
        else:symbolic_keep=symbolic_keep.to(hidden)
        symbolic=torch.zeros_like(audio_context)
        last_heads=hidden.new_zeros(batch,8)
        previous_heads=hidden.new_zeros(batch,8)
        memory = safe_padding = None
        history_reference_beat = None
        legacy_history_attended = torch.zeros_like(audio_context)
        if history is not None and history.size(1):
            if history.ndim==2:history=history.unsqueeze(-1)
            padding=history_padding_mask if history_padding_mask is not None else torch.zeros(history.shape[:2],device=history.device,dtype=torch.bool)
            lengths=(~padding).sum(1)
            if self.song_paired:
                history_reference_beat=quality_inputs['quality_history_clock'][..., 0].gather(1, (lengths-1).clamp_min(0)[:, None]).squeeze(1)
            positions=(lengths[:,None]-1-torch.arange(history.size(1),device=history.device)[None]).clamp_min(0)
            # The checkpoint policy selects historical digit anchors or parsed
            # head angles. Factors and the original embedding retain note kinds,
            # modifiers and touch rings; center Touch has no outer angle.
            anchors=self.action_factors.history_anchor_embedding(history,self.event_embedding)
            hist_emb=(.75*anchors+.25*self.event_embedding(history)+self.action_factors.embed(history,composed_weight=factor_weight))*history.ne(0)[...,None]
            memory=hist_emb.sum(-2)+self.position_embedding(positions.clamp_max(self.config.history_events+3))
            if history_features is not None:memory=memory+self.history_feature_projection(history_features)
            safe_padding=padding.clone()
            if self.training and self.config.history_dropout>0.:
                dropped=torch.rand(padding.shape,device=hidden.device)<self.config.history_dropout
                if getattr(self,'local_lead_in',None) is not None:
                    # The explicit lead-in needs the actual recent sequence.
                    # Older context retains dropout; symbolic_keep still masks
                    # the entire symbolic branch coherently when requested.
                    recent=torch.arange(history.size(1),device=history.device)[None] >= (lengths[:,None]-LOCAL_LEAD_IN_EVENTS)
                    dropped &= ~recent
                safe_padding |= dropped
            rows=torch.arange(batch,device=history.device)
            tail_valid=(~safe_padding[rows,(lengths-1).clamp_min(0)])&lengths.gt(0)
            all_masked=safe_padding.all(dim=1)
            memory=memory.masked_fill(safe_padding.unsqueeze(-1),0.)
            attn_padding=safe_padding.clone()
            attn_padding[:,0]&=~all_masked
            memory=self.history_encoder(memory,src_key_padding_mask=attn_padding)
            if self.config.history_memory_policy=='rms_cap_v1':
                memory=bounded_history_memory(memory,safe_padding,self.config.history_memory_rms_bound)
            with history_attention_context(self.config.history_attention_policy,audio_context.device.type):
                attended,_=self.history_attention(audio_context,memory,memory,key_padding_mask=attn_padding,need_weights=False)
            attended=attended*(~all_masked)[:,None,None]
            tail=memory[rows,(lengths-1).clamp_min(0)]*tail_valid[:,None]
            legacy_history_attended=self.history_gate.tanh()*attended*lengths.gt(0)[:,None,None]
            symbolic=symbolic+tail[:,None]*lengths.gt(0)[:,None,None]+legacy_history_attended
            last_ids=history[rows,(lengths-1).clamp_min(0)]
            last_heads=self.action_factors.atom_head_lanes[last_ids].any(-2).to(hidden)*tail_valid[:,None]
            landing_heads=None
            if getattr(self,'local_lead_in',None) is not None:
                lead_in_res,landing_heads=self.local_lead_in(
                    memory,history,history_features,safe_padding,lengths,
                    self.action_factors.atom_head_lanes,self.action_factors.atom_end_lanes,
                    candidate_features[...,self.config.audio_dim:],valid,history_beat_ages=history_beat_ages)
                symbolic=symbolic+lead_in_res
            if hasattr(self,'cross_event_relative_factors') or self.cooperative_geometry is not None:
                previous_heads=( (last_heads+landing_heads*tail_valid[:,None]).clamp_min(0.)
                                 if landing_heads is not None else last_heads )
        if action_state is not None:
            projected_state=self.action_projection(action_state)
            symbolic=symbolic+(projected_state[:,None] if action_state.ndim==2 else projected_state)
        timing=(merge_separate_head_timing(head_timing) if self.config.head_timing_policy=='separate_pads_v2' else head_timing)
        symbolic=symbolic+self.head_timing_projection(timing.to(hidden))
        if hasattr(self,'head_timing_detail_projection'):
            symbolic=symbolic+self.head_timing_detail_projection(head_timing.to(hidden))
        trailing=self.predict_trailing(trailing_audio,effective_condition) if trailing_audio is not None else None
        decision_trailing=(trailing if trailing_override is None
                           else trailing_override.to(trailing if trailing is not None else hidden))
        if ledger is not None:
            keep=((torch.rand(batch,1,1,device=hidden.device)>=self.config.ledger_dropout).to(hidden)
                if self.training else 1.)
            ledger_context=self.ledger_projection(ledger.to(hidden))
            if decision_trailing is not None:
                difference=decision_trailing[:,:count].detach()-ledger[...,:4].float().clamp_min(0).log1p()
                ledger_context=ledger_context+float(curve_strength)*self.discrepancy_projection(difference.to(hidden))
            symbolic=symbolic+keep*ledger_context
        load_features=None
        if hasattr(self,'button_load'):
            from .button_load import button_load_features
            hist=history.unsqueeze(-1) if history.ndim==2 else history
            load_valid=(~history_padding_mask if history_padding_mask is not None else
                torch.ones(hist.shape[:2],device=hist.device,dtype=torch.bool))
            load_features=button_load_features(self.action_factors.atom_head_lanes[hist].any(-2),
                history_beat_ages,load_valid,head_timing,candidate_features[...,self.config.audio_dim],
                history_second_ages=history_second_ages,candidate_gap_seconds=candidate_features[...,self.config.audio_dim+1],
                extended=self.compact_actions)
            if hasattr(self,'button_load_context'):
                # Recent-run evidence should decay when the next query waits.
                context_features=load_features.clone()
                ages=torch.expm1(head_timing[...,head_timing.size(-1)//3:head_timing.size(-1)//3+8].float())
                context_features[...,18]=context_features[...,18]*torch.exp(-ages/.25)
                summary=torch.cat((context_features.mean(-2),context_features.amax(-2)),-1)
                symbolic=symbolic+8.*self.button_load_context(summary.to(symbolic))
            load_features=load_features*symbolic_keep[:,None,None]
        symbolic=symbolic*symbolic_keep[:,None]
        gap_col=candidate_features[...,self.config.audio_dim].contiguous()
        integrated=getattr(self.config,'timing_distribution_policy','hierarchical_v1')=='survival_integrated_v2'
        survival=integrated or getattr(self.config,'timing_distribution_policy','hierarchical_v1')=='survival_v1'
        if self.compact_actions and self.config.future_horizon_policy!='off':
            audio_context=audio_context+self.fixed_future(audio_context,quality_inputs)
        if survival and hasattr(self,'future_horizon'):
            audio_context=audio_context+self.future_horizon(
                audio_context,candidate_features,valid,audio_dim=self.config.audio_dim)
        bin_edges=(self._bin_edges if self._bin_edges.device==gap_col.device and self._bin_edges.dtype==gap_col.dtype
                   else gap_col.new_tensor(BIN_EDGES))
        bins=torch.bucketize(gap_col,bin_edges)
        fine=self.pointer(audio_context).squeeze(-1).float()
        if hasattr(self,'fine_timing_gain'):
            from maichart.training.music_plan_targets import FINE_RHYTHM_GAP_CEILING
            local_fine=read_plan(phrase_grid,planner_indices,planner_mix)[...,8].detach().float()
            # event_ctx's existing window-relative fine target. The learned
            # scalar begins neutral, never creates a global subdivision quota,
            # and contributes at most +/-2 to the audio timing score.
            local_odds=torch.logit(local_fine.clamp(1e-5,1.-1e-5)).clamp(-2.,2.)*.5
            fine_gap=(gap_col>0.) & (gap_col<=float(FINE_RHYTHM_GAP_CEILING))
            contrast=fine_gap.to(fine)*2.-1.
            fine=fine+2.*torch.tanh(8.*self.fine_timing_gain)*local_odds*contrast
        bin_hidden=bin_pool(audio_context,bins,valid)+self.bin_embedding.weight
        coarse=self.bin_head(bin_hidden).squeeze(-1).float()
        fine_res=self.config.history_residual_bound*torch.tanh(self.history_pointer(symbolic).squeeze(-1).float())
        bin_res=self.config.history_residual_bound*torch.tanh(self.history_bin(bin_pool(symbolic,bins,valid)).squeeze(-1).float())
        audio_logits=hierarchical_log_probs(fine,coarse,bins,valid)
        echo_lane_bias=None
        if hasattr(self,'echo_prior'):
            if history_beat_ages is None:
                raise ValueError('Phase echo requires exact history beat ages')
            hist=(history.unsqueeze(-1) if history.ndim==2 else history)
            echo_valid=(~history_padding_mask if history_padding_mask is not None else
                torch.ones(hist.shape[:2],device=hist.device,dtype=torch.bool))
            echo_lanes=self.action_factors.atom_head_lanes[hist].any(-2)
            echo_lane_bias,p_echo=self.echo_prior(echo_lanes,history_beat_ages,echo_valid,gap_col,audio_context,symbolic_keep)
            bin_echo=torch.full_like(coarse,float('-inf')).scatter_reduce(1,bins,
                p_echo.masked_fill(~valid,float('-inf')),reduce='amax',include_self=True)
            bin_echo=torch.where(torch.isfinite(bin_echo),bin_echo,torch.zeros_like(bin_echo))
            proposed=hierarchical_log_probs(fine+fine_res+p_echo,coarse+bin_res+bin_echo,bins,valid)
        else:
            proposed=hierarchical_log_probs(fine+fine_res,coarse+bin_res,bins,valid)
        # Bound the FINAL candidate residual too. Bounding intermediate heads
        # alone does not bound odds after per-bin normalizers are subtracted.
        delta=(proposed-audio_logits)*valid
        delta=delta-delta.sum(-1,keepdim=True)/valid.sum(-1,keepdim=True).clamp_min(1)
        delta=delta*valid
        correction=self.config.history_residual_bound*torch.tanh(delta/self.config.history_residual_bound)
        full_logits=(audio_logits+correction).masked_fill(~valid,-1e4).log_softmax(-1)
        survival_result={}
        if survival:
            from .event_survival import event_survival,normalize_survival
            if integrated:
                from .event_survival import integrated_survival,redistribute_survival
                if any(value is None for value in (survival_audio,survival_edges,survival_candidate_seconds,survival_start_seconds)):
                    raise ValueError('Integrated survival requires its fixed absolute-time audio grid')
                grid_plan=(read_plan(plan,survival_plan_indices,survival_plan_mix)
                           if survival_plan_indices is not None else None)
                grid_rates=(self.predict_survival_grid(survival_audio,effective_condition,grid_plan)
                            if survival_grid_rates_override is None else survival_grid_rates_override.float())
                if self.training and survival_grid_rates_override is not None:
                    raise ValueError('Cached survival rates are inference-only')
                base,tail,exposure=integrated_survival(grid_rates,survival_edges,
                    survival_candidate_seconds,valid,survival_start_seconds)
                survival_result['survival_grid_rates']=grid_rates
                feedback_base,feedback_tail=base,tail
                if self.config.density_feedback_policy=='causal_count_debt_v1':
                    from .event_survival import causal_count_feedback
                    if survival_prefix_counts is None:raise ValueError('Count feedback requires actual committed prefix counts')
                    effective_rates,penalty=causal_count_feedback(grid_rates,survival_edges,survival_prefix_counts,symbolic_keep)
                    feedback_base,feedback_tail,_=integrated_survival(effective_rates,survival_edges,
                        survival_candidate_seconds,valid,survival_start_seconds)
                    survival_result['density_feedback_log_penalty']=penalty
            else:
                rates=torch.expm1(decision_workload[:,:count,0].float().clamp(0.,8.))
                base,tail,exposure=event_survival(rates,candidate_features[...,self.config.audio_dim+1],valid,survival_start)
            # The event-rate distribution owns timing. The existing acoustic
            # shape and symbolic adjustments retain bounded log-odds influence.
            raw_audio_timing=audio_logits
            shape=audio_logits-(audio_logits*valid).sum(-1,keepdim=True)/valid.sum(-1,keepdim=True).clamp_min(1)
            # Only symbolic history has a bounded residual contract. Clipping
            # the audio owner's score to +/-1 erased onset confidence and its
            # gradient, leaving smooth density to choose individual notes.
            if not integrated:shape=torch.tanh(shape)*valid
            if integrated:
                # A density prior supplies exposure, not the whole rhythm.
                # Keep the learned distance classifier separate from fine
                # positions; multiplying full candidate distributions would
                # penalize bands merely for having more candidate points.
                coarse_adjustment=coarse.log_softmax(-1).gather(1,bins)
                timed_base,_=redistribute_survival(base,tail,coarse_adjustment,valid)
                audio_logits,audio_tail=redistribute_survival(timed_base,tail,shape,valid,bins=bins)
                if self.config.density_feedback_policy=='causal_count_debt_v1':
                    conditioned,_=redistribute_survival(feedback_base,feedback_tail,coarse_adjustment,valid)
                    conditioned,_=redistribute_survival(conditioned,feedback_tail,shape,valid,bins=bins)
                    full_logits,full_tail=redistribute_survival(conditioned,feedback_tail,correction,valid)
                else:
                    full_logits,full_tail=redistribute_survival(audio_logits,tail,correction,valid)
                survival_result['audio_timing_log_probs']=raw_audio_timing
            else:
                audio_logits,audio_tail=normalize_survival(base+shape,tail)
                full_logits,full_tail=normalize_survival(base+shape,tail,correction)
            survival_result.update(next_event_log_probs=full_logits,audio_next_event_log_probs=audio_logits,
                no_event_log_prob=full_tail,audio_no_event_log_prob=audio_tail,event_exposure=exposure)
        if structure_term is not None:
            # Explicit configuration path, after the timing decision logits.
            # This also survives selected-prefix reconstruction below.
            audio_context=audio_context+structure_term
        if hasattr(self,'future_horizon') and not survival:
            audio_context=audio_context+self.future_horizon(
                audio_context,candidate_features,valid,audio_dim=self.config.audio_dim)
        quality={}
        quality_residual=quality_audio_residual=0.
        if self.quality_context is not None:
            if quality_inputs is None:
                raise ValueError('Configuration quality requires causal configuration quality inputs')
            quality=self.quality_context(audio_context,effective_condition,quality_inputs,
                anchor_embeddings=(None if self.song_paired else self._embedding(quality_inputs['quality_anchor_ids'],factor_weight=factor_weight)),
                symbolic_keep=symbolic_keep,valid_mask=valid,
                context_strength=quality_context_strength,plan_strength=quality_plan_strength,
                placement_probs=place_probs,attack=candidate_features[:, :count, 92],
                song_state=(candidate_song if self.song_paired else None), phrase_prediction=phrase_prediction,
                history_memory=memory,history_padding=safe_padding,history_reference_beat=history_reference_beat,
                history_song=(read_plan(song_state,quality_inputs['quality_history_plan_indices'],quality_inputs['quality_history_plan_mix']) if self.song_paired and memory is not None else None))
            quality_residual=quality['quality_residual']
            quality_audio_residual=quality['quality_audio_residual']
        audio_action_context=self.candidate_fusion(audio_context+quality_audio_residual)
        action_symbolic = _action_history_context(symbolic, legacy_history_attended, symbolic_keep, self.song_paired)
        action_base=audio_context+action_symbolic+self.action_relation(last_heads)[:,None]*symbolic_keep[:,None]+quality_residual
        action_context=self.candidate_fusion(action_base)
        result={'candidate_logits':full_logits,'candidate_context':action_context,
            'audio_action_context':audio_action_context,
            'audio_candidate_logits':audio_logits,'candidate_bins':bins,
            'bin_logits':coarse+bin_res,'placement_logits':placement,'plan':plan,'workload':workload,
            'pointer_residual_rms':(correction.square()*valid).sum().sqrt()/valid.sum().clamp_min(1).sqrt(),
            'plan_context_rms':candidate_plan.square().mean().sqrt()}
        result.update(quality)
        if plan_rest_logits is not None:result['plan_rest_logits']=plan_rest_logits
        result.update(survival_result)
        if self.compact_actions and empty_gap_audio is not None and not integrated:
            # Authored gap probes are supervision only. They never enter the
            # candidate context, history, pointer or configuration logits.
            gap_prediction=self.predict_workload(empty_gap_audio,effective_condition)
            result['empty_gap_rates']=torch.expm1(gap_prediction[...,0].float().clamp(0.,8.))
        if load_features is not None:result['button_load_features']=load_features
        if echo_lane_bias is not None:result['echo_lane_bias']=echo_lane_bias
        if trailing is not None:result['trailing_workload']=trailing
        if hasattr(self,'cross_event_relative_factors') or self.cooperative_geometry is not None:
            result['previous_heads']=previous_heads*symbolic_keep
        if self.config.action_state_policy=='selected_prefix_v3':
            # Keep the committed snapshot in the pointer. Only the chosen
            # action query replaces it with state advanced to that time.
            state_term=(projected_state[:,None] if action_state.ndim==2 else projected_state) if action_state is not None else 0.
            result['selected_state_base']=action_base-state_term*symbolic_keep[:,None]
            result['selected_state_keep']=symbolic_keep
        if decode_actions:
            result.update(self.decode_candidate(action_context,candidate_indices,
                action_targets=action_targets,audio_context=audio_action_context,project_audio_actions=audio_auxiliary,
                motif_ids=motif_ids,motif_relative_beats=motif_relative_beats,motif_mask=motif_mask,
                motif_strength=motif_strength,motif_keep=symbolic_keep,
                selected_action_state=selected_action_state,state_base=result.get('selected_state_base'),
                state_keep=result.get('selected_state_keep'),factor_weight=factor_weight,
                previous_heads=result.get('previous_heads'),echo_heads=echo_lane_bias,button_load_features=load_features,packed_actions=packed_actions,
                packed_slot_indices=packed_slot_indices,packed_first_indices=packed_first_indices))
        return result

    def predict_survival_grid(self,audio,condition,plan_at_grid=None):
        from .event_survival import fuse_survival_rates
        return fuse_survival_rates(self.predict_workload(audio,condition)[...,0],plan_at_grid)

    def decode_candidate(self,candidate_context,candidate_indices=None,*,action_targets=None,project_actions=True,audio_context=None,
                         motif_ids=None,motif_relative_beats=None,motif_mask=None,motif_strength=1.,motif_keep=None,
                         selected_action_state=None,state_base=None,state_keep=None,factor_weight=None,
                         project_audio_actions=True,previous_heads=None,echo_heads=None,button_load_features=None,packed_actions=False,
                         packed_slot_indices=None,packed_first_indices=None):
        if (hasattr(self,'cross_event_relative_factors') or self.cooperative_geometry is not None) and previous_heads is None:
            raise ValueError('Cross-event action decoding requires causal previous_heads from the same forward')
        if previous_heads is not None and previous_heads.shape!=(candidate_context.size(0),8):
            raise ValueError('previous_heads must match the forward batch [B,8]')
        if project_actions and factor_weight is None:
            factor_weight=self.action_factors.composed_weight()
        if candidate_indices is not None:
            rows=torch.arange(candidate_context.size(0),device=candidate_context.device)
            candidate_context=candidate_context[rows,candidate_indices]
            if audio_context is not None:
                audio_context=audio_context[rows,candidate_indices]
            if state_base is not None:state_base=state_base[rows,candidate_indices]
            if echo_heads is not None and echo_heads.ndim==3:echo_heads=echo_heads[rows,candidate_indices]
            if button_load_features is not None and button_load_features.ndim==4:
                button_load_features=button_load_features[rows,candidate_indices]
        result={'future':self.future_head(candidate_context),'eos_logits':self.eos_head(candidate_context).squeeze(-1),'action_context':candidate_context}
        if previous_heads is not None:result['previous_heads']=previous_heads
        if echo_heads is not None:result['echo_heads']=echo_heads
        if button_load_features is not None:result['button_load_features']=button_load_features
        if self.config.action_state_policy=='selected_prefix_v3':
            if selected_action_state is None or state_base is None:
                raise ValueError('selected_prefix_v3 requires a causal state at the selected beat')
            if selected_action_state.shape!=(*state_base.shape[:-1],self.config.action_dim):
                raise ValueError('Selected action state must match the selected candidate shape')
            keep=state_keep if state_keep is not None else state_base.new_ones(state_base.shape[0],1)
            while keep.ndim<state_base.ndim:keep=keep.unsqueeze(-2)
            candidate_context=self.candidate_fusion(state_base+self.action_projection(selected_action_state)*keep)
            result['action_context']=candidate_context
        # Motifs guide configurations only. The learned timing pointer, EOS,
        # future summary and audio-only action reference retain their inputs.
        if hasattr(self,'motif_memory') and motif_ids is not None:
            if motif_relative_beats is None or motif_mask is None:
                raise ValueError('Motif IDs require relative timing and visibility mask')
            motif_residual=self.motif_memory(
                candidate_context,lambda ids:self._embedding(ids,factor_weight=factor_weight),
                motif_ids,motif_relative_beats,motif_mask)
            # The archive is symbolic history too. Coherent history dropout
            # must not leave a second token-history route into action decoding.
            if motif_keep is not None:
                keep=motif_keep.to(motif_residual)
                while keep.ndim<motif_residual.ndim:keep=keep.unsqueeze(-2)
                motif_residual=motif_residual*keep
            candidate_context=candidate_context+float(motif_strength)*motif_residual
            result['action_context']=candidate_context
        if audio_context is not None:
            result['audio_action_context']=audio_context
        if project_actions:
            slot_embeddings=self.slot_embedding.weight
            if action_targets is not None:
                if not 1<=action_targets.size(-1)<=self.config.action_slots:
                    raise ValueError('Action target slots exceed the configured capacity')
                slot_embeddings=slot_embeddings[:action_targets.size(-1)]
            contexts=candidate_context.unsqueeze(-2)+slot_embeddings
            audio_contexts=(audio_context.unsqueeze(-2)+slot_embeddings
                            if audio_context is not None and project_audio_actions else None)
            relation=None
            prefix_ends=None
            geometry_active=None
            if action_targets is not None:
                embedded=self._embedding(action_targets,factor_weight=factor_weight)
                prefix=embedded.cumsum(-2)-embedded
                contexts=contexts+prefix
                if audio_contexts is not None:
                    audio_contexts=audio_contexts+prefix
                heads=self.action_factors.atom_head_lanes[action_targets].to(contexts)
                relation=(heads.cumsum(-2)-heads)>0
                if self.cooperative_geometry is not None:
                    ends=self.cooperative_geometry.prefix_end_lanes[action_targets].to(contexts)
                    prefix_ends=(ends.cumsum(-2)-ends)>0
                    # Include the first STOP, and ignore all slots after it.
                    geometry_active=torch.ones_like(action_targets,dtype=torch.bool)
                    geometry_active[...,1:]=action_targets[...,:-1].ne(0).cumprod(-1).bool()
            slot_indices=None
            if packed_actions:
                if action_targets is None or action_targets.ndim!=2 or contexts.ndim!=3:
                    raise ValueError('Packed actions require selected candidates and [B,S] teacher targets')
                if packed_slot_indices is None:
                    active=torch.ones_like(action_targets,dtype=torch.bool)
                    # Legacy activity contains joint first-STOP supervision.
                    active[:,1:]=action_targets[:,:-1].ne(0)
                    slot_indices=active.nonzero(as_tuple=False)
                else:
                    slot_indices=packed_slot_indices
                row,slot=slot_indices.unbind(-1)
                contexts=contexts[row,slot]
                if audio_contexts is not None:audio_contexts=audio_contexts[row,slot]
                if relation is not None:relation=relation[row,slot]
                if prefix_ends is not None:prefix_ends=prefix_ends[row,slot]
                if geometry_active is not None:geometry_active=geometry_active[row,slot]
                result['action_slot_indices']=slot_indices
            contexts=self.slot_fusion(contexts)
            full_logits=F.linear(contexts,factor_weight)
            if not self.compact_actions:full_logits=full_logits+self.action_head(contexts)
            conserved_tables=self.conserved_action_head.detail_tables() if hasattr(self,'conserved_action_head') else None
            if self.compact_actions:
                geometry_slots=(slot if slot_indices is not None else torch.arange(contexts.size(-2),device=contexts.device))
                h,d=self._compact_geometry(contexts,
                    previous_heads=previous_heads[row] if slot_indices is not None else previous_heads,
                    prefix_heads=relation,prefix_ends=prefix_ends,slots=geometry_slots,active=geometry_active,
                    echo_heads=echo_heads[row] if slot_indices is not None and echo_heads is not None else echo_heads,
                    button_features=button_load_features[row] if slot_indices is not None and button_load_features is not None else button_load_features)
                full_logits=self.conserved_action_head(full_logits,contexts,tables=conserved_tables,head_bias=h,detail_bias=d)
                result['action_logits']=full_logits
                if audio_contexts is not None:result['audio_action_logits']=self.conserved_action_head.audio_logits(self.slot_fusion(audio_contexts))
                return result
            if conserved_tables is not None:
                full_logits=self.conserved_action_head(full_logits,contexts,tables=conserved_tables)
            if relation is not None:
                # Same-event head geometry is also a macro position decision,
                # not a trajectory/duration score to normalize away per pad.
                full_logits=self._same_event_logits(full_logits,contexts,relation)
            if hasattr(self,'cross_event_relative_factors'):
                # Relative movement is a macro starting-pad decision. Applying
                # its per-pad shift before within-group normalization would
                # cancel the entire learned branch in the grouped head.
                if slot_indices is None:
                    first=self._cross_event_logits(full_logits[...,0,:],contexts[...,0,:],previous_heads)
                    full_logits=torch.cat((first.unsqueeze(-2),full_logits[...,1:,:]),-2)
                else:
                    first_rows=(slot.eq(0).nonzero(as_tuple=False).flatten()
                        if packed_first_indices is None else packed_first_indices)
                    first=self._cross_event_logits(full_logits[first_rows],contexts[first_rows],previous_heads[row[first_rows]])
                    full_logits=full_logits.index_copy(0,first_rows,first)
            if self.cooperative_geometry is not None:
                geometry_slots=(slot if slot_indices is not None else
                    torch.arange(contexts.size(-2),device=contexts.device))
                full_logits=self.cooperative_geometry(full_logits,contexts,
                    previous_heads=(previous_heads[row] if slot_indices is not None else previous_heads),
                    prefix_heads=relation,prefix_ends=prefix_ends,slots=geometry_slots,active=geometry_active)
            lane_bias=None
            if echo_heads is not None:
                lane_bias=(echo_heads[row] if slot_indices is not None else echo_heads.unsqueeze(-2))
            if button_load_features is not None:
                loads=button_load_features[row] if slot_indices is not None else button_load_features
                load_scores=self.button_load(loads,contexts)
                lane_bias=load_scores if lane_bias is None else lane_bias+load_scores
            if lane_bias is not None:full_logits=self._echo_logits(full_logits,lane_bias)
            result['action_logits']=full_logits
            if audio_contexts is not None:
                audio_contexts=self.slot_fusion(audio_contexts)
                audio_logits=self.action_head(audio_contexts)+F.linear(audio_contexts,factor_weight)
                if conserved_tables is not None:
                    audio_logits=self.conserved_action_head(audio_logits,audio_contexts,tables=conserved_tables)
                if relation is not None:
                    audio_logits=self._same_event_logits(audio_logits,audio_contexts,relation)
                if self.cooperative_geometry is not None and (prefix_ends is not None or relation is not None):
                    # Current-event choices also exist on the audio-only path;
                    # committed historical heads deliberately do not.
                    audio_logits=self.cooperative_geometry(audio_logits,audio_contexts,
                        prefix_heads=relation,prefix_ends=prefix_ends,slots=geometry_slots,active=geometry_active)
                result['audio_action_logits']=audio_logits
        return result

    def _same_event_logits(self,logits,context,prefix_heads):
        delta=self.action_factors.relative_logits(context,prefix_heads)
        if getattr(self.config,'configuration_quality_policy','off') not in {'cooperative_v2','song_paired_v3'}:
            return logits+delta
        result=logits
        for kind in range(3):
            ids=getattr(self.cooperative_geometry,f'kind_{kind}')
            if not ids.numel():continue
            original=logits.index_select(-1,ids).float()
            within_delta=delta.index_select(-1,ids).float()
            within_delta=within_delta-within_delta.mean(-1,keepdim=True)
            adjusted=original+within_delta
            adjusted=adjusted+(original.logsumexp(-1,keepdim=True)-adjusted.logsumexp(-1,keepdim=True))
            result=result.index_copy(-1,ids,adjusted.to(logits.dtype))
        return result

    def _compact_geometry(self,context,*,previous_heads=None,prefix_heads=None,prefix_ends=None,
                          slots=0,active=None,echo_heads=None,button_features=None,history_strength=1.):
        def offsets(source,table):
            source=source.to(context)
            while source.ndim<context.ndim:source=source.unsqueeze(-2)
            source=source/source.sum(-1,keepdim=True).clamp_min(1)
            return (F.linear(context,table.weight)[...,self._compact_offsets]*source[...,None]).sum(-2)
        heads=context.new_zeros((*context.shape[:-1],8));ends=None
        if prefix_heads is not None:heads=heads+offsets(prefix_heads,self.action_factors.relative_factors)
        first=slots==0 if isinstance(slots,int) else slots.eq(0).unsqueeze(-1)
        if previous_heads is not None and hasattr(self,'cross_event_relative_factors'):
            heads=heads+offsets(previous_heads,self.cross_event_relative_factors)*first*float(history_strength)
        g=self.cooperative_geometry
        if g is not None:
            if previous_heads is not None:heads=heads+offsets(previous_heads,g.later_cross_factors)*(~first if torch.is_tensor(first) else not first)*float(history_strength)
            if prefix_ends is not None:
                heads=heads+offsets(prefix_ends,g.end_to_head_factors)
                ends=offsets(prefix_ends,g.end_to_end_factors)
            if prefix_heads is not None:
                add=offsets(prefix_heads,g.head_to_end_factors)
                ends=add if ends is None else ends+add
        if active is not None:
            heads=heads*active[...,None]
            if ends is not None:ends=ends*active[...,None]
        if echo_heads is not None:
            while echo_heads.ndim<heads.ndim:echo_heads=echo_heads.unsqueeze(-2)
            heads=heads+echo_heads*float(history_strength)
        head_bias=heads.unsqueeze(-2).expand(*heads.shape[:-1],3,8)
        if button_features is not None:head_bias=head_bias+self.button_load(button_features,context)*float(history_strength)
        detail=None if ends is None else F.linear(ends,g.target_ends.to(ends))
        return head_bias,detail

    def audio_reference_slot(self,context,slot,selected_ids):
        if not self.compact_actions:return self.action_slot(context,slot,selected_ids)
        if selected_ids:
            ids=torch.as_tensor(selected_ids,dtype=torch.long,device=context.device)
            context=context+self._embedding(ids).sum(0)
        return self.conserved_action_head.audio_logits(self.slot_fusion(context+self.slot_embedding.weight[slot]))

    def _cross_event_logits(self,logits,context,previous_heads,strength=1.):
        """Learn button movement without changing Tap/Hold/Slide/Touch masses."""
        delta=float(strength)*self.action_factors.relative_logits(context,previous_heads,
            factor_embedding=self.cross_event_relative_factors,target_geometry=self.cross_event_geometry)
        result=logits
        for kind in range(3):
            ids=getattr(self,f'cross_event_kind_{kind}')
            if not ids.numel():continue
            original=logits.index_select(-1,ids).float()
            adjusted=original+delta.index_select(-1,ids).float()
            shift=original.logsumexp(-1,keepdim=True)-adjusted.logsumexp(-1,keepdim=True)
            adjusted=adjusted+shift
            result=result.index_copy(-1,ids,adjusted.to(logits.dtype))
        return result

    def _echo_logits(self,logits,echo_heads):
        bias=echo_heads.float() @ self.action_factors.atom_head_lanes.float().T
        result=logits
        for kind in range(3):
            ids=getattr(self,f'echo_kind_{kind}')
            if not ids.numel():continue
            original=logits.index_select(-1,ids).float()
            adjusted=original+bias.index_select(-1,ids)
            adjusted=adjusted+(original.logsumexp(-1,keepdim=True)-adjusted.logsumexp(-1,keepdim=True))
            result=result.index_copy(-1,ids,adjusted.to(logits.dtype))
        return result

    def action_slot(self,context,slot,selected_ids,*,audio_context=None,history_strength=1.,previous_heads=None,echo_heads=None,button_load_features=None):
        if audio_context is not None and float(history_strength)<1.:
            alpha=max(0.,min(1.,float(history_strength)))
            context=alpha*context+(1.-alpha)*audio_context
        if selected_ids:
            ids=torch.as_tensor(selected_ids,dtype=torch.long,device=context.device)
            context=context+self._embedding(ids).sum(0)
        slot_context=self.slot_fusion(context+self.slot_embedding.weight[slot])
        if self.compact_actions:
            ids=torch.as_tensor(selected_ids,dtype=torch.long,device=context.device)
            prefix_heads=self.action_factors.atom_head_lanes[ids].any(0) if selected_ids else None
            prefix_ends=self.cooperative_geometry.prefix_end_lanes[ids].any(0) if selected_ids and self.cooperative_geometry is not None else None
            h,d=self._compact_geometry(slot_context,previous_heads=previous_heads,prefix_heads=prefix_heads,
                prefix_ends=prefix_ends,slots=slot,echo_heads=echo_heads,button_features=button_load_features,history_strength=history_strength)
            return self.conserved_action_head(self.action_factors(slot_context),slot_context,head_bias=h,detail_bias=d)
        logits=self.action_head(slot_context)+self.action_factors(slot_context)
        if hasattr(self,'conserved_action_head'):
            logits=self.conserved_action_head(logits,slot_context)
        if selected_ids:
            prefix_heads=self.action_factors.atom_head_lanes[ids].any(0)
            logits=self._same_event_logits(logits,slot_context,prefix_heads)
        if slot==0 and previous_heads is not None and hasattr(self,'cross_event_relative_factors'):
            logits=self._cross_event_logits(logits,slot_context,previous_heads,history_strength)
        if self.cooperative_geometry is not None and slot>0:
            prefix_ends=(self.cooperative_geometry.prefix_end_lanes[ids].any(0)
                         if selected_ids else None)
            logits=self.cooperative_geometry(logits,slot_context,previous_heads=previous_heads,
                prefix_heads=prefix_heads if selected_ids else None,prefix_ends=prefix_ends,
                slots=slot,history_strength=history_strength)
        lane_bias=echo_heads
        if button_load_features is not None:
            load_scores=self.button_load(button_load_features,slot_context)
            lane_bias=load_scores if lane_bias is None else lane_bias+load_scores
        if lane_bias is not None:logits=self._echo_logits(logits,lane_bias*float(history_strength))
        return logits
