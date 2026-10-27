"""Compact event-level composer model with a candidate pointer and action head."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import torch
from torch import nn

from . import INPUT_CONTRACT
from .acoustics import CONTEXT_DIM, INTERVAL_DIM, TRAILING_CONTEXT_DIM
from .events import HEAD_TIMING_DIM,HEAD_TIMING_POLICIES,head_timing_dim_for_policy

WORKLOAD_PROFILES = {'audio_workload_v1', 'music_context_v2'}
GROUNDED_PROFILE = 'audio_grounded_v3'
WORKLOAD_PROFILES.add(GROUNDED_PROFILE)
MUSIC_PROFILES = {'music_context_v2', GROUNDED_PROFILE}
HISTORY_ANCHOR_POLICIES = {'legacy_digit_v1', 'parsed_heads_v2', 'parsed_geometry_v3'}
STATE_POLICIES = {'committed_legacy_v1', 'committed_causal_v2', 'candidate_prefix_v2', 'selected_prefix_v3'}
MOTIF_POLICIES = {'off', 'audio_recurrence_archive_v1'}
SLIDE_FACTOR_POLICIES = {'marginal_v1', 'branch_joint_v2'}
HISTORY_MEMORY_POLICIES = {'unbounded_v1', 'rms_cap_v1'}
HISTORY_ATTENTION_POLICIES = {'auto_v1', 'math_v1'}
COMPOSER_POLICY_FIELDS = {
    'action_state_policy': (STATE_POLICIES, 'committed_legacy_v1', 'committed_causal_v2'),
    'motif_memory_policy': (MOTIF_POLICIES, 'off', 'audio_recurrence_archive_v1'),
    'slide_factor_policy': (SLIDE_FACTOR_POLICIES, 'marginal_v1', 'branch_joint_v2'),
    'history_memory_policy': (HISTORY_MEMORY_POLICIES, 'unbounded_v1', 'rms_cap_v1'),
    'history_attention_policy': (HISTORY_ATTENTION_POLICIES, 'auto_v1', 'math_v1'),
    'head_timing_policy': (HEAD_TIMING_POLICIES, 'merged_pads_v1', 'separate_pads_v2'),
    'cross_event_relation_policy': ({'off','relative_heads_v1'}, 'off', 'relative_heads_v1'),
}


def validate_supported_context_policies(config):
    """Reject context policies that the selected architecture cannot execute."""
    values=config if isinstance(config,dict) else config.to_dict()
    if values.get('architecture_profile')==GROUNDED_PROFILE:return
    unsupported = {
        'motif_memory_policy': values.get('motif_memory_policy','off')!='off',
        'action_state_policy': values.get('action_state_policy','committed_legacy_v1') in {'candidate_prefix_v2','selected_prefix_v3'},
        'slide_factor_policy': values.get('slide_factor_policy','marginal_v1')!='marginal_v1',
        'history_memory_policy': values.get('history_memory_policy','unbounded_v1')!='unbounded_v1',
        'history_attention_policy': values.get('history_attention_policy','auto_v1')!='auto_v1',
        'head_timing_policy':values.get('head_timing_policy','merged_pads_v1')!='merged_pads_v1',
        'cross_event_relation_policy':values.get('cross_event_relation_policy','off')!='off',
    }
    for name, invalid in unsupported.items():
        if invalid:raise ValueError(f'{name}={values[name]} requires audio_grounded_v3')


def resolve_composer_policy(name, requested='auto', parent_config=None, *, resume=False, architecture_profile=GROUNDED_PROFILE):
    choices, legacy, fresh = COMPOSER_POLICY_FIELDS[name]
    if not isinstance(requested,str) or requested not in {'auto', *choices}:
        raise ValueError(f'Unknown {name}')
    if name in {'motif_memory_policy','slide_factor_policy','history_memory_policy','history_attention_policy','head_timing_policy','cross_event_relation_policy'} and architecture_profile!=GROUNDED_PROFILE:
        fresh=legacy
    inherited = parent_config.get(name, legacy) if parent_config else None
    if not resume and name in {'head_timing_policy','cross_event_relation_policy'}:
        inherited=None
    resolved = (inherited or fresh) if requested == 'auto' else requested
    if resume and inherited is not None and resolved != inherited:
        raise ValueError(f'Resume inherits {name}; use explicit finetune in a new directory')
    return resolved


def resolve_history_anchor_policy(requested='auto', parent_config=None, *, resume=False):
    """Fresh models use parsed heads; continuation keeps the parent's meaning."""
    if not isinstance(requested,str) or requested not in {'auto', *HISTORY_ANCHOR_POLICIES}:
        raise ValueError('Unknown history_anchor_policy')
    parent_config = parent_config or {}
    inherited = parent_config.get('history_anchor_policy', 'legacy_digit_v1') if parent_config else None
    resolved = (inherited or 'parsed_geometry_v3') if requested == 'auto' else requested
    if resume and inherited is not None and resolved != inherited:
        raise ValueError('Resume inherits history_anchor_policy; use explicit finetune in a new directory to change it')
    return resolved


@dataclass(slots=True)
class ComposerConfig:
    atom_vocab_size: int
    d_model: int = 192
    nhead: int = 6
    layers: int = 3
    dim_feedforward: int = 576
    dropout: float = .1
    audio_dim: int = 120
    condition_dim: int = 16
    action_dim: int = 50
    history_events: int = 128
    candidate_count: int = 96
    action_slots: int = 8
    candidate_feature_dim: int = 141
    history_feature_dim: int = 8
    future_dim: int = 8
    architecture_profile: str = 'audio_workload_v1'
    symbolic_dropout: float = .15
    head_timing_dim: int = 0
    lens_width: int = 64
    lens_chunk_size: int = 32
    history_residual_bound: float = 1.
    history_dropout: float = .3
    ledger_dropout: float = .5
    history_anchor_policy: str = 'parsed_geometry_v3'
    action_state_policy: str = 'committed_causal_v2'
    motif_memory_policy: str = 'off'
    motif_max_events: int = 16
    slide_factor_policy: str = 'marginal_v1'
    history_memory_policy: str = 'unbounded_v1'
    history_memory_rms_bound: float = 8.
    history_attention_policy: str = 'auto_v1'
    head_timing_policy: str = 'merged_pads_v1'
    cross_event_relation_policy: str = 'off'

    def __post_init__(self):
        if self.history_anchor_policy not in HISTORY_ANCHOR_POLICIES:
            raise ValueError('Unknown history_anchor_policy')
        for name, (choices, _, _) in COMPOSER_POLICY_FIELDS.items():
            if getattr(self, name) not in choices:
                raise ValueError(f'Unknown {name}')
        if (isinstance(self.history_memory_rms_bound, bool)
                or not math.isfinite(self.history_memory_rms_bound)
                or not 1. <= self.history_memory_rms_bound <= 64.):
            raise ValueError('history_memory_rms_bound must be finite and in [1,64]')
        if isinstance(self.motif_max_events, bool) or not 1 <= self.motif_max_events <= 32:
            raise ValueError('motif_max_events must be in [1,32]')
        if not 1 <= self.action_slots <= 64:
            raise ValueError('Composer action_slots must be in [1,64]')
        self.head_timing_dim = int(self.head_timing_dim)
        if self.head_timing_dim < 0:
            raise ValueError('head_timing_dim must be non-negative')
        if self.head_timing_dim and self.architecture_profile not in MUSIC_PROFILES:
            raise ValueError('head_timing_dim is only supported by music_context_v2')
        expected_head_dim=head_timing_dim_for_policy(self.head_timing_policy)
        if self.head_timing_dim and self.head_timing_dim != expected_head_dim:
            raise ValueError(f'head_timing_dim must equal {expected_head_dim} for {self.head_timing_policy}')
        if self.architecture_profile==GROUNDED_PROFILE:
            if self.head_timing_dim!=expected_head_dim:raise ValueError('audio_grounded_v3 requires head timing')
            if self.lens_width<8 or self.lens_chunk_size<1:raise ValueError('Invalid audio lens size')
            if not math.isfinite(self.history_residual_bound) or not 0<self.history_residual_bound<=4:
                raise ValueError('history_residual_bound must be in (0,4]')
            if any(not math.isfinite(x) or not 0<=x<=.9 for x in (self.history_dropout,self.ledger_dropout)):
                raise ValueError('Grouped dropout must be in [0,.9]')

    def to_dict(self): return asdict(self)


def build_model(config,atom_catalog=None):
    validate_supported_context_policies(config)
    if config.architecture_profile==GROUNDED_PROFILE:
        from .grounded_model import AudioGroundedComposer
        if atom_catalog is None or len(atom_catalog)+1!=config.atom_vocab_size:
            raise ValueError('audio_grounded_v3 requires its exact atom catalog')
        return AudioGroundedComposer(config,atom_catalog)
    return ChartComposer(config)


class ChartComposer(nn.Module):
    architecture = 'chart_composer_v1'

    def __init__(self, config: ComposerConfig):
        # AudioGroundedComposer temporarily uses a music-context backbone,
        # then installs the grounded implementations of these policies.
        if type(self) is ChartComposer:validate_supported_context_policies(config)
        super().__init__(); self.config=config
        width=config.d_model
        self.audio_projection=nn.Linear(config.audio_dim,width)
        self.condition_projection=nn.Linear(config.condition_dim,width)
        self.action_projection=nn.Linear(config.action_dim,width)
        self.time_projection=nn.Linear(config.candidate_feature_dim-config.audio_dim,width)
        self.event_embedding=nn.Embedding(config.atom_vocab_size,width)
        self.position_embedding=nn.Embedding(config.history_events+4,width)
        self.history_feature_projection=nn.Linear(config.history_feature_dim,width)
        layer=nn.TransformerEncoderLayer(width,config.nhead,config.dim_feedforward,
                                         dropout=config.dropout,activation='gelu',batch_first=True,norm_first=True)
        self.audio_encoder=nn.TransformerEncoder(layer,config.layers,enable_nested_tensor=False)
        self.history_encoder=nn.TransformerEncoder(layer,config.layers,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(width)
        self.pointer=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,1))
        self.action_head=nn.Linear(width,config.atom_vocab_size if config.architecture_profile=='music_context_v2'
                                   else config.action_slots*config.atom_vocab_size)
        self.future_head=nn.Linear(width,8)
        self.eos_head=nn.Linear(width,1)
        if config.head_timing_dim:
            self.head_timing_projection=nn.Linear(HEAD_TIMING_DIM if config.head_timing_policy=='separate_pads_v2'
                                                   else config.head_timing_dim,width)
            nn.init.zeros_(self.head_timing_projection.weight)
            nn.init.zeros_(self.head_timing_projection.bias)
        if config.architecture_profile in WORKLOAD_PROFILES:
            self.workload_head = nn.Sequential(nn.Linear(CONTEXT_DIM + config.condition_dim, width // 2),
                nn.GELU(), nn.Linear(width // 2, 5))
            self.workload_projection = nn.Linear(5, width)
            self.ledger_projection = nn.Linear(6, width)
            self.candidate_fusion = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
            self.slot_fusion = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
        if config.architecture_profile == 'music_context_v2':
            self.slot_embedding = nn.Embedding(config.action_slots, width)
            self.history_attention = nn.MultiheadAttention(width, config.nhead, dropout=config.dropout, batch_first=True)
            self.history_gate = nn.Parameter(torch.tensor(.1))
            self.interval_projection = nn.Linear(INTERVAL_DIM, width)
            self.trailing_head = nn.Sequential(nn.Linear(TRAILING_CONTEXT_DIM + config.condition_dim, width // 2),
                                              nn.GELU(), nn.Linear(width // 2, 4))
            self.discrepancy_projection = nn.Linear(4, width)
            # New evidence paths start neutral for migration/finetune.
            nn.init.zeros_(self.interval_projection.weight); nn.init.zeros_(self.interval_projection.bias)
            nn.init.zeros_(self.discrepancy_projection.weight); nn.init.zeros_(self.discrepancy_projection.bias)

    def predict_workload(self, audio_context, condition):
        controls = condition[:, None].expand(-1, audio_context.size(1), -1)
        inputs = torch.cat((audio_context.float(), controls.float()), -1)
        # Keep this small auxiliary head in FP32 under CUDA autocast.  Its
        # output is later cast by the surrounding model, while its regression
        # target and gradients remain well-conditioned in BF16 training.
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            raw = self.workload_head(inputs)
            result = torch.cat((torch.nn.functional.softplus(raw[..., :2]), raw[..., 2:].sigmoid()), -1)
        return result

    def forward(self, candidate_features, history, condition, action_state=None, future_state=None,
                history_features=None, candidate_mask=None, history_padding_mask=None,
                candidate_indices=None, decode_actions=True, workload_audio=None, ledger=None,
                action_targets=None, curve_strength=1., interval_audio=None, trailing_audio=None,
                workload_override=None, symbolic_keep=None, audio_auxiliary=True,
                head_timing=None):
        batch=candidate_features.size(0)
        acoustic=candidate_features[...,:self.config.audio_dim]
        candidate_hidden=self.audio_projection(acoustic)+self.time_projection(candidate_features[...,self.config.audio_dim:])
        candidate_hidden=self.audio_encoder(candidate_hidden,
            src_key_padding_mask=candidate_mask)
        valid=~candidate_mask if candidate_mask is not None else torch.ones(candidate_hidden.shape[:2],dtype=torch.bool,device=candidate_hidden.device)
        audio_summary=(candidate_hidden*valid[...,None]).sum(1)/valid.sum(1).clamp_min(1)[:,None]
        memory = None
        safe_padding = None
        if history is None or history.size(1)==0:
            history_summary=audio_summary
        else:
            if history.ndim == 2:
                history = history.unsqueeze(-1)
            positions=torch.arange(history.size(1),device=history.device)[None]
            if history_padding_mask is None:
                lengths=torch.full((history.size(0),),history.size(1),device=history.device,dtype=torch.long)
            else:
                lengths=(~history_padding_mask).sum(1)
            positions=(lengths[:,None]-1-positions).clamp_min(0).clamp_max(self.config.history_events+3)
            event_embeddings=self.event_embedding(history)
            event_embeddings=event_embeddings.masked_fill(history.eq(0).unsqueeze(-1),0.)
            history_hidden=event_embeddings.sum(dim=2)+self.position_embedding(positions)
            if history_features is not None:
                if self.training and self.config.architecture_profile=='audio_workload_v1' and history_features.size(1)>2:
                    history_features=history_features.clone()
                    hide=torch.rand(history_features.shape[:2],device=history_features.device)<.15
                    hide &= positions.ge(2)
                    history_features[...,:2]=history_features[...,:2].masked_fill(hide[...,None],0.)
                history_hidden=history_hidden+self.history_feature_projection(history_features)
            safe_padding=history_padding_mask
            if safe_padding is not None:
                empty=(~safe_padding).sum(1)==0
                safe_padding=safe_padding.clone()
                history_hidden=history_hidden.clone()
                safe_padding[:,0]=safe_padding[:,0] & ~empty
                history_hidden[:,0]=history_hidden[:,0].masked_fill(empty[:,None],0.)
            history_hidden=self.history_encoder(history_hidden,
                src_key_padding_mask=safe_padding)
            memory = history_hidden
            if safe_padding is None:
                lengths=torch.full((history.size(0),),history.size(1)-1,device=history.device,dtype=torch.long)
            else:
                lengths=(~safe_padding).sum(1).clamp_min(1)-1
            gathered=history_hidden[torch.arange(history.size(0),device=history.device),lengths]
            if safe_padding is not None:
                valid_rows=(~history_padding_mask).any(1)
                history_summary=torch.where(valid_rows[:,None],gathered,audio_summary)
            else:
                history_summary=gathered
        music_v2 = self.config.architecture_profile == 'music_context_v2'
        if music_v2:
            # Drop all chart-derived representations together, preserving the
            # actual committed history/state and the authored target alignment.
            # This is modality dropout, not invented corrupted-prefix labels.
            if symbolic_keep is None:
                symbolic_keep = ((torch.rand(batch, 1, device=candidate_hidden.device) >= self.config.symbolic_dropout)
                    if self.training else candidate_hidden.new_ones(batch, 1))
            symbolic_keep = symbolic_keep.to(candidate_hidden)
            if history is None or history.size(1)==0:
                history_summary = torch.zeros_like(audio_summary)
            elif history_padding_mask is not None:
                history_summary = history_summary * (~history_padding_mask).any(1)[:,None]
            history_summary = history_summary * symbolic_keep
        context=self.norm(history_summary+audio_summary+self.condition_projection(condition))
        if action_state is not None:
            projected=self.action_projection(action_state)
            context=context+projected*(symbolic_keep if music_v2 else 1.)
        # Future intent is an auxiliary prediction.  It is deliberately not
        # fed back into the current decision: the old teacher-forced path used
        # a target-derived previous state that included the current event,
        # while free generation used the model's own prediction.
        candidate_context=context[:,None,:]+candidate_hidden
        if self.config.head_timing_dim:
            if head_timing is None:
                raise ValueError('head_timing is required when head_timing_dim is enabled')
            if head_timing.ndim != 3 or head_timing.shape[:2] != candidate_hidden.shape[:2] \
                    or head_timing.size(-1) != self.config.head_timing_dim:
                raise ValueError('head_timing shape does not match the Composer contract')
            timing_context=self.head_timing_projection(head_timing.to(candidate_hidden))
            if music_v2:
                timing_context=timing_context*symbolic_keep[:,None]
            candidate_context=candidate_context+timing_context
        elif head_timing is not None:
            raise ValueError('head_timing was supplied to a checkpoint without head_timing_dim')
        workload = None
        if self.config.architecture_profile in WORKLOAD_PROFILES:
            if workload_audio is None:
                workload_audio = candidate_features.new_zeros(batch, candidate_features.size(1), CONTEXT_DIM)
            workload = self.predict_workload(workload_audio, condition)
            decision_workload = workload if workload_override is None else workload_override.to(workload)
            # The explicit chart-calibrated head gets only its own regression
            # gradient; token CE cannot turn it into an opaque hidden channel.
            candidate_context = candidate_context + float(curve_strength) * self.workload_projection(
                decision_workload[:, :candidate_features.size(1)].detach())
            if ledger is not None:
                candidate_context = candidate_context + self.ledger_projection(ledger.to(candidate_context)) * (symbolic_keep[:,None] if music_v2 else 1.)
            if music_v2:
                if interval_audio is None:
                    interval_audio = candidate_features.new_zeros(batch, candidate_features.size(1), INTERVAL_DIM)
                interval = self.interval_projection(interval_audio.to(candidate_hidden))
                candidate_context = candidate_context + interval
                if audio_auxiliary:
                    audio_context = self.norm(audio_summary + self.condition_projection(condition))[:,None] + candidate_hidden + interval
                    audio_context = audio_context + float(curve_strength) * self.workload_projection(
                        decision_workload[:, :candidate_features.size(1)].detach())
                if memory is not None:
                    attended, _ = self.history_attention(candidate_hidden, memory, memory,
                        key_padding_mask=safe_padding, need_weights=False)
                    has_history = ((~history_padding_mask).any(1)[:,None,None]
                                   if history_padding_mask is not None else 1.)
                    candidate_context = candidate_context + self.history_gate.tanh() * attended * symbolic_keep[:,None] * has_history
                if trailing_audio is not None:
                    trailing = self.predict_trailing(trailing_audio, condition)
                    if ledger is not None:
                        difference = trailing[:, :candidate_features.size(1)].detach() - ledger[...,:4].float().clamp_min(0).log1p()
                        candidate_context = candidate_context + float(curve_strength) * self.discrepancy_projection(difference.to(candidate_context)) * symbolic_keep[:,None]
            candidate_context = self.candidate_fusion(candidate_context)
        pointer=self.pointer(candidate_context).squeeze(-1)
        if candidate_mask is not None:pointer=pointer.masked_fill(candidate_mask,-1e4)
        result={'candidate_logits':pointer,'candidate_context':candidate_context}
        if workload is not None: result['workload'] = workload
        if music_v2:
            if audio_auxiliary:
                audio_logits = self.pointer(self.candidate_fusion(audio_context)).squeeze(-1)
                result['audio_candidate_logits'] = audio_logits.masked_fill(candidate_mask,-1e4) if candidate_mask is not None else audio_logits
            if trailing_audio is not None: result['trailing_workload'] = trailing
        if decode_actions:
            result.update(self.decode_candidate(candidate_context,candidate_indices,action_targets=action_targets))
        return result

    def predict_trailing(self, audio_context, condition):
        controls = condition[:,None].expand(-1, audio_context.size(1), -1)
        with torch.autocast(device_type=audio_context.device.type, enabled=False):
            return torch.nn.functional.softplus(self.trailing_head(torch.cat((audio_context.float(), controls.float()), -1)))

    def decode_candidate(self, candidate_context, candidate_indices=None, *, action_targets=None, project_actions=True, audio_context=None):
        """Project only supervised/chosen candidates without changing the heads.

        Pointer scores still use every candidate. The large action vocabulary
        needs one projection per example during training and free generation,
        rather than one projection for each of the 96 proposed positions.
        """
        if candidate_indices is not None:
            rows=torch.arange(candidate_context.size(0),device=candidate_context.device)
            candidate_context=candidate_context[rows,candidate_indices]
        result={'future':self.future_head(candidate_context),
                'eos_logits':self.eos_head(candidate_context).squeeze(-1),'action_context':candidate_context}
        if not project_actions: return result
        action_logits = None
        if self.config.architecture_profile in WORKLOAD_PROFILES and action_targets is not None:
            # Training conditions each slot on the preceding authored atoms;
            # inference uses exactly the atoms already selected at this beat.
            embedded = self.event_embedding(action_targets) * action_targets.ne(0).unsqueeze(-1)
            prefix = torch.cumsum(embedded, dim=-2) - embedded
            contexts = candidate_context.unsqueeze(-2) + prefix
            if self.config.architecture_profile == 'music_context_v2':
                contexts = contexts + self.slot_embedding.weight
                action_logits = self.action_head(self.slot_fusion(contexts))
            else:
                contexts = self.slot_fusion(contexts)
                weights = self.action_head.weight.view(self.config.action_slots,self.config.atom_vocab_size,-1)
                action_logits = torch.einsum('...sd,svd->...sv', contexts, weights)
                action_logits = action_logits + self.action_head.bias.view(self.config.action_slots,-1)
        else:
            if self.config.architecture_profile == 'music_context_v2':
                # Diagnostic marginal with an empty within-event prefix.
                contexts = self.slot_fusion(candidate_context.unsqueeze(-2) + self.slot_embedding.weight)
                action_logits = self.action_head(contexts)
            else:
                action_logits=self.action_head(candidate_context).reshape(
                    *candidate_context.shape[:-1],self.config.action_slots,self.config.atom_vocab_size)
        result['action_logits']=action_logits
        return result

    def action_slot(self, context, slot, selected_ids, *, audio_context=None, history_strength=1.):
        if self.config.architecture_profile in WORKLOAD_PROFILES:
            if selected_ids:
                ids = torch.as_tensor(selected_ids, dtype=torch.long, device=context.device)
                context = context + self.event_embedding(ids).sum(0)
            if self.config.architecture_profile == 'music_context_v2':
                context = context + self.slot_embedding.weight[slot]
            context = self.slot_fusion(context)
        if self.config.architecture_profile == 'music_context_v2':
            return self.action_head(context)
        start = slot * self.config.atom_vocab_size
        return torch.nn.functional.linear(context, self.action_head.weight[start:start+self.config.atom_vocab_size],
            self.action_head.bias[start:start+self.config.atom_vocab_size])
