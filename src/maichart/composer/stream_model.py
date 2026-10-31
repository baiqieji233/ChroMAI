"""Causal event-set stream: choose time, then a complete unordered event.

The audio cache belongs to a caller/forward pass, never to model training state.
Full songs use linear local encoding and at most 128 summary attention tokens.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import math
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from .events import event_metadata,head_pad_indices_for_policy
from .stream_continuation import continuation_dim
from .stream_timing import AbsoluteTimingHead, absolute_target_indices, grouped_logsumexp, TIMING_POLICIES, TIMING_GROUP_POLICIES

PAD, BOS, NOTE, WAIT, EOS = range(5)


def to_device_batch(batch, device, *, non_blocking=False):
    """Move tensors including CPU-prepared sparse detail groups, preserving IDs."""
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, dict):
        return {key: to_device_batch(value, device, non_blocking=non_blocking) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(to_device_batch(value, device, non_blocking=non_blocking) for value in batch)
    if isinstance(batch, list):
        return [to_device_batch(value, device, non_blocking=non_blocking) for value in batch]
    return batch


@dataclass
class StreamComposerConfig:
    architecture_profile: str = 'event_set_stream_v1'
    d_model: int = 256
    nhead: int = 8
    layers: int = 4
    dim_feedforward: int = 768
    dropout: float = .1
    audio_dim: int = 238
    audio_grid_seconds: float = .01
    condition_dim: int = 16
    history_events: int = 64
    history_window_policy: str = 'block_aligned_v2'
    continuation_policy: str = 'pattern_runs_v2'
    sensor_load_policy: str = 'separate_touch_v1'
    lane_load_policy: str = 'direct_buttons_v1'
    gap_continuation_policy: str = 'probability_v2'
    history_phase_policy: str = 'orbit_average_v1'
    phase_output_policy: str = 'orbit_reference_v1'
    clock_policy: str = 'musical_v2'
    timing_audio_policy: str = 'delta_local_v2'
    timing_output_policy: str = 'absolute_grid_v1'
    timing_phase_catalog: tuple[str, ...] = ()
    timing_grid_audio_policy: str = 'candidate_pool_v1'
    timing_group_policy: str = 'phase_gap_v2'
    detail_policy: str = 'phase_autoregressive_v2'
    literal_output_policy: str = 'type_blocks_v1'
    event_family_policy: str = 'audio_prior_transition_v3'
    event_family_audio_weight: float = .25
    window_balance_policy: str = 'sqrt_window_v2'
    button_load_policy: str = 'target_fast_v2'
    plan_kind_policy: str = 'rate_integral_v2'
    motion_continuation_policy: str = 'context_motion_v3'
    recovery_policy: str = 'mixed_exit_v2'
    prefix_augmentation_rate: float = .15
    prefix_augmentation_input_policy: str = 'phase_recovery_v4'
    attention_policy: str = 'bf16_dropout_math_v1'
    decoder_policy: str = 'prenorm_independent_v2'
    transition_policy: str = 'off'
    rollout_steps: int = 16
    audio_summary_tokens: int = 128
    state_dim: int = 50
    max_event_atoms: int = 32
    delta_beats: tuple[str, ...] = ('0', '1/4', '1/2', '1', '2')
    plan_weight: float = .1
    plan_target_units: str = 'rates_per_second_v2'
    plan_loss_normalization: str = 'per_second_v2'
    plan_blank_window_seconds: float = .5
    recovery_weight: float = .1
    transition_exit_weight: float = .5
    phrase_context_policy: str = 'off'
    phrase_loss_weight: float = .05
    duration_audio_policy: str = 'off'
    physical_policy: str = 'legacy_report_v1'

    def __post_init__(self):
        if self.duration_audio_policy not in ('off', 'interval_v1'):
            raise ValueError('Unsupported duration audio policy')
        if self.physical_policy not in ('legacy_report_v1', 'runtime_hard_v2'):
            raise ValueError('Unsupported physical policy')
        if self.phrase_context_policy not in ('off', 'causal_phrase_v1'):
            raise ValueError('Unsupported phrase context policy')
        if not math.isfinite(self.phrase_loss_weight) or self.phrase_loss_weight < 0:
            raise ValueError('Phrase loss weight must be finite and nonnegative')
        self.delta_beats = tuple(str(x) for x in self.delta_beats)
        self.timing_phase_catalog = tuple(str(x) for x in self.timing_phase_catalog)
        if self.timing_output_policy not in TIMING_POLICIES:raise ValueError('Unsupported timing output policy')
        if self.timing_grid_audio_policy not in ('off','candidate_pool_v1','query_pool_v2','query_pool_v3'):raise ValueError('Unsupported timing grid audio policy')
        if self.timing_group_policy not in TIMING_GROUP_POLICIES:raise ValueError('Unsupported timing group policy')
        if not math.isfinite(self.audio_grid_seconds) or not .005<=self.audio_grid_seconds<=.05:
            raise ValueError('Audio grid seconds must be positive and finite in [0.005, 0.05]')
        if self.history_phase_policy not in ('canonical_v1','orbit_average_v1'):
            raise ValueError('Unsupported history phase policy')
        if self.phase_output_policy not in ('legacy_canonical_v1','orbit_reference_v1'):
            raise ValueError('Unsupported phase output policy')
        for name,allowed in (('clock_policy',{'legacy_v1','musical_v2'}),
            ('timing_audio_policy',{'legacy_v1','delta_local_v2'}),
            ('detail_policy',{'independent_v1','phase_autoregressive_v2'}),
            ('literal_output_policy',{'flat_family_v1','type_blocks_v1'}),
            ('event_family_policy',{'legacy_v1','semantic_mass_v1','audio_semantic_v2','audio_prior_transition_v3'}),
            ('window_balance_policy',{'sqrt_window_v2','equal_window_v1'}),
            ('button_load_policy',{'prefix_v1','target_fast_v2'}),
            ('history_window_policy',{'rolling_legacy_v1','block_aligned_v2'}),
            ('continuation_policy',{'off','typed_runs_v1','pattern_runs_v2'}),
            ('sensor_load_policy',{'button_only_v1','separate_touch_v1'}),
            ('lane_load_policy',{'off','direct_buttons_v1'}),
            ('gap_continuation_policy',{'additive_v1','probability_v2'}),
            ('plan_kind_policy',{'off','rate_integral_v2'}),
            ('recovery_policy',{'persistent_v1','mixed_exit_v2','greedy_mixed_v3','greedy_mixed_v4'}),
            ('prefix_augmentation_input_policy',{'musical_loop_recovery_v8','musical_loop_recovery_v7','musical_loop_recovery_v6','musical_loop_recovery_v5','phase_recovery_v4','causal_window_config_time_first_decision_v3'}),
            ('motion_continuation_policy',{'off','motion_continuation_v1','motion_distinct_v2','context_motion_v3'})):
            if getattr(self,name) not in allowed:raise ValueError(f'Unsupported {name}')
        if self.attention_policy not in ('auto_v1','bf16_dropout_math_v1'):raise ValueError('Unsupported attention policy')
        if self.decoder_policy not in ('postnorm_v1','prenorm_independent_v2'):raise ValueError('Unsupported decoder policy')
        if self.transition_policy not in ('off','music_exit_v1','music_exit_v2','music_exit_v3'):raise ValueError('Unsupported transition policy')
        if self.transition_policy in ('music_exit_v1','music_exit_v2','music_exit_v3') and self.event_family_policy=='legacy_v1':
            raise ValueError(f'{self.transition_policy} requires semantic event families')
        if not math.isfinite(self.event_family_audio_weight) or not 0<=self.event_family_audio_weight<=1:
            raise ValueError('Event family audio weight must be finite in [0, 1]')
        if self.plan_target_units not in ('counts_per_audio_cell','rates_per_second_v2'):
            raise ValueError('Unsupported stream plan target units')
        if self.plan_loss_normalization not in ('cell_mean_v1','per_second_v2'):
            raise ValueError('Unsupported stream plan loss normalization')
        if not math.isfinite(self.plan_blank_window_seconds) or self.plan_blank_window_seconds<0:
            raise ValueError('Plan blank window must be finite and nonnegative')
        if not math.isfinite(self.transition_exit_weight) or self.transition_exit_weight<0:
            raise ValueError('Transition exit weight must be finite and nonnegative')
        if not math.isfinite(self.prefix_augmentation_rate) or not 0<=self.prefix_augmentation_rate<=1:
            raise ValueError('Prefix augmentation rate must be in [0, 1]')
        if self.architecture_profile != 'event_set_stream_v1':
            raise ValueError('Unsupported stream architecture')
        if self.d_model % self.nhead or not 1 <= self.audio_summary_tokens <= 128:
            raise ValueError('Invalid stream width/summary size')
        if not self.delta_beats or Fraction(self.delta_beats[0]) != 0:
            raise ValueError('Delta class zero must mean zero beats')
        if any(Fraction(x) < 0 for x in self.delta_beats[1:]):
            raise ValueError('Delta values must be nonnegative')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        # Initial scratch checkpoints encoded audio at 50 ms before this field.
        return cls(**{'audio_grid_seconds':.05,'history_phase_policy':'canonical_v1',
                      'phase_output_policy':'legacy_canonical_v1','clock_policy':'legacy_v1',
                      'timing_audio_policy':'legacy_v1','detail_policy':'independent_v1','literal_output_policy':'flat_family_v1','event_family_policy':'legacy_v1',
                      'timing_output_policy':'relative_delta_v1','timing_phase_catalog':(),
                      'timing_grid_audio_policy':'off',
                      'timing_group_policy':'phase_only_v1',
                      'event_family_audio_weight':0.,'window_balance_policy':'equal_window_v1',
                      'button_load_policy':'prefix_v1','plan_kind_policy':'off',
                      'motion_continuation_policy':'off',
                      'history_window_policy':'rolling_legacy_v1','continuation_policy':'off',
                      'sensor_load_policy':'button_only_v1','gap_continuation_policy':'additive_v1',
                      'lane_load_policy':'off',
                      'recovery_policy':'persistent_v1','prefix_augmentation_rate':0.,
                      'prefix_augmentation_input_policy':'causal_window_config_time_first_decision_v3',
                      'transition_exit_weight':0.,
                      'attention_policy':'auto_v1',
                      'decoder_policy':'postnorm_v1','transition_policy':'off',
                      'plan_target_units':'counts_per_audio_cell','plan_blank_window_seconds':0.,
                      'plan_loss_normalization':'cell_mean_v1',**values})


class StreamComposer(nn.Module):
    def __init__(self, config: StreamComposerConfig, codec):
        super().__init__()
        self.config, self.codec = config, codec
        d = config.d_model
        self._audio_tracking_strength = 0.
        if config.duration_audio_policy == 'interval_v1':
            from .stream_duration import DurationAudioScorer
            self.duration_audio = DurationAudioScorer(d, codec)
        if config.phrase_context_policy == 'causal_phrase_v1':
            from .stream_phrase_model import CausalPhraseContext
            self.phrase_context = CausalPhraseContext(d, config.condition_dim)
        self.audio_input = nn.Sequential(nn.Linear(config.audio_dim, d), nn.GELU(), nn.LayerNorm(d))
        self.audio_local = nn.Conv1d(d, d, 5, padding=2, groups=d)
        self.audio_clock = nn.Linear(14 if config.clock_policy=='musical_v2' else 4, d)
        self.global_audio = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d, config.nhead, config.dim_feedforward, config.dropout, batch_first=True), 1,
            enable_nested_tensor=False)
        self.kind_embedding = nn.Embedding(5, d, padding_idx=PAD)
        self.shape_embedding = nn.Embedding(codec.num_shapes, d)
        self.phase_embedding = nn.Embedding(8, d)
        self.phase_change = nn.Embedding(8, d)
        self.delta_embedding = nn.Embedding(len(config.delta_beats), d)
        self.register_buffer('delta_values', torch.tensor([float(Fraction(x)) for x in config.delta_beats],dtype=torch.float64), persistent=False)
        self.atom_embedding = nn.Embedding(codec.num_atoms, d, padding_idx=0)
        self.condition = nn.Linear(config.condition_dim, d)
        if config.transition_policy in ('music_exit_v1','music_exit_v2','music_exit_v3'):
            clock_dim=14 if config.clock_policy=='musical_v2' else 4
            inputs=32*9+clock_dim+2+config.condition_dim+continuation_dim(config.continuation_policy)+38
            self.music_exit_head=nn.Sequential(nn.Linear(inputs,32),nn.GELU(),nn.Linear(32,3))
            nn.init.zeros_(self.music_exit_head[2].weight);nn.init.zeros_(self.music_exit_head[2].bias)
        self.cursor_clock = nn.Linear(18 if config.clock_policy=='musical_v2' else 8, d)
        self.state_projection = nn.Linear(config.state_dim, d)
        self.load_projection = nn.Linear(32 if config.button_load_policy=='target_fast_v2' else 16, d)
        if config.lane_load_policy=='direct_buttons_v1':
            self.lane_load_projection=nn.Linear(4,1,bias=False)
            nn.init.zeros_(self.lane_load_projection.weight)
        if config.sensor_load_policy=='separate_touch_v1':
            self.touch_load_projection=nn.Linear(132,d)
            nn.init.zeros_(self.touch_load_projection.weight);nn.init.zeros_(self.touch_load_projection.bias)
        if config.continuation_policy!='off':
            features=continuation_dim(config.continuation_policy)
            self.continuation_projection=nn.Linear(features,d)
            nn.init.zeros_(self.continuation_projection.weight);nn.init.zeros_(self.continuation_projection.bias)
            if config.continuation_policy=='pattern_runs_v2':
                self.action_continuation_projection=nn.Linear(features,d,bias=False)
                self.kind_continuation_projection=nn.Linear(features,5,bias=False)
                self.gap_continuation_projection=nn.Linear(features,1,bias=False)
                for module in (self.action_continuation_projection,self.kind_continuation_projection,self.gap_continuation_projection):
                    nn.init.zeros_(module.weight)
        offsets=[0,.25,.5,.75,1,1.25,1.5,1.75,2,3,4,6,8] if config.timing_audio_policy=='delta_local_v2' else list(range(9))
        self.register_buffer('near_offsets',torch.tensor(offsets),persistent=False)
        self.near_audio = nn.Linear(len(offsets) * d, d)
        if config.timing_audio_policy=='delta_local_v2':
            self.delta_audio_key=nn.Linear(d,32)
            self.delta_query=nn.Linear(d,32)
        prenorm=config.decoder_policy=='prenorm_independent_v2'
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d, config.nhead, config.dim_feedforward, config.dropout, batch_first=True,
            norm_first=prenorm), config.layers, norm=nn.LayerNorm(d) if prenorm else None)
        if prenorm:
            # Torch clones the same initial layer. Give a fresh model genuinely
            # independent matrices instead of repeatedly applying one nearly
            # identical operator. Stage migration subsequently loads every
            # existing layer tensor by its unchanged parameter name.
            for layer in self.decoder.layers:
                for name,parameter in layer.named_parameters():
                    if parameter.ndim>1:nn.init.xavier_uniform_(parameter)
                    elif name.endswith('weight') and 'norm' in name:nn.init.ones_(parameter)
                    else:nn.init.zeros_(parameter)
        self.kind_head = nn.Linear(d, 5)
        self.delta_head = nn.Linear(d, len(config.delta_beats))
        if config.timing_output_policy=='absolute_grid_v1':
            self.timing_grid=AbsoluteTimingHead(d,config.delta_beats,config.timing_phase_catalog,
                                               audio_policy=config.timing_grid_audio_policy,group_policy=config.timing_group_policy)
            self.history_gap_projection=nn.Linear(13,d,bias=False)
            nn.init.zeros_(self.history_gap_projection.weight)
        self.action_fusion = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.LayerNorm(d))
        self.shape_head = nn.Linear(d, codec.num_shapes)
        if config.event_family_policy!='legacy_v1':
            from .stream_actions import semantic_catalog
            atom_families,shape_families=semantic_catalog(codec)
            self._semantic_atom_families=atom_families.tolist()
            self._semantic_shape_families=shape_families.tolist()
            self.register_buffer('semantic_atom_families',atom_families,persistent=False)
            self.register_buffer('semantic_shape_families',shape_families,persistent=False)
            routes=torch.cat((shape_families[1:],torch.arange(1,32)))
            self.register_buffer('semantic_route_families',routes,persistent=False)
            available=0
            for value in atom_families.tolist():available|=value
            supported=torch.tensor([mask & ~available == 0 and mask.bit_count()<=config.max_event_atoms
                                    for mask in range(1,32)],dtype=torch.bool)
            self.register_buffer('semantic_family_supported',supported,persistent=False)
            self.event_family_head=nn.Linear(d,31)
            self.literal_route_head=nn.Linear(d,31)
            self.literal_family_embedding=nn.Embedding(32,d)
            for module in (self.event_family_head,self.literal_route_head):
                nn.init.zeros_(module.weight);nn.init.zeros_(module.bias)
            nn.init.zeros_(self.literal_family_embedding.weight)
            if config.event_family_policy in ('audio_semantic_v2','audio_prior_transition_v3'):
                # Current music is the nonzero identity basis. New projections
                # start at zero, without creating a double-zero gradient path.
                self.family_audio_projection=nn.Linear(5*d,d,bias=False)
                self.audio_event_family_head=nn.Linear(d,31)
                modules=[self.family_audio_projection,self.audio_event_family_head]
                if config.event_family_policy=='audio_semantic_v2':
                    self.family_history_projection=nn.Linear(30,d,bias=False)
                    modules.append(self.family_history_projection)
                else:
                    self.family_continuity_head=nn.Linear(2*d+1,1)
                    modules.append(self.family_continuity_head)
                    clock_width=14 if config.clock_policy=='musical_v2' else 4
                    self.family_rhythm_projection=nn.Linear(clock_width+5,d,bias=False)
                    modules.append(self.family_rhythm_projection)
                for module in modules:
                    nn.init.zeros_(module.weight)
                    if getattr(module,'bias',None) is not None:nn.init.zeros_(module.bias)
        self.phase_head = nn.Linear(d, 8)
        if config.phase_output_policy=='orbit_reference_v1':
            self.reference_phase_head=nn.Linear(d,8)
        self.absolute_phase_bias = nn.Parameter(torch.zeros(8))
        if config.motion_continuation_policy!='off':
            self.motion_head=nn.Linear(d,3)
            nn.init.zeros_(self.motion_head.weight);nn.init.zeros_(self.motion_head.bias)
        if config.motion_continuation_policy=='context_motion_v3':
            # Geometry is evidence for the main predictor. It no longer forms
            # a second unconstrained logit reward for KEEP/REVERSE/STAY.
            self.motion_context_projection=nn.Linear(24,d,bias=False)
            nn.init.zeros_(self.motion_context_projection.weight)
        if config.motion_continuation_policy!='off' or config.gap_continuation_policy=='probability_v2':
            self.keep_gap_head=nn.Linear(d,1)
            nn.init.zeros_(self.keep_gap_head.weight);nn.init.zeros_(self.keep_gap_head.bias)
        if config.gap_continuation_policy=='probability_v2':
            self.keep_gap_audio=nn.Linear(32,1)
            nn.init.zeros_(self.keep_gap_audio.weight);nn.init.zeros_(self.keep_gap_audio.bias)
        if config.detail_policy=='independent_v1':
            self.detail_heads = nn.ModuleList(nn.ModuleList(nn.Linear(d, size) for size in slots)
                                              for slots in codec.detail_sizes)
        else:
            self.detail_embeddings=nn.ModuleList(nn.ModuleList(nn.Embedding(size,32) for size in slots)
                                                  for slots in codec.detail_sizes)
            self.detail_bias=nn.ModuleList(nn.ParameterList(nn.Parameter(torch.zeros(size)) for size in slots)
                                           for slots in codec.detail_sizes)
            self.detail_input=nn.Linear(32,d,bias=False)
            self.detail_output=nn.Linear(d,32)
            self.detail_cell=nn.GRUCell(d,d)
            self.detail_start=nn.Parameter(torch.zeros(d))
        self.literal_cell = nn.GRU(d, d, batch_first=True)
        self.literal_start = nn.Parameter(torch.zeros(d))
        self.literal_head = nn.Linear(d, codec.num_atoms)
        if config.literal_output_policy=='type_blocks_v1' and config.event_family_policy!='legacy_v1':
            self.literal_stop_head=nn.Linear(d,5)
            self.literal_type_embedding=nn.Embedding(5,d)
            nn.init.zeros_(self.literal_stop_head.weight);nn.init.zeros_(self.literal_stop_head.bias)
            nn.init.zeros_(self.literal_type_embedding.weight)
            atom_types=self.semantic_atom_families.tolist()
            if any(value not in (0,1,2,4,8,16) for value in atom_types):
                raise ValueError('Type-block literal atoms must have exactly one parsed note type')
            local=torch.zeros(codec.num_atoms,dtype=torch.long)
            self._literal_type_ids=[]
            for type_id,bit in enumerate((1,2,4,8,16)):
                ids=torch.tensor([i for i,value in enumerate(atom_types) if value==bit],dtype=torch.long)
                self.register_buffer(f'literal_type_atoms_{type_id}',ids,persistent=False)
                local[ids]=torch.arange(ids.numel())
                self._literal_type_ids.append(tuple(ids.tolist()))
            self.register_buffer('literal_type_local',local,persistent=False)
        self.plan_head = nn.Linear(d, 3)
        if config.plan_kind_policy=='rate_integral_v2':self.plan_kind_features=nn.Linear(3,5)
        lanes = torch.zeros(codec.num_atoms, 8)
        touch_lanes = torch.zeros(codec.num_atoms,33)
        literal_bits=[0]*codec.num_atoms
        for i, text in enumerate(codec.atom_catalog, 1):
            pads=head_pad_indices_for_policy(event_metadata(text),'separate_pads_v2')
            literal_bits[i]=sum(1<<pad for pad in pads)
            for pad in pads:
                if pad>=8:touch_lanes[i,pad-8]=1.
            for head in event_metadata(text).heads:
                if str(head) in tuple(str(x) for x in range(1, 9)):
                    lanes[i, int(head) - 1] = 1.
        self.register_buffer('atom_lanes', lanes, persistent=False)
        self.register_buffer('atom_touch',touch_lanes,persistent=False)
        shape_lanes=torch.zeros(codec.num_shapes,8,8)
        for shape,slots in enumerate(codec.shape_catalog):
            if shape==0:continue
            for _,head in slots:
                for phase in range(8):shape_lanes[shape,phase,(head+phase)%8]=1.
        self.register_buffer('shape_button_lanes',shape_lanes,persistent=False)
        # A motion decision concerns typed head geometry, not modifier/detail
        # spelling. Close KEEP over each shape's geometric rotation orbit so
        # equivalent physical configurations cannot land in opposite groups.
        stabilizers=torch.zeros(codec.num_shapes,8,dtype=torch.bool)
        stabilizers[:,0]=True
        for shape,slots in enumerate(codec.shape_catalog):
            if shape==0:continue
            original=tuple(sorted(slots))
            for shift in range(1,8):
                stabilizers[shape,shift]=tuple(sorted((kind,(head+shift)%8) for kind,head in slots))==original
        self.register_buffer('shape_motion_stabilizer',stabilizers,persistent=False)
        angles=torch.arange(8)
        self.register_buffer('motion_rotation_differences',(angles[:,None]-angles[None,:])%8,persistent=False)
        self._literal_bits=literal_bits
        self.register_buffer('literal_head_bits',torch.tensor(literal_bits,dtype=torch.long),persistent=False)

    def _clock(self,seconds, beats):
        if self.config.clock_policy=='musical_v2':
            periods=beats.new_tensor([.25,.5,1.,2.,3.,4.])
            angles=2*math.pi*beats[...,None]/periods
            return torch.cat((torch.sign(seconds)[...,None]*torch.log1p(seconds.abs())[...,None],
                torch.sign(beats)[...,None]*torch.log1p(beats.abs())[...,None],angles.sin(),angles.cos()),-1).float()
        return torch.stack((torch.sign(seconds)*torch.log1p(seconds.abs()),
                            torch.sign(beats)*torch.log1p(beats.abs()), beats.sin(), beats.cos()), -1).float()

    def _step_position(self,index):
        frequencies=torch.exp(torch.arange(0,self.config.d_model,2,device=index.device).float()*(-math.log(10000.)/self.config.d_model))
        angles=index.float()[...,None]*frequencies
        return torch.stack((angles.sin(),angles.cos()),-1).flatten(-2)[...,:self.config.d_model]

    def attention_context(self,device_type):
        # The reproduced dropout cancellation defect concerns CUDA BF16 SDPA
        # backward. Preserve fast inference/FP32/dropout-free paths and keep
        # the surrounding projections/FFN in the selected autocast precision.
        if (self.config.attention_policy=='bf16_dropout_math_v1' and self.training
                and self.config.dropout>0 and device_type=='cuda'
                and torch.is_autocast_enabled('cuda') and torch.get_autocast_dtype('cuda')==torch.bfloat16):
            from .stability import history_attention_context
            return history_attention_context('math_v1',device_type)
        return nullcontext()

    def encode_audio(self, song_audio, song_mask, song_seconds, song_beats):
        valid = song_mask.bool()
        torch._assert_async(valid.any(-1).all(), 'Each song needs at least one valid audio frame')
        x = self.audio_input(song_audio.float()) + self.audio_clock(self._clock(song_seconds, song_beats))
        x = x * valid[..., None]
        x = (x + F.gelu(self.audio_local(x.transpose(1, 2)).transpose(1, 2))) * valid[..., None]
        # Mask-aware pooling does not depend on the other songs' padded length.
        u, frames, _ = x.shape
        n = min(self.config.audio_summary_tokens, frames)
        ranks = (valid.long().cumsum(1) - 1).clamp_min(0)
        bins = (ranks * n // valid.sum(1, keepdim=True).clamp_min(1)).clamp_max(n-1)
        assignment = F.one_hot(bins, n).to(x.dtype) * valid[..., None]
        counts = assignment.sum(1)
        summaries = torch.bmm(assignment.transpose(1, 2), x) / counts.clamp_min(1)[..., None]
        summary_valid = counts > 0
        with self.attention_context(x.device.type):
            summaries = self.global_audio(summaries, src_key_padding_mask=~summary_valid)
        result={'local': x, 'global': summaries, 'global_mask': summary_valid,
                'seconds': song_seconds, 'beats': song_beats, 'mask': valid}
        if hasattr(self,'delta_audio_key'):result['delta_keys']=self.delta_audio_key(x)
        if self.config.timing_grid_audio_policy in ('candidate_pool_v1','query_pool_v2','query_pool_v3') and self.config.timing_output_policy=='absolute_grid_v1':
            # Reuse the projected keys. Older local-only audio policies can
            # provide a bounded 32d view without another learned backbone.
            result['grid_audio_keys']=(result['delta_keys'] if 'delta_keys' in result else
                x[...,:32] if x.size(-1)>=32 else F.pad(x,(0,32-x.size(-1))))
        if self.config.plan_kind_policy!='off':result['plan_base_logits']=self.plan_head(x)
        return result

    @staticmethod
    def _read_audio(cache, queries, inverse, *, clock='seconds',field='local'):
        # Search only clocks per row, never duplicate the full feature bank B times.
        times = cache[clock][inverse].masked_fill(~cache['mask'][inverse], torch.inf).contiguous()
        lengths = cache['mask'].sum(-1)[inverse]
        q = queries.to(times).reshape(queries.size(0), -1).contiguous()
        index = torch.searchsorted(times, q, right=True)
        right = torch.minimum(index, (lengths-1)[:, None]).clamp_min(0)
        left = (right-1).clamp_min(0)
        lo, hi = times.gather(1, left), times.gather(1, right)
        mix = ((q-lo)/(hi-lo).clamp_min(1e-8)).clamp(0, 1).float()
        bank = cache[field]
        a, b = bank[inverse[:, None], left], bank[inverse[:, None], right]
        return (a+(b-a)*mix[..., None]).reshape(*queries.shape, bank.size(-1))

    def _prefix_load(self, atoms, kinds, seconds,query_seconds=None):
        return self._pad_load(self.atom_lanes,atoms,kinds,seconds,query_seconds,
                              fast=self.config.button_load_policy=='target_fast_v2')

    @staticmethod
    def _pad_load(pad_map, atoms, kinds, seconds,query_seconds=None,*,fast=True):
        heads = pad_map[atoms].amax(-2) * kinds.eq(NOTE)[..., None]
        query=seconds if query_seconds is None else query_seconds
        ages = (query[:, :, None]-seconds[:, None, :]).float()
        t = seconds.size(1)
        causal = torch.ones(t, t, device=seconds.device, dtype=torch.bool).tril()
        values = []
        for scale in ((.15,1.,4.) if fast else (1.,4.)):
            weights = torch.exp(-ages.clamp_min(0)/scale)*causal
            values.append(torch.bmm(weights, heads.float()))
        if fast:
            last=seconds[...,None].expand(-1,-1,pad_map.size(-1)).masked_fill(heads<=0,-torch.inf).cummax(1).values
            gap=torch.where(torch.isfinite(last),(query[...,None]-last).clamp_min(0).log1p(),torch.zeros_like(last))
            values.append(gap.float())
        return torch.cat(values, -1)

    def timing_audio_scores(self,hidden,beats,cache,inverse):
        candidates=self._read_audio(cache,beats[...,None]+self.delta_values,inverse,clock='beats',field='delta_keys')
        return (self.delta_query(hidden).float()[...,None,:]*candidates.float()).sum(-1)/math.sqrt(32)

    def configure_timing_phases(self, phases):
        if self.config.timing_output_policy!='absolute_grid_v1':return
        self.timing_grid.configure(phases)
        self.config.timing_phase_catalog=tuple(str(x) for x in self.timing_grid.phase_catalog)

    def _continuation_input(self,batch,like):
        values=batch.get('continuation_features')
        width=continuation_dim(self.config.continuation_policy)
        if values is None:return like.new_zeros(*like.shape[:-1],width)
        if values.shape!=(*like.shape[:-1],width):
            raise ValueError('Continuation features do not match the checkpoint policy/history')
        return values.float()

    def repeat_delta_log_probs(self,hidden,batch,logits,allowed,cache,inverse,*,symbolic_only=False):
        """A calibrated continue/switch decision with a conditional switch law.

        The copy class receives exactly sigmoid(gate), independently of how
        many alternatives compete with it. WAIT breaks continuation. No target
        timing or target action is consulted by this factorization.
        """
        previous=batch['history_delta'].clamp_min(0)
        copy=F.one_hot(previous,logits.size(-1)).bool()
        eligible=(batch['history_kind'].eq(NOTE)&previous.gt(0)
                  &allowed.gather(-1,previous[...,None]).squeeze(-1))
        gate=self.keep_gap_head(hidden).float().squeeze(-1)
        if hasattr(self,'gap_continuation_projection'):
            gate=gate+self.gap_continuation_projection(self._continuation_input(batch,hidden)).float().squeeze(-1)
        if not symbolic_only and 'delta_keys' in cache:
            keys=self._read_audio(cache,batch['step_beat'][...,None]+self.delta_values,
                                  inverse,clock='beats',field='delta_keys').float()
            previous_key=keys.gather(-2,previous[...,None,None].expand(*previous.shape,1,keys.size(-1))).squeeze(-2)
            mean=(keys*allowed[...,None]).sum(-2)/allowed.sum(-1).clamp_min(1)[...,None]
            gate=gate+self.keep_gap_audio(previous_key-mean).float().squeeze(-1)
        ordinary=logits.float().masked_fill(~allowed,-1e4).log_softmax(-1)
        other=allowed&~copy
        conditional=logits.float().masked_fill(~other,-1e4).log_softmax(-1)
        mixture=torch.where(copy,F.logsigmoid(gate)[...,None],
                            F.logsigmoid(-gate)[...,None]+conditional)
        # Degenerate support has a deterministic copy outcome, with no false
        # switch probability disappearing into illegal alternatives.
        mixture=torch.where((~other.any(-1))[...,None],torch.zeros_like(mixture),mixture)
        result=torch.where(eligible[...,None],mixture,ordinary).masked_fill(~allowed,-1e4)
        return result,torch.where(eligible,previous,torch.full_like(previous,-1))

    def transition_reference_seconds(self,batch,event_seconds):
        """Select the committed music reference without reading future labels.

        At a V3 NOTE cursor, use the strictly earlier NOTE: the latest NOTE
        is the current music itself. WAIT and future action queries still
        use the latest NOTE. Missing references compare music to itself.
        V1/V2 retain their original last-NOTE/song-start contract.
        """
        kinds=batch['history_kind'];note=kinds.eq(NOTE)&batch['step_mask'].bool()
        positions=torch.arange(kinds.size(1),device=kinds.device)[None].expand_as(kinds)
        last=positions.masked_fill(~note,-1).cummax(-1).values
        if self.config.transition_policy=='music_exit_v3':
            earlier=torch.cat((last.new_full((last.size(0),1),-1),last[:,:-1]),1)
            cursor_note=(event_seconds.double()-batch['step_second'].double()).abs().lt(1e-7)&note
            last=torch.where(cursor_note,earlier,last)
        previous=batch['step_second'].gather(1,last.clamp_min(0))
        if self.config.transition_policy=='music_exit_v3':
            return torch.where(last.ge(0),previous,event_seconds.to(previous))
        return previous.masked_fill(last.lt(0),0.)

    def transition_logits_at(self,cache,event_seconds,inverse,batch):
        """Shared music/run-conditioned time, motion and template retention.

        Uses only committed notes and encoded song reads, never target labels.
        Six fixed future probes provide evidence for stopping a regular run.
        """
        kinds=batch['history_kind']
        previous=self.transition_reference_seconds(batch,event_seconds)
        if 'delta_keys' in cache:keys=cache
        else:
            local=cache['local'];keys={**cache,'transition_keys':F.pad(local,(0,max(0,32-local.size(-1))))[...,:32]}
        field='delta_keys' if 'delta_keys' in keys else 'transition_keys'
        current=self._read_audio(keys,event_seconds,inverse,field=field).float()
        old=self._read_audio(keys,previous,inverse,field=field).float()
        beat_cache={**cache,'transition_beats':cache['beats'][...,None]}
        beats=self._read_audio(beat_cache,event_seconds,inverse,field='transition_beats').squeeze(-1)
        offsets=beats.new_tensor([.25,.5,1.,2.,4.,8.])
        future=self._read_audio(keys,beats[...,None]+offsets,inverse,clock='beats',field=field).float().flatten(-2)
        elapsed=(event_seconds.double()-previous).clamp_min(0).log1p().float()
        gap=batch.get('history_gap_beats',self.delta_values[batch['history_delta'].clamp_min(0)]).float().clamp_min(0).log1p()
        condition=batch['condition'].float()[:,None].expand(-1,kinds.size(1),-1)
        values=torch.cat((current,current-old,current*old,future,self._clock(event_seconds,beats),
            elapsed[...,None],gap[...,None],condition,self._continuation_input(batch,current),
            self.transition_symbol_features(batch)),-1)
        return self.music_exit_head(F.layer_norm(values,(values.size(-1),))).float()

    def transition_symbol_features(self,batch):
        """Relative movement/type/size identifies what the music must change.

        Average complete phase orbits; never invent a left/right hand label
        or choose an arbitrary anchor for a symmetric chord.
        """
        kinds=batch['history_kind'];note=kinds.eq(NOTE)
        shapes=batch['history_shape']
        masks=batch.get('history_phase_mask',F.one_hot(batch['history_phase'].clamp_min(0),8).bool()).bool()
        symmetry=self.shape_motion_stabilizer[shapes]
        masks=(masks[...,self.motion_rotation_differences]&symmetry[...,None,:]).any(-1).float()
        mass=masks/masks.sum(-1,keepdim=True).clamp_min(1)
        before=torch.cat((mass.new_zeros(mass.size(0),1,8),mass[:,:-1]),1)
        lanes=torch.arange(8,device=mass.device)
        bins=F.one_hot((lanes[None,:]-lanes[:,None])%8,8).float()
        direction=torch.einsum('bti,btj,ijk->btk',before,mass,bins)
        comparable=self.motion_reference_shapes(batch).gt(0)
        direction=direction*comparable[...,None]
        from .stream_actions import family_masks_from_atoms
        families=family_masks_from_atoms(batch['history_atoms'],self.semantic_atom_families)
        bits=families.new_tensor([1,2,4,8,16])
        types=(families[...,None]&bits).ne(0).float()*note[...,None]
        heads=(self.atom_lanes[batch['history_atoms']].amax(-2).sum(-1)
            +self.atom_touch[batch['history_atoms']].amax(-2).sum(-1)).float().log1p()*note
        atom_types=self.semantic_atom_families[batch['history_atoms']]
        button_types=atom_types[...,None].eq(atom_types.new_tensor([1,2,4]))
        typed_heads=(self.atom_lanes[batch['history_atoms']][...,None,:]
            *button_types[...,None]).amax(-3).float().flatten(-2)*note[...,None]
        return torch.cat((direction,types,heads[...,None],typed_heads),-1)

    @staticmethod
    def motion_reference_shapes(batch):
        shapes=batch['history_shape'];note=batch['history_kind'].eq(NOTE)&shapes.gt(0)
        prior=torch.cat((shapes.new_full((shapes.size(0),1),-1),shapes[:,:-1]),1)
        previous_note=torch.cat((note.new_zeros(note.size(0),1),note[:,:-1]),1)
        return shapes.masked_fill(~(note&previous_note&prior.eq(shapes)),-1)

    def motion_keep_mask(self,shape,motion_phase_distribution,motion_reference_shape):
        if motion_phase_distribution is None or motion_reference_shape is None:return None
        shape=torch.as_tensor(shape,device=motion_phase_distribution.device,dtype=torch.long)
        raw=motion_phase_distribution[...,0,:].gt(0)
        if self.config.transition_policy in ('music_exit_v2','music_exit_v3'):
            # Distinct motion features reserve KEEP for nonstationary pairs.
            # A repeated physical head still has a continuation: STAY. Without
            # this branch, jacks would have no learnable motion exit at all.
            raw=torch.where(raw.any(-1,keepdim=True),raw,motion_phase_distribution[...,2,:].gt(0))
        symmetries=self.shape_motion_stabilizer[shape.clamp(0,self.codec.num_shapes-1)]
        # Last axes are output phase j and symmetry shift s: raw[j-s].
        closed=(raw[...,self.motion_rotation_differences]&symmetries[...,None,:]).any(-1)
        return closed&(shape.gt(0)&shape.eq(motion_reference_shape))[...,None]

    def action_context_at(self, hidden, event_seconds, audio_cache, song_inverse,batch=None,*,return_button_load=False):
        music = self._read_audio(audio_cache, event_seconds, song_inverse)
        result=self.action_fusion(torch.cat((hidden, music.to(hidden)), -1))
        if hasattr(self,'action_continuation_projection') and batch is not None:
            result=result+self.action_continuation_projection(self._continuation_input(batch,hidden)).to(result)
        button_load=None
        if batch is not None:
            if self.config.button_load_policy=='target_fast_v2':
                flat_load=self._prefix_load(batch['history_atoms'],batch['history_kind'],
                    batch['step_second'],event_seconds)
                result=result+self.load_projection(flat_load).to(result)
                if return_button_load:button_load=flat_load.reshape(*flat_load.shape[:-1],4,8).transpose(-1,-2)
            elif return_button_load:
                flat_load=self._pad_load(self.atom_lanes,batch['history_atoms'],batch['history_kind'],
                    batch['step_second'],event_seconds)
                button_load=flat_load.reshape(*flat_load.shape[:-1],4,8).transpose(-1,-2)
        if hasattr(self,'touch_load_projection') and batch is not None:
            result=result+self.touch_load_projection(self._pad_load(self.atom_touch,
                batch['history_atoms'],batch['history_kind'],batch['step_second'],event_seconds)).to(result)
        return (result,button_load) if return_button_load else result

    def action_context_for_delta(self, output, batch, delta_ids, audio_cache, *, event_seconds=None,return_button_load=False):
        """Condition actions on sampled time; exact TempoClock seconds may override interpolation."""
        if event_seconds is None:
            if 'candidate_beats' in output:
                query=output['candidate_beats'].gather(-1,delta_ids[...,None]).squeeze(-1)
            else:
                delta = self.delta_values.to(dtype=batch['step_beat'].dtype)
                query = batch['step_beat'] + delta[delta_ids]
            inverse = batch['song_inverse']
            beats = audio_cache['beats'][inverse].masked_fill(~audio_cache['mask'][inverse], torch.inf).contiguous()
            length = audio_cache['mask'].sum(-1)[inverse]
            right = torch.minimum(torch.searchsorted(beats, query.contiguous(), right=True), (length-1)[:, None]).clamp_min(0)
            left = (right-1).clamp_min(0)
            lo, hi = beats.gather(1, left), beats.gather(1, right)
            mix = ((query-lo)/(hi-lo).clamp_min(1e-8)).clamp(0, 1)
            seconds = audio_cache['seconds'][inverse]
            event_seconds = seconds.gather(1, left)+(seconds.gather(1, right)-seconds.gather(1, left))*mix
        return self.action_context_at(output['hidden'], event_seconds, audio_cache, batch['song_inverse'],batch,
                                      return_button_load=return_button_load)

    def _condition_plan(self,cache,condition,inverse):
        base=cache['plan_base_logits'] if 'plan_base_logits' in cache else self.plan_head(cache['local'])
        return base[inverse]+F.linear(self.condition(condition.float()),self.plan_head.weight)[:,None]

    @staticmethod
    def _plan_integral_state(logits,cache,inverse,end_seconds):
        seconds=cache['seconds'][inverse].double()
        valid=cache['mask'][inverse]
        last=valid.sum(-1).clamp_min(1)-1
        widths=torch.cat(((seconds[:,1:]-seconds[:,:-1]).clamp_min(0),seconds.new_zeros(seconds.size(0),1)),1)
        widths=widths.scatter(1,last[:,None],(end_seconds.double()-seconds.gather(1,last[:,None]).squeeze(1)).clamp_min(0)[:,None])*valid
        rates=F.softplus(logits[...,1:].detach().double())*valid[...,None]
        prefix=torch.cat((rates.new_zeros(rates.size(0),1,2),(rates*widths[...,None]).cumsum(1)),1)
        return {'rates':rates,'prefix':prefix,'widths':widths,'seconds':seconds,'valid':valid}

    @torch.no_grad()
    def prepare_plan_cache(self,cache,condition,song_inverse,audio_end_sec):
        """Caller-owned inference cache for a fixed audio/condition pairing."""
        if self.training:raise ValueError('Conditioned plan cache is inference-only')
        result=dict(cache)
        logits=self._condition_plan(cache,condition,song_inverse)
        result['conditioned_plan']={'logits':logits,**self._plan_integral_state(logits,cache,song_inverse,audio_end_sec)}
        return result

    def _plan_at_cursor(self,logits,cache,batch,inverse,state=None):
        state=state or self._plan_integral_state(logits,cache,inverse,batch['audio_end_sec'])
        start=batch['step_second'].double()
        future_beat=torch.minimum(batch['step_beat']+2,batch['audio_end_beat'][:,None])
        seconds_cache={**cache,'second_values':cache['seconds'][...,None]}
        stop=self._read_audio(seconds_cache,future_beat,inverse,clock='beats',field='second_values').squeeze(-1).double()
        last=cache['mask'].sum(-1)[inverse].clamp_min(1)-1
        last_beat=cache['beats'][inverse].gather(1,last[:,None])
        last_second=cache['seconds'][inverse].gather(1,last[:,None])
        tail_mix=((future_beat-last_beat)/(batch['audio_end_beat'][:,None]-last_beat).clamp_min(1e-10)).clamp(0,1)
        tail_second=last_second+tail_mix*(batch['audio_end_sec'][:,None]-last_second)
        stop=torch.where(future_beat>=last_beat,tail_second,stop)
        def integral(query):
            query=torch.minimum(query,batch['audio_end_sec'][:,None]).clamp_min(0)
            times=state['seconds'].masked_fill(~state['valid'],torch.inf).contiguous()
            index=(torch.searchsorted(times,query.contiguous(),right=True)-1).clamp_min(0)
            index=torch.minimum(index,last[:,None])
            gathered=index[...,None].expand(-1,-1,2)
            return state['prefix'].gather(1,gathered)+state['rates'].gather(1,gathered)*(query-times.gather(1,index))[...,None]
        intensity=(integral(stop)-integral(start)).clamp_min(0).float()
        event=intensity[...,1]
        prior=torch.zeros(*event.shape,5,device=event.device,dtype=event.dtype)
        prior[...,WAIT]=-event
        prior[...,NOTE]=(-torch.expm1(-event)).clamp_min(1e-8).log()
        plan_cache={'local':logits.detach(),'seconds':cache['seconds'][inverse],
                    'mask':cache['mask'][inverse]}
        identity=torch.arange(inverse.size(0),device=inverse.device)
        rest=self._read_audio(plan_cache,start,identity)[...,0].sigmoid()
        features=torch.cat((intensity.log1p(),rest[...,None]),-1)
        return prior.detach(),features.detach(),state['widths'].float()

    def _plan_candidate_expected_counts(self,state,cache,batch,inverse,candidate_beats):
        """Reuse detached plan prefix sums for candidate interval evidence.

        This reads scalar clocks/counts, not another audio encoding or a new
        planner. The timing head learns what the count implies; no density
        penalty or mandatory note threshold is imposed here.
        """
        seconds_cache={**cache,'second_values':cache['seconds'][...,None]}
        stops=self._read_audio(seconds_cache,candidate_beats,inverse,clock='beats',field='second_values').squeeze(-1).double()
        last=cache['mask'].sum(-1)[inverse].clamp_min(1)-1
        last_beat=cache['beats'][inverse].gather(1,last[:,None])[:,None]
        last_second=cache['seconds'][inverse].gather(1,last[:,None])[:,None]
        end_beat=batch['audio_end_beat'][:,None,None]
        end_second=batch['audio_end_sec'][:,None,None]
        tail_mix=((candidate_beats-last_beat)/(end_beat-last_beat).clamp_min(1e-10)).clamp(0,1)
        stops=torch.where(candidate_beats>=last_beat,last_second+tail_mix*(end_second-last_second),stops)
        times=state['seconds'].masked_fill(~state['valid'],torch.inf).contiguous()
        prefix=state['prefix'][...,1]
        rates=state['rates'][...,1]
        def integral(query):
            shape=query.shape
            values=query.double().reshape(query.size(0),-1)
            values=torch.minimum(values,batch['audio_end_sec'][:,None]).clamp_min(0)
            index=(torch.searchsorted(times,values.contiguous(),right=True)-1).clamp_min(0)
            index=torch.minimum(index,last[:,None])
            result=prefix.gather(1,index)+rates.gather(1,index)*(values-times.gather(1,index))
            return result.reshape(shape)
        expected=(integral(stops)-integral(batch['step_second'].double())[...,None]).clamp_min(0)
        return expected.float().detach()

    def event_phase_logits(self, context, shape, reference_phase, reference_phase_mask=None,motion_phase_distribution=None,button_load=None,
                           motion_keep_logit=None,motion_reference_shape=None):
        if hasattr(self,'motion_context_projection') and motion_phase_distribution is not None:
            context=context+self.motion_context_projection(motion_phase_distribution.float().flatten(-2)).to(context)
        relative = self.phase_head(context+self.shape_embedding(shape))
        absolute = torch.arange(8, device=context.device)
        motion=0.
        if (hasattr(self,'motion_head') and motion_phase_distribution is not None
                and self.config.motion_continuation_policy!='context_motion_v3'):
            motion=torch.einsum('...r,...rk->...k',self.motion_head(context).float(),motion_phase_distribution.float())
        lane=0.
        if hasattr(self,'lane_load_projection') and button_load is not None:
            scores=self.lane_load_projection(button_load.float()).squeeze(-1)
            heads=self.shape_button_lanes[shape]
            lane=(heads*scores[...,None,:]).sum(-1)/heads.sum(-1).clamp_min(1)
        if self.config.phase_output_policy=='legacy_canonical_v1':
            indices = (absolute-reference_phase[..., None]) % 8
            result=relative.gather(-1, indices) + self.absolute_phase_bias+motion+lane
            return self._motion_transition(result,shape,motion_phase_distribution,motion_keep_logit,motion_reference_shape)
        if reference_phase_mask is None:
            raise ValueError('orbit_reference_v1 requires reference_phase_mask')
        mask=reference_phase_mask.to(device=context.device,dtype=torch.bool)
        # No preceding rotatable NOTE has a conventional zero reference, not
        # eight interchangeable invented previous orientations.
        origin=torch.zeros_like(mask);origin[...,0]=True
        mask=torch.where(mask.any(-1,keepdim=True),mask,origin)
        reference=self.reference_phase_head(context).float().masked_fill(~mask,-torch.inf).log_softmax(-1)
        relative=relative.float().log_softmax(-1)
        # rows r: reference orientation; columns j: absolute output orientation.
        differences=(absolute[None,:]-absolute[:,None])%8
        joint=reference[..., :,None]+relative[...,differences]
        result=joint.logsumexp(-2)+self.absolute_phase_bias+motion+lane
        return self._motion_transition(result,shape,motion_phase_distribution,motion_keep_logit,motion_reference_shape)

    def _motion_transition(self,logits,shape,distribution,keep_logit,reference_shape):
        if self.config.transition_policy not in ('music_exit_v1','music_exit_v2','music_exit_v3') or keep_logit is None:return logits
        from .stream_transition import two_group_log_probs
        mask=self.motion_keep_mask(shape,distribution,reference_shape)
        return logits if mask is None else two_group_log_probs(logits,mask,self.transition_keep_logit(logits,mask,keep_logit))

    def transition_keep_logit(self,base,mask,correction,*,allowed=None):
        if self.config.transition_policy in ('music_exit_v2','music_exit_v3'):
            from .stream_transition import residual_keep_logit
            return residual_keep_logit(base,mask,correction,gain=8.,allowed=allowed)
        return correction

    @staticmethod
    def motion_phase_features(kinds,shapes,masks,*,distinct=False,adjacent=False):
        valid=kinds.eq(NOTE)&shapes.gt(0)
        positions=torch.arange(kinds.size(1),device=kinds.device)[None].expand_as(kinds)
        last=positions.masked_fill(~valid,-1).cummax(-1).values
        preceding=torch.cat((last.new_full((last.size(0),1),-1),last[:,:-1]),1)
        older=preceding.gather(1,last.clamp_min(0))
        first=masks.gather(1,older.clamp_min(0)[...,None].expand(-1,-1,8)).float()
        second=masks.gather(1,last.clamp_min(0)[...,None].expand(-1,-1,8)).float()
        first=first/first.sum(-1,keepdim=True).clamp_min(1)
        second=second/second.sum(-1,keepdim=True).clamp_min(1)
        angles=torch.arange(8,device=kinds.device)
        before,after=angles[:,None].expand(8,8),angles[None,:].expand(8,8)
        destinations=torch.stack(((2*after-before)%8,before,after),-1)
        pushforward=F.one_hot(destinations,8).float()
        if distinct:
            # KEEP and REVERSE describe movement; stationary pairs have only
            # STAY. Apply this before orbit marginalization for symmetric sets.
            moving=before.ne(after)
            pushforward=pushforward*torch.stack((moving,moving,torch.ones_like(moving)),-1)[...,None]
        distribution=torch.einsum('bti,btj,ijrk->btrk',first,second,pushforward)
        usable=last.ge(0)&older.ge(0)
        if adjacent:
            # A WAIT, Touch or unsupported event breaks the local two-event
            # movement relation. Never revive stale geometry after a rest.
            previous_valid=torch.cat((valid.new_zeros(valid.size(0),1),valid[:,:-1]),1)
            usable=usable&valid&previous_valid
        return distribution*usable[...,None,None]

    @staticmethod
    def reference_phase_mask(kinds,shapes,phase,phase_mask=None):
        rotatable=kinds.eq(NOTE)&shapes.gt(0)
        positions=torch.arange(kinds.size(1),device=kinds.device)[None].expand_as(kinds)
        latest=positions.masked_fill(~rotatable,-1).cummax(-1).values
        masks=F.one_hot(phase,8).bool() if phase_mask is None else phase_mask.bool()
        reference=masks.gather(1,latest.clamp_min(0)[...,None].expand(-1,-1,8))
        origin=torch.zeros_like(reference);origin[...,0]=True
        return torch.where(latest.ge(0)[...,None],reference,origin)

    def history_phase_features(self, kinds, shapes, phase, phase_mask=None):
        """Marginalize complete-event symmetry without picking a movement anchor."""
        rotatable = kinds.eq(NOTE) & shapes.gt(0)
        pos = torch.arange(kinds.size(1), device=kinds.device)[None].expand_as(kinds)
        latest = pos.masked_fill(~rotatable, -1).cummax(-1).values
        previous = torch.cat((latest.new_full((latest.size(0), 1), -1), latest[:, :-1]), 1)
        reference = phase.gather(1, latest.clamp_min(0)).masked_fill(latest < 0, 0)
        active = rotatable & previous.ge(0)
        if self.config.history_phase_policy == 'canonical_v1':
            old_phase = phase.gather(1, previous.clamp_min(0))
            return self.phase_embedding(phase), self.phase_change((phase-old_phase)%8)*active[...,None], reference
        if phase_mask is None:
            raise ValueError('orbit_average_v1 requires complete-event history_phase_mask')
        mass = phase_mask.to(self.phase_embedding.weight.dtype)*rotatable[..., None]
        probability = mass/mass.sum(-1,keepdim=True).clamp_min(1)
        old_probability = probability.gather(1,previous.clamp_min(0)[...,None].expand(-1,-1,8))
        angles = torch.arange(8,device=phase.device)
        difference = (angles[:,None]-angles[None,:])%8
        movement = torch.einsum('bti,btj,ijd->btd',probability,old_probability,self.phase_change.weight[difference])
        return probability @ self.phase_embedding.weight, movement*active[...,None], reference

    def set_audio_tracking_strength(self, strength=0.):
        strength = float(strength)
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError('Audio tracking strength must be finite in [0, 1]')
        self._audio_tracking_strength = strength

    def duration_context_at(self, output, batch, event_seconds, audio_cache):
        """Prepare bounded reads using the audio bank's local tempo estimate.

        A bank cell straddling a tempo change has an averaged beat length. This
        auxiliary score never replaces the native parser's physical timings.
        """
        if not hasattr(self, 'duration_audio'): return None
        inverse = batch.get('song_inverse')
        if inverse is None:
            inverse = torch.arange(event_seconds.size(0), device=event_seconds.device)
        seconds = audio_cache['seconds'][inverse]
        beats = audio_cache['beats'][inverse]
        lengths = audio_cache['mask'].sum(-1)[inverse]
        ordered = seconds.masked_fill(~audio_cache['mask'][inverse], torch.inf).contiguous()
        right = torch.searchsorted(ordered, event_seconds.to(ordered).contiguous(), right=True)
        right = torch.minimum(right, (lengths-1)[:, None]).clamp_min(1)
        right = right.clamp_max(seconds.size(1)-1)
        left = (right-1).clamp_min(0)
        ds = seconds.gather(1, right)-seconds.gather(1, left)
        db = beats.gather(1, right)-beats.gather(1, left)
        valid = (lengths[:, None] >= 2) & (db > 0) & (ds > 0)
        spb = (ds/db.clamp_min(1e-8)).float()
        intent = output.get('phrase_intent')
        attention = self.kind_embedding.weight.new_ones(*event_seconds.shape, 32)
        if intent is not None:
            strength = torch.as_tensor(batch.get('audio_tracking_strength', self._audio_tracking_strength),
                device=attention.device,dtype=attention.dtype)
            if strength.ndim == 1: strength = strength[:, None]
            if strength.ndim == 2: strength = strength[..., None]
            attention = attention+strength*(2*self.phrase_context.tracking_acoustic(intent).sigmoid()-1)
        inv = inverse[:, None].expand_as(event_seconds)
        end = seconds.gather(1, (lengths-1).clamp_min(0)[:, None]).expand_as(event_seconds)
        return dict(cache=audio_cache, inverse=inv.flatten(), second=event_seconds.float().flatten(),
                    seconds_per_beat=spb.flatten(), valid=valid.flatten(),
                    attention=attention.flatten(0, 1), end_second=end.float().flatten())

    def duration_detail_logits(self, logits, shape, slot, duration_context=None):
        if not hasattr(self, 'duration_audio'): return logits
        return self.duration_audio(self, logits, shape, slot, duration_context)

    def forward(self, batch, audio_cache=None, *, symbolic_only=False):
        symbolic_only = symbolic_only or batch.get('symbolic_only', False)
        if symbolic_only:
            # A zero one-frame bank makes every current/future read zero and
            # avoids encoding or materializing the full song during pretraining.
            u = batch['song_audio'].size(0)
            local = self.kind_embedding.weight.new_zeros(u, 1, self.config.d_model)
            cache = {'local': local, 'global': local,
                     'global_mask': torch.ones(u, 1, device=local.device, dtype=torch.bool),
                     'mask': torch.ones(u, 1, device=local.device, dtype=torch.bool),
                     'seconds': local.new_zeros(u, 1), 'beats': local.new_zeros(u, 1)}
            if hasattr(self,'delta_audio_key'):cache['delta_keys']=local.new_zeros(u,1,32)
        else:
            cache = audio_cache if audio_cache is not None else self.encode_audio(
                batch['song_audio'], batch['song_mask'], batch['song_seconds'], batch['song_beats'])
        inverse = batch.get('song_inverse')
        if inverse is None:
            inverse = torch.arange(batch['history_kind'].size(0), device=batch['history_kind'].device)
        kinds, atoms = batch['history_kind'], batch['history_atoms']
        valid = batch['step_mask'].bool()
        phase = batch['history_phase'].clamp_min(0)
        note = kinds.eq(NOTE)
        phase_context,movement,reference_phase = self.history_phase_features(
            kinds,batch['history_shape'],phase,batch.get('history_phase_mask'))
        reference_phase_mask=self.reference_phase_mask(kinds,batch['history_shape'],phase,batch.get('history_phase_mask'))
        motion_distribution=None
        if hasattr(self,'motion_head'):
            masks=batch.get('history_phase_mask',F.one_hot(phase,8).bool())
            motion_distribution=self.motion_phase_features(kinds,batch['history_shape'],masks,
                distinct=self.config.motion_continuation_policy in ('motion_distinct_v2','context_motion_v3'),
                adjacent=self.config.motion_continuation_policy=='context_motion_v3')
        atom_emb = self.atom_embedding(atoms).sum(-2) / atoms.ne(0).sum(-1).clamp_min(1)[..., None]
        x = self.kind_embedding(kinds) + self.delta_embedding(batch['history_delta'].clamp_min(0))
        x = x + (self.shape_embedding(batch['history_shape'].clamp_min(0)) + phase_context + atom_emb)*note[..., None] + movement
        sec, beat = batch['step_second'].float(), batch['step_beat'].float()
        history_gap=batch.get('history_gap_beats',self.delta_values[batch['history_delta'].clamp_min(0)])
        if hasattr(self,'history_gap_projection'):
            periods=history_gap.new_tensor([.25,.5,1.,2.,3.,4.])
            angles=2*math.pi*history_gap[...,None]/periods
            gap_features=torch.cat((history_gap.clamp_min(0).log1p()[...,None],angles.sin(),angles.cos()),-1).float()
            x=x+self.history_gap_projection(gap_features).to(x)
        elapsed = torch.cat((sec.new_zeros(sec.size(0), 1), sec[:, 1:]-sec[:, :-1]), -1)
        end_sec = (batch['audio_end_sec'] if 'audio_end_sec' in batch else batch['song_seconds'].amax(-1)[inverse]).float()
        end_beat = (batch['audio_end_beat'] if 'audio_end_beat' in batch else batch['song_beats'].amax(-1)[inverse]).to(batch['step_beat'])
        clock = torch.cat((self._clock(sec, beat), elapsed.clamp_min(0).log1p()[..., None],
                           history_gap.float().clamp_min(0).log1p()[..., None],
                           (end_sec[:,None]-sec).clamp_min(0).log1p()[...,None],
                           (end_beat.float()[:,None]-beat).clamp_min(0).log1p()[...,None]), -1)
        x = x + self.cursor_clock(clock) + self.condition(batch['condition'].float())[:, None]
        if self.config.clock_policy=='musical_v2':
            index=batch.get('step_index')
            if index is None:index=torch.arange(kinds.size(1),device=kinds.device)[None].expand_as(kinds)
            x=x+self._step_position(index).to(x)
        if 'action_state' in batch:
            state = batch['action_state'].float()
            state_valid = ~batch['action_state_mask'].bool()
            pooled = (state * state_valid[..., None]).sum(-2)/state_valid.sum(-1).clamp_min(1)[..., None]
            x = x + self.state_projection(pooled)
        x = x + self.load_projection(self._prefix_load(atoms, kinds, sec))
        if hasattr(self,'touch_load_projection'):
            x=x+self.touch_load_projection(self._pad_load(self.atom_touch,atoms,kinds,sec)).to(x)
        if hasattr(self,'continuation_projection'):
            x=x+self.continuation_projection(self._continuation_input(batch,x)).to(x)
        offsets = self.near_offsets.to(x)
        nearby = self._read_audio(cache, beat[..., None]+offsets, inverse, clock='beats')
        x = x + self.near_audio(nearby.flatten(-2))
        phrase_output = {}
        if hasattr(self, 'phrase_context'):
            phrase_residual, phrase_output = self.phrase_context(self, cache, inverse, batch,
                tracking_strength=batch.get('audio_tracking_strength', self._audio_tracking_strength))
            x = x + phrase_residual.to(x)
        t = x.size(1)
        causal = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
        safe_valid = valid.clone()
        safe_valid[:, 0] = True
        with self.attention_context(x.device.type):
            hidden = self.decoder(x, cache['global'][inverse], tgt_mask=causal,
                tgt_key_padding_mask=~safe_valid, memory_key_padding_mask=~cache['global_mask'][inverse])
        hidden = hidden * valid[..., None]
        # target_second is intentionally read only after time logits exist.
        kind_logits, delta_logits = self.kind_head(hidden).float(), self.delta_head(hidden).float()
        if hasattr(self,'kind_continuation_projection'):
            kind_logits=kind_logits+self.kind_continuation_projection(self._continuation_input(batch,hidden)).float()
        if self.config.timing_output_policy=='relative_delta_v1' and hasattr(self,'keep_gap_head') and self.config.gap_continuation_policy=='additive_v1':
            positions=torch.arange(kinds.size(1),device=kinds.device)[None].expand_as(kinds)
            last_note=positions.masked_fill(~kinds.eq(NOTE),-1).cummax(-1).values
            gap=batch['history_delta'].gather(1,last_note.clamp_min(0))
            matches=F.one_hot(gap,len(self.config.delta_beats)).to(delta_logits)*last_note.ge(0)[...,None]
            delta_logits=delta_logits+self.keep_gap_head(hidden).float()*matches
        if self.config.timing_output_policy=='relative_delta_v1' and hasattr(self,'delta_audio_key') and not symbolic_only:
            delta_logits=delta_logits+self.timing_audio_scores(hidden,batch['step_beat'],cache,inverse)
        plan=None;plan_exposure=None;plan_state=None
        if not symbolic_only and (self.config.plan_kind_policy!='off' or 'audio_plan_target' in batch):
            prepared=cache.get('conditioned_plan')
            if prepared is not None and self.training:raise ValueError('Conditioned plan cache is inference-only')
            plan=prepared['logits'] if prepared is not None else self._condition_plan(cache,batch['condition'],inverse)
            plan_state=prepared
            if self.config.timing_output_policy=='absolute_grid_v1' and self.config.timing_group_policy=='phase_gap_v2' and plan_state is None:
                plan_state=self._plan_integral_state(plan,cache,inverse,batch['audio_end_sec'])
            if self.config.plan_kind_policy=='rate_integral_v2':
                prior,features,plan_exposure=self._plan_at_cursor(plan,cache,batch,inverse,plan_state)
                kind_logits=kind_logits+prior+self.plan_kind_features(features).float()
        # Match the generation support without infinities in padded/zero-loss
        # frames. WAIT is a fixed two-beat transition; EOS owns the final tail.
        remaining = end_beat.to(batch['step_beat'])[:, None]-batch['step_beat']
        kind_allowed = torch.ones_like(kind_logits, dtype=torch.bool)
        kind_allowed[..., :2] = False
        kind_allowed[..., EOS] = remaining <= 2
        kind_allowed[..., WAIT] = remaining > 2
        kind_logits = kind_logits.masked_fill(~kind_allowed, -1e4)
        absolute_timing=self.config.timing_output_policy=='absolute_grid_v1'
        values = self.delta_values.to(batch['step_beat'])
        first = kinds.eq(BOS) & batch['step_beat'].eq(0)
        candidate_beats=candidate_deltas=None
        if absolute_timing:
            candidate_beats,candidate_deltas=self.timing_grid.candidates(batch['step_beat'])
            delta_allowed=(candidate_deltas>0)|(candidate_deltas.eq(0)&first[...,None])
            delta_allowed=delta_allowed&(candidate_beats<end_beat[:,None,None])
            delta_allowed[...,0]=False
            query=(self.delta_query(hidden) if hasattr(self,'delta_query') else
                   F.pad(hidden,(0,max(0,32-hidden.size(-1))))[...,:32])
            audio_scores=torch.zeros_like(candidate_beats,dtype=torch.float32)
            candidate_audio=None
            if not symbolic_only and 'delta_keys' in cache:
                keys=self._read_audio(cache,candidate_beats,inverse,clock='beats',field='delta_keys').float()
                candidate_audio=keys
                audio_scores=(query.float()[...,None,:]*keys).sum(-1)/math.sqrt(32)
            elif not symbolic_only and self.config.timing_grid_audio_policy in ('candidate_pool_v1','query_pool_v2','query_pool_v3'):
                if 'grid_audio_keys' in cache:
                    candidate_audio=self._read_audio(cache,candidate_beats,inverse,clock='beats',field='grid_audio_keys').float()
                else:
                    local=cache['local']
                    grid_cache={**cache,'grid_audio_keys':F.pad(local,(0,max(0,32-local.size(-1))))[...,:32]}
                    candidate_audio=self._read_audio(grid_cache,candidate_beats,inverse,clock='beats',field='grid_audio_keys').float()
            expected=(self._plan_candidate_expected_counts(plan_state,cache,batch,inverse,candidate_beats)
                if self.config.timing_group_policy=='phase_gap_v2' and plan_state is not None else None)
            delta_logits,candidate_beats,candidate_deltas=self.timing_grid(hidden,batch['step_beat'],
                delta_logits,self.delta_values,query,audio_scores,delta_allowed,candidate_audio=candidate_audio,
                candidate_expected_count=expected)
        else:
            delta_allowed = (values[None, None] > 0) | (values[None, None].eq(0) & first[..., None])
            delta_allowed = delta_allowed & (batch['step_beat'][..., None]+values < end_beat[:, None, None])
            delta_allowed[..., 0] = False
            delta_logits = delta_logits.masked_fill(~delta_allowed, -1e4)
        repeat_delta_id=torch.full_like(kinds,-1)
        if self.config.gap_continuation_policy=='probability_v2' and not absolute_timing:
            delta_logits,repeat_delta_id=self.repeat_delta_log_probs(
                hidden,batch,delta_logits,delta_allowed,cache,inverse,symbolic_only=symbolic_only)
        transition_options={};time_keep_mask=None
        if self.config.transition_policy in ('music_exit_v1','music_exit_v2','music_exit_v3'):
            from .stream_transition import two_group_log_probs
            time_decision=self.transition_logits_at(cache,batch['step_second'],inverse,batch)
            gaps=candidate_deltas if absolute_timing else values
            previous_gap=batch.get('history_gap_beats',values[batch['history_delta'].clamp_min(0)])
            time_keep_mask=(gaps-previous_gap[...,None]).abs().lt(1e-7)&kinds.eq(NOTE)[...,None]&delta_allowed
            delta_logits=two_group_log_probs(delta_logits,time_keep_mask,
                self.transition_keep_logit(delta_logits,time_keep_mask,time_decision[...,0],allowed=delta_allowed),allowed=delta_allowed)
            action_decision=self.transition_logits_at(cache,batch.get('target_second',batch['step_second']),inverse,batch)
            transition_options=dict(template_keep_logit=action_decision[...,2],
                template_reference_shape=batch['history_shape'].masked_fill(~kinds.eq(NOTE),-1))
            motion_options=dict(motion_keep_logit=action_decision[...,1],
                motion_reference_shape=self.motion_reference_shapes(batch))
        else:motion_options={}
        if hasattr(self,'lane_load_projection'):
            action,button_load=self.action_context_at(hidden,batch.get('target_second',batch['step_second']),
                cache,inverse,batch,return_button_load=True)
        else:
            action=self.action_context_at(hidden,batch.get('target_second',batch['step_second']),cache,inverse,batch)
            button_load=None
        shape_logits = None
        event_routes=None
        family_context=None
        if self.config.event_family_policy!='legacy_v1':
            if self.config.event_family_policy in ('audio_semantic_v2','audio_prior_transition_v3'):
                family_context=self.family_context_at(cache,batch.get('target_second',batch['step_second']),inverse,batch)
            event_routes=self.event_route_log_probs(action,
                family_context=family_context['full'] if family_context is not None else None,
                family_logits=(family_context['full_logits'] if family_context is not None and 'full_logits' in family_context else None),**transition_options)
            event_routes=event_routes.masked_fill(~torch.isfinite(event_routes),-1e4)
            # Preserve the public shape view for diagnostics. The supervised
            # likelihood below uses the exact semantic literal route.
            literal=event_routes[...,self.codec.num_shapes-1:].logsumexp(-1,keepdim=True)
            shape_logits=torch.cat((literal,event_routes[...,:self.codec.num_shapes-1]),-1)
        else:
            shape_logits=self.shape_head(action)
        shapes = (batch['target_shape'] if 'target_shape' in batch else shape_logits.argmax(-1)).clamp_min(0)
        result = {'kind_logits': kind_logits, 'delta_logits': delta_logits,
                  'shape_logits': shape_logits,
                  'phase_logits': self.event_phase_logits(action, shapes, reference_phase,reference_phase_mask,motion_distribution,button_load,**motion_options),
                  'hidden': hidden, 'action_context': action, 'reference_phase': reference_phase,
                  'reference_phase_mask':reference_phase_mask,'repeat_delta_id':repeat_delta_id}
        result.update(phrase_output)
        if hasattr(self, 'duration_audio'):
            result['duration_audio_context'] = self.duration_context_at(result, batch,
                batch.get('target_second', batch['step_second']), cache)
        if motion_distribution is not None:result['motion_phase_distribution']=motion_distribution
        if time_keep_mask is not None:
            result.update(timing_keep_mask=time_keep_mask,**motion_options,**transition_options)
        if event_routes is not None:
            result['event_route_log_probs']=event_routes
            result['event_family_logits']=(family_context['full_logits'] if family_context is not None and 'full_logits' in family_context
                else self.event_family_scores(family_context['full'] if family_context is not None else action))
            if self.config.transition_policy in ('music_exit_v1','music_exit_v2','music_exit_v3'):
                result['event_family_logits']=grouped_logsumexp(event_routes,self.semantic_route_families-1,groups=31)
            if family_context is not None:
                result['event_family_audio_logits']=(family_context['audio_logits'] if 'audio_logits' in family_context
                    else self.event_family_scores(family_context['audio']))
                if 'continuity_logits' in family_context:
                    result['event_family_continuity_logits']=family_context['continuity_logits']
                    result['event_family_previous']=family_context['previous_family']
                    result['event_family_has_previous']=family_context['has_previous']
        if absolute_timing:
            result.update(candidate_beats=candidate_beats,candidate_deltas=candidate_deltas,
                          timing_grid_ids=self.timing_grid.candidate_groups(candidate_deltas),
                          timing_group_count=self.timing_grid.num_groups)
        if button_load is not None:result['button_load']=button_load
        if plan is not None:
            result['plan_logits'] = plan
            if plan_exposure is not None:result['plan_exposure']=plan_exposure
        return result

    def _family_history_features(self,batch,event_seconds):
        """Five-type causal summaries; no lane, ring, or configuration input.

        Prefix sums and row-index caps prevent a candidate's future time from
        reading later teacher-forced events. Work is linear in sequence size.
        """
        from .stream_actions import family_masks_from_atoms
        kinds=batch['history_kind']
        valid=batch['step_mask'].bool()
        note=kinds.eq(NOTE)&valid
        family=family_masks_from_atoms(batch['history_atoms'],self.semantic_atom_families)
        bits=family.new_tensor([1,2,4,8,16])
        present=(family[...,None]&bits).ne(0)&note[...,None]
        values=present.float()
        b,t=kinds.shape
        prefix=torch.cat((values.new_zeros(b,1,5),values.cumsum(1)),1)
        note_prefix=torch.cat((values.new_zeros(b,1),note.float().cumsum(1)),1)
        upper=torch.arange(1,t+1,device=kinds.device)[None].expand(b,-1)
        seconds=batch['step_second'].double()
        ordered=seconds.masked_fill(~valid,torch.inf).contiguous()
        features=[]
        for window in (.5,2.,8.):
            lower=torch.searchsorted(ordered,(event_seconds.double()-window).contiguous(),right=False)
            lower=torch.minimum(lower,upper)
            left=prefix.gather(1,lower[...,None].expand(-1,-1,5))
            right=prefix[:,1:]
            count=note_prefix[:,1:]-note_prefix.gather(1,lower)
            features.append((right-left)/count.clamp_min(1)[...,None])
        indices=torch.arange(t,device=kinds.device)[None].expand(b,-1)
        latest=indices.masked_fill(~note,-1).cummax(1).values
        last=values.gather(1,latest.clamp_min(0)[...,None].expand(-1,-1,5))
        features.append(last*latest.ge(0)[...,None])
        type_seconds=seconds[...,None].expand(-1,-1,5).masked_fill(~present,-torch.inf).cummax(1).values
        seen=torch.isfinite(type_seconds)
        ages=(event_seconds.double()[...,None]-type_seconds.masked_fill(~seen,0.)).clamp_min(0)
        features.extend((seen.float(),(ages.log1p()*seen).float()))
        return torch.cat(features,-1)

    def family_context_at(self,audio_cache,event_seconds,song_inverse,batch):
        """Music-first semantic input, separate from the geometry decoder."""
        if self.config.event_family_policy not in ('audio_semantic_v2','audio_prior_transition_v3'):
            raise ValueError('Music-first family context requires an audio semantic policy')
        music=self._read_audio(audio_cache,event_seconds,song_inverse)
        beat_cache={**audio_cache,'family_beat_values':audio_cache['beats'][...,None]}
        beats=self._read_audio(beat_cache,event_seconds,song_inverse,field='family_beat_values').squeeze(-1)
        offsets=beats.new_tensor([-.5,.5,-2.,2.])
        nearby=self._read_audio(audio_cache,beats[...,None]+offsets,song_inverse,clock='beats')
        lengths=audio_cache['mask'].sum(-1)[song_inverse].clamp_min(1)
        times=audio_cache['seconds'][song_inverse].masked_fill(~audio_cache['mask'][song_inverse],torch.inf).contiguous()
        frame=(torch.searchsorted(times,event_seconds.to(times).contiguous(),right=True)-1).clamp_min(0)
        frame=torch.minimum(frame,(lengths-1)[:,None])
        bins=(frame*audio_cache['global'].size(1)//lengths[:,None]).clamp_max(audio_cache['global'].size(1)-1)
        summaries=audio_cache['global'][song_inverse]
        global_music=summaries.gather(1,bins[...,None].expand(-1,-1,summaries.size(-1)))
        condition=self.condition(batch['condition'].float())[:,None].expand_as(music)
        projected=self.family_audio_projection(torch.cat((music,nearby[...,:2,:].mean(-2),
            nearby[...,2:,:].mean(-2),global_music,condition.to(music)),-1))
        rhythm=None
        if self.config.event_family_policy=='audio_prior_transition_v3':
            rhythm=self._family_rhythm_features(audio_cache,event_seconds,beats,song_inverse,batch,condition)
            projected=projected+self.family_rhythm_projection(rhythm).to(projected)
        audio=F.layer_norm(music+condition.to(music)+projected,(self.config.d_model,))
        if self.config.event_family_policy=='audio_prior_transition_v3':
            from .stream_actions import refresh_family_log_probs
            previous_family,previous_seconds,has_previous=self._family_previous_note(batch)
            previous_music=self._read_audio(audio_cache,previous_seconds,song_inverse)
            elapsed=(event_seconds.double()-previous_seconds.double()).clamp_min(0)
            # No family identity, count, sensor, or decoder state enters rho.
            evidence=torch.cat(((music-previous_music).float(),(music*previous_music).float(),
                elapsed.log1p().float()[...,None]),-1)
            continuity=self.family_continuity_head(evidence).float().squeeze(-1)
            audio_logits=self.event_family_scores(audio)
            full_logits=refresh_family_log_probs(audio_logits,continuity,previous_family,has_previous,
                detach_audio=self.config.event_family_audio_weight>0)
            return dict(full=audio,audio=audio,full_logits=full_logits,audio_logits=audio_logits,
                continuity_logits=continuity,previous_family=previous_family,has_previous=has_previous,
                rhythm_features=rhythm)
        history=self.family_history_projection(self._family_history_features(batch,event_seconds)).to(audio)
        full=F.layer_norm(audio+history,(self.config.d_model,))
        return {'full':full,'audio':audio}

    def _family_rhythm_features(self,audio_cache,event_seconds,event_beats,song_inverse,batch,condition):
        """Actual candidate gap, absolute clock, and predicted per-second plan.

        This reuses music/condition predictions. No target family, authored
        future count, key position, or occupancy enters the semantic input.
        Planner values are detached, retaining its independent count labels.
        """
        gap_seconds=(event_seconds.double()-batch['step_second'].double()).clamp_min(0)
        gap_beats=(event_beats.double()-batch['step_beat'].double()).clamp_min(0)
        gaps=torch.stack((gap_beats.log1p(),gap_seconds.log1p()),-1).float()
        clock=self._clock(event_seconds,event_beats)
        plan=clock.new_zeros(*clock.shape[:-1],3)
        if self.config.plan_kind_policy!='off':
            base=audio_cache.get('plan_base_logits')
            if base is None:base=self.plan_head(audio_cache['local'])
            source={**audio_cache,'family_plan_values':base.detach()}
            logits=self._read_audio(source,event_seconds,song_inverse,field='family_plan_values').float()
            logits=logits+F.linear(condition[:,0].float(),self.plan_head.weight).float().detach()[:,None]
            logits=logits.detach()
            rates=F.softplus(logits[...,1:])
            if self.config.plan_target_units!='rates_per_second_v2':
                seconds=audio_cache['seconds'][song_inverse]
                valid=audio_cache['mask'][song_inverse]
                lengths=valid.sum(-1).clamp_min(1)
                ordered=seconds.masked_fill(~valid,torch.inf).contiguous()
                index=(torch.searchsorted(ordered,event_seconds.to(ordered).contiguous(),right=True)-1).clamp_min(0)
                index=torch.minimum(index,(lengths-1)[:,None])
                nxt=torch.minimum(index+1,(lengths-1)[:,None])
                exposure=seconds.gather(1,nxt)-seconds.gather(1,index)
                last=seconds.gather(1,(lengths-1)[:,None]).squeeze(1)
                end=batch.get('audio_end_sec',last+self.config.audio_grid_seconds).to(last)
                exposure=torch.where(index.eq((lengths-1)[:,None]),end[:,None]-last[:,None],exposure)
                rates=rates/exposure.clamp_min(1e-5).float()[...,None]
            plan=torch.cat((logits[...,:1].sigmoid(),rates.log1p()),-1)
        return torch.cat((clock,gaps,plan),-1)

    def _family_previous_note(self,batch):
        """Actual previous NOTE at each causal position; WAIT is not a type."""
        from .stream_actions import family_masks_from_atoms
        families=family_masks_from_atoms(batch['history_atoms'],self.semantic_atom_families)
        kinds=batch['history_kind']
        note=kinds.eq(NOTE)&batch['step_mask'].bool()&families.gt(0)
        b,t=kinds.shape
        positions=torch.arange(t,device=kinds.device)[None].expand(b,-1)
        latest=positions.masked_fill(~note,-1).cummax(1).values
        previous=families.gather(1,latest.clamp_min(0)).masked_fill(latest.lt(0),0)
        seconds=batch['step_second'].gather(1,latest.clamp_min(0)).masked_fill(latest.lt(0),0.)
        supported=self.semantic_family_supported[(previous-1).clamp_min(0)]
        return previous,seconds,latest.ge(0)&supported

    @property
    def active_event_family_head(self):
        return self.audio_event_family_head if self.config.event_family_policy in ('audio_semantic_v2','audio_prior_transition_v3') else self.event_family_head

    def event_family_scores(self,family_context):
        return self.active_event_family_head(family_context).float().masked_fill(~self.semantic_family_supported,-1e4)

    @torch.no_grad()
    def sample_event_family(self,family_context=None,*,family_logits=None,temperature=1.,generator=None):
        if family_logits is None:
            if self.config.event_family_policy=='audio_prior_transition_v3':
                raise ValueError('audio_prior_transition_v3 requires mixed family_logits')
            if family_context is None:raise ValueError('Semantic family selection requires its own context')
            family_logits=self.event_family_scores(family_context)
        logits=family_logits.float().reshape(-1,31)
        if logits.size(0)!=1:raise ValueError('Sample one event family at a time')
        scores=logits[0].masked_fill(~self.semantic_family_supported,-torch.inf)
        if not bool(torch.isfinite(scores).any()):return None
        if temperature<=0:return int(scores.argmax())+1
        return int(torch.multinomial(scores.softmax(-1),1,generator=generator))+1

    def event_route_log_probs(self,contexts,*,family_context=None,family_logits=None,family_override=None,
                             temperature=1.,available_shapes=None,blocked_families=(),
                             template_keep_logit=None,template_reference_shape=None,template_keep_override=None):
        from .stream_actions import grouped_event_log_probs
        base=self.shape_head(contexts).float()
        scores=torch.cat((base[...,1:],base[...,:1]+self.literal_route_head(contexts).float()),-1)
        if available_shapes is not None:
            available_shapes=torch.as_tensor(available_shapes,device=scores.device,dtype=torch.bool)
            mask=torch.cat((available_shapes[1:],available_shapes[:1].expand(31)))
            scores=scores.masked_fill(~mask,-torch.inf)
        if blocked_families:
            indices=torch.as_tensor([self.codec.num_shapes-2+int(family) for family in blocked_families],device=scores.device)
            scores=scores.index_fill(-1,indices,-torch.inf)
        if family_override is not None:
            family_id=torch.as_tensor(family_override,device=scores.device,dtype=torch.long)
            scores=scores.masked_fill(self.semantic_route_families!=family_id[...,None],-torch.inf)
        if family_logits is None:
            if self.config.event_family_policy=='audio_prior_transition_v3':
                raise ValueError('audio_prior_transition_v3 requires mixed family_logits')
            if self.config.event_family_policy=='audio_semantic_v2' and family_context is None:
                raise ValueError('audio_semantic_v2 requires candidate music family_context')
            family_logits=self.event_family_scores(contexts if family_context is None else family_context)
        result=grouped_event_log_probs(scores,family_logits,
            self.semantic_route_families,self.semantic_family_supported,temperature=temperature)
        if self.config.transition_policy in ('music_exit_v1','music_exit_v2','music_exit_v3') and template_keep_logit is not None:
            from .stream_transition import two_group_log_probs
            mask=self.template_keep_mask(template_reference_shape,result)
            result=two_group_log_probs(result,mask,self.transition_keep_logit(result,mask,template_keep_logit))
            if template_keep_override is not None:
                selected=mask if bool(template_keep_override) else ~mask
                result=result.masked_fill(~selected,-torch.inf)
        return result

    def template_keep_mask(self,reference_shape,scores):
        if reference_shape is None:return torch.zeros_like(scores,dtype=torch.bool)
        ids=torch.cat((torch.arange(1,self.codec.num_shapes,device=scores.device),
            torch.zeros(31,device=scores.device,dtype=torch.long)))
        reference=torch.as_tensor(reference_shape,device=scores.device)
        return ids.eq(reference[...,None])&reference.gt(0)[...,None]

    @torch.no_grad()
    def sample_event_intent(self,context,*,family_logits,template_keep_logit,template_reference_shape,
                            temperature=1.,generator=None):
        """Choose template continuation, then semantic family; retain both on retry."""
        routes=self.event_route_log_probs(context.reshape(1,-1),family_logits=family_logits,
            template_keep_logit=template_keep_logit,template_reference_shape=template_reference_shape).reshape(-1)
        mask=self.template_keep_mask(template_reference_shape,routes)
        masses=torch.stack((routes.masked_fill(~mask,-torch.inf).logsumexp(-1),
            routes.masked_fill(mask,-torch.inf).logsumexp(-1)))
        if not bool(torch.isfinite(masses).any()):return None,None
        group=int(masses.argmax()) if temperature<=0 else int(torch.multinomial(masses.softmax(-1),1,generator=generator))
        routes=routes.masked_fill(~(mask if group==0 else ~mask),-torch.inf)
        scores=grouped_logsumexp(routes,self.semantic_route_families-1,groups=31)
        return self.sample_event_family(family_logits=scores,temperature=temperature,generator=generator),group==0

    def event_family_targets(self,batch):
        from .stream_actions import family_masks_from_atoms
        shapes=batch['target_shape'].clamp_min(0)
        literal=family_masks_from_atoms(batch['target_details'],self.semantic_atom_families)
        return torch.where(shapes.gt(0),self.semantic_shape_families[shapes],literal)

    @torch.no_grad()
    def initialize_event_family_prior(self,family_counts,literal_counts=None):
        """Initialize new family probabilities from authored weighted events.

        The literal-vs-regular odds are data based, with the parent's shape
        bias scale retained as a baseline. This is a learned classifier
        initialization, not a generation cap or a desired Touch percentage.
        """
        counts=torch.as_tensor(family_counts,device=self.active_event_family_head.bias.device,dtype=torch.float32).reshape(31)
        prior=(counts+.5).log()
        self.active_event_family_head.bias.copy_(prior-prior.logsumexp(-1))
        if literal_counts is None:return
        literals=torch.as_tensor(literal_counts,device=counts.device,dtype=torch.float32).reshape(31)
        for family in range(1,32):
            regular=self.semantic_shape_families.eq(family)
            if regular.any():
                regular_mass=(counts[family-1]-literals[family-1]).clamp_min(0)
                reference=self.shape_head.bias[regular].float().logsumexp(0)-self.shape_head.bias[0].float()
                self.literal_route_head.bias[family-1].copy_(
                    (literals[family-1]+.5).log()-(regular_mass+.5*regular.sum()).log()+reference)

    @torch.no_grad()
    def initialize_literal_stop_prior(self,block_counts,extra_atom_counts):
        """Fit each type's STOP hazard after its mandatory first atom."""
        if not hasattr(self,'literal_stop_head'):return
        counts=torch.as_tensor(block_counts,device=self.literal_stop_head.bias.device,dtype=torch.float32).reshape(5)
        extras=torch.as_tensor(extra_atom_counts,device=counts.device,dtype=torch.float32).reshape(5)
        self.literal_stop_head.bias.copy_((counts+.5).log()-(extras+.5).log())

    def literal_block_nll(self,contexts,targets,families=None,groups=None,*,return_stop=False):
        """Exact typed-block NLL with a separately learned continuation law.

        All real atoms remain supervised. Stopping a Touch block can be
        learned before TouchHold has appeared, because the next block owns
        that type. CPU packing keeps GPU work proportional to real blocks.
        """
        from .stream_actions import family_masks_from_atoms
        from .stream_literal import tensor_literal_type_blocks
        if families is None:families=family_masks_from_atoms(targets,self.semantic_atom_families)
        if groups is None:groups=tensor_literal_type_blocks(targets,self.semantic_atom_families)
        result=contexts.sum(-1)*0.
        stop_result=contexts.sum(-1)*0.
        for type_id,(rows,target,prefix) in enumerate(groups):
            if not rows.numel() or not self._literal_type_ids[type_id]:continue
            ids=getattr(self,f'literal_type_atoms_{type_id}')
            base=contexts.index_select(0,rows)+self.literal_family_embedding(families.index_select(0,rows))
            present=prefix.gt(0)
            earlier=(self.atom_embedding(prefix.clamp_min(0))*present[...,None]).sum(-2)/present.sum(-1).clamp_min(1)[...,None]
            base=base+earlier+self.literal_type_embedding.weight[type_id]
            prior=self.atom_embedding(target.clamp_min(0))
            shifted=torch.cat((self.literal_start.expand(base.size(0),1,-1),prior[:,:-1]),1)
            hidden,_=self.literal_cell(shifted,base[None])
            logits=F.linear(hidden,self.literal_head.weight.index_select(0,ids),self.literal_head.bias.index_select(0,ids))
            local=self.literal_type_local[target.clamp_min(0)]
            identity=F.cross_entropy(logits.flatten(0,1).float(),local.flatten(),reduction='none').reshape_as(target)
            stop=F.linear(hidden,self.literal_stop_head.weight[type_id:type_id+1],
                self.literal_stop_head.bias[type_id:type_id+1]).squeeze(-1).float()
            hazard=F.binary_cross_entropy_with_logits(stop,target.eq(0).float(),reduction='none')
            valid_block=target.gt(0).any(-1)
            later=torch.arange(target.size(1),device=target.device)[None].gt(0)
            stop_nll=(hazard*(target.ge(0)&later)).sum(-1)*valid_block
            nll=(identity*target.gt(0)).sum(-1)*valid_block+stop_nll
            result=result.index_add(0,rows,nll)
            stop_result=stop_result.index_add(0,rows,stop_nll)
        return (result,stop_result) if return_stop else result

    def _decode_literal_blocks(self,context,family,choose,temperature):
        details=[];used_heads=0
        for type_id,bit in enumerate((1,2,4,8,16)):
            if not family&bit:continue
            ids=getattr(self,f'literal_type_atoms_{type_id}')
            if not ids.numel():return None
            earlier=(self.atom_embedding.weight[details].mean(0,keepdim=True) if details else torch.zeros_like(context))
            base=context+self.literal_family_embedding.weight[family]+self.literal_type_embedding.weight[type_id]+earlier
            hidden=base[None];previous=self.literal_start.reshape(1,1,-1);count=0
            for _ in range(self.config.max_event_atoms+1):
                step,hidden=self.literal_cell(previous,hidden)
                available=(self.literal_head_bits.index_select(0,ids)&used_heads).eq(0)
                if not bool(available.any()):
                    if count:break
                    return None
                if count:
                    stop=F.linear(step[:,0],self.literal_stop_head.weight[type_id:type_id+1],
                        self.literal_stop_head.bias[type_id:type_id+1]).reshape(())
                    hazard=torch.stack((stop,torch.zeros_like(stop)))
                    # Positive detail temperature does not change event
                    # cardinality. The binary hazard stays calibrated at T=1.
                    if choose(hazard,sampling_temperature=0. if temperature<=0 else 1.)==0:break
                if len(details)>=self.config.max_event_atoms:return None
                logits=F.linear(step[:,0],self.literal_head.weight.index_select(0,ids),
                    self.literal_head.bias.index_select(0,ids)).masked_fill(~available[None],-torch.inf)
                atom=self._literal_type_ids[type_id][choose(logits)]
                details.append(atom);used_heads|=self._literal_bits[atom];count+=1
                previous=self.atom_embedding.weight[atom].reshape(1,1,-1)
            else:return None
        return details+[0] if details else None

    def literal_logits(self, contexts, targets, families=None):
        if self.config.event_family_policy!='legacy_v1':
            from .stream_actions import family_masks_from_atoms,prefix_union
            families=(family_masks_from_atoms(targets,self.semantic_atom_families) if families is None else families).clamp_min(1)
            contexts=contexts+self.literal_family_embedding(families)
        previous = self.atom_embedding(targets.clamp_min(0))
        shifted = torch.cat((self.literal_start.expand(contexts.size(0), 1, -1), previous[:, :-1]), 1)
        hidden, _ = self.literal_cell(shifted, contexts[None])
        logits=self.literal_head(hidden)
        if self.config.event_family_policy!='legacy_v1':
            atom_types=self.semantic_atom_families
            types=atom_types[targets.clamp(0,atom_types.numel()-1)]*targets.gt(0)
            bits=types.new_tensor([1,2,4,8,16])
            seen=prefix_union(types,bits)
            allowed=(atom_types[None,None].gt(0)&
                     (atom_types[None,None]&families[:,None,None]).eq(atom_types[None,None]))
            allowed=allowed.expand(-1,targets.size(1),-1).clone()
            allowed[...,0]=seen.eq(families[:,None])
            # Type conditioning defines this output factorization. Physical
            # playability remains the generation guard's responsibility; do
            # not turn an authored token into a fixed 1e4 teacher loss here.
            logits=logits.float().masked_fill(~allowed,-1e4)
        return logits

    def regular_detail_log_probs(self,context,shape,phases,targets,duration_context=None):
        """Exact teacher-forced detail sequence likelihood at each real phase."""
        count=context.size(0)
        if self.config.detail_policy=='independent_v1':
            result=context.new_zeros(count)
            for slot,head in enumerate(self.detail_heads[shape]):
                logits=self.duration_detail_logits(head(context),shape,slot,duration_context)
                result=result-F.cross_entropy(logits.float(),targets[:,slot],reduction='none')
            return result
        hidden=context+self.shape_embedding.weight[shape]+self.phase_embedding(phases)
        previous=self.detail_start.expand(count,-1)
        result=context.new_zeros(count,dtype=torch.float32)
        for slot,embedding in enumerate(self.detail_embeddings[shape]):
            hidden=self.detail_cell(previous,hidden)
            logits=F.linear(self.detail_output(hidden),embedding.weight,self.detail_bias[shape][slot])
            logits=self.duration_detail_logits(logits,shape,slot,duration_context)
            result=result-F.cross_entropy(logits.float(),targets[:,slot],reduction='none')
            previous=self.detail_input(embedding(targets[:,slot]))
        return result

    def loss(self, output, batch):
        return stream_loss(self, output, batch)

    @torch.no_grad()
    def decode_event(self, context, *, reference_phase=0, reference_phase_mask=None,temperature=1., generator=None,
                     blocked_shape_phases=None,shape_override=None,phase_override=None,motion_phase_distribution=None,button_load=None,
                     family_context=None,family_logits=None,family_override=None,
                     motion_keep_logit=None,motion_reference_shape=None,template_keep_logit=None,
                     template_reference_shape=None,template_keep_override=None,duration_context=None):
        """Decode one entire NOTE; callers choose its time before this method."""
        context = context.reshape(-1, self.config.d_model)
        if context.size(0) != 1:
            refs = torch.as_tensor(reference_phase).expand(context.size(0)).tolist()
            masks=[None]*len(refs) if reference_phase_mask is None else torch.as_tensor(reference_phase_mask).reshape(-1,8).expand(context.size(0),-1)
            motions=[None]*len(refs) if motion_phase_distribution is None else motion_phase_distribution.reshape(-1,3,8).expand(context.size(0),-1,-1)
            loads=[None]*len(refs) if button_load is None else button_load.reshape(-1,8,4).expand(context.size(0),-1,-1)
            family_contexts=[None]*len(refs) if family_context is None else family_context.reshape(-1,self.config.d_model).expand(context.size(0),-1)
            family_scores=[None]*len(refs) if family_logits is None else family_logits.reshape(-1,31).expand(context.size(0),-1)
            overrides=[None]*len(refs) if family_override is None else torch.as_tensor(family_override).reshape(-1).expand(context.size(0)).tolist()
            def per_row(value):return [None]*len(refs) if value is None else torch.as_tensor(value).reshape(-1).expand(len(refs)).tolist()
            decisions=list(zip(per_row(motion_keep_logit),per_row(motion_reference_shape),per_row(template_keep_logit),per_row(template_reference_shape),per_row(template_keep_override)))
            return [self.decode_event(row, reference_phase=ref,reference_phase_mask=mask,
                temperature=temperature, generator=generator,blocked_shape_phases=blocked_shape_phases,
                shape_override=shape_override,phase_override=phase_override,motion_phase_distribution=motion,button_load=load,
                family_context=family_ctx,family_logits=family_score,family_override=override,
                motion_keep_logit=decision[0],motion_reference_shape=decision[1],template_keep_logit=decision[2],template_reference_shape=decision[3],template_keep_override=decision[4],
                duration_context=({key:value if key=='cache' else value[row_index:row_index+1]
                    for key,value in duration_context.items()} if duration_context is not None else None))
                for row_index,(row, ref,mask,motion,load,family_ctx,family_score,override,decision) in
                    enumerate(zip(context, refs,masks,motions,loads,family_contexts,family_scores,overrides,decisions))]
        def choose(logits,sampling_temperature=None):
            scale=temperature if sampling_temperature is None else sampling_temperature
            if scale <= 0:
                return int(logits.argmax(-1).item())
            return int(torch.multinomial((logits.float()/scale).softmax(-1), 1, generator=generator).item())
        blocked=set(blocked_shape_phases or ())
        available=[any((shape,phase) not in blocked for phase in (range(8) if shape else (0,))) for shape in range(self.codec.num_shapes)]
        semantic=self.config.event_family_policy!='legacy_v1'
        family=None
        if semantic:
            if (self.config.transition_policy in ('music_exit_v1','music_exit_v2','music_exit_v3') and template_keep_logit is not None
                    and family_override is None and shape_override is None):
                family_override,template_keep_override=self.sample_event_intent(context,family_logits=family_logits,
                    template_keep_logit=template_keep_logit,template_reference_shape=template_reference_shape,
                    temperature=temperature,generator=generator)
                if family_override is None:return None
            blocked_families={entry[2] for entry in blocked if len(entry)==3 and entry[:2]==(0,0)}
            route_available=list(available)
            if shape_override is not None:
                shape=int(shape_override)
                if not 0<=shape<self.codec.num_shapes:raise ValueError('Invalid shape override')
                route_available=[allowed and index==shape for index,allowed in enumerate(route_available)]
            routes=self.event_route_log_probs(context,temperature=temperature,
                available_shapes=route_available,blocked_families=blocked_families,
                family_context=family_context,family_logits=family_logits,family_override=family_override,
                template_keep_logit=template_keep_logit,template_reference_shape=template_reference_shape,
                template_keep_override=template_keep_override)
            if not bool(torch.isfinite(routes).any()):return None
            if temperature<=0:
                # Select the family before its route so its number of shapes
                # cannot change which semantic category wins greedily.
                masses=grouped_logsumexp(routes,self.semantic_route_families-1,groups=31)
                best_family=masses.argmax(-1,keepdim=True)+1
                routes=routes.masked_fill(self.semantic_route_families!=best_family,-torch.inf)
            route=choose(routes,sampling_temperature=0. if temperature<=0 else 1.)
            if route<self.codec.num_shapes-1:
                shape=route+1
                family=self._semantic_shape_families[shape]
            else:
                shape=0;family=route-(self.codec.num_shapes-1)+1
        elif shape_override is not None:
            shape=int(shape_override)
            if not 0<=shape<self.codec.num_shapes:raise ValueError('Invalid shape override')
            if not available[shape]:return None
        else:
            if not any(available):return None
            shape_logits=self.shape_head(context).masked_fill(~torch.tensor(available,device=context.device)[None],-torch.inf)
            shape=choose(shape_logits)
        if phase_override is not None:
            phase=int(phase_override)
            if not 0<=phase<8 or (shape==0 and phase!=0):raise ValueError('Invalid phase override')
            if (shape,phase) in blocked:return None
        elif shape:
            phase_logits=self.event_phase_logits(context,torch.tensor([shape],device=context.device),
                torch.tensor([reference_phase],device=context.device),None if reference_phase_mask is None else
                torch.as_tensor(reference_phase_mask,device=context.device).reshape(1,8),
                None if motion_phase_distribution is None else motion_phase_distribution.to(context.device).reshape(1,3,8),
                None if button_load is None else button_load.to(context.device).reshape(1,8,4),
                motion_keep_logit=None if motion_keep_logit is None else torch.as_tensor(motion_keep_logit,device=context.device).reshape(1),
                motion_reference_shape=None if motion_reference_shape is None else torch.as_tensor(motion_reference_shape,device=context.device).reshape(1))
            phase_logits=phase_logits.masked_fill(torch.tensor([(shape,p) in blocked for p in range(8)],device=context.device)[None],-torch.inf)
            if (self.config.transition_policy in ('music_exit_v1','music_exit_v2','music_exit_v3') and motion_keep_logit is not None
                    and motion_reference_shape is not None and motion_phase_distribution is not None):
                from .stream_transition import sample_two_group
                keep=self.motion_keep_mask(torch.tensor(shape,device=context.device),motion_phase_distribution,
                    torch.as_tensor(motion_reference_shape,device=context.device))
                phase=(sample_two_group(phase_logits,keep,temperature=temperature,generator=generator)
                    if keep is not None else choose(phase_logits))
            else:phase=choose(phase_logits)
        else:phase=0
        if shape:
            if self.config.detail_policy=='independent_v1':
                details = [choose(self.duration_detail_logits(head(context),shape,slot,duration_context))
                           for slot,head in enumerate(self.detail_heads[shape])]
            else:
                details=[]
                hidden=context+self.shape_embedding.weight[shape]+self.phase_embedding.weight[phase]
                previous=self.detail_start.expand(1,-1)
                for slot,embedding in enumerate(self.detail_embeddings[shape]):
                    hidden=self.detail_cell(previous,hidden)
                    logits=F.linear(self.detail_output(hidden),embedding.weight,self.detail_bias[shape][slot])
                    logits=self.duration_detail_logits(logits,shape,slot,duration_context)
                    detail=choose(logits);details.append(detail)
                    previous=self.detail_input(embedding.weight[detail][None])
        elif semantic and hasattr(self,'literal_stop_head'):
            details=self._decode_literal_blocks(context,family,choose,temperature)
            if details is None:return None
        else:
            details = []
            used_heads=seen_types=0
            literal_context=context+self.literal_family_embedding.weight[family] if semantic else context
            hidden = literal_context[None]
            previous = self.literal_start.reshape(1, 1, -1)
            for slot in range(self.config.max_event_atoms+1):
                if semantic and slot==self.config.max_event_atoms:
                    if seen_types!=family:return None
                    details.append(0);break
                step, hidden = self.literal_cell(previous, hidden)
                logits = self.literal_head(step[:, 0])
                # A NOTE is a set of physical onsets. A second atom may change
                # modifiers/duration, but cannot add the same head twice.
                logits=logits.masked_fill((self.literal_head_bits & used_heads).ne(0)[None],-torch.inf)
                if not details:
                    logits[:, 0] = -torch.inf  # NOTE cannot be an empty event.
                if semantic:
                    types=self.semantic_atom_families
                    supported=types.gt(0)&(types&family).eq(types)
                    supported[0]=seen_types==family
                    logits=logits.masked_fill(~supported[None],-torch.inf)
                    if not bool(torch.isfinite(logits).any()):return None
                atom = choose(logits)
                if atom == 0:
                    details.append(0)
                    break
                if slot == self.config.max_event_atoms:
                    details.append(0)
                    break
                details.append(atom)
                used_heads |= self._literal_bits[atom]
                if semantic:seen_types|=self._semantic_atom_families[atom]
                previous = self.atom_embedding.weight[atom].reshape(1, 1, -1)
        text = self.codec.decode(shape, phase, details)
        encoded = self.codec.encode(text)
        result={'shape': shape, 'phase': phase, 'details': details,
                'atoms': encoded['atoms'], 'text': text}
        if semantic:result['family']=family
        return result


def stream_loss(model, output, batch):
    """Joint sequence likelihood, uniform-time plan and optional recovery group."""
    weights = batch['loss_mask'].float()*batch.get('sample_weight', torch.ones_like(batch['loss_mask'])).float()
    # Dataset target weights total K per fixed time window. Renormalizing by
    # their sum here would cancel that balancing in homogeneous/single batches.
    denominator = float(batch.get('_loss_normalizer', weights.size(0)*model.config.rollout_steps))
    if denominator <= 0:raise ValueError('Stream loss normalizer must be positive')
    def reduce(values):
        return (values*weights).sum()/denominator
    kind = batch['target_kind']
    note = kind.eq(NOTE)
    kind_nll=F.cross_entropy(output['kind_logits'].flatten(0, 1), kind.flatten(), reduction='none').reshape_as(weights)
    kind_loss = reduce(kind_nll)
    # WAIT's duration is fixed by the stream contract and never sampled from
    # this head. Only NOTE targets supervise its conditional time distribution.
    if 'candidate_beats' in output:
        target_beats=batch.get('target_beat')
        if target_beats is None:target_beats=batch['step_beat']+model.delta_values[batch['target_delta'].clamp_min(0)]
        delta_targets=absolute_target_indices(output['candidate_beats'],target_beats,note&weights.gt(0))
    else:delta_targets=batch['target_delta'].clamp_min(0)
    delta_nll=F.cross_entropy(output['delta_logits'].flatten(0, 1), delta_targets.flatten(), reduction='none').reshape_as(weights)*note
    delta_loss = reduce(delta_nll)
    timing_grid_loss=delta_loss.new_zeros(())
    timing_within_grid_loss=delta_loss.new_zeros(())
    if 'timing_grid_ids' in output:
        groups=output['timing_grid_ids']
        group_log_probs=grouped_logsumexp(output['delta_logits'].float().log_softmax(-1),groups,
            groups=output.get('timing_group_count',len(model.timing_grid.grid_head.bias)))
        target_group=groups.expand_as(output['delta_logits']).gather(-1,delta_targets[...,None]).squeeze(-1)
        group_nll=-group_log_probs.gather(-1,target_group[...,None]).squeeze(-1)*note
        timing_grid_loss=reduce(group_nll)
        timing_within_grid_loss=delta_loss-timing_grid_loss
    family_loss=kind_loss.new_zeros(())
    family_full_loss=kind_loss.new_zeros(())
    family_audio_loss=kind_loss.new_zeros(())
    if model.config.event_family_policy!='legacy_v1':
        families=model.event_family_targets(batch)
        shapes=batch['target_shape'].clamp_min(0)
        routes=torch.where(shapes.gt(0),shapes-1,model.codec.num_shapes-2+families.clamp_min(1))
        shape_nll=-output['event_route_log_probs'].gather(-1,routes[...,None]).squeeze(-1)
        shape_nll=shape_nll.masked_fill(~(note&weights.gt(0)),0.)
        family_nll=F.cross_entropy(output['event_family_logits'].flatten(0,1),
            (families.clamp_min(1)-1).flatten(),reduction='none').reshape_as(weights)
        family_full_loss=reduce(family_nll*note)
        if model.config.event_family_policy in ('audio_semantic_v2','audio_prior_transition_v3'):
            audio_family_nll=F.cross_entropy(output['event_family_audio_logits'].flatten(0,1),
                (families.clamp_min(1)-1).flatten(),reduction='none').reshape_as(weights)
            alpha=model.config.event_family_audio_weight
            # Reallocate the existing family budget; do not stack a second
            # full family CE onto the configuration objective. Exit targets
            # below consume this same budgeted shape_nll.
            shape_nll=shape_nll+alpha*(audio_family_nll-family_nll).masked_fill(~(note&weights.gt(0)),0.)
            family_audio_loss=reduce(audio_family_nll*note)
            family_loss=(1-alpha)*family_full_loss+alpha*family_audio_loss
        else:family_loss=family_full_loss
    else:
        shape_nll=F.cross_entropy(output['shape_logits'].flatten(0, 1), batch['target_shape'].clamp_min(0).flatten(), reduction='none').reshape_as(weights)*note
    shape_loss = reduce(shape_nll)
    phase_mask = batch['target_phase_mask'].bool()
    regular = note & batch['target_shape'].gt(0) & weights.gt(0)
    # Give non-NOTE/fallback rows a finite dummy marginal before masking them.
    safe_mask = phase_mask | ~regular[..., None]
    phase_nll = -output['phase_logits'].float().log_softmax(-1).masked_fill(~safe_mask, -torch.inf).logsumexp(-1)
    phase_loss = reduce(phase_nll*regular)
    detail_sum = output['action_context'].sum()*0.
    context = output['action_context'].flatten(0, 1)
    details = batch['target_details'].flatten(0, 1)
    flat_weights = weights.flatten()
    detail_per_row=context.sum(-1)*0.
    # Collation already knows the target grouping on CPU. No device-to-host
    # shape discovery or dynamically sized boolean indexing in this hot path.
    sparse_groups = batch.get('_detail_phase_groups')
    groups = batch['_detail_groups'] if sparse_groups is None else sparse_groups
    for group in groups:
        shape, indices = group[:2]
        ctx = context.index_select(0, indices)
        duration_context=None
        if output.get('duration_audio_context') is not None:
            from .stream_duration import select_duration_context
            duration_context=select_duration_context(output['duration_audio_context'],indices)
        target, mass = details.index_select(0, indices), flat_weights.index_select(0, indices)
        if model.config.detail_policy=='independent_v1':
            for slot, head in enumerate(model.detail_heads[shape]):
                logits=model.duration_detail_logits(head(ctx),shape,slot,duration_context)
                slot_nll=F.cross_entropy(logits.float(), target[:, slot], reduction='none')
                detail_sum = detail_sum + (slot_nll*mass).sum()
                detail_per_row=detail_per_row.index_add(0,indices,slot_nll)
        else:
            if sparse_groups is None:
                phases=torch.arange(8,device=ctx.device)[None].expand(ctx.size(0),-1)
                expanded_duration=(select_duration_context(duration_context,
                    torch.arange(ctx.size(0),device=ctx.device).repeat_interleave(8))
                    if duration_context is not None else None)
                phase_details=model.regular_detail_log_probs(
                    ctx[:,None].expand(-1,8,-1).reshape(-1,ctx.size(-1)),shape,phases.reshape(-1),
                    target[:,None].expand(-1,8,-1).reshape(-1,target.size(-1)),
                    duration_context=expanded_duration).reshape(-1,8)
            else:
                rows, phases = group[2:]
                packed = model.regular_detail_log_probs(ctx.index_select(0, rows), shape,
                    phases, target.index_select(0, rows),duration_context=(select_duration_context(duration_context,rows)
                        if duration_context is not None else None))
                phase_details = packed.new_full((ctx.size(0), 8), -torch.inf).index_put(
                    (rows, phases), packed)
            phase_log_probs=output['phase_logits'].flatten(0,1).index_select(0,indices).float().log_softmax(-1)
            allowed=phase_mask.flatten(0,1).index_select(0,indices)
            joint=-(phase_log_probs+phase_details).masked_fill(~allowed,-torch.inf).logsumexp(-1)
            independent=phase_nll.flatten().index_select(0,indices)
            # phase_loss + this conditional increment is exactly the joint
            # orbit likelihood; no separately averaged canonical-detail loss.
            detail_sum=detail_sum+((joint-independent)*mass).sum()
            detail_per_row=detail_per_row.index_add(0,indices,joint-independent)
    literal = batch['_literal_indices']
    literal_stop_loss=detail_sum.new_zeros(())
    if literal.numel():
        target = details.index_select(0, literal)
        literal_families=(families.flatten().index_select(0,literal) if model.config.event_family_policy!='legacy_v1' else None)
        if hasattr(model,'literal_stop_head'):
            literal_nll,stop_nll=model.literal_block_nll(context.index_select(0,literal),target,
                literal_families,batch.get('_literal_type_blocks'),return_stop=True)
            literal_stop_loss=(stop_nll*flat_weights.index_select(0,literal)).sum()/denominator
        else:
            logits = model.literal_logits(context.index_select(0, literal), target,literal_families)
            active = target.ge(0)
            ce = F.cross_entropy(logits.flatten(0, 1).float(), target.clamp_min(0).flatten(), reduction='none').reshape_as(target)
            literal_nll=(ce*active).sum(-1)
        detail_sum = detail_sum + (literal_nll*flat_weights.index_select(0, literal)).sum()
        detail_per_row=detail_per_row.index_add(0,literal,literal_nll)
    detail_loss = detail_sum/denominator
    plan_loss = kind_loss.new_zeros(())
    plan_rate_loss=kind_loss.new_zeros(())
    plan_blank_loss=kind_loss.new_zeros(())
    if 'plan_logits' in output and 'audio_plan_target' in batch:
        logits, target = output['plan_logits'].float(), batch['audio_plan_target'].float()
        blank = F.binary_cross_entropy_with_logits(logits[..., 0], target[..., 0], reduction='none')
        # Predict counts for the same real cell exposure as the dataset labels.
        rates = F.softplus(logits[..., 1:])
        if model.config.plan_target_units=='rates_per_second_v2':
            exposure=batch.get('audio_plan_exposure',output.get('plan_exposure'))
            if exposure is None:raise ValueError('Rate plan supervision requires real audio cell exposure')
            rates=rates*exposure.float()[...,None]
        counts = target[..., 1:]
        poisson = rates-counts*rates.clamp_min(1e-8).log()+torch.lgamma(counts+1.)
        mask = batch['audio_plan_weight'].float()
        blank_mask=batch.get('audio_plan_blank_weight',mask).float()
        rate_mask=mask
        if (model.config.plan_target_units=='rates_per_second_v2'
                and model.config.plan_loss_normalization=='per_second_v2'):
            # The dataset already stores dt / known_song_seconds (plus chart
            # and chunk weight). Poisson's expected count also contains dt.
            # Integrate count NLL per known second, not another dt-weighted
            # cell mean, so changing audio resolution does not shrink learning.
            safe_exposure=exposure.float().masked_fill(exposure<=0,1.)
            rate_mask=(mask/safe_exposure).masked_fill(exposure<=0,0.)
        plan_blank_loss=(blank*blank_mask).sum()/max(1,mask.size(0))
        plan_rate_loss=(poisson.sum(-1)*rate_mask).sum()/max(1,mask.size(0))
        plan_loss=plan_blank_loss+plan_rate_loss
    time_loss = kind_loss+delta_loss
    configuration_loss = shape_loss+phase_loss+detail_loss
    recovery = output.get('recovery_loss', kind_loss.new_zeros(()))
    base_loss=time_loss+configuration_loss+model.config.plan_weight*plan_loss
    exit_mask=(batch.get('transition_exit_mask',torch.zeros_like(note)).bool()
               &batch.get('transition_exit_labels',torch.zeros_like(note)).bool()
               &(note|kind.eq(WAIT)|kind.eq(EOS))&weights.gt(0))
    joint_nll=kind_nll+delta_nll+shape_nll+phase_nll*regular+detail_per_row.reshape_as(weights)
    transition_exit_loss=model.config.transition_exit_weight*reduce(joint_nll*exit_mask)
    phrase_loss = kind_loss.new_zeros(())
    if 'phrase_prediction' in output:
        from .stream_phrase_model import phrase_auxiliary_loss
        phrase_loss = phrase_auxiliary_loss(output, batch)
    total = base_loss+transition_exit_loss+model.config.recovery_weight*recovery+model.config.phrase_loss_weight*phrase_loss
    continuation_metrics={}
    if model.config.transition_policy in ('music_exit_v1','music_exit_v2','music_exit_v3'):
        # These are components/diagnostics of the existing joint NLL. They
        # are not another loss added to it and cannot double the exit budget.
        with torch.no_grad():
            def describe(name,logits,keep,truth,active):
                scores=logits.float().log_softmax(-1)
                keep=keep.expand_as(scores)
                available=torch.isfinite(scores)&scores.gt(-9999)
                eligible=(keep&available).any(-1)&(~keep&available).any(-1)&active
                target_keep=(truth&keep).any(-1)
                target_exit=(truth&~keep).any(-1)
                masses=torch.stack((scores.masked_fill(~keep,-torch.inf).logsumexp(-1),
                    scores.masked_fill(keep,-torch.inf).logsumexp(-1)),-1)
                accepted=torch.stack((target_keep,target_exit),-1)
                nll=-masses.masked_fill(~accepted,-torch.inf).logsumexp(-1)
                continuation_metrics[name+'_decision_loss']=reduce(nll.masked_fill(~eligible,0.))
                continuation_metrics[name+'_exit_targets']=(eligible&target_exit&~target_keep).sum().float()
            truth=F.one_hot(delta_targets,output['delta_logits'].size(-1)).bool()
            describe('time',output['delta_logits'],output['timing_keep_mask'],truth,note&weights.gt(0))
            motion=model.motion_keep_mask(batch['target_shape'],output.get('motion_phase_distribution'),output['motion_reference_shape'])
            if motion is not None:describe('motion',output['phase_logits'],motion,phase_mask,regular)
            routes=torch.where(batch['target_shape'].gt(0),batch['target_shape']-1,
                model.codec.num_shapes-2+families.clamp_min(1))
            keep=model.template_keep_mask(output['template_reference_shape'],output['event_route_log_probs'])
            describe('template',output['event_route_log_probs'],keep,
                F.one_hot(routes,output['event_route_log_probs'].size(-1)).bool(),note&weights.gt(0))
    metrics = {name: value.detach() for name, value in dict(loss=total, time_loss=time_loss,
        kind_loss=kind_loss, delta_loss=delta_loss, timing_grid_loss=timing_grid_loss,
        timing_within_grid_loss=timing_within_grid_loss, configuration_loss=configuration_loss,
        shape_loss=shape_loss, event_family_loss=family_loss,
        event_family_full_loss=family_full_loss,event_family_audio_loss=family_audio_loss,
        event_route_conditional_loss=shape_loss-family_loss, phase_loss=phase_loss, detail_loss=detail_loss,
        literal_stop_loss=literal_stop_loss,
        plan_loss=plan_loss,plan_rate_loss=plan_rate_loss,plan_blank_loss=plan_blank_loss,
        base_loss=base_loss,phrase_loss=phrase_loss,phrase_weighted_loss=model.config.phrase_loss_weight*phrase_loss,
        transition_exit_loss=transition_exit_loss,exit_count=exit_mask.sum(),
        exit_note_count=(exit_mask&note).sum(),exit_wait_count=(exit_mask&kind.eq(WAIT)).sum(),
        exit_eos_count=(exit_mask&kind.eq(EOS)).sum(),
        recovery_loss=model.config.recovery_weight*recovery,**continuation_metrics).items()}
    return total, metrics


# Explicit name for callers that distinguish this model from legacy Composer.
EventSetStreamComposer = StreamComposer
