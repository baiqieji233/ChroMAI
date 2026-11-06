"""Shared exact absolute onset support and learned metrical probability mass.

The catalog is a static training vocabulary of absolute beat phases. It never
depends on a row's target. Each phase has two forward occurrences, so a cursor
off the authored grid can select the original grid without an invented delta
class, rounding a generated event, or a second decoder pass.
"""
from fractions import Fraction
import math
import torch
from torch import nn
from torch.nn import functional as F

TIMING_POLICIES = ('absolute_grid_v1', 'relative_delta_v1')
TIMING_GROUP_POLICIES = ('phase_gap_v2', 'phase_only_v1')
TIMING_AUDIO_POLICIES = ('off','candidate_pool_v1','query_pool_v2','query_pool_v3')
QUERY_INTERACTION_GAIN = 8.
GRID_DENOMINATORS = (1, 2, 4, 8, 16, 32, 3, 6, 12, 24)
GRID_FAMILIES = (*map(str, GRID_DENOMINATORS), 'other')


def ternary_probability_stages(raw, mode, curved, group_ids):
    """Read stage probability mass; never modify support or sampling."""
    ids=group_ids.remainder(len(GRID_FAMILIES))
    ternary=(ids>=GRID_DENOMINATORS.index(3))&(ids<=GRID_DENOMINATORS.index(24))
    return (torch.stack((raw,mode,curved)).float().softmax(-1)*ternary).sum(-1)


def timing_phases(delta_beats, explicit=()):
    return tuple(sorted({Fraction(str(x)) % 1 for x in (*delta_beats, *explicit)} | {Fraction(0)}))


def phase_group(value):
    denominator = (Fraction(str(value)) % 1).denominator
    return GRID_DENOMINATORS.index(denominator) if denominator in GRID_DENOMINATORS else len(GRID_DENOMINATORS)


def timing_group(phase, gap, policy='phase_gap_v2'):
    """Authored label grouping shared with training-prior initialization."""
    if policy not in TIMING_GROUP_POLICIES:raise ValueError('Unsupported timing group policy')
    return phase_group(phase)+(len(GRID_FAMILIES) if policy=='phase_gap_v2' and Fraction(str(gap))>=1 else 0)


def encode_history_gap(vocab, gap, policy='relative_delta_v1'):
    """An OOV exact gap uses no categorical embedding in the new policy.

    Its exact value still enters history_gap_beats and the continuous clock.
    Relative-only checkpoints retain their strict vocabulary contract.
    """
    try:
        return vocab.encode(gap)
    except ValueError:
        value = Fraction(str(gap))
        if policy != 'absolute_grid_v1' or not 0 <= value <= 2:
            raise
        return 0


def current_timing_deltas(model, output, cursor):
    """Exact Fraction candidates in the same order as the model tensors."""
    if getattr(model.config, 'timing_output_policy', 'relative_delta_v1') != 'absolute_grid_v1':
        return tuple(Fraction(str(x)) for x in model.config.delta_beats)
    phases = model.timing_grid.phase_catalog
    origin = cursor.numerator // cursor.denominator
    first = tuple(Fraction(origin) + phase + (1 if Fraction(origin) + phase <= cursor else 0)
                  for phase in phases)
    return (Fraction(0), Fraction(0), *(beat-cursor for beat in first), *(beat+1-cursor for beat in first))


def grouped_logsumexp(scores, group_ids, groups=len(GRID_FAMILIES)):
    ids = group_ids.expand_as(scores)
    maximum = scores.new_full((*scores.shape[:-1], groups), -torch.inf)
    maximum = maximum.scatter_reduce(-1, ids, scores, reduce='amax', include_self=True)
    gathered = maximum.gather(-1, ids)
    finite = torch.isfinite(scores) & torch.isfinite(gathered)
    shifted = torch.where(finite, scores-gathered, torch.zeros_like(scores))
    summed = scores.new_zeros(maximum.shape).scatter_add(-1, ids, shifted.exp()*finite)
    return torch.where(summed.gt(0), maximum+summed.clamp_min(1e-30).log(), -torch.inf)


def temper_grid_logits(logits, temperature, group_ids, group_count=None):
    """Temperature changes grid and within-grid decisions without size bias."""
    scores = logits.float()
    if temperature <= 0:
        return scores
    if group_count is None:group_count=2*len(GRID_FAMILIES) if group_ids.ndim>1 else len(GRID_FAMILIES)
    if scores.ndim==1 and group_ids.ndim==3 and group_ids.size(0)==1:
        # Single-step generation/recovery consume the final causal position.
        group_ids=group_ids[0,-1]
    groups = grouped_logsumexp(scores, group_ids, groups=group_count)
    macro = (groups/temperature).log_softmax(-1)
    details = scores/temperature
    conditional_lse = grouped_logsumexp(details, group_ids, groups=group_count)
    values = details-conditional_lse.gather(-1, group_ids.expand_as(scores))
    result = values+macro.gather(-1, group_ids.expand_as(scores))
    return result.masked_fill(~torch.isfinite(scores), -torch.inf)


class AbsoluteTimingHead(nn.Module):
    def __init__(self, width, delta_beats, phases=(), *, audio_policy='off',group_policy='phase_only_v1'):
        super().__init__()
        if audio_policy not in TIMING_AUDIO_POLICIES:
            raise ValueError('Unsupported timing grid audio policy')
        self.audio_policy=audio_policy
        if group_policy not in TIMING_GROUP_POLICIES:raise ValueError('Unsupported timing group policy')
        self.group_policy=group_policy
        self.num_groups=len(GRID_FAMILIES)*(2 if group_policy=='phase_gap_v2' else 1)
        self.grid_head = nn.Linear(width, len(GRID_FAMILIES))
        self.feature_key = nn.Linear(14, 32, bias=False)
        nn.init.zeros_(self.grid_head.weight); nn.init.zeros_(self.grid_head.bias)
        nn.init.zeros_(self.feature_key.weight)
        if group_policy=='phase_gap_v2':
            self.bucket_head=nn.Linear(width,self.num_groups)
            self.interval_candidate_projection=nn.Linear(2,1,bias=False)
            self.interval_group_projection=nn.Linear(2,1,bias=False)
            for module in (self.bucket_head,self.interval_candidate_projection,self.interval_group_projection):
                nn.init.zeros_(module.weight)
                if module.bias is not None:nn.init.zeros_(module.bias)
        if audio_policy in ('candidate_pool_v1','query_pool_v2','query_pool_v3'):
            self.audio_projection=nn.Linear(32,1,bias=False)
            nn.init.zeros_(self.audio_projection.weight)
        if audio_policy in ('query_pool_v2','query_pool_v3'):
            self.query_projection=nn.Linear(1,1,bias=False)
            nn.init.zeros_(self.query_projection.weight)
        if audio_policy=='query_pool_v3':
            self.query_interaction_projection=nn.Linear(32,1,bias=False)
            nn.init.zeros_(self.query_interaction_projection.weight)
        self.register_buffer('phases', torch.empty(0, dtype=torch.float64), persistent=False)
        self.register_buffer('group_ids', torch.empty(0, dtype=torch.long), persistent=False)
        self.configure(timing_phases(delta_beats, phases))

    def configure(self, phases):
        self.phase_catalog = timing_phases((), phases)
        self.phases = torch.tensor([float(x) for x in self.phase_catalog], dtype=torch.float64, device=self.phases.device)
        ids = [phase_group(x) for x in self.phase_catalog]
        self.group_ids = torch.tensor([0, 0, *ids, *ids], dtype=torch.long, device=self.phases.device)

    def candidates(self, cursor):
        origin = cursor.floor()[..., None]
        phase = self.phases.to(cursor)
        anchored = origin+phase
        first = anchored+(anchored <= cursor[..., None]+1e-9).to(cursor)
        beats = torch.cat((cursor[..., None].expand(*cursor.shape, 2), first, first+1), -1)
        return beats, (beats-cursor[..., None]).clamp_min(0)

    def candidate_groups(self,gaps):
        if self.group_policy=='phase_only_v1':return self.group_ids
        # Subtracting float64 non-binary phases may yield 1-1ULP for an
        # exact one-beat move. Match candidate construction tolerance.
        return self.group_ids+gaps.ge(1.-1e-9).long()*len(GRID_FAMILIES)

    def _group_means(self,values,allowed,counts,group_ids):
        ids=group_ids.expand(*values.shape[:-1])[...,None].expand_as(values)
        pooled=values.new_zeros((*values.shape[:-2],self.num_groups,values.size(-1)))
        pooled=pooled.scatter_add(-2,ids,values*allowed[...,None])
        return pooled/counts.clamp_min(1)[...,None]

    @staticmethod
    def _center_groups(pooled,counts):
        valid=counts.gt(0)
        common=(pooled*valid[...,None]).sum(-2,keepdim=True)/valid.sum(-1,keepdim=True).clamp_min(1)[...,None]
        return (pooled-common)*valid[...,None]

    def pooled_audio_logits(self, candidate_audio, allowed, counts,group_ids=None):
        """Direct candidate music evidence for the grid decision.

        Each grid receives one count-normalized 32d readout, independently of
        how many phases it owns. Centering removes the common song/background
        vector. This is separate from the within-grid interval scores.
        """
        keys=candidate_audio.float()
        group_ids=self.group_ids if group_ids is None else group_ids
        pooled=self._group_means(keys,allowed,counts,group_ids)
        centered=self._center_groups(pooled,counts)
        return self.audio_projection(centered.to(self.audio_projection.weight.dtype)).float().squeeze(-1)

    def query_audio_logits(self,audio_scores,allowed,counts,group_ids):
        """Context/music compatibility can change macro timing decisions.

        The same scores already enter the conditional choice inside a grid.
        That contribution cancels for singleton grids, so its count-normalized
        log-mean-exp must separately reach the macro classifier. Repeated
        identical candidates have identical evidence rather than extra mass.
        """
        values=audio_scores.float().masked_fill(~allowed,-torch.inf)
        pooled=grouped_logsumexp(values,group_ids,groups=self.num_groups)
        pooled=(pooled-counts.clamp_min(1).float().log()).masked_fill(counts.eq(0),0.)
        centered=self._center_groups(pooled[...,None],counts)
        return self.query_projection(centered.to(self.query_projection.weight.dtype)).float().squeeze(-1)

    def query_interaction_logits(self,query,candidate_audio,allowed,counts,group_ids):
        """Learn the macro readout of all 32 existing query/key interactions.

        Reuse the same candidate keys and query as the dot-product score.
        A count-normalized mean keeps duplicated candidates from adding
        evidence. Zero initialization preserves the V2 probability contract;
        the fixed gain scales learning rather than imposing a rhythm rule.
        """
        interaction=query.float()[...,None,:]*candidate_audio.float()/math.sqrt(32)
        interaction=interaction.masked_fill(~allowed[...,None],0.)
        pooled=self._group_means(interaction,allowed,counts,group_ids)
        centered=self._center_groups(pooled,counts)
        return QUERY_INTERACTION_GAIN*self.query_interaction_projection(
            centered.to(self.query_interaction_projection.weight.dtype)).float().squeeze(-1)

    def forward(self, hidden, cursor, relative_logits, relative_values, query, audio_scores, allowed,
                *, candidate_audio=None,candidate_expected_count=None,return_intensity=False):
        beats, gaps = self.candidates(cursor)
        group_ids=self.candidate_groups(gaps)
        # Preserve the old model's useful interval evidence only within a
        # chosen absolute-grid family. It cannot steal mass from another grid.
        values = relative_values.to(cursor)
        right = torch.searchsorted(values.contiguous(), gaps.contiguous(), right=True).clamp(1, values.numel()-1)
        left = right-1
        lo, hi = values[left], values[right]
        mix = ((gaps-lo)/(hi-lo).clamp_min(1e-9)).clamp(0, 1).float()
        scores = relative_logits.gather(-1, left)+(relative_logits.gather(-1, right)-relative_logits.gather(-1, left))*mix
        periods = gaps.new_tensor([.25, .5, 1., 2., 3., 4.])
        angles = 2*math.pi*beats[..., None]/periods
        features = torch.cat((gaps[..., None], gaps.log1p()[..., None], angles.sin(), angles.cos()), -1).float()
        keys = self.feature_key(features).float()
        scores = scores.float()+(query.float()[..., None, :]*keys).sum(-1)/math.sqrt(32)+audio_scores.float()
        interval_features=None
        if self.group_policy=='phase_gap_v2' and candidate_expected_count is not None:
            expected=candidate_expected_count.detach().float().clamp_min(0)
            interval_features=torch.stack((expected.log1p(),expected),-1)
            scores=scores+self.interval_candidate_projection(interval_features).float().squeeze(-1)
        # Finite sentinels keep padded targets and unsupported tail families
        # numerically safe during BF16 backward.
        masked = scores.masked_fill(~allowed, -1e4)
        counts = torch.zeros((*scores.shape[:-1], self.num_groups), device=scores.device, dtype=torch.long)
        counts = counts.scatter_add(-1, group_ids.expand_as(allowed), allowed.long())
        lse = grouped_logsumexp(masked, group_ids,groups=self.num_groups)
        macro=self.grid_head(hidden).float()
        if self.group_policy=='phase_gap_v2':
            macro=torch.cat((macro,macro),-1)+self.bucket_head(hidden).float()
        if self.audio_policy in ('candidate_pool_v1','query_pool_v2','query_pool_v3') and candidate_audio is not None:
            macro=macro+self.pooled_audio_logits(candidate_audio,allowed,counts,group_ids)
        if self.audio_policy in ('query_pool_v2','query_pool_v3'):
            macro=macro+self.query_audio_logits(audio_scores,allowed,counts,group_ids)
        if self.audio_policy=='query_pool_v3' and candidate_audio is not None:
            macro=macro+self.query_interaction_logits(query,candidate_audio,allowed,counts,group_ids)
        if interval_features is not None:
            pooled=self._group_means(interval_features,allowed,counts,group_ids)
            macro=macro+self.interval_group_projection(self._center_groups(pooled,counts)).float().squeeze(-1)
        if return_intensity:
            return (scores+macro.gather(-1,group_ids.expand_as(scores))).masked_fill(~allowed,-1e4),beats,gaps
        macro=macro.masked_fill(counts.eq(0), -1e4).log_softmax(-1)
        logits = masked-lse.gather(-1, group_ids.expand_as(masked))+macro.gather(-1, group_ids.expand_as(masked))
        return logits.masked_fill(~allowed, -1e4), beats, gaps


def absolute_target_indices(candidate_beats, target_beats, active):
    distance = (candidate_beats-target_beats[..., None]).abs()
    distance[..., 0] = torch.inf
    nearest, index = distance.min(-1)
    torch._assert_async((~active | nearest.lt(1e-7)).all(), 'Authored onset is absent from the absolute phase catalog')
    return index


def initialize_timing_prior(head, ds, *, batch_size=256):
    """Fit the static v6 first-arrival baseline, without audio/model forwards.

    A NOTE contributes survival exposure before its midpoint cell and the
    discrete event likelihood inside that cell. WAIT/EOS are right censored.
    Candidate construction and seconds exposure match first_arrival; batching
    bounds memory and avoids a Python loop over the phase catalog per target.
    Other learned music/history scores remain evidence around this baseline.
    """
    import numpy as np
    from .stream_data import NOTE, WAIT, EOS
    phases=np.asarray([float(x) for x in head.phase_catalog],dtype=np.float64)
    catalog=np.asarray(head.group_ids.detach().cpu(),dtype=np.int64)
    groups=head.num_groups
    exposure=np.zeros(groups,dtype=np.float64)
    event_widths=[];event_groups=[];event_weights=[]
    for song,stream in enumerate(ds.streams):
        for begin in range(0,len(stream),batch_size):
            rows=stream[begin:begin+batch_size]
            cursor=np.asarray([float(s.start) for s in rows])
            target=np.asarray([float(s.end) for s in rows])
            kinds=np.asarray([s.kind for s in rows])
            weights=np.asarray([float(s.weight)*ds.target_window_weights[song][begin+i]*ds.chart_weights[song]
                if not ds.unknown_steps[song][begin+i] else 0. for i,s in enumerate(rows)])
            weights*=np.isin(kinds,(NOTE,WAIT,EOS))
            anchored=np.floor(cursor)[:,None]+phases
            first=anchored+(anchored<=cursor[:,None]+1e-9)
            beats=np.concatenate((np.repeat(cursor[:,None],2,axis=1),first,first+1),axis=1)
            gaps=np.maximum(beats-cursor[:,None],0.)
            end=float(ds._end_beats[song])
            allowed=((gaps>0)|((gaps==0)&((begin+np.arange(len(rows)))==0)[:,None]&(cursor==0)[:,None]))&(beats<end)
            allowed[:,0]=False
            clock=ds.clocks[song]
            seconds=clock.at_floats(beats)
            start=np.asarray([s.start_second for s in rows])
            stop=np.maximum(start,clock.at_floats(np.minimum(cursor+2,end)))
            valid=allowed&(seconds>=start[:,None]-1e-9)&(seconds<=stop[:,None]+1e-9)
            seconds=np.minimum(np.maximum(seconds,start[:,None]),stop[:,None])
            order=np.argsort(np.where(valid,seconds,np.inf),axis=-1,kind='stable')
            sec=np.take_along_axis(seconds,order,-1)
            legal=np.take_along_axis(valid,order,-1)
            future=np.where(legal,sec,np.inf)
            following=np.concatenate((future[:,1:],np.full((len(rows),1),np.inf)),axis=1)
            right=np.where(np.isfinite(following),.5*(sec+following),stop[:,None])
            right=np.maximum.accumulate(np.where(legal,np.minimum(right,stop[:,None]),-np.inf),axis=-1)
            previous=np.concatenate((start[:,None],right[:,:-1]),axis=-1)
            previous=np.where(np.isfinite(previous),previous,start[:,None])
            width=np.where(legal,np.maximum(right-previous,0.),0.)
            ids=np.broadcast_to(catalog,beats.shape).copy()
            if head.group_policy=='phase_gap_v2':ids+=(gaps>=1.-1e-9)*len(GRID_FAMILIES)
            ids=np.take_along_axis(ids,order,-1)
            distance=np.abs(beats-target[:,None]);distance[:,0]=np.inf
            target_id=distance.argmin(-1)
            target_position=(order==target_id[:,None]).argmax(-1)
            is_note=(kinds==NOTE)&(weights>0)
            if np.any(is_note&((distance.min(-1)>=1e-7)|~valid[np.arange(len(rows)),target_id])):
                raise ValueError('Timing prior target absent from legal absolute catalog')
            before=(np.arange(beats.shape[-1])[None,:]<target_position[:,None])|~is_note[:,None]
            exposure+=np.bincount(ids.ravel(),weights=(width*before*weights[:,None]).ravel(),minlength=groups)
            event_widths.append(width[np.arange(len(rows)),target_position][is_note])
            event_groups.append(ids[np.arange(len(rows)),target_position][is_note])
            event_weights.append(weights[is_note])
    widths=np.concatenate(event_widths) if event_widths else np.empty(0)
    ids=np.concatenate(event_groups) if event_groups else np.empty(0,dtype=np.int64)
    weights=np.concatenate(event_weights) if event_weights else np.empty(0)
    if np.any(widths<=0):raise ValueError('Timing prior NOTE has zero exposure')
    # A weak Gamma(shape=1.5, rate=.5 seconds) prior makes absent/uncensored
    # groups finite. The exact cell likelihood has a monotone derivative:
    # sum(weight * width / expm1(rate * width)) - survival_exposure.
    counts=np.bincount(ids,weights=weights,minlength=groups)
    low=np.zeros(groups);high=(counts+.5)/(exposure+.5)
    for _ in range(48):
        rate=(low+high)*.5
        x=rate[ids]*widths
        derivative=np.bincount(ids,weights=weights*widths/np.expm1(np.minimum(x,700.)),minlength=groups)+.5/rate-exposure-.5
        low=np.where(derivative>0,rate,low);high=np.where(derivative>0,high,rate)
    rate=(low+high)*.5
    bias=rate+np.log(-np.expm1(-rate)) # stable inverse softplus
    with torch.no_grad():
        head.grid_head.bias.copy_(torch.as_tensor(bias[:len(GRID_FAMILIES)],device=head.grid_head.bias.device,dtype=head.grid_head.bias.dtype))
        if hasattr(head,'bucket_head'):
            residual=bias-np.tile(bias[:len(GRID_FAMILIES)],2)
            head.bucket_head.bias.copy_(torch.as_tensor(residual,device=head.bucket_head.bias.device,dtype=head.bucket_head.bias.dtype))
    return dict(timing_prior_policy='seconds_cell_first_arrival_map_v1',timing_prior_survival_seconds=exposure.tolist(),
        timing_prior_event_counts=counts.tolist(),timing_prior_rates=rate.tolist())
