"""Next-onset probabilities from event rates integrated over real seconds.

The final survival mass means no onset in this search window, not song EOS.
Candidates must be chronological. Midpoint cells partition time, so inserting
extra candidate points cannot create additional exposure or event mass.
"""
import torch


def event_survival(rates, seconds, valid, start=None):
    rates,seconds=rates.float(),seconds.float()
    b,c=seconds.shape
    origin=seconds.new_zeros(b,1) if start is None else torch.as_tensor(start,device=seconds.device).float().reshape(b,1)
    positions=torch.arange(c,device=seconds.device)[None].expand(b,-1)
    prev=positions.masked_fill(~valid,-1).cummax(-1).values
    prev=torch.cat((prev.new_full((b,1),-1),prev[:,:-1]),-1)
    nxt=positions.masked_fill(~valid,c).flip(-1).cummin(-1).values.flip(-1)
    nxt=torch.cat((nxt[:,1:],nxt.new_full((b,1),c)),-1)
    before=seconds.gather(1,prev.clamp_min(0))
    after=seconds.gather(1,nxt.clamp_max(c-1))
    left=torch.where(prev>=0,(before+seconds)*.5,origin)
    right=torch.where(nxt<c,(seconds+after)*.5,seconds)
    exposure=(right-torch.maximum(left,origin)).clamp_min(0.)*valid
    hazard=rates.clamp_min(0.)*exposure
    cumulative=hazard.cumsum(-1)
    first=(-(cumulative-hazard)+(-torch.expm1(-hazard)).clamp_min(1e-30).log()).masked_fill(~valid,-1e4)
    tail=-hazard.sum(-1)
    return first,tail,exposure


def normalize_survival(first,tail,correction=None):
    if correction is not None:first=first+correction
    joint=torch.cat((first,tail[:,None]),-1).log_softmax(-1)
    return joint[:,:-1],joint[:,-1]


def fuse_survival_rates(workload_lograte, plan=None):
    """Use the existing whole-song rest prediction as a soft rate prior.

    This stays independent of an editor curve. The planner retains its own
    supervised objective; onset loss must not learn to erase its rest signal.
    """
    local = torch.expm1(workload_lograte.float().clamp(0., 8.))
    if plan is None:
        return local
    reference = plan.detach().float()
    planned = torch.expm1(reference[..., 0].clamp(0., 5.))
    rest = reference[..., 5].clamp(0., 1.)
    gate = (1. - rest).square().clamp_min(.02)
    return (.65 * local + .35 * planned) * gate


def causal_count_feedback(rates,edges,counts,keep=None):
    """Soft event_ctx-style spent-versus-expected feedback, never a hard quota.

    One-standard-deviation slack permits ordinary bursts. Only excess actual
    starts lower intensity; undershooting never forces new notes. Expectations
    are detached so the onset objective cannot inflate its past allowance.
    """
    edge=edges.double()
    width=(edge[:,1:]-edge[:,:-1]).clamp_min(0.)
    reference=rates.detach().double().masked_fill(width<=0.,0.)
    prefix=torch.cat((reference.new_zeros(reference.size(0),1),(reference*width).cumsum(-1)),-1)
    centers=(edge[:,:-1]+edge[:,1:])*.5
    def integral(q):
        q=torch.maximum(edge[:,:1],torch.minimum(q,edge[:,-1:]))
        index=(torch.searchsorted(edge.contiguous(),q.contiguous(),right=True)-1).clamp(0,rates.size(1)-1)
        return prefix.gather(1,index)+reference.gather(1,index)*(q-edge.gather(1,index))
    stop=integral(centers)
    expected=torch.stack([(stop-integral(centers-w)).clamp_min(0.) for w in (1.,4.)],-1)
    sigma=(expected+1.).sqrt()
    debt=((counts.double()-expected-sigma).clamp_min(0.)/sigma).amax(-1)
    penalty=(4.*torch.tanh(debt/4.)).float().masked_fill(width<=0.,0.)
    if keep is not None:penalty=penalty*keep.to(penalty)
    return rates*torch.exp(-penalty),penalty


def integrated_survival(rates, edges, seconds, valid, start):
    """First-event mass on right-ended candidate cells of a fixed rate grid.

    Grid cells and queries use absolute audio seconds. A committed candidate
    ends the consumed interval exactly; no midpoint's future half-cell is
    integrated twice. Padding has repeated edges and therefore zero exposure.
    Prefix arithmetic uses float64; the small MLP remains in its usual dtype.
    """
    rates = rates.double().clamp_min(0.)
    edges = edges.double()
    seconds = seconds.double()
    start = start.double().reshape(-1, 1)
    widths = (edges[:, 1:] - edges[:, :-1]).clamp_min(0.)
    rates = rates.masked_fill(widths <= 0., 0.)
    prefix = torch.cat((rates.new_zeros(rates.size(0), 1), (rates * widths).cumsum(-1)), -1)

    def integral(query):
        query = torch.maximum(edges[:, :1], torch.minimum(query, edges[:, -1:]))
        idx = torch.searchsorted(edges.contiguous(), query.contiguous(), right=True) - 1
        idx = idx.clamp(0, rates.size(1) - 1)
        return prefix.gather(1, idx) + rates.gather(1, idx) * (query - edges.gather(1, idx))

    cumulative = (integral(seconds) - integral(start)).clamp_min(0.)
    positions = torch.arange(seconds.size(1), device=seconds.device)[None].expand_as(seconds)
    previous = positions.masked_fill(~valid, -1).cummax(-1).values
    previous = torch.cat((previous.new_full((rates.size(0), 1), -1), previous[:, :-1]), -1)
    before = cumulative.gather(1, previous.clamp_min(0))
    before = torch.where(previous >= 0, before, torch.zeros_like(before))
    cell = (cumulative - before).clamp_min(0.)
    first = -before + (-torch.expm1(-cell)).clamp_min(1e-30).log()
    first = first.masked_fill(~valid, -1e4)
    horizon = cumulative.masked_fill(~valid, 0.).amax(-1)
    return first.float(), -horizon.float(), cell.float()


def redistribute_survival(base, tail, delta, valid, bins=None):
    """Reorder event times while preserving the independently learned tail."""
    # base and tail come from the same integral, already summing to one.
    # No softmax with the tail: symbolic or rhythmic gains cannot consume it.
    event_mass = (-torch.expm1(tail.double())).clamp_min(1e-30).log().float()
    shifted = (base.float() + delta.float()).masked_fill(~valid, -1e4)
    if bins is not None:
        # The distance stage owns band mass; the acoustic onset scorer owns
        # precise positions inside a band. Its uncalibrated global offset must
        # not turn the whole song into a handful of far-anchor jumps.
        membership=torch.nn.functional.one_hot(bins,6).bool() & valid[...,None]
        original=base.float()[...,None].masked_fill(~membership,-1e4).logsumexp(1)
        moved=shifted[...,None].masked_fill(~membership,-1e4).logsumexp(1)
        events=shifted-moved.gather(1,bins)+original.gather(1,bins)
        return events.masked_fill(~valid,-1e4),tail.float()
    conditional = shifted - shifted.logsumexp(-1, keepdim=True)
    events = (conditional + event_mass[:, None]).masked_fill(~valid, -1e4)
    return events, tail.float()
