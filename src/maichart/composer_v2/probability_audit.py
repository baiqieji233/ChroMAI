"""Optional scoring of proposals, without changing draws or candidate ranking."""
import hashlib
import math
import torch

SCHEMA='composer-v2-probability-v1'


def file_digest(path):
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


class EventProbability:
    def __init__(self):
        self.rows=[];self.tokens=[];self.roles=[];self.forced=[]

    def record(self,logits,token,grammar,*,role,forced=False,proposal_logits=None,
               allowed=None,temperature=1.,top_p=1.):
        values=logits.detach().float()
        grammar=sorted(set(grammar))
        if token not in grammar:raise ValueError('Recorded token is not in its unconditioned grammar')
        selected=values[token]
        raw=selected-torch.logsumexp(values,0)
        legal=selected-torch.logsumexp(values[grammar],0)
        if forced or temperature==0:
            proposal=raw.new_zeros(())
        else:
            allowed=sorted(set(allowed));position=allowed.index(token)
            proposal_values=proposal_logits.detach().float()[allowed]/temperature
            probabilities,order=torch.sort(torch.softmax(proposal_values,0),descending=True)
            probabilities=probabilities*(probabilities.cumsum(0)-probabilities<top_p)
            probabilities=probabilities/probabilities.sum()
            proposal=torch.where(order==position,probabilities,torch.zeros_like(probabilities)).sum().log()
        self.rows.append(torch.stack((raw,legal,proposal)))
        self.tokens.append(int(token));self.roles.append(role);self.forced.append(bool(forced))


def summarize(events,*,boundary_tokens=0):
    """One small device transfer per candidate, including explicitly forced IDs.

    Raw = configured model output before grammar/retries/temperature/top-p.
    Grammar = that output conditioned on legal syntax and head uniqueness.
    Proposal = actual temperature/top-p/retry draw; forced tokens have mass one.
    These are different distributions and must never be silently substituted.
    """
    logs=[event for event in events if event is not None]
    missing=sum(event is None for event in events)
    rows=[row for event in logs for row in event.rows]
    numbers=torch.stack(rows).double().cpu().tolist() if rows else []
    tokens=[t for event in logs for t in event.tokens]
    roles=[t for event in logs for t in event.roles]
    forced=[t for event in logs for t in event.forced]
    result=dict(schema=SCHEMA,coverage='partial' if missing else 'sampled_and_forced_complete',
        missing_events=missing,range_scope='scored_before_stop_boundary',boundary_tokens_excluded=boundary_tokens,
        meaning='Scored span only, not full sequence MAP; boundary crossing and EOS are reported separately.')
    for key,role in (('all_tokens',None),('timing','timing'),('configuration','configuration')):
        indices=[i for i,value in enumerate(roles) if role is None or value==role]
        summary=dict(tokens=len(indices),forced_tokens=sum(forced[i] for i in indices))
        for column,name in enumerate(('raw_logprob_sum','grammar_logprob_sum','proposal_logprob_sum')):
            values=[numbers[i][column] for i in indices]
            summary[name]=sum(values) if not missing and all(math.isfinite(v) for v in values) else None
        result[key]=summary
    result['token_records']=[dict(token_id=t,role=r,forced=f,
        **{name:value if math.isfinite(value) else None for name,value in zip(
            ('raw_logprob','grammar_logprob','proposal_logprob'),row)})
        for t,r,f,row in zip(tokens,roles,forced,numbers)]
    return result


def span_diagnostics(points,codec,start,stop):
    from maichart.composer.event_set_codec import _atoms
    from .timing_metrics import rhythm_profile
    points=[p for p in points if start<=p.beat<stop]
    gaps=[float(b.beat-a.beat) for a,b in zip(points,points[1:])]
    profile=rhythm_profile(points)
    return dict(events_in_span=len(points),atoms_in_span=sum(len(_atoms(p.note_text)) for p in points),
        tokens_in_span=len(codec.encode_points(points)),
        mean_gap_beats=sum(gaps)/len(gaps) if gaps else None,max_gap_beats=max(gaps) if gaps else None,
        phase_counts=profile['phase_counts'],interval_counts=profile['interval_counts'])
