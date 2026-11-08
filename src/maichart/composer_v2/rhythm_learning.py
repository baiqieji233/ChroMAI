"""Authored timing emphasis, with no new events or acoustic-peak pseudo labels."""
from fractions import Fraction
import numpy as np

POLICY = 'mixed_phase_authored_gaps_v1'
PROPER_POLICY = 'authored_mixed_context_proper_ce_v2'


def supervision_weights(codec, ids, labels, target, cursor, base_weights=None, context_phases=()):
    """Redistribute timing weight inside a window; leave action weights intact.

    Mixed quarter/sixth phases receive more practice. Authored gaps and EOS
    retain their targets, including when music is audible. Total timing mass
    stays unchanged, so dense/mixed windows do not gain extra whole-chart weight.
    Validation does not use this training-only emphasis.
    """
    result=np.ones(len(labels),np.float32)
    phase_ids=codec.phases
    phases=[*context_phases, *[p.beat % 1 for p in target]]
    quarter=any(p and (4*p).denominator==1 and (2*p).denominator!=1 for p in phases)
    sixth=any(p and (6*p).denominator==1 and (2*p).denominator!=1 for p in phases)
    timing=np.array([int(t)>=0 and (int(t) in phase_ids or
        codec.tokens[int(t)].startswith('NUM:') or int(t) in (codec.at_id,codec.phase_id,codec.eos_id))
        for t in labels],dtype=bool)
    if quarter and sixth:
        result[np.array([int(t) in phase_ids for t in labels])]=2.
    # Each authored future event supplies its own incoming gap. No waveform
    # peak is treated as a mandatory note; this includes deliberate omissions.
    pos=int(np.flatnonzero(np.asarray(labels)>=0)[0]) if np.any(np.asarray(labels)>=0) else 0
    previous=Fraction(cursor)
    for point in target:
        encoded=codec.encode_points([point])
        if point.beat-previous>=2:
            stop=pos+encoded.index(codec.ids['[PHASE]'])+1
            result[pos:min(stop,len(result))]*=2.
        previous=point.beat;pos+=len(encoded)
    result[np.asarray(labels)==codec.eos_id]*=2.
    base=np.ones(len(labels),np.float32) if base_weights is None else np.asarray(base_weights,dtype=np.float32)
    mass=float(base[timing].sum());denominator=float((base[timing]*result[timing]).sum())
    if denominator>0:result[timing]*=mass/denominator
    result[~timing]=1.
    return result
