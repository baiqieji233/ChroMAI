"""Conservative contact diagnostics for simultaneous Composer actions.

The reference sensor coordinates and Touch-group adjacency describe judgement,
not hand reach. In particular, an adjacent Touch group can clear after only a
subset of its sensors is hit. No calibrated hand-reach or cabinet/button
geometry is available here. A deliberately narrow, uncalibrated reach template
assumes widely separated outer buttons and center C require three contacts. This
is a physical assumption, not proven by sensor geometry or player trials; it
can produce false positives. Other Touch combinations are explicitly unknown.
``minimum_contacts`` is conditional on that template, not a verified hand count.

``occupied`` contains active source heads (e.g. Holds), as supplied by the
caller; Slides do not become occupied merely because they exist. ``blocked``
contains temporarily unavailable heads. Both use the grammar's head identity;
button ``1`` and Touch ``A1`` are distinct. This module does not replace the
sampler's existing head and button-budget masks.
"""
from __future__ import annotations
import numpy as np

POLICIES = frozenset({'off', 'diagnostic_v1', 'contact_groups_v1'})
_FAR_BUTTONS=np.asarray([any((bits & (1<<a)) and (bits & (1<<b))
    and min(b-a,8-(b-a))>=3 for a in range(8) for b in range(a+1,8))
    for bits in range(256)],dtype=bool)


class ContactBudget:
    """Cache immutable grammar head metadata for repeated candidate checks."""

    def __init__(self, grammar):
        self.heads = {str(atom): frozenset(map(str, heads or ()))
                      for atom, heads in grammar.heads.items()}
        self._catalogs={}

    def mask(self, atoms, selected_atoms=(), *, occupied=(), policy='contact_groups_v1'):
        """Vectorized catalog mask; no per-atom grammar parsing in a slot."""
        if policy not in POLICIES:raise ValueError('unknown contact policy')
        if policy!='contact_groups_v1':return np.ones(len(atoms)+1,dtype=bool)
        key=id(atoms)
        cached=self._catalogs.get(key)
        if cached is None or cached[0] is not atoms:
            heads=[frozenset()]+[self.heads.get(str(a),frozenset()) for a in atoms]
            buttons=np.asarray([sum(1<<(int(h)-1) for h in row if h in '12345678' and len(h)==1)
                                for row in heads],dtype=np.uint16)
            center=np.asarray(['C' in row for row in heads])
            other=np.asarray([any(not h.isdigit() and h!='C' for h in row) for row in heads])
            cached=(atoms,buttons,center,other);self._catalogs[key]=cached
        _,buttons,center,other=cached
        prior=set(map(str,occupied))
        for atom in selected_atoms:prior.update(self.heads.get(str(atom),()))
        prior_bits=sum(1<<(int(h)-1) for h in prior if h in '12345678' and len(h)==1)
        union=buttons|prior_bits
        far=_FAR_BUTTONS[union]
        reject=far&(center|('C' in prior))&~(other|any(not h.isdigit() and h!='C' for h in prior))
        result=~reject;result[0]=True
        return result

    def decision(self, selected_atoms, candidate_atom, *, occupied=(), blocked=(),
                 policy='contact_groups_v1'):
        if policy not in POLICIES:
            raise ValueError(f'unknown contact policy: {policy!r}')
        selected = tuple(selected_atoms)
        candidate = str(candidate_atom)
        occupied = frozenset(map(str, occupied))
        blocked = frozenset(map(str, blocked))
        candidate_heads = self.heads.get(candidate)
        selected_heads = [self.heads.get(str(atom)) for atom in selected]
        known = candidate_heads is not None and bool(candidate_heads) and all(selected_heads)
        previous = set(occupied)
        for heads in selected_heads:
            if heads:
                previous.update(heads)
        all_heads = previous | (candidate_heads or frozenset())
        result = {'allowed': True, 'verdict': 'unknown', 'reason': 'unresolved_heads',
                  'minimum_contacts': None,
                  'policy': policy}
        if policy == 'off':
            result.update(verdict='allow', reason='policy_off', minimum_contacts=None)
            return result
        if not known:
            return result
        collision = candidate_heads & previous
        unavailable = candidate_heads & blocked
        if collision or unavailable:
            result.update(verdict='reject', reason='same_head' if collision else 'blocked_head',
                          conflicting_heads=sorted(collision or unavailable),
                          allowed=policy != 'contact_groups_v1')
            return result
        touches = {head for head in all_heads if not head.isdigit()}
        buttons = sorted(int(head) for head in all_heads if head in {str(i) for i in range(1, 9)})
        # This template is conditional on a fixed reach assumption. Other
        # Touches could participate in group auto-judgement and must not be
        # ignored; do not apply the template if any is present.
        far_pair = any(min(abs(a-b), 8-abs(a-b)) >= 3
                       for i, a in enumerate(buttons) for b in buttons[i+1:])
        if touches == {'C'} and far_pair:
            result.update(verdict='reject', reason='far_buttons_center_assumed_three_contacts',
                          minimum_contacts=3, allowed=policy != 'contact_groups_v1',
                          assumption='uncalibrated_human_reach_template')
            return result
        if touches:
            result.update(verdict='unknown', reason='touch_reach_unvalidated')
            return result
        # Even for button-only input we make no assertion that the count of
        # heads equals the number of hands. The sampler owns its button cap.
        result.update(verdict='allow', reason='no_additional_contact_constraint',
                      minimum_contacts=2 if far_pair else 1 if all_heads else 0)
        return result


def contact_decision(selected_atoms, candidate_atom, grammar, *, occupied=(),
                     blocked=(), policy='contact_groups_v1'):
    """Evaluate one candidate; reuse ContactBudget for a decoding loop."""
    return ContactBudget(grammar).decision(selected_atoms, candidate_atom,
                                           occupied=occupied, blocked=blocked,
                                           policy=policy)
