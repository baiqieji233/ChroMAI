from dataclasses import replace
from fractions import Fraction

import numpy as np

from maichart.composer.motif_memory import motif_relations_fingerprint
from maichart.form.recurrence import Relation,SongRelations
from maichart.form.timing import MetricSpan


def relations():
    spans=tuple(MetricSpan(i,Fraction(i*4),Fraction(i*4+4),4,4,(),Fraction(4),(),False,1.,'test') for i in range(3))
    return SongRelations(spans,np.zeros((3,504)),[Relation(0,2,.9,.8,1.,0,24)])


def test_motif_input_identity_changes_with_graph_bounds_phase_or_ranking():
    reference=relations();baseline=motif_relations_fingerprint([(0,reference)])
    for graph in (
        replace(reference,links=[]),
        replace(reference,links=[replace(reference.links[0],phase_shift_bins=1)]),
        replace(reference,spans=(*reference.spans[:2],replace(reference.spans[2],end=Fraction(13)))),
    ):
        assert motif_relations_fingerprint([(0,graph)])!=baseline
    ranked=replace(reference,links=[*reference.links,Relation(1,2,.8,.8,1.,0,24)])
    reordered=replace(ranked,links=[ranked.links[0],replace(ranked.links[1],similarity=.95)])
    assert motif_relations_fingerprint([(0,ranked)])!=motif_relations_fingerprint([(0,reordered)])


def test_motif_input_identity_ignores_serialization_order_and_unused_diagnostics():
    a=relations();b=replace(a,spans=tuple(reversed(a.spans)),audit={'irrelevant':'changed'},
        descriptors=np.ones((3,504)),links=[replace(a.links[0],similarity=.8)])
    assert motif_relations_fingerprint([(2,a),(0,None)])==motif_relations_fingerprint([(0,None),(2,b)])
