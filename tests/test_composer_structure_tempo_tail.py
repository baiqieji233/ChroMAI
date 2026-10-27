from fractions import Fraction
from types import SimpleNamespace
import numpy as np
from maichart.event_timing import EventTempo
from maichart.composer.structure import _build


def test_future_tempo_anchor_does_not_shorten_structure_analysis_of_current_audio():
    audio=SimpleNamespace(times=np.linspace(0,5,101),features=np.zeros((101,104)),duration_sec=5.)
    song=SimpleNamespace(audio=audio,first=0.,tempo=[EventTempo(Fraction(0),120.),EventTempo(Fraction(100),240.)])
    _,_,grid,_=_build(song,None,{'name':'native_multicycle_v1'})
    assert grid[-1]==10.
