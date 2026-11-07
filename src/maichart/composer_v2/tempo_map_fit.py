"""Audio-only piecewise periodic clocks; no meter or authored chart input."""
import numpy as np
from scipy.ndimage import gaussian_filter1d,uniform_filter1d
from maichart.composer_v2.tempo_fit import _refine,_wrapped
from fractions import Fraction
from maichart.event_timing import EventTempo
from maichart.tempo_clock import TempoClock

POLICY='audio_verified_piecewise_clock_v1'


def refine_acoustic(times,flux,bpm,phase,*,start=0.,end=None,fixed_bpm=False):
    """Bounded full-span slope/phase search; no unconditional integer snapping."""
    end=float(times[-1]) if end is None else end
    f=gaussian_filter1d(flux.astype(float),.8)
    f=4*np.tanh(f/np.maximum(uniform_filter1d(f,201),.03)/4)
    center=phase+round(((start+end)/2-phase)*bpm/60)*60/bpm
    k=np.arange(np.ceil((start+.15-center)*bpm/60),np.floor((end-.15-center)*bpm/60))
    if len(k)<12:return bpm,phase,dict(applied=False,reason='insufficient_span')
    shifts=np.linspace(-.09,.09,37)
    def score(b):
        query=center+shifts[:,None]+k[None,:]*60/b
        values=np.interp(query,times,f);scores=values.mean(1);i=int(scores.argmax())
        return float(scores[i]),float(center+shifts[i])
    old=score(bpm);best=old;chosen=bpm
    for b in ([bpm] if fixed_bpm else np.linspace(bpm-.3,bpm+.3,121)):
        s=score(b)
        if s[0]>best[0]:best=s;chosen=float(b)
    if best[0]<1.1:return bpm,phase,dict(applied=False,reason='weak_periodic_onset_support')
    # A correction must have audible evidence rather than an arbitrary grid winner.
    if best[0]<.8 or best[0]<old[0]*1.003:
        chosen=bpm;best=old
    raw=chosen;integer=round(chosen)
    drift=(end-start)*abs(integer/chosen-1)
    integer_score=score(integer) if integer>0 else (-1,0)
    snapped=not fixed_bpm and drift<=.015 and integer_score[0]>=best[0]*.995
    if snapped:chosen=float(integer);best=integer_score
    return chosen,best[1]%(60/chosen),dict(applied=True,input_bpm=bpm,continuous_bpm=raw,
        output_bpm=chosen,integer_selected=bool(snapped),equivalent_song_drift_sec=drift,
        input_score=old[0],output_score=best[0],method='native_flux_period_phase_search_v1')


def fit_variable_clock(beats,bpm,phase,flux=None,flux_times=None,first_override=None):
    segments,gain=_segment(beats,bpm,phase,flux,flux_times)
    alternate=None
    if flux is not None and len(segments)>1 and max(s['p'] for s in segments)/min(s['p'] for s in segments)>=1.01:
        import librosa
        dynamic=librosa.feature.tempo(onset_envelope=flux.astype(float),sr=100,hop_length=1,
            start_bpm=bpm,aggregate=None,max_tempo=500.)
        _,alternate=librosa.beat.beat_track(onset_envelope=flux.astype(float),sr=100,hop_length=1,
            bpm=dynamic,trim=False,units='time')
        alternate=alternate[(alternate>=beats[0])&(alternate<=beats[-1])]
        if len(alternate)>=8:segments,gain=_segment(beats,bpm,phase,flux,flux_times,alternate)
    calibrations=[]
    if flux is not None:
        for s in segments:
            refined,aligned,cal=refine_acoustic(flux_times,flux,60/s['p'],s['phase'],start=s['start'],end=s['end'])
            s.update(p=60/refined,phase=aligned);calibrations.append(cal)
    report=dict(policy=POLICY,candidate_segments=[dict(start_sec=s['start'],end_sec=s['end'],bpm=60/s['p'],phase=s['phase']) for s in segments],
                segmentation_gain=gain,automatic_tempo_map_applied=False,
                independent_tracker='librosa_dynamic_dp' if alternate is not None else None,
                segment_calibration=calibrations)
    # Near-identical slopes with different phases are not a tempo-change map.
    if len(segments)<2 or max(s['p'] for s in segments)/min(s['p'] for s in segments)<1.01:
        report['decision']='constant_or_phase_variation';return None,None,report
    if len(segments)>12 or gain<8:
        report['decision']='insufficient_segment_evidence';return None,None,report
    first=segments[0]['phase'] if first_override is None else float(first_override)
    changes=[EventTempo(Fraction(0),60/segments[0]['p'])]
    boundaries=[]
    for previous,current in zip(segments,segments[1:]):
        clock=TempoClock(changes,first);boundary=float(current['start'])
        # Search nearby intersections of the two fitted pulse grids. Do not
        # impose bar lengths or 4/4. The 1/48 search is a detection resolution,
        # independent of the generated event vocabulary.
        beat_at= float(changes[-1].beat)+(boundary-clock.at(changes[-1].beat))/clock._scale_arr[-1]
        candidates=np.arange(np.floor((beat_at-3)*48),np.ceil((beat_at+3)*48)+1)/48
        candidates=candidates[candidates>float(changes[-1].beat)]
        seconds=clock.at_floats(candidates)
        # A fractional change is not itself an integer beat of the new clock.
        # Compare the implied beat-zero intercept, not the boundary timestamp.
        errors=abs(_wrapped(seconds-candidates*current['p'],current['p'],current['phase']))
        cost=errors+.008*abs(seconds-boundary)
        if not len(cost):report['decision']='invalid_boundary';return None,None,report
        best=int(cost.argmin())
        if errors[best]>.025 or abs(seconds[best]-boundary)>1.5:
            report['decision']='boundary_not_phase_consistent';return None,None,report
        changes.append(EventTempo(Fraction(int(round(candidates[best]*48)),48),60/current['p']))
        boundaries.append(dict(seconds=float(seconds[best]),beat=str(changes[-1].beat),phase_error_sec=float(errors[best]),
                               coarse_boundary_sec=boundary,search_resolution_beats=1/48,search_radius_sec=1.5))
    clock=TempoClock(changes,first)
    ix=np.clip(np.searchsorted(clock._seconds_arr,beats,side='right')-1,0,len(changes)-1)
    beat_values=clock._beat_floats[ix]+(beats-clock._seconds_arr[ix])/clock._scale_arr[ix]
    residual=abs(beat_values-np.round(beat_values))*clock._scale_arr[ix]
    coverage=float((residual<.05).mean())
    report.update(boundaries=boundaries,map_peak_coverage=coverage,map_residual_p90_sec=float(np.quantile(residual,.9)))
    if coverage<.80 or np.quantile(residual,.9)>.07:
        report['decision']='uncertain_pulse_or_missing_beats';return None,None,report
    report.update(automatic_tempo_map_applied=True,decision='accepted')
    return changes,first,report

def _candidates(t,global_bpm):
    hypotheses=[];windows=[]
    for start in np.arange(t[0],t[-1]-5,2.):
        times=t[(t>=start)&(t<start+8)]
        if len(times)<8:continue
        median=np.median(np.diff(times));fit=_refine(times,median)
        if fit is None or fit['keep'].mean()<.8 or np.quantile(abs(fit['residual'][fit['keep']]),.9)>.03:continue
        b=60/fit['period']
        scale=min((.5,1,2),key=lambda s:abs(np.log(s*b/global_bpm)))
        if abs(np.log(scale*b/global_bpm))<.03:
            fit=_refine(times,fit['period']/scale)
            if fit is None:continue
        windows.append((start,start+8,fit))
    groups=[]
    for a,b,f in windows:
        if groups and a<=groups[-1][1] and abs(f['period']/groups[-1][2]['period']-1)<.006:
            old=groups[-1];times=t[(t>=old[0])&(t<b)];merged=_refine(times,old[2]['period'])
            if merged is not None and merged['keep'].mean()>.75:groups[-1]=(old[0],b,merged);continue
        groups.append((a,b,f))
    for a,b,f in groups:
        if b-a<8:continue
        hypotheses.append(dict(p=f['period'],phase=f['first'],start=a,end=b))
    return hypotheses

def _segment(t,bpm,first,flux=None,ft=None,alternate=None):
    hs=[dict(p=60/bpm,phase=first,start=t[0],end=t[-1]),*_candidates(t,bpm)]
    if alternate is not None:
        extra=_candidates(alternate,bpm)
        for h in extra:
            center=(h['start']+h['end'])/2
            nearby=[r for r in hs if r['start']<=center<=r['end']]
            choices=[(abs(np.log(h['p']*factor/r['p'])),factor) for r in nearby for factor in (.5,1.,2.)]
            if choices:
                error,factor=min(choices)
                if error<.03 and factor!=1:
                    sample=alternate[(alternate>=h['start'])&(alternate<h['end'])]
                    fitted=_refine(sample,h['p']*factor)
                    if fitted is not None:h.update(p=fitted['period'],phase=fitted['first'])
        hs.extend(extra)
    n=max(7,int(np.floor(t[-1]))+1);cost=np.zeros((len(hs),n));sigma=.025
    if flux is not None:
        f=gaussian_filter1d(flux.astype(float),.8);f=4*np.tanh(f/np.maximum(uniform_filter1d(f,201),.03)/4)
    for k,h in enumerate(hs):
        sources=[t] if alternate is None else [t,alternate]
        for source in sources:
            residual=_wrapped(source,h['p'],h['phase']);w=np.exp(-.5*(residual/sigma)**2)
            count=np.bincount(source.astype(int),minlength=n)
            cost[k]-=np.bincount(source.astype(int),weights=w,minlength=n)/np.maximum(count,1)/len(sources)
        if flux is not None:
            grid=np.arange(h['phase'],t[-1],h['p']);values=np.interp(grid,ft,f)
            freq=np.bincount(grid.astype(int),minlength=n)
            onset=np.bincount(grid.astype(int),weights=values,minlength=n)/np.maximum(freq,1)
            cost[k]-=.6*onset
    prefix=np.pad(np.cumsum(cost,axis=1),((0,0),(1,0)))
    dp=np.full(n+1,np.inf);dp[0]=-5.;backs={}
    for end in range(6,n+1):
        starts=np.arange(end-5)
        values=prefix[:,end,None]-prefix[:,starts]+dp[starts]+5
        k,i=np.unravel_index(values.argmin(),values.shape)
        dp[end]=values[k,i];backs[end]=(int(starts[i]),int(k))
    ranges=[];end=n
    while end:
        start,k=backs[end];ranges.append(dict(start=start,end=end,**{key:hs[k][key] for key in ('p','phase')}));end=start
    ranges=list(reversed(ranges))
    return ranges, float(prefix[0,-1]-dp[n])
