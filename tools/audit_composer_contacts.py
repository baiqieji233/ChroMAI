"""Measure contact-policy interceptions on authored chart events."""
import argparse,json,sys
import heapq
from bisect import bisect_left
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--cache-dir',default='cache/chart_composer_v1')
    p.add_argument('--limit-charts',type=int,default=0);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--include-active-holds',action=argparse.BooleanOptionalAction,default=True)
    a=p.parse_args(argv)
    from maichart.form.data import FormDataset
    from maichart.composer.contacts import ContactBudget
    from maichart.composer.events import event_metadata
    from maichart.generation.event_cli import _NoteGrammar
    from maichart.tempo_clock import TempoClock
    ds=FormDataset(a.manifest,cache_dir=a.cache_dir,limit_charts=a.limit_charts,
        max_tokens=1_000_000,max_region_seconds=999.,preparation_workers=a.workers,
        audio_cache_gb=.25,quality_profile='musical_evidence_v1',
        audio_profile='rational_audio_queries_v1',preserve_excluded_context=True)
    counts=Counter();examples=[]
    for song_index,record in enumerate(ds.songs):
        points=record[1]
        atoms=sorted({atom for point in points for atom in event_metadata(point.note_text).atoms})
        budget=ContactBudget(_NoteGrammar.from_atoms(atoms))
        song=record[0];clock=TempoClock(song.tempo,song.first)
        markers=[]
        if a.include_active_holds:
            starts=[float(clock.at(p.beat)) for p in points]
            for source_marker in record[4].markers:
                if source_marker.get('type') not in {'hold','touch_hold'}:continue
                marker=dict(source_marker)
                if 'source_beat' not in marker:
                    t=float(marker.get('second',0.));index=bisect_left(starts,t)
                    candidates=[i for i in (index-1,index) if 0<=i<len(starts)]
                    nearest=min(candidates,key=lambda i:abs(starts[i]-t)) if candidates else None
                    if nearest is None or abs(starts[nearest]-t)>1e-4:
                        counts['unresolved_hold_sources']+=1;continue
                    marker['source_beat']=points[nearest].beat
                markers.append(marker)
            markers.sort(key=lambda m:m['source_beat'])
        cursor=0;ends=[];active=Counter()
        for point in points:
            second=float(clock.at(point.beat))
            while cursor<len(markers) and markers[cursor]['source_beat']<point.beat:
                marker=markers[cursor];cursor+=1
                pad=(str(marker['lane']) if marker['type']=='hold' else str(marker.get('touch_pad','')))
                end=float(marker.get('end_second',-1.))
                if pad and end>second+1e-8:
                    active[pad]+=1;heapq.heappush(ends,(end,pad))
            while ends and ends[0][0]<=second+1e-8:
                _,pad=heapq.heappop(ends);active[pad]-=1
                if not active[pad]:del active[pad]
            occupied=set(active)
            selected=[];verdict='allow';reason=None
            for atom in event_metadata(point.note_text).atoms:
                result=budget.decision(selected,atom,occupied=occupied,policy='contact_groups_v1')
                if result['verdict']=='reject':verdict='reject';reason=result['reason'];break
                if result['verdict']=='unknown':verdict='unknown'
                selected.append(atom)
            counts['events']+=1;counts[verdict]+=1
            counts['events_with_active_hold_sources']+=bool(occupied)
            if verdict=='reject':
                counts[reason]+=1
                if len(examples)<40:examples.append({'song_index':song_index,'beat':str(point.beat),
                    'note_text':point.note_text,'reason':reason,'occupied':sorted(occupied)})
    report={'schema':'composer-contact-corpus-audit-v2','manifest':str(Path(a.manifest).resolve()),
        'charts':len(ds.songs),'counts':dict(counts),'examples':examples,
        'include_active_holds':a.include_active_holds,
        'scope':'authored events with exact prior Hold/TouchHold source occupancy; Slide waiting is not a locked hand; no human reach verification'}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'charts':report['charts'],'counts':report['counts'],'report':str(out)},ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
