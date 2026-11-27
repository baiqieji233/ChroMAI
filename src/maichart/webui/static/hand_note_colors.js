function simulationColorTable(markers, notes, abstainOnMissing=false) {
  const epsilon = .00012;
  const equal = (a,b) => Number.isFinite(a) && Number.isFinite(b) && Math.abs(a-b) < epsilon;
  const padKey = p => /^C[12]?$/.test(p) ? 'C' : p;
  const byPad = new Map();
  for (const note of notes) {
    const pad = padKey(note.pad);
    if (!byPad.has(pad)) byPad.set(pad, new Map());
    const times = byPad.get(pad), tick = Math.floor(note.start/epsilon);
    if (!times.has(tick)) times.set(tick, []);
    times.get(tick).push(note);
  }
  const routePad = ([x,y]) => 'A' + ((Math.round((Math.atan2(y,x) + Math.PI*3/8)/(Math.PI/4)) + 8) % 8 + 1);
  const uniqueSide = hits => {
    if (abstainOnMissing && hits.some(n=>!n.head)) return 0;
    const sides = new Set(hits.map(n=>n.head).filter(Boolean));
    return sides.size === 1 ? sides.values().next().value : 0;
  };
  const rows = [];
  for (const [id,type,pad,t,trace,end,endpoints,key,hasHead,mine,headMine,trackMine] of markers) {
    const moving = type === 'slide' || type === 'wifi';
    const times = byPad.get(padKey(pad)), tick = Math.floor(t/epsilon);
    const coincident = [tick-1,tick,tick+1].flatMap(k=>times?.get(k) || []).filter(n=>equal(n.start,t));
    const shape = key.match(/^[1-8](V[1-8]|pp|qq|.)/)?.[1];
    const sameTrack = coincident.filter(n=>n.kind === type && equal(n.trace,trace) && equal(n.end,end));
    function matchTrack(candidate) {
      if (candidate.length <= 1) return candidate[0];
      const shaped = candidate.filter(n=>n.shape === shape);
      return shaped.length === 1 ? shaped[0] : undefined;
    }
    function findLane(point) {
      return point ? matchTrack(sameTrack.filter(n=>!n.centerOnly && n.endPad===routePad(point))) : undefined;
    }
    const regular = type==='slide' && findLane(endpoints.at(-1));
    const headHits = moving
      ? coincident.filter(n=>n.kind === 'star' || (n.kind === type && n.hasHead &&
        (type === 'wifi' ? sameTrack.includes(n) : n === regular)))
      : coincident.filter(n=>n.kind === type || (equal(t,end) &&
        ((type==='hold' && n.kind==='tap') || (type==='touch_hold' && n.kind==='touch'))));
    const head = (moving ? hasHead && !headMine : !mine) ? uniqueSide(headHits) : 0;
    const row = [id,head,0,0,[]];
    if (type==='slide' && !trackMine && regular?.track?.length) {
      row[2] = regular.track[0][1];
      let side = row[2];
      for (const [second,nextSide] of regular.track.slice(1)) {
        if (second >= end || nextSide === side) continue;
        row[4].push([Math.max(trace,second),nextSide]);
        side = nextSide;
      }
      if (Number.isFinite(regular.trackStop) && regular.trackStop < end-epsilon)
        row[4].push([Math.max(trace,regular.trackStop),0]);
    } else if (type==='wifi' && !trackMine) {
      const first=findLane(endpoints[0]), last=findLane(endpoints.at(-1));
      const center=sameTrack.find(n=>n.centerOnly) || findLane(endpoints[1]);
      // The body simulator provides only the center estimate. Fly-rg
      // provides individually judged paths, so missing branches stay neutral.
      const judgeBranch = n => {
        if (!n?.track?.length || (Number.isFinite(n.trackStop) && n.trackStop < end-epsilon)) return 0;
        const sides = new Set(n.track.map(h=>h[1]));
        return sides.size === 1 ? sides.values().next().value : 0;
      };
      row[2]=judgeBranch(first);
      row[3]=judgeBranch(last);
      row.push(judgeBranch(center));
    }
    if (row[1] || row[2] || row[3] || row[5]) rows.push(row);
  }
  return rows;
}
