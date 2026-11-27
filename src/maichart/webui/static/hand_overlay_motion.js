const HAND_TRAIL_SECONDS = .82;
const HAND_LOOKAHEAD_SECONDS = .34;

function handFrameIndex(rows, second) {
  let lo = 0, hi = rows.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (rows[mid].second <= second + 1e-7) lo = mid + 1;
    else hi = mid;
  }
  return lo - 1;
}

function handPosition(hit) {
  const rim = ['button', 'hold', 'hypothesis'].includes(hit.kind) ||
    (hit.kind === 'ActionPress' && /^A[1-8]$/.test(hit.note));
  const radius = rim ? 1 : 205 / 240;
  return [hit.x * radius, hit.y * radius];
}

function handSamples(frames, side) {
  return frames.flatMap(frame => {
    const contacts = frame[side] || [];
    if (!contacts.length) return [];
    const points = contacts.map(handPosition);
    const center = points.reduce(([x,y], [a,b])=>[x+a/points.length,y+b/points.length],[0,0]);
    return [{second:frame.second,contacts,points,center}];
  });
}

function handMovement(start, end) {
  const gap = end.second - start.second;
  if (gap <= 0) return null;
  const sameJob = start.contacts.length === 1 && end.contacts.length === 1 &&
    start.contacts[0].job >= 0 && start.contacts[0].job === end.contacts[0].job &&
    start.contacts[0].phase !== 'release';
  if (gap > .9 && !sameJob) return null;
  const rimStart = start.contacts.length === 1 && start.contacts[0].kind === 'button' && start.contacts[0].lane;
  const rimEnd = end.contacts.length === 1 && end.contacts[0].kind === 'button' && end.contacts[0].lane;
  const step = rimStart && rimEnd ? ((rimEnd-rimStart+12)%8)-4 : 0;
  const alongRim = Math.abs(step) === 1 && gap <= .28;
  // Keep an active Slide on its sampled native path. During a longer idle
  // interval, show only the final approach to the next known contact.
  const began = sameJob ? start.second : Math.max(start.second,end.second-.30);
  const pointAt = second => {
    const fraction = Math.max(0,Math.min(1,(second-began)/(end.second-began)));
    const u = sameJob || alongRim ? fraction : fraction*fraction*(3-2*fraction);
    const [x,y] = start.center, [a,b] = end.center;
    if (alongRim) {
      const angle = Math.atan2(y,x)+step*Math.PI/4*u;
      const radius = Math.hypot(x,y)*(1-u)+Math.hypot(a,b)*u;
      return [radius*Math.cos(angle),radius*Math.sin(angle)];
    }
    return [x+(a-x)*u,y+(b-y)*u];
  };
  return {began,stop:end.second,pointAt};
}

function handMarkers(rows, index, movement, second) {
  const recent = rows[index];
  if (recent && second-recent.second <= (recent.points.length > 1 ? .012 : 1e-7)) return recent.points;
  if (movement && second >= movement.began) return [movement.pointAt(second)];
  if (recent && second-recent.second <= .20) return recent.points;
  return [];
}
