// Contacts from the hand solver are in units of the outer sensor radius (205).
// MiaView's judgment line is at 42% of the short side of its native viewport.
function mountHandOverlay(iframe) {
  const document = iframe.contentDocument;
  const viewport = document?.getElementById('viewport');
  if (!viewport) throw new Error('未找到 MiaView 的谱面画布');
  const canvas = document.createElement('canvas');
  canvas.id = 'chromai-hand-overlay';
  canvas.setAttribute('aria-hidden', 'true');
  Object.assign(canvas.style, {
    position: 'absolute', inset: '0', width: '100%', height: '100%',
    pointerEvents: 'none', zIndex: '2',
  });
  viewport.appendChild(canvas);
  const g = canvas.getContext('2d');

  const histories = new WeakMap();

  function arrow(a, b, color, scale) {
    const angle = Math.atan2(b[1] - a[1], b[0] - a[0]);
    const [x,y] = b;
    g.beginPath();
    g.moveTo(x, y);
    for (const direction of [-.54, .54]) {
      g.lineTo(x - 14 * scale * Math.cos(angle + direction), y - 14 * scale * Math.sin(angle + direction));
    }
    g.closePath();
    g.fillStyle = color;
    g.fill();
  }

  function trail(movement, start, stop, screen, color, scale, future) {
    if (stop <= start) return;
    const steps = Math.max(1, Math.ceil((stop-start)/.016));
    const points = Array.from({length:steps+1}, (_,i)=>screen(movement.pointAt(start+(stop-start)*i/steps)));
    const distance = points.slice(1).reduce((acc,[x,y],i)=>acc+Math.hypot(x-points[i][0],y-points[i][1]),0);
    if (distance < 3*scale) return;
    g.beginPath();
    g.moveTo(...points[0]);
    for (const point of points.slice(1)) g.lineTo(...point);
    g.setLineDash(future ? [9*scale,7*scale] : []);
    g.strokeStyle = '#10151d';
    g.lineWidth = (future ? 8 : 12) * scale;
    g.stroke();
    g.strokeStyle = color;
    g.lineWidth = (future ? 4 : 7) * scale;
    g.stroke();
    g.setLineDash([]);
    if (distance >= 30*scale) {
      const middle = Math.ceil(steps*.58);
      arrow(points[middle-1],points[middle],color,scale);
    }
  }

  function draw(source, position, {visible = true, upcoming = true} = {}) {
    const width = viewport.clientWidth, height = viewport.clientHeight;
    if (!width || !height) return;
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
    g.clearRect(0, 0, width, height);
    canvas.hidden = !visible;
    if (!visible || !source?.frames.length) return;
    const size = Math.min(width, height);
    const scale = size / 720;
    const radius = size * .42;
    const screenPoint = ([x,y]) => [width/2+x*radius,height/2+y*radius];
    const frames = source.frames;
    if (source.external) {
      const frame = frames[handFrameIndex(frames,position)];
      if (!frame || position - frame.second > .12) return;
      for (const hit of frame.contacts) {
        const [x,y] = screenPoint(handPosition(hit));
        g.beginPath();
        g.arc(x,y,22 * scale,0,Math.PI * 2);
        g.lineWidth = 5 * scale;
        g.strokeStyle = '#efe7ae';
        g.stroke();
      }
      return;
    }
    const styles = source.anonymous
      ? [['left', '#6ddcc5', 'A'], ['right', '#d1a4ff', 'B']]
      : [['left', '#55bdff', 'L'], ['right', '#ff9d50', 'R']];
    const markers = [];
    g.lineCap = 'round';
    g.lineJoin = 'round';
    if (!histories.has(source)) {
      histories.set(source,Object.fromEntries(styles.map(([side])=>[side,handSamples(frames,side)])));
    }
    for (const [side,color,letter] of styles) {
      const rows = histories.get(source)[side];
      const index = handFrameIndex(rows,position);
      const end = upcoming ? position+HAND_LOOKAHEAD_SECONDS : position;
      const start = position-HAND_TRAIL_SECONDS;
      let active = null;
      for (let i = Math.max(0,handFrameIndex(rows,start)); i+1 < rows.length && rows[i].second < end; i++) {
        const motion = handMovement(rows[i],rows[i+1]);
        if (i === index) active = motion;
        if (!motion) continue;
        const playedEnd = Math.min(position,motion.stop);
        g.globalAlpha = Math.max(.27,.91-(position-playedEnd)*.78);
        trail(motion,Math.max(start,motion.began),playedEnd,screenPoint,color,scale,false);
        if (upcoming) {
          g.globalAlpha = .54;
          trail(motion,Math.max(position,motion.began),Math.min(end,motion.stop),screenPoint,color,scale,true);
        }
      }
      for (const point of handMarkers(rows,index,active,position)) {
        const [x,y] = screenPoint(point);
        markers.push({x,y,color,letter});
      }
    }
    g.globalAlpha = 1;
    for (const marker of markers) {
      const {x,y,color,letter} = marker;
      const separation = markers.reduce((distance,other)=>other.letter===letter ? distance :
        Math.min(distance,Math.hypot(other.x-x,other.y-y)/scale),27);
      const offset = (letter==='L'||letter==='A'?-14:14) * scale * (1-separation/27);
      g.beginPath();
      g.arc(x+offset,y,25*scale,0,Math.PI*2);
      g.globalAlpha = .20;
      g.fillStyle = color;
      g.fill();
      g.globalAlpha = 1;
      g.lineWidth = 10*scale;
      g.strokeStyle = '#10151d';
      g.stroke();
      g.lineWidth = 5*scale;
      g.strokeStyle = color;
      g.stroke();
      g.font = `bold ${21*scale}px sans-serif`;
      g.textAlign = 'center';
      g.textBaseline = 'middle';
      g.lineWidth = 6*scale;
      g.strokeStyle = '#10151d';
      g.strokeText(letter,x+offset,y);
      g.fillStyle = color;
      g.fillText(letter,x+offset,y);
    }
  }
  return {draw};
}
