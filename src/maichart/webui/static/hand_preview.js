const el = id => document.getElementById(id);
const slots = ['', 'EASY', 'BASIC', 'ADVANCED', 'EXPERT', 'MASTER', 'Re:MASTER', 'UTAGE'];
const mediaPattern = /\.(mp3|ogg|wav|flac|m4a|aac|opus|mp4)$/i;
const pathOf = file => file.webkitRelativePath || file.name;
const parentOf = file => pathOf(file).split('/').slice(0, -1).join('/');
const basename = file => file.name.replace(/\.[^.]+$/, '').toLowerCase();
let charts = [], media = [], matching = [], meta, runtime, playback;
const chartTitles = new WeakMap();
let selection = 0, request, working = false;

function message(text, error=false) {
  el('progress').hidden = !text;
  el('progress').textContent = text;
  el('progress').classList.toggle('error', error);
}
function enable() {
  el('calculate').disabled = working || !meta || !runtime?.available;
}
async function json(url, options={}) {
  const response = await fetch(url, options);
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || response.statusText);
  return value;
}
function post(url, body, signal) {
  return json(url, {method:'POST', body, signal, headers:{'X-ChroMAI-Preview':'1'}});
}
async function checkRuntime() {
  try {
    runtime = await json('/api/renderer');
    el('install').hidden = runtime.available || runtime.installing;
    el('runtimeStatus').textContent = runtime.available
      ? (runtime.tinted ? '' : '当前 MiaView 支持原色与手部轨迹。音符着色需要手色版播放器。')
      : (runtime.error || runtime.message || '首次使用需要安装本地播放器');
    if (runtime.installing) setTimeout(checkRuntime, 800);
  } catch (e) { el('runtimeStatus').textContent = '无法连接预览服务 · '+e.message; }
  enable();
}
el('install').onclick = async () => {
  try { await post('/api/renderer/install'); await checkRuntime(); }
  catch (e) { message(e.message, true); }
};
function reset() {
  selection++;
  request?.abort();
  request = new AbortController();
  working = false;
  playback?.dispose();
  playback = null;
  el('mount').replaceChildren();
  el('stage').hidden = true;
  enable();
  return selection;
}
function audioOptions(chart) {
  matching = media.filter(file => parentOf(file) === parentOf(chart));
  el('audio').replaceChildren(new Option('无配乐', '-1'), ...matching.map((f,i)=>new Option(f.name, i)));
  const index = matching.findIndex(f => basename(f)==='track');
  const same = matching.findIndex(f => basename(f)===basename(chart));
  el('audio').value = String(index>=0 ? index : same>=0 ? same : matching.length===1 ? 0 : -1);
}
async function inspect() {
  const file = charts[Number(el('chart').value)];
  meta = null;
  const version = reset();
  el('slot').replaceChildren();
  audioOptions(file);
  message('正在读取谱面…');
  try {
    const info = await post('/api/chart', file, request.signal);
    if (version !== selection) return;
    meta = info;
    if (info.title) chartTitles.set(file, info.title);
    for (const {slot,level} of info.slots) el('slot').add(new Option(slots[slot]+' '+level, slot));
    el('slot').value = String(info.slots.find(d=>d.slot===5)?.slot ?? info.slots.at(-1).slot);
    el('chart').selectedOptions[0].textContent = (info.title ? info.title+' · ' : '')+pathOf(file);
    message('');
  } catch (e) { if (version===selection) message(e.message, true); }
  enable();
}
function filterCharts(force=false) {
  const old = el('chart').value;
  const query = el('search').value.trim().toLowerCase();
  const options = [];
  charts.forEach((file,i)=>{
    const title = chartTitles.get(file);
    const label = title ? title+' · '+pathOf(file) : pathOf(file);
    if (label.toLowerCase().includes(query)) options.push(new Option(label, i));
  });
  el('chart').replaceChildren(...options);
  el('fileCount').textContent = options.length+' / '+charts.length+' 个谱面文件';
  if(!options.length) {
    meta=null;reset();el('slot').replaceChildren();el('audio').replaceChildren();message('没有匹配的谱面。');
    return;
  }
  if (!force && options.some(option=>option.value===old)){el('chart').value=old;return;}
  inspect();
}
el('search').oninput = () => filterCharts();
function acceptFiles(list) {
  const all = Array.from(list);
  const found = all.filter(f=>/\.(txt|maidata)$/i.test(f.name));
  if (!found.length) { message('所选内容中没有 .txt 或 .maidata 谱面。', true); return; }
  charts = found.sort((a,b)=>Number(/^maidata\./i.test(b.name))-Number(/^maidata\./i.test(a.name))
                                 || pathOf(a).localeCompare(pathOf(b),'zh-CN'));
  media = all.filter(f=>mediaPattern.test(f.name));
  el('selection').hidden = false;
  el('searchWrap').hidden = charts.length<2;
  el('search').value = '';
  filterCharts(true);
}
for (const [button, input] of [['pickFolder','folder'],['pickFiles','files'],['pickAudio','audioFile']]) {
  el(button).onclick = () => el(input).click();
}
for (const input of ['folder','files']) {
  el(input).onchange = () => {
    if (el(input).files.length) acceptFiles(el(input).files);
    el(input).value = '';
  };
}
el('audioFile').onchange = () => {
  const file = el('audioFile').files[0];
  if (!file) return;
  const index = matching.push(file)-1;
  el('audio').add(new Option(file.name, index));
  el('audio').value = String(index);
  el('audioFile').value = '';
  reset(); message('');
};
el('chart').onchange = inspect;
for (const id of ['slot','audio']) el(id).onchange = () => { reset(); message(''); };

el('calculate').onclick = async () => {
  if (!meta || working) return;
  const version = reset();
  const signal = request.signal;
  working = true; enable();
  try {
    const slot = Number(el('slot').value), sources = {};
    for (const [style,name] of [['continuous','连扫优先'],['relay','分段换手']]) {
      message('正在计算「'+name+'」 · '+(style==='continuous'?'1/2':'2/2'));
      sources[style] = await post('/api/compute/'+slot+'/'+style, meta.text, signal);
      if (version !== selection) return;
    }
    message('正在比较左右手候选，生成置信标注…');
    const annotation = await post('/api/compute/'+slot+'/labels', meta.text, signal);
    if(version!==selection)return;
    message('正在载入 MiaView…');
    const sdk = await import(runtime.sdk_url);
    if (version !== selection) return;
    el('title').textContent = meta.title || charts[Number(el('chart').value)].name;
    const difficulty = meta.slots.find(d=>d.slot===slot);
    el('details').textContent = [slots[slot]+' '+difficulty.level,meta.artist,difficulty.designer].filter(Boolean).join(' · ');
    el('stage').hidden = false;
    el('stage').inert = true;
    const player = sdk.createPreview(el('mount'), {rendererUrl:runtime.renderer_url,
      resolution:{width:720,height:720},maxFps:'unlimited',theme:'dark',diagnostics:false});
    playback = createPlayback(player, sources, matching[Number(el('audio').value)], annotation.labels);
    await playback.load(meta.text, slot);
    if (version===selection) { el('stage').inert=false; message(''); }
  } catch (e) {
    if (version===selection) { reset(); message('预览失败 · '+e.message, true); }
  } finally { if (version===selection) { working=false; enable(); } }
};

function createPlayback(player, sources, file, labels) {
  let alive=true, ready=false, playing=false, rate=1, position=0, stamp=performance.now();
  let duration=1, overlay, renderer, markers=[], frameId, interval, polling=false, revision=0;
  let pendingSeek=null, seeking=false, lastSeek=0, lastDiff='', lastLabel=-1;
  const colors={}, lookup={}, slow = new Audio();
  const blob = file ? URL.createObjectURL(file) : null;
  if (blob) slow.src=blob;
  const now = () => Math.max(0,Math.min(duration,playing && rate!==1 && blob && !slow.ended ? slow.currentTime :
                              position+(playing?(performance.now()-stamp)*rate/1000:0)));
  function set(t, isPlaying) { position=t; playing=isPlaying; stamp=performance.now(); }
  function fail(e) { if(alive) message('播放失败 · '+e.message,true); }
  async function sync() {
    if (!ready || rate!==1 || polling || seeking) return;
    polling=true;
    const was=revision;
    try { const s=await player.getState(); if(alive && was===revision && rate===1) set(s.currentTime,s.playing); }
    catch(e){fail(e)} finally{polling=false}
  }
  // Coalesce scrub and slow-motion seeks so the WASM player never queues up a replay.
  async function move(t) {
    pendingSeek=t;
    if(seeking)return;
    seeking=true;
    try {
      while(alive && pendingSeek!==null){ const target=pendingSeek;pendingSeek=null;await player.seek(target); }
    } catch(e){fail(e)} finally{seeking=false}
  }
  function seek(t) {
    revision++; t=Math.max(0,Math.min(duration,t)); set(t,playing);
    if(blob && rate!==1)slow.currentTime=t;
    return move(t);
  }
  async function pause() {
    revision++; const t=now(); slow.pause();set(t,false);await player.pause();return t;
  }
  async function play() {
    if(!ready)return;
    const t=now()>=duration-.01?0:now();revision++;
    if(rate===1){await player.seek(t);await player.play();}
    else {
      await player.pause();await player.seek(t);
      if(blob){slow.playbackRate=rate;slow.currentTime=t;await slow.play();}
    }
    if(alive)set(t,true);
  }
  function showDiff(t, style) {
    const element=el('diff'), other=style==='continuous'?'relay':'continuous';
    const key=style+Math.floor(t*20);
    if(lastDiff===key)return;
    lastDiff=key;
    const differences=[];
    for(const [id,kind,pad,onset,trace,end,,shape] of markers){
      if(onset>t+.48 || Math.max(onset,end)<t-.025)continue;
      const a=lookup[style].get(id), b=lookup[other].get(id);
      if(!a||!b)continue;
      const compare=(label,x,y)=>{if(x&&y&&x!==y)differences.push(label+' '+(x===1?'左':'右')+'手')};
      if(onset>=t-.025 || kind.includes('hold')) compare(kind==='slide'||kind==='wifi'?shape+' 星头':pad,a[1],b[1]);
      if(kind==='slide'){
        const owner=row=>row[4].reduce((h,[sec,next])=>sec<=Math.max(trace,t)?next:h,row[2]);
        compare(shape+' 滑行',owner(a),owner(b));
      } else if(kind==='wifi')compare(shape+' 中线',a[5],b[5]);
    }
    element.hidden=!differences.length;
    if(differences.length)element.textContent='与另一方案分手不同 · '+differences.slice(0,5).join('　');
  }
  function draw() {
    if(!alive || !ready)return;
    const t=now(),style=el('style').value,mode=el('mode').value;
    overlay.draw(sources[style],t,{visible:mode==='paths',upcoming:el('future').checked});
    if(renderer && mode==='notes')showDiff(t,style);
    if(mode==='labels' && Math.floor(t*30)!==lastLabel){
      lastLabel=Math.floor(t*30);
      const nearby=labels.filter(n=>(n.trace??n.start)<=t+.12 && n.end>=t-.09);
      const grades={supported:'较强',lean:'倾向',uncertain:'不确定'};
      const reasons={hold_anchor:'重复键接 Hold，另一键在松手时应答', styles_differ:'两种方案不同',
        partial_route:'整条动作无法归给一只手', missing:'缺少分手记录', assignment_changed:'不同求解阶段分手不同',
        insufficient_margin:'左右候选接近或证据不足', rival_paths:'比较落点及后续候选'};
      el('labelStatus').textContent='原色保留不确定项 · '+nearby.slice(0,6).map(n=>{
        const pad=n.pad+(n.endPad?'→'+n.endPad:n.kind==='hold'?'h':'');
        return pad+' '+(n.hand===1?'左手 · ':n.hand===2?'右手 · ':'')+grades[n.confidence];
      }).join('　');
      el('labelStatus').title=nearby.slice(0,6).map(n=>n.pad+' · '+reasons[n.reason]).join('\n');
    }
    el('clock').textContent=Math.floor(t/60)+':'+(t%60).toFixed(3).padStart(6,'0');
    el('bar').value=t;el('play').textContent=playing?'暂停':'播放';
    if(playing && rate!==1){
      if(t>=duration-.005){slow.pause();set(t,false);}
      if(performance.now()-lastSeek>=32){lastSeek=performance.now();move(t);}
    }
    frameId=requestAnimationFrame(draw);
  }
  function recolor() {
    const mode=el('mode').value;
    renderer?.assign(mode==='labels'?colors.labels:mode==='notes'?colors[el('style').value]:[]);
    el('futureWrap').hidden=mode!=='paths'; el('legend').hidden=mode!=='notes'&&mode!=='labels';
    el('labelStatus').hidden=mode!=='labels';lastLabel=-1;
    el('diff').hidden=true;lastDiff='';
  }
  async function load(text,slot){
    await player.ready;
    if(!alive)return;
    const frame=el('mount').querySelector('iframe');
    overlay=mountHandOverlay(frame);
    await player.load({maidata:text,difficulty:slot,...(file?{audio:file}:{})});
    if(!alive)return;
    const state=await player.getState();
    if(!alive)return;
    duration=state.duration || sources.continuous.duration || 1;
    renderer=frame.contentWindow.miaHandNoteColors;
    if(renderer){
      markers=renderer.notes();
      const marked=labels.map(n=>n.trace===undefined
        ? {...n,head:n.hand} : {...n,track:n.hand===null?[]:[[n.trace,n.hand]]});
      colors.labels=simulationColorTable(markers,marked,true);
      for(const [style,source] of Object.entries(sources)){
        colors[style]=simulationColorTable(markers,source.noteHands);
        lookup[style]=new Map(colors[style].map(row=>[row[0],row]));
      }
    }
    el('mode').querySelector('option[value=notes]').disabled=!renderer;
    el('mode').querySelector('option[value=labels]').disabled=!renderer;
    el('mode').value=renderer?'labels':'paths';
    el('style').value='continuous';el('bar').max=duration;el('speed').value='1';
    ready=true;recolor();draw();interval=setInterval(sync,180);
    el('mode').onchange=recolor;
    el('style').onchange=()=>{if(el('mode').value==='labels')el('mode').value='notes';recolor()};
    el('compare').onclick=()=>{el('style').value=el('style').value==='continuous'?'relay':'continuous';el('style').onchange()};
    el('bar').oninput=()=>seek(Number(el('bar').value));
    el('play').onclick=()=>(playing?pause():play()).catch(fail);
    for(const [key,offset] of [['prev',-1/48],['next',1/48],['start',null]]){
      el(key).onclick=async()=>{try{const t=await pause();if(alive)await seek(offset===null?0:t+offset);}catch(e){fail(e)}};
    }
    el('speed').onchange=async()=>{
      try{const active=playing,t=await pause();if(!alive)return;rate=Number(el('speed').value);await seek(t);if(active&&alive)await play();}
      catch(e){fail(e)}
    };
    player.on('playing',()=>{if(!alive)return;if(rate!==1){slow.pause();rate=1;el('speed').value='1';}sync()});
    player.on('paused',sync);player.on('seeked',sync);
  }
  function dispose(){
    alive=false;ready=false;clearInterval(interval);cancelAnimationFrame(frameId);slow.pause();
    if(blob){slow.removeAttribute('src');slow.load();URL.revokeObjectURL(blob);}
    player.destroy();
  }
  return {load, dispose};
}
el('full').onclick=()=>document.fullscreenElement?document.exitFullscreen():el('stage').requestFullscreen();
checkRuntime();
