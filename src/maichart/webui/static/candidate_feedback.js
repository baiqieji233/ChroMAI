"use strict";
function candidateSavedReviews(records) {
  const positive=records.filter(row=>row.label==='good'),rest=[];
  for(const row of records.filter(item=>item.label==='uncertain')) {
    let pieces=[[row.start_sec,row.end_sec]];
    for(const approved of positive)pieces=pieces.flatMap(([start,end])=>{
      if(approved.end_sec<=start||approved.start_sec>=end)return [[start,end]];
      return [...(start<approved.start_sec?[[start,approved.start_sec]]:[]),...(end>approved.end_sec?[[approved.end_sec,end]]:[])];
    });
    for(const [start_sec,end_sec] of pieces)rest.push({...row,start_sec,end_sec});
  }
  const seen=new Set();
  return [...positive,...rest].sort((a,b)=>a.start_sec-b.start_sec).filter(row=>{
    const key=JSON.stringify([row.label,row.start_sec,row.end_sec,row.reason||'']);if(seen.has(key))return false;seen.add(key);return true;
  });
}
// Review complete charts first; one shared audio timeline defines an optional span.
(() => {
  const node=id=>document.getElementById(id),canvas=node('feedbackTimeline');
  if(!canvas)return;
  let entry=null,currentIndex=0,anchorIndex=0,range=null,media=null,mediaAudio=null,viewStart=0,viewEnd=0,drag=null,mediaSerial=0,busy=false,comparisonSwitch=false,timeline=null,timelineSerial=0,playhead=0,playing=false;
  let navigationTarget=null,navigationFrame=null,navigationBusy=false,navigationSerial=0;
  let playbackDuration=0;
  let collectionSerial=0,trainingBusy=false,collection={items:[],regions:0};
  let entryChoiceKey='';
  const chart=()=>entry?.charts[currentIndex];
  const audioKey=version=>version?.audio_sha256||version?.audio_path||version?.audio_url||'';
  const complete=version=>version && !version.partial_preview && version.complete!==false && (!version.content_kind || version.content_kind==='complete_chart');
  const status=text=>{node('feedbackStatus').textContent=text;};
  const duration=()=>Math.max(0,playbackDuration,Number(timeline?.duration_sec||media?.duration_sec||chart()?.audio_duration_sec||chart()?.duration_sec||0));
  const validRange=()=>range && Number.isFinite(range.start_sec) && Number.isFinite(range.end_sec) && range.start_sec>=0 && range.end_sec>range.start_sec;
  function audition() {
    if(!validRange())return null;
    const context=node('feedbackContext').checked?4:0,limit=Math.max(duration(),range.end_sec);
    return {startSec:Math.max(0,range.start_sec-context),endSec:limit?Math.min(limit,range.end_sec+context):range.end_sec+context};
  }
  function updateRange() {
    const source=entry?.source_kind==='training_source';
    node('feedbackStart').value=range?.start_sec??'';node('feedbackEnd').value=range?.end_sec??'';
    node('feedbackRange').textContent=validRange()?`你正在评价：${range.start_sec.toFixed(2)}–${range.end_sec.toFixed(2)} 秒${selectionSummary()}`:range?'请选择终点':source?'拖选片段后评价':'你正在评价：整张谱面';
    const listen=audition();node('feedbackAuditionRange').textContent=listen?`试听范围：${listen.startSec.toFixed(2)}–${listen.endSec.toFixed(2)} 秒；反馈只标记所选范围。`:source?'在时间轴拖选片段后，可以试听并标注。':'当前未选择片段，评价范围为整张谱面。';
    node('feedbackPlayRange').disabled=!validRange();
    node('feedbackTrainingRange').textContent=validRange()?`当前选择：${range.start_sec.toFixed(2)}–${range.end_sec.toFixed(2)} 秒${selectionSummary()}`:'请先在谱面时间轴选择片段。';
    trainingButtons();
    feedbackButtons();
    node('feedbackSelectedSource').textContent=validRange()?selectedNotes().map(note=>`${note.start_sec.toFixed(3)}s · 第${note.beat}拍 · ${note.raw}`).filter((line,index,all)=>all.indexOf(line)===index).join('\n')||'所选范围没有音符起点。':'尚未选择片段。';draw();
  }
  const colors={tap:'#d84b89',hold:'#d59c24',slide:'#299bdb',touch:'#8e61c9',touch_hold:'#8e61c9'};
  const noteLane=note=>String(note.kind).startsWith('touch')?8:Math.max(0,Math.min(7,Number(note.head)-1));
  const beatNumber=value=>{const parts=String(value).split('/').map(Number);return parts.length===2?parts[0]/parts[1]:parts[0];};
  function selectedNotes() {return !validRange()?[]:(timeline?.notes||[]).filter(note=>note.start_sec>=range.start_sec-1e-7 && note.start_sec<range.end_sec-1e-7);}
  function selectionSummary() {
    const notes=selectedNotes();if(!notes.length)return ' · 无音符起点';
    const ordered=[...notes].sort((a,b)=>beatNumber(a.beat)-beatNumber(b.beat));
    const last=notes.reduce((latest,note)=>beatNumber(note.end_beat||note.beat)>beatNumber(latest.end_beat||latest.beat)?note:latest,ordered[0]);
    return ` · 第${ordered[0].beat}–${last.end_beat||last.beat}拍 · ${notes.length}音符／${new Set(notes.map(note=>note.start_sec)).size}组`;
  }
  function clearLoop() {globalThis.ChartPreview?.clearRange();}
  function resetEntryChoice() {const input=node('feedbackLearnEntry');if(input)input.checked=false;}
  function setRange(value) {clearLoop();resetEntryChoice();range=value;updateRange();}
  globalThis.FeedbackRange={get:()=>validRange()?{...range}:null,set:value=>{setRange(value);if(validRange())navigate(range.start_sec);}};
  window.addEventListener('feedback-set-range',event=>globalThis.FeedbackRange.set(event.detail));
  function setView(start,span) {
    const full=duration();if(!full)return;
    const width=Math.min(full,Math.max(.25,span));viewStart=Math.max(0,Math.min(full-width,start));viewEnd=viewStart+width;draw();
  }
  function resetNavigation() {
    ++navigationSerial;navigationTarget=null;
    if(navigationFrame!==null)cancelAnimationFrame(navigationFrame);
    navigationFrame=null;
  }
  function scheduleNavigation() {
    if(navigationBusy || navigationFrame!==null || navigationTarget===null)return;
    navigationFrame=requestAnimationFrame(async()=>{
      navigationFrame=null;
      if(navigationBusy || navigationTarget===null)return;
      const target=navigationTarget,serial=navigationSerial;
      navigationTarget=null;navigationBusy=true;
      try {await globalThis.ChartPreview?.seek(target);}
      catch(error) {if(serial===navigationSerial)status(error.message);}
      finally {navigationBusy=false;scheduleNavigation();}
    });
  }
  // There is one time cursor for the editor and the player. Fast wheel/slider
  // gestures replace the pending seek rather than building a stale IPC queue.
  function navigate(second,{start=null}={}) {
    if(!timeline || !Number.isFinite(second))return;
    clearLoop();playhead=Math.max(0,Math.min(duration(),second));
    const span=viewEnd-viewStart||16;
    setView(start===null?playhead-span*.25:start,span);
    navigationTarget=playhead;scheduleNavigation();
  }
  function zoom(span) {
    const ratio=(playhead-viewStart)/Math.max(.01,viewEnd-viewStart);
    setView(playhead-Math.max(0,Math.min(1,ratio))*span,span);
  }
  function draw() {
    const box=canvas.getBoundingClientRect();if(!box.width)return;
    canvas.width=Math.max(300,Math.round(box.width));canvas.height=Math.max(120,Math.round(box.height));const ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height,left=48,top=46,row=(h-top-14)/9,span=Math.max(.01,viewEnd-viewStart),x=time=>left+(time-viewStart)/span*(w-left-8),y=lane=>top+(lane+.5)*row;
    ctx.fillStyle='#f8fafc';ctx.fillRect(0,0,w,h);ctx.font='11px sans-serif';
    if(node('feedbackShowWave').checked && media && mediaAudio===audioKey(chart())) {
      const wave=media.waveform;ctx.strokeStyle='#7d92a133';ctx.beginPath();
      for(let px=left;px<w;px++) {const i=Math.floor((viewStart+(px-left)/(w-left)*span)/wave.step_sec),low=wave.minimum[i]||0,high=wave.maximum[i]||0;ctx.moveTo(px,h/2-high*h*.35);ctx.lineTo(px,h/2-low*h*.35);}ctx.stroke();
    }
    for(let lane=0;lane<9;lane++) {ctx.strokeStyle='#dce3eb';ctx.beginPath();ctx.moveTo(left,top+lane*row);ctx.lineTo(w,top+lane*row);ctx.stroke();ctx.fillStyle='#526174';ctx.fillText(lane===8?'Touch':String(lane+1),6,y(lane)+4);}
    const secondStep=span>80?20:span>32?10:span>12?4:span>5?2:1;
    for(let second=Math.ceil(viewStart/secondStep)*secondStep;second<=viewEnd;second+=secondStep) {ctx.strokeStyle='#ced7e3';ctx.beginPath();ctx.moveTo(x(second),top);ctx.lineTo(x(second),h);ctx.stroke();ctx.fillStyle='#536376';ctx.fillText(second.toFixed(secondStep<1?1:0)+'s',x(second)+2,12);}
    let lastBeatX=-999;
    for(const beat of timeline?.beats||[]) {if(beat.time_sec<viewStart || beat.time_sec>viewEnd)continue;const px=x(beat.time_sec);if(px-lastBeatX<42)continue;lastBeatX=px;ctx.fillStyle='#78859a';ctx.fillText('拍'+beat.beat,px+2,30);ctx.strokeStyle='#e4e9f0';ctx.beginPath();ctx.moveTo(px,top-5);ctx.lineTo(px,top);ctx.stroke();}
    if(validRange()) {ctx.fillStyle='#6491e22b';ctx.fillRect(x(range.start_sec),top,x(range.end_sec)-x(range.start_sec),h-top);}
    ctx.save();ctx.beginPath();ctx.rect(left,top,w-left,h-top);ctx.clip();
    for(const note of timeline?.notes||[]) {
      if((note.end_sec??note.start_sec)<viewStart || note.start_sec>viewEnd)continue;
      const px=x(note.start_sec),py=y(noteLane(note)),color=note.is_break?'#e77632':colors[note.kind]||'#747f91';ctx.strokeStyle=color;ctx.fillStyle=color;
      if(Number.isFinite(note.end_sec)) {
        if(note.kind==='slide') {
          const trace=Number.isFinite(note.trace_start_sec)?note.trace_start_sec:note.start_sec;
          ctx.setLineDash([3,3]);ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(px,py);ctx.lineTo(x(trace),py);ctx.stroke();ctx.setLineDash([]);
          const segments=note.segments?.length?note.segments:[{start_sec:trace,end_sec:note.end_sec,head:note.head,end_head:note.end_head}];
          for(const part of segments) {
            const fromY=y(Math.max(0,Math.min(7,Number(part.head)-1))),toY=y(Math.max(0,Math.min(7,Number(part.end_head)-1))),fromX=x(part.start_sec),toX=x(part.end_sec);
            ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(fromX,fromY);ctx.lineTo(toX,toY);ctx.stroke();const angle=Math.atan2(toY-fromY,toX-fromX);ctx.beginPath();ctx.moveTo(toX,toY);ctx.lineTo(toX-7*Math.cos(angle-.55),toY-7*Math.sin(angle-.55));ctx.lineTo(toX-7*Math.cos(angle+.55),toY-7*Math.sin(angle+.55));ctx.closePath();ctx.fill();
          }
        } else {ctx.globalAlpha=.42;ctx.fillRect(px,py-5,Math.max(2,x(note.end_sec)-px),10);ctx.globalAlpha=1;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x(note.end_sec),py-7);ctx.lineTo(x(note.end_sec),py+7);ctx.stroke();}
      }
      if(note.has_head===false)continue;
      ctx.lineWidth=2;ctx.beginPath();
      if(note.kind==='slide') {for(let point=0;point<10;point++){const angle=-Math.PI/2+point*Math.PI/5,r=point%2?3.5:8;const hx=px+Math.cos(angle)*r,hy=py+Math.sin(angle)*r;if(!point)ctx.moveTo(hx,hy);else ctx.lineTo(hx,hy);}ctx.closePath();ctx.stroke();}
      else {ctx.arc(px,py,note.kind.startsWith('touch')?6:6.5,0,Math.PI*2);ctx.stroke();if(note.kind==='hold'||note.kind==='touch_hold'){ctx.beginPath();ctx.arc(px,py,2.5,0,Math.PI*2);ctx.fill();}}
      if(note.unsupported){ctx.fillStyle='#9a504b';ctx.fillText('?',px+6,py-6);}
    }
    if(playhead>=viewStart && playhead<=viewEnd) {ctx.strokeStyle='#273fca';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(x(playhead),top);ctx.lineTo(x(playhead),h);ctx.stroke();}
    ctx.restore();
    canvas.dataset.viewStart=String(viewStart);canvas.dataset.viewEnd=String(viewEnd);
    node('feedbackViewRange').textContent=`当前位置 ${playhead.toFixed(2)}秒 · 视图 ${viewStart.toFixed(1)}–${viewEnd.toFixed(1)}秒`;
    const scroll=node('feedbackScroll');scroll.max=String(duration());scroll.value=String(playhead);scroll.disabled=duration()<=0;
    const overview=node('feedbackOverview');overview.width=w;const oc=overview.getContext('2d'),oh=overview.height,full=duration()||1;oc.fillStyle='#edf1f7';oc.fillRect(0,0,w,oh);oc.strokeStyle='#8394b5';oc.beginPath();
    for(const note of timeline?.notes||[]) {const px=left+note.start_sec/full*(w-left-8);oc.moveTo(px,oh-5);oc.lineTo(px,Math.max(3,oh-8-noteLane(note)*2.5));}oc.stroke();oc.fillStyle='#5c86de33';oc.fillRect(left+viewStart/full*(w-left-8),0,span/full*(w-left-8),oh);oc.strokeStyle='#577abd';oc.strokeRect(left+viewStart/full*(w-left-8),1,span/full*(w-left-8),oh-2);
    oc.strokeStyle='#273fca';oc.lineWidth=2;oc.beginPath();const overviewCursor=left+playhead/full*(w-left-8);oc.moveTo(overviewCursor,0);oc.lineTo(overviewCursor,oh);oc.stroke();
  }
  async function loadTimeline({preserveView=false}={}) {
    const version=chart(),request=++timelineSerial;if(!version)return;
    timeline=null;node('feedbackTimelineStatus').textContent='正在载入谱面音符…';draw();
    try {
      const url=version.timeline_url||`/api/preview/timeline?id=${encodeURIComponent(entry.task_id)}&version=${encodeURIComponent(version.version_id)}`;
      let data=version.timeline;
      if(!data) {const response=await fetch(url,{cache:'no-store'});data=await response.json();if(!response.ok)throw new Error(data.error||'谱面时间轴载入失败');}
      if(request!==timelineSerial || version.version_id!==chart()?.version_id || version.path!==chart()?.path)return;
      timeline=data;
      if(!preserveView || viewEnd<=viewStart)setView(Math.max(0,playhead-2),16);else setView(viewStart,viewEnd-viewStart);
      node('feedbackTimelineStatus').textContent=`${data.notes.length} 个音符。滚轮前后定位；点击音符或拖选片段，同步定位到选区起点。缩放请用下方按钮。${data.warnings?.length?'部分写法仅显示可识别的音头，详情见提示。':''}`;
      updateRange();
    } catch(error) {if(request===timelineSerial){timeline=null;node('feedbackTimelineStatus').textContent=error.message;draw();}}
  }
  function refreshChoices() {
    const current=entry?.charts[anchorIndex],key=audioKey(current),available=(entry?.charts||[]).map((version,index)=>({version,index})).filter(({version})=>complete(version) && key && audioKey(version)===key);
    const previousB=node('feedbackCandidateB').value;
    for(const id of ['feedbackCandidateA','feedbackCandidateB']) {
      const select=node(id);select.replaceChildren();
      for(const {version,index} of available) {
        const option=document.createElement('option');option.value=String(index);option.textContent=version.label||version.title||`完整谱面 ${index+1}`;select.append(option);
      }
    }
    node('feedbackCandidateA').value=String(anchorIndex);
    const b=available.find(item=>String(item.index)===previousB && item.index!==anchorIndex)||available.find(item=>item.index!==anchorIndex);
    node('feedbackCandidateB').value=b?String(b.index):'';
    const paired=!!b;node('feedbackComparisonControls').hidden=!paired;node('feedbackPairChoices').hidden=!paired;node('feedbackCandidateB').disabled=!paired;node('feedbackPreviewB').disabled=!paired;
    const side=currentIndex===Number(node('feedbackCandidateA').value)?'版本 A':currentIndex===Number(node('feedbackCandidateB').value)?'版本 B':`第 ${currentIndex+1} 张`;
    node('feedbackCurrentVersion').textContent=entry?.source_kind==='training_source'?`训练素材 · ${chart()?.title||''}`:`正在预览：${side} · ${chart()?.label||chart()?.title||''}`;
    for(const [id,index] of [['feedbackPreviewA',anchorIndex],['feedbackPreviewB',b?.index]])node(id).setAttribute('aria-pressed',String(currentIndex===index));
    node('feedbackComparisonNotice').textContent=paired?'这两张完整谱面使用同一音频。切换时沿用位置、播放／暂停状态与所选范围。':'目前只有这一张完整谱面；可以直接标记这一段。有另一张同音频的完整版本后再比较。';
  }
  function refresh(value) {
    resetNavigation();
    resetEntryChoice();
    playhead=0;playing=false;playbackDuration=0;
    if(entry?.task_id!==value.task_id) {range=null;clearLoop();}
    collection={items:[],regions:0};++collectionSerial;node('feedbackTrainingList').replaceChildren();
    const previous=chart()?.path,previousAudio=audioKey(chart());entry=value;
    currentIndex=Math.max(0,value.charts.findIndex(version=>version.path===previous));anchorIndex=currentIndex;
    if(previousAudio!==audioKey(chart())) {playhead=0;playing=false;range=null;media=null;mediaAudio=null;clearLoop();++mediaSerial;}
    refreshChoices();updateRange();void loadTimeline();
    const source=entry.source_kind==='training_source';
    node('feedbackTraining').hidden=source;node('feedbackComparison').hidden=source;
    status(source?'训练素材 · 圈选后保存片段标注。':'生成结果 · 记录评价，或展开下方“收集训练示范”。');
    renderSaved(source?entry.saved_reviews||entry.annotations||[]:[]);
    if(!source)void loadCollection();
  }
  async function choose(index,{comparison=false}={}) {
    if(!complete(entry?.charts[index]))return;
    const oldAudio=audioKey(chart());comparisonSwitch=comparison;
    try {
      const bounds=validRange()?{startSec:range.start_sec,endSec:range.end_sec}:{};
      await globalThis.ChartPreview?.chooseChart(index,{...bounds,restart:false,source:comparison?'comparison':'main'});
      currentIndex=index;resetEntryChoice();if(!comparison)anchorIndex=index;
      if(oldAudio!==audioKey(chart())) {range=null;media=null;mediaAudio=null;clearLoop();++mediaSerial;}
      refreshChoices();updateRange();
    } catch(error) {status(error.message);} finally {comparisonSwitch=false;}
  }
  window.addEventListener('chart-preview-updated',event=>refresh(event.detail));
  window.addEventListener('training-source-annotated',event=>{
    const detail=event.detail;if(entry?.source_kind!=='training_source'||entry.training_context?.manifest!==detail.manifest||entry.training_context?.chart_id!==detail.chart_id)return;
    if(!Array.isArray(detail.result?.records))return;
    const record=detail.result.records.find(row=>row.chart_id===detail.chart_id);
    entry.annotations=record?.regions||[];
    entry.saved_reviews=entry.training_context?.candidate_pool?[...entry.annotations,...(record?.candidate_reviews||[]).filter(row=>row.label==='uncertain')]:entry.annotations;
    renderSaved(entry.saved_reviews);
  });
  window.addEventListener('chart-preview-versionchanged',event=>{
    const index=Number(event.detail?.index);if(!entry?.charts[index])return;
    resetNavigation();
    const oldAudio=audioKey(chart());currentIndex=index;
    if(!comparisonSwitch && event.detail?.source!=='comparison')anchorIndex=index;
    if(oldAudio!==audioKey(chart())) {playhead=0;playing=false;range=null;media=null;mediaAudio=null;clearLoop();++mediaSerial;}
    refreshChoices();updateRange();void loadTimeline({preserveView:oldAudio===audioKey(chart())});
  });
  node('feedbackCandidateA').addEventListener('change',()=>void choose(Number(node('feedbackCandidateA').value)));
  node('feedbackCandidateB').addEventListener('change',()=>{if(node('feedbackCandidateB').value===node('feedbackCandidateA').value)refreshChoices();});
  node('feedbackPreviewA').addEventListener('click',()=>void choose(Number(node('feedbackCandidateA').value),{comparison:true}));
  node('feedbackPreviewB').addEventListener('click',()=>void choose(Number(node('feedbackCandidateB').value),{comparison:true}));
  for(const boundary of ['Start','End'])node('feedbackMark'+boundary).addEventListener('click',async()=>{
    try {
      const state=await globalThis.ChartPreview?.getState(),time=Number(state?.currentTime);
      if(!state || !Number.isFinite(time))throw new Error('请先载入预览播放器。');
      if(boundary==='Start')range={start_sec:time,end_sec:null};
      else {const start=range?.start_sec??0;range={start_sec:Math.min(start,time),end_sec:Math.max(start,time)};}
      clearLoop();updateRange();
    } catch(error) {status(error.message);}
  });
  for(const id of ['feedbackStart','feedbackEnd'])node(id).addEventListener('change',()=>{
    const start=node('feedbackStart').value,end=node('feedbackEnd').value;
    setRange(start==='' && end===''?null:{start_sec:start===''?null:Number(start),end_sec:end===''?null:Number(end)});
  });
  node('feedbackContext').addEventListener('change',()=>{clearLoop();updateRange();});
  node('feedbackClear').addEventListener('click',()=>{setRange(null);});
  node('feedbackPlayRange').addEventListener('click',async()=>{try {const bounds=audition();if(bounds)await globalThis.ChartPreview?.playRange(bounds);}catch(error){status(error.message);}});
  node('feedbackPause').addEventListener('click',()=>void globalThis.ChartPreview?.pause());
  node('feedbackZoomIn').addEventListener('click',()=>zoom(Math.max(.25,(viewEnd-viewStart)/2)));
  node('feedbackZoomStepOut').addEventListener('click',()=>zoom(Math.min(duration(),(viewEnd-viewStart)*2)));
  node('feedbackZoomOut').addEventListener('click',()=>setView(0,duration()));
  node('feedbackDetailView').addEventListener('click',()=>setView(playhead-2,16));
  node('feedbackScroll').addEventListener('input',()=>navigate(Number(node('feedbackScroll').value)));
  node('feedbackFollowPlayback').addEventListener('change',()=>{if(node('feedbackFollowPlayback').checked)setView(playhead-(viewEnd-viewStart)*.25,viewEnd-viewStart);});
  node('feedbackShowWave').addEventListener('change',async()=>{
    if(!node('feedbackShowWave').checked){draw();return;}
    const version=chart();if(!version?.audio_path){status('当前谱面没有可用的音频路径。');node('feedbackShowWave').checked=false;return;}
    if(media && mediaAudio===audioKey(version)){draw();return;}
    const request=++mediaSerial,key=audioKey(version);
    try {
      status('正在载入可选波形背景（CPU，复用音频缓存）…');
      const created=await api('/api/composer-v2/media',{model_family:'chart_composer_v2',audio:version.audio_path,cache_dir:valueOrDefault('#v2GenerateCacheDir','cache/chart_composer_v2')});
      while(request===mediaSerial && key===audioKey(chart())) {
        const task=await api('/api/task?id='+encodeURIComponent(created.task_id));
        if(['failed','cancelled'].includes(task.status))throw new Error(task.error||'波形载入已取消');
        if(task.status==='completed') {media=task.result.editor_media;if(!media)throw new Error('没有返回波形数据');mediaAudio=key;draw();status('波形仅作对齐背景；直接选择谱面音符。');return;}
        await new Promise(resolve=>setTimeout(resolve,500));
      }
    } catch(error) {if(request===mediaSerial)status(error.message);}
  });
  function time(event) {const box=canvas.getBoundingClientRect();return Math.max(0,viewStart+Math.max(0,Math.min(1,(event.clientX-box.left-48)/(box.width-56)))*(viewEnd-viewStart));}
  function hit(event) {const box=canvas.getBoundingClientRect(),at=time(event),lane=Math.floor((event.clientY-box.top-46)/((canvas.height-60)/9)),tolerance=(viewEnd-viewStart)*9/Math.max(1,box.width-56);return (timeline?.notes||[]).filter(note=>note.has_head!==false && noteLane(note)===lane && Math.abs(note.start_sec-at)<=tolerance).sort((a,b)=>Math.abs(a.start_sec-at)-Math.abs(b.start_sec-at))[0];}
  function onsetRange(start,end,{clicked=false}={}) {
    const notes=timeline?.notes||[],onsets=[...new Set(notes.map(note=>note.start_sec))].sort((a,b)=>a-b);
    if(!onsets.length)return {start_sec:Math.min(start,end),end_sec:Math.max(start,end)};
    // Snap near actual notes; a drag inside a genuine gap must stay in that gap.
    const tolerance=Math.min(.2,(viewEnd-viewStart)*10/Math.max(1,canvas.width-56));
    const nearest=value=>{const closest=onsets.reduce((best,item)=>Math.abs(item-value)<Math.abs(best-value)?item:best,onsets[0]);return Math.abs(closest-value)<=tolerance?closest:value;};
    let a=Math.max(0,nearest(Math.min(start,end))),b=Math.max(0,nearest(Math.max(start,end)));
    if(clicked || b<=a) {
      const chord=notes.filter(note=>Math.abs(note.start_sec-a)<1e-7),next=onsets.find(item=>item>a+1e-7);
      b=Math.max(next??Math.min(duration(),a+.25),...chord.map(note=>note.end_sec??a));
      if(b<=a)b=a+.05;
    }
    return {start_sec:a,end_sec:b};
  }
  canvas.addEventListener('pointerdown',event=>{if(!timeline)return;event.preventDefault();canvas.setPointerCapture(event.pointerId);drag={start:time(event),x:event.clientX,y:event.clientY,note:hit(event),moved:false};});
  canvas.addEventListener('pointermove',event=>{
    const note=hit(event);canvas.title=note?`${note.kind} · ${note.head} · 第${note.beat}拍 · ${note.start_sec.toFixed(3)}秒${note.end_sec!=null?' → '+note.end_sec.toFixed(3)+'秒':''}\n${note.raw}${note.path?'\n路径 '+note.path:''}${note.unsupported?'\n未完整识别时值／轨迹':''}`:'';
    if(!drag)return;drag.moved ||= Math.hypot(event.clientX-drag.x,event.clientY-drag.y)>3;
    if(drag.moved)setRange(onsetRange(drag.start,time(event)));
  });
  canvas.addEventListener('pointerup',event=>{
    if(!drag)return;const previous=drag;drag=null;canvas.releasePointerCapture(event.pointerId);
    if(previous.moved) {setRange(onsetRange(previous.start,time(event)));navigate(range.start_sec,{start:viewStart});}
    else if(previous.note) {setRange(onsetRange(previous.note.start_sec,previous.note.start_sec,{clicked:true}));navigate(range.start_sec,{start:viewStart});}
    else navigate(time(event),{start:viewStart});
  });
  canvas.addEventListener('pointercancel',()=>{drag=null;});
  canvas.addEventListener('wheel',event=>{
    if(!timeline || event.ctrlKey || event.metaKey)return;
    event.preventDefault();const span=viewEnd-viewStart,delta=Math.abs(event.deltaX)>Math.abs(event.deltaY)?event.deltaX:event.deltaY;
    const pixels=delta*(event.deltaMode===1?16:event.deltaMode===2?canvas.clientWidth:1),seconds=pixels/Math.max(1,canvas.clientWidth-56)*span;
    const target=Math.max(0,Math.min(duration(),playhead+seconds));
    navigate(target,{start:viewStart+target-playhead});
  },{passive:false});
  node('feedbackOverview').addEventListener('pointerdown',event=>{const box=event.currentTarget.getBoundingClientRect();navigate(Math.max(0,Math.min(1,(event.clientX-box.left-48)/(box.width-56)))*duration());});
  window.addEventListener('chart-preview-playback',event=>{
    const state=event.detail||{};playing=!!state.playing;
    if(Number.isFinite(state.duration))playbackDuration=Math.max(0,state.duration);
    if(navigationBusy || navigationTarget!==null)return;
    if(Number.isFinite(state.currentTime))playhead=state.currentTime;
    const span=viewEnd-viewStart||16,outside=playhead<viewStart || playhead>viewEnd;
    if(timeline && !drag && ((!playing && outside) || (playing && node('feedbackFollowPlayback').checked && (outside || playhead>viewStart+span*.8))))setView(playhead-span*.25,span);
    else draw();
  });
  const resize=new ResizeObserver(()=>draw());resize.observe(canvas);
  const trainingNames={timing:'只学采音',layout:'只学配置',both:'采音和配置'};
  function feedbackButtons() {
    const source=entry?.source_kind==='training_source';
    for(const button of document.querySelectorAll('[data-generation-feedback]'))button.disabled=busy || (source && !validRange());
    for(const button of document.querySelectorAll('[data-feedback-choice]'))button.disabled=busy;
  }
  async function locateReview(item) {
    const task=entry.task_id;
    const index=item.path?entry.charts.findIndex(version=>version.path===item.path && (version.difficulty_index??5)===item.difficulty_index):currentIndex;
    if(index<0)return;
    if(index!==currentIndex)await choose(index,{comparison:true});
    if(task!==entry.task_id)return;
    if(Number.isFinite(item.start_sec) && Number.isFinite(item.end_sec)) {
      setRange({start_sec:item.start_sec,end_sec:item.end_sec});await globalThis.ChartPreview?.seek(item.start_sec);
    } else setRange(null);
  }
  function renderSaved(records) {
    if(entry?.training_context?.candidate_pool)records=candidateSavedReviews(records);
    const list=node('feedbackSavedList');list.replaceChildren();
    const names={good:'好',problem:'有问题',uncertain:'不确定'},reasons={timing:'采音',layout:'配置',comfort:'手感'};
    for(const item of records) {
      const li=document.createElement('li'),button=document.createElement('button');button.type='button';button.className='small-button';
      button.textContent=`${names[item.label]||item.label} · ${Number.isFinite(item.start_sec)?item.start_sec.toFixed(2)+'–'+item.end_sec.toFixed(2)+'秒':'整张谱面'}${item.reason?' · '+(reasons[item.reason]||item.reason):''}`;
      button.addEventListener('click',()=>void locateReview(item).catch(error=>status(error.message)));
      li.append(button);list.append(li);
    }
    node('feedbackSavedSummary').textContent=`已保存的标注（${records.length}）`;
    node('feedbackSavedEmpty').hidden=!!records.length;
  }
  function trainingButtons() {
    node('feedbackCollectTraining').disabled=trainingBusy || !validRange() || !selectedNotes().length || !node('feedbackTrainingUse').value;
    node('feedbackAddTraining').disabled=trainingBusy || !(collection.regions || collection.published);
    if(node('feedbackExportBundle'))node('feedbackExportBundle').disabled=trainingBusy||!collection.regions;
    const entryChoice=node('feedbackLearnEntry');
    if(entryChoice) {
      const key=validRange()?`${chart()?.path}:${chart()?.file_sha256}:${range.start_sec}:${range.end_sec}`:'';
      if(key!==entryChoiceKey){entryChoice.checked=false;entryChoiceKey=key;}
      const timing=['timing','both'].includes(node('feedbackTrainingUse').value);
      entryChoice.disabled=trainingBusy || !validRange() || !selectedNotes().length || !timing;
      if(!timing || !validRange())entryChoice.checked=false;
    }
  }
  async function trainingRequest(endpoint,payload) {
    const response=await fetch('/api/quality/'+endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const result=await response.json();
    if(!response.ok)throw new Error(response.status===404?'当前后台尚未支持训练示范，请在方便时重启 WebUI 后再使用。':result.error||'训练示范操作失败');
    return result;
  }
  const collectionCandidates=()=>entry?.charts.filter(complete)||[];
  const reviewPayload=(version,bounds)=>({path:version.path,difficulty_index:version.difficulty_index??5,
    audio_path:version.audio_path,audio_sha256:version.audio_sha256,chart_sha256:version.chart_sha256,
    file_sha256:version.file_sha256,version_id:version.version_id,...bounds});
  async function loadCollection() {
    if(!entry)return;
    const request=++collectionSerial,task=entry.task_id;
    try {
      const result=await trainingRequest('reviewed-collection',{candidates:collectionCandidates(),task_id:task});
      if(request!==collectionSerial || task!==entry.task_id)return;
      collection={...result,items:result.items||[],regions:result.regions||0};
      if(entry?.source_kind!=='training_source')renderSaved(result.reviews||[]);
      const list=node('feedbackTrainingList');list.replaceChildren();
      for(const item of collection.items) {
        const li=document.createElement('li'),view=document.createElement('button'),remove=document.createElement('button');
        const use=item.timing_weight && item.component_weight?'both':item.timing_weight?'timing':'layout';
        view.type=remove.type='button';view.className=remove.className='small-button';
        view.textContent=`${item.start_sec.toFixed(2)}–${item.end_sec.toFixed(2)}秒 · ${trainingNames[use]}${Number.isFinite(item.entry_start_sec)?' ＋前文接入':''}`;
        view.title=`${item.title} · 点击定位这段谱面`;
        view.addEventListener('click',()=>void locateReview(item).catch(error=>{node('feedbackTrainingStatus').textContent=error.message;}));
        remove.textContent='移出';remove.title='不再将这段作为训练示范，保留原谱面';
        remove.addEventListener('click',()=>void changeCollection(reviewPayload(item,item),'none'));
        li.append(view,remove);list.append(li);
      }
      node('feedbackTrainingStatus').textContent=collection.published?(collection.needs_update?'收集内容已变化，请更新示范库，才会用于下次训练。':`本任务 ${collection.regions} 段已整理到示范库。`):collection.regions?`已收集 ${collection.regions} 段；尚未加入训练来源。`:'拖选并试听后，选择值得学习的内容。';
      node('feedbackAddTraining').textContent=collection.published?'更新示范库':'加入下次训练来源';
      trainingButtons();
    } catch(error) {if(request===collectionSerial)node('feedbackTrainingStatus').textContent=error.message;}
  }
  async function changeCollection(payload,use,learnEntry=false) {
    if(trainingBusy)return;
    const task=entry?.task_id;trainingBusy=true;trainingButtons();
    try {
      await trainingRequest('generation-feedback',{...payload,label:'good',reason:'',training_use:use,learn_entry:learnEntry});
      if(task===entry?.task_id)await loadCollection();
    } catch(error) {if(task===entry?.task_id)node('feedbackTrainingStatus').textContent=error.message;}
    finally {trainingBusy=false;trainingButtons();}
  }
  node('feedbackTrainingUse').addEventListener('change',trainingButtons);
  node('feedbackCollectTraining').addEventListener('click',()=>{
    if(!validRange() || !selectedNotes().length)return;
    void changeCollection(reviewPayload(chart(),{start_sec:range.start_sec,end_sec:range.end_sec}),node('feedbackTrainingUse').value,!!node('feedbackLearnEntry')?.checked);
  });
  node('feedbackExportBundle')?.addEventListener('click',()=>void globalThis.ReviewExchange?.export({candidates:collectionCandidates()},
    {status:node('feedbackBundleStatus'),download:node('feedbackBundleDownload')}));
  node('feedbackAddTraining').addEventListener('click',async()=>{
    if(trainingBusy)return;
    const task=entry?.task_id;trainingBusy=true;trainingButtons();
    try {
      node('feedbackTrainingStatus').textContent='正在核对已确认范围并保存校准素材…';
      const result=await trainingRequest('export-reviewed',{candidates:collectionCandidates(),task_id:task});
      if(task!==entry?.task_id)return;
      const added=await addReviewedTrainingSource(result);
      if(task===entry?.task_id) {
        await loadCollection();
        node('feedbackTrainingStatus').textContent=`示范库现有 ${result.regions} 段；训练页合并显示为一个来源（当前共 ${added.sources} 个来源）。请到训练页检查来源，在新目录 finetune。`;
      }
    } catch(error) {if(task===entry?.task_id)node('feedbackTrainingStatus').textContent=error.message;}
    finally {trainingBusy=false;trainingButtons();}
  });
  async function save(button,pair) {
    if(busy)return;const identity=chart()?.path;
    const task=entry?.task_id,sourceContext=entry?.source_kind==='training_source'?{...entry.training_context}:null;
    try {
      const version=chart();if(!complete(version))throw new Error('请先载入一张完整谱面。');
      if(range && !validRange())throw new Error('请标记有效的起点和终点，或清除选择以评价整张谱面。');
      const listen=audition();
      const bounds=validRange()?{start_sec:range.start_sec,end_sec:range.end_sec,context_start_sec:listen.startSec,context_end_sec:listen.endSec}:{};
      let payload,url;
      if(pair) {
        const a=Number(node('feedbackCandidateA').value),b=Number(node('feedbackCandidateB').value),left=entry.charts[a],right=entry.charts[b];
        if(a===b || !complete(left) || !complete(right) || !audioKey(left) || audioKey(left)!==audioKey(right))throw new Error('请选择同一音频的两张不同完整谱面。');
        payload={audio_path:left.audio_path,candidates:[left,right].map(item=>({path:item.path,difficulty_index:item.difficulty_index??5,audio_path:item.audio_path,audio_sha256:item.audio_sha256,chart_sha256:item.chart_sha256,file_sha256:item.file_sha256,version_id:item.version_id})),preferred:button.dataset.feedbackChoice,reason:node('feedbackReason').value,...bounds};url='/api/quality/candidate-feedback';
      } else {
        payload={path:version.path,difficulty_index:version.difficulty_index??5,audio_path:version.audio_path,audio_sha256:version.audio_sha256,chart_sha256:version.chart_sha256,file_sha256:version.file_sha256,version_id:version.version_id,label:button.dataset.generationFeedback,reason:node('feedbackReason').value,...bounds};url='/api/quality/generation-feedback';
        if(sourceContext) {
          if(!validRange())throw new Error('请先在谱面时间轴圈选片段；整谱质量请在素材管理中设置。');
          Object.assign(payload,sourceContext);url='/api/quality/source-feedback';
        }
      }
      busy=true;feedbackButtons();
      const response=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}),result=await response.json();
      if(!response.ok)throw new Error(result.error||'保存失败');
      if(chart()?.path===identity && entry.task_id===task)status(sourceContext?'已保存质量标注。导出时，在训练素材勾选该来源，点“导出已确认片段”。':'已记录评价。要学习或导出这段，请展开“收集训练示范”。');
      if(sourceContext) {
        const record=result.records?.find(row=>row.chart_id===sourceContext.chart_id);
        if(chart()?.path===identity && entry.task_id===task) {entry.annotations=record?.regions||[];renderSaved(entry.annotations);}
        window.dispatchEvent(new CustomEvent('training-source-annotated',{detail:{...sourceContext,result}}));
      }
      else if(!pair)void loadCollection();
    } catch(error) {if(chart()?.path===identity)status(error.message);}finally {busy=false;feedbackButtons();}
  }
  for(const button of document.querySelectorAll('[data-feedback-choice]'))button.addEventListener('click',()=>void save(button,true));
  for(const button of document.querySelectorAll('[data-generation-feedback]'))button.addEventListener('click',()=>void save(button,false));
  const initial=globalThis.ChartPreview?.getEntry();if(initial)refresh(initial);updateRange();
})();

// Training spans reuse the selected chart audio and cached CPU waveform service.
(() => {
  const node=id=>document.getElementById(id),audio=node('qualityDetailAudio'),canvas=node('qualitySpanWave');
  if(!audio || !canvas)return;
  let saving=false;
  let spanStart=null,spanEnd=null,media=null,mediaChart=null,viewStart=0,viewEnd=0,dragStart=null,serial=0;
  const status=text=>{node('qualitySpanStatus').textContent=text;};
  function update() {
    node('qualitySpanRange').textContent=spanStart===null?'尚未选择片段':`${spanStart.toFixed(2)} 秒 → ${spanEnd===null?'待标终点':spanEnd.toFixed(2)+' 秒'}`;
    draw();
  }
  function draw() {
    if(!media || canvas.hidden)return;
    canvas.width=Math.max(300,Math.round(canvas.getBoundingClientRect().width));const w=canvas.width,h=canvas.height,ctx=canvas.getContext('2d');
    ctx.fillStyle='#f4f7f8';ctx.fillRect(0,0,w,h);ctx.strokeStyle='#408e97';ctx.beginPath();
    const wave=media.waveform,step=wave.step_sec,range=Math.max(.01,viewEnd-viewStart);
    for(let x=0;x<w;x++) {
      const first=Math.floor((viewStart+x/w*range)/step),last=Math.min(wave.maximum.length,Math.max(first+1,Math.ceil((viewStart+(x+1)/w*range)/step)));
      let low=0,high=0;for(let i=first;i<last;i++){low=Math.min(low,wave.minimum[i]||0);high=Math.max(high,wave.maximum[i]||0);}
      ctx.moveTo(x,h/2-high*h*.42);ctx.lineTo(x,h/2-low*h*.42);
    }
    ctx.stroke();
    if(spanStart!==null) {const x=(spanStart-viewStart)/range*w,end=((spanEnd??spanStart)-viewStart)/range*w;ctx.fillStyle='#5d86e244';ctx.fillRect(x,0,end-x,h);}
    ctx.fillStyle='#586674';ctx.fillText(viewStart.toFixed(1)+'s',4,12);ctx.fillText(viewEnd.toFixed(1)+'s',w-55,12);
  }
  function reset() {++serial;spanStart=spanEnd=null;media=null;mediaChart=null;canvas.hidden=true;update();status('');}
  audio.addEventListener('loadstart',reset);
  node('qualitySpanStart').addEventListener('click',()=>{spanStart=audio.currentTime;spanEnd=null;update();});
  node('qualitySpanEnd').addEventListener('click',()=>{if(spanStart===null)spanStart=0;spanEnd=audio.currentTime;if(spanEnd<spanStart)[spanStart,spanEnd]=[spanEnd,spanStart];update();});
  node('qualitySpanWaveLoad').addEventListener('click',async()=>{
    const chart=qualityState.detail;if(!chart?.audio_path){status('当前谱面没有音频路径。');return;}
    const request=++serial,chartId=chart.chart_id,button=node('qualitySpanWaveLoad');button.disabled=true;
    try {
      status('正在准备波形（CPU，复用音频缓存）…');
      const created=await api('/api/composer-v2/media',{model_family:'chart_composer_v2',audio:chart.audio_path,cache_dir:valueOrDefault('#v2GenerateCacheDir','cache/chart_composer_v2')});
      while(request===serial && chartId===qualityState.detail?.chart_id) {
        const task=await api('/api/task?id='+encodeURIComponent(created.task_id));
        if(['failed','cancelled'].includes(task.status))throw new Error(task.error||'波形准备已取消');
        if(task.status==='completed') {
          media=task.result.editor_media;if(!media)throw new Error('波形数据未返回');
          mediaChart=chartId;viewStart=0;viewEnd=media.duration_sec;canvas.hidden=false;draw();status('在波形拖选后，点击评价即可保存。');return;
        }
        await new Promise(resolve=>setTimeout(resolve,500));
      }
    } catch(error) {if(request===serial)status(error.message);} finally {button.disabled=false;}
  });
  node('qualitySpanZoomIn').addEventListener('click',()=>{if(media && spanStart!==null && spanEnd>spanStart){viewStart=spanStart;viewEnd=spanEnd;draw();}});
  node('qualitySpanZoomOut').addEventListener('click',()=>{if(media){viewStart=0;viewEnd=media.duration_sec;draw();}});
  function time(event) {const rect=canvas.getBoundingClientRect();return viewStart+Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width))*(viewEnd-viewStart);}
  canvas.addEventListener('pointerdown',event=>{if(!media || mediaChart!==qualityState.detail?.chart_id)return;canvas.setPointerCapture(event.pointerId);dragStart=time(event);spanStart=dragStart;spanEnd=dragStart;update();});
  canvas.addEventListener('pointermove',event=>{if(dragStart===null)return;const current=time(event);spanStart=Math.min(dragStart,current);spanEnd=Math.max(dragStart,current);update();});
  canvas.addEventListener('pointerup',event=>{if(dragStart===null)return;const current=time(event);spanStart=Math.min(dragStart,current);spanEnd=Math.max(dragStart,current);dragStart=null;canvas.releasePointerCapture(event.pointerId);if(spanEnd-spanStart<.01){audio.currentTime=spanStart;spanEnd=null;}update();});
  const quickButtons=[...document.querySelectorAll('[data-quality-span-tag]')];
  for(const button of quickButtons)button.addEventListener('click',async()=>{
    if(saving)return;
    const chartId=qualityState.detail?.chart_id,detailRequest=qualityState.detailRequest;
    try {
      if(!qualityState.detail?.chart_id || spanStart===null || spanEnd===null || spanEnd<=spanStart)throw new Error('请先标记起点、终点，或在波形拖选片段。');
      const tag=button.dataset.qualitySpanTag,label=tag==='good'?'good':tag==='uncertain'?'uncertain':'problem',reason=label==='problem'?tag:'';
      const region={start_sec:spanStart,end_sec:spanEnd,label,reason,weight:1,timing_weight:label==='good'?1.5:reason==='timing'?0:1,component_weight:label==='good'?1.5:['layout','comfort'].includes(reason)?0:1};
      saving=true;quickButtons.forEach(node=>node.disabled=true);
      qualityState.regions=replaceQualityRegionSpan(collectQualityRegions(),region);updateQualityDetail();
      await saveQualityRegions();if(chartId===qualityState.detail?.chart_id)status('已保存片段评价。');
    } catch(error) {if(chartId===qualityState.detail?.chart_id && detailRequest===qualityState.detailRequest)status(error.message);} finally {saving=false;quickButtons.forEach(node=>node.disabled=false);}
  });
})();

// BEGIN CANDIDATE POOL WORKBENCH
"use strict";
// Candidate regions navigate a complete source chart; the shared timeline owns selection.
(() => {
  const $=id=>document.getElementById(id);
  let enabled=false,active=null,current=null,queue=[],seen=new Set(),undo=null,busy=false,scanning=false,serial=0;
  let budgetPending=false,budgetTaskId=null;
  let scanTimer=null;
  const isPool=item=>!!(item?.candidate_pool || item?.candidate_pool_summary);
  const identity=()=>active?.manifest;
  const message=text=>{$('candidatePoolReviewStatus').textContent=text;};
  const describe=value=>Array.isArray(value)?value.map(describe).join('；'):typeof value==='string'?
    ({existing_exact_chart:'与已有资料为同一张谱面',duplicate_source_chart:'与已有资料指向同一谱面槽位'}[value]||value):
    value?.message||value?.label||'需要结合音频核对';
  const token=item=>`${item.chart_id}:${item.start_sec}:${item.end_sec}`;
  function controls() {
    const source=globalThis.ChartPreview?.getEntry();
    const matches=current && source?.training_context?.manifest===identity() && source?.training_context?.chart_id===current.chart_id;
    $('candidatePoolReview').hidden=!matches;
    const ordinary=$('feedbackStatus')?.closest('.review-tool-group');if(ordinary)ordinary.hidden=!!matches;
    const error=current?.candidate_status==='error'||current?.status==='error';
    for(const button of document.querySelectorAll('[data-pool-learn]'))button.disabled=busy||scanning||!matches||error;
    $('candidatePoolWhole').disabled=busy||scanning||!matches||error;
    $('candidatePoolUndo').disabled=busy||!undo;
    for(const id of ['candidatePoolNext','candidatePoolSync','candidatePoolExcluded'])$(id).disabled=busy||!enabled;
    $('candidatePoolBudget').disabled=budgetPending||!enabled;
    if(scanning||$('candidatePoolStatusFilter')?.value==='error')$('candidatePoolNext').disabled=true;
  }
  function summary(data) {
    const counts=data.counts||data;
    const scan=data.scan_state||data.scan?.state||'';
    scanning=['queued','running'].includes(scan)||data.scan_complete===false;
    const progress=data.scan_task?.progress||{},stage=progress.stage;
    const stages={starting:'启动检查',references:'核对已有资料',identities:'核对源谱与音频',annotations:'整理已有标注',scanning:'检查谱面与音频',finalizing:'汇总检查结果',complete:'检查完成'};
    const duration=seconds=>seconds<60?`${Math.max(1,Math.ceil(seconds))} 秒`:`${Math.ceil(seconds/60)} 分钟`;
    let scanMessage='';
    if(['queued','running'].includes(scan)) {
      scanMessage=stage?`${stages[stage]||'正在检查'}${progress.total>0?` ${progress.completed||0}/${progress.total}`:''}`:`扫描${scan==='queued'?'排队中':'进行中'}`;
      if(Number.isFinite(progress.eta_sec))scanMessage+=` · 本阶段预计剩余 ${duration(progress.eta_sec)}`;
      else scanMessage+=' · 正在估算用时';
      if(stage==='scanning'&&progress.workers>1)scanMessage+=` · ${progress.workers} 路并行`;
      if(progress.title)scanMessage+=` · ${progress.title}`;
      scanMessage+='，完成后可开始审阅。 ';
    } else if(data.scan_complete===false) {
      scanMessage=scan==='failed'?'检查失败，查看任务日志或点击“重新扫描／同步”重试。 ':'上次检查已中断，点击“重新扫描／同步”继续。 ';
    }
    if(stage==='scanning'&&['queued','running'].includes(scan)&&progress.counts) {
      Object.assign(counts,progress.counts,{pending:Math.max(0,Number(progress.total||0)-Number(progress.completed||0))});
    }
    $('candidatePoolSummary').textContent=scanMessage+`只学习已确认片段 · 已确认 ${data.confirmed_regions??data.reviewed_regions??0} 段 · 未检查 ${counts.pending??0}／错误 ${counts.error??0}／待核对 ${counts.suspicious??0}／低风险 ${counts.ok??0} · 可审阅 ${data.reviewable??data.reviewable_charts??'待扫描'} 张`;
    controls();
  }
  async function refresh() {
    const manifest=identity(),request=serial;if(!manifest||!enabled)return;
    const data=await api('/api/candidates/status',{manifest});
    if(request===serial&&manifest===identity()) {
      summary(data);
      if(scanTimer!==null&&typeof clearTimeout==='function')clearTimeout(scanTimer);
      scanTimer=null;
      if(['queued','running'].includes(data.scan_state)&&typeof setTimeout==='function')scanTimer=setTimeout(()=>{
        scanTimer=null;void refresh().catch(error=>message(`读取检查进度失败：${error.message}`));
      },2000);
    }
  }
  async function run(action) {
    if(busy)return;busy=true;controls();
    try {await action();}catch(error){message(error.message);if($('candidatePoolReview').hidden)$('candidatePoolSummary').textContent=error.message;}
    finally {busy=false;controls();}
  }
  async function next() {
    const manifest=identity(),request=serial;if(!manifest)return;
    if(!queue.length) {
      // Requery from the beginning: confirmation can shrink the server queue.
      let offset=0,total=Infinity;
      while(!queue.length&&offset<total) {
        const data=await api('/api/candidates/queue',{manifest,offset,limit:100,candidate_status:$('candidatePoolStatusFilter')?.value||''});
        if(request!==serial)return;
        const items=data.items||[];total=Number(data.total??items.length);
        queue=items.filter(item=>!seen.has(token(item)));offset+=items.length;
        if(!items.length)break;
      }
    }
    if(!queue.length){message($('candidatePoolStatusFilter')?.value==='error'?'错误谱面请先在目录查看原因，修正源资料后再同步；修正前不确认学习。':'这个筛选下暂无未审片段。可换一个检查状态，或在目录打开整谱自行圈选。');return;}
    if(current) {const same=queue.findIndex(item=>item.chart_id===current.chart_id);if(same>0)queue.unshift(...queue.splice(same,1));}
    const item=queue.shift();
    if(current?.chart_id!==item.chart_id||globalThis.ChartPreview?.getEntry()?.training_context?.manifest!==manifest||globalThis.ChartPreview?.getEntry()?.training_context?.chart_id!==item.chart_id) {
      const preview=await api('/api/quality/chart-preview',{manifest,chart_id:item.chart_id});
      if(request!==serial)return;
      globalThis.ChartPreview?.openSource(preview,{scroll:true});
    }
    if(request!==serial)return;
    current=item;seen.add(token(item));
    $('feedbackContext').checked=true;
    globalThis.FeedbackRange?.set({start_sec:Number(item.start_sec),end_sec:Number(item.end_sec)});
    $('candidatePoolWhy').textContent=`${Number(item.start_sec).toFixed(2)}–${Number(item.end_sec).toFixed(2)} 秒 · ${describe(item.why||'代表性片段')}。可在时间轴微调，预览始终保留整谱。`;
    $('candidatePoolIssues').textContent=(Array.isArray(item.issues)?item.issues:[item.issues]).filter(Boolean).map(describe).join('；')||'扫描未列出问题；请结合音频与前后文判断。';
    $('candidatePoolEntry').checked=false;message('试听后选择学习内容；只有明确确认的范围才进入监督。');controls();
    await globalThis.ChartPreview?.seek?.(Number(item.start_sec));
  }
  async function save(use,whole=false) {
    const manifest=identity(),item=current,request=serial;
    const context=globalThis.ChartPreview?.getEntry()?.training_context;
    if(context?.manifest!==manifest||context?.chart_id!==item?.chart_id)throw new Error('预览已切换，请重新打开候选片段。');
    if(!item)throw new Error('请先打开候选片段。');
    const version=globalThis.ChartPreview?.getEntry()?.charts?.[0];
    if(!version?.file_sha256||!version?.chart_sha256||!version?.audio_sha256||!version?.audio_path)throw new Error('预览缺少谱面或音频身份，请重新打开候选片段后再保存。');
    if(item.candidate_status==='error'||item.status==='error')throw new Error('该谱有阻断错误，请修正源谱并同步后再审阅。');
    let range=globalThis.FeedbackRange?.get();
    if(whole) {
      const source=globalThis.ChartPreview?.getEntry(),version=source?.charts?.find(chart=>chart.chart_id===item.chart_id)||source?.charts?.[0];
      let timeline=version?.timeline;
      if(!timeline && version?.timeline_url){const response=await fetch(version.timeline_url);timeline=await response.json();if(!response.ok)throw new Error(timeline.error||'无法读取整谱范围');}
      const notes=timeline?.notes||[];
      if(!notes.length)throw new Error('尚未取得整谱音符范围，请使用时间轴圈选。');
      range={start_sec:Math.min(...notes.map(n=>Number(n.start_sec))),end_sec:Math.max(...notes.map(n=>Number(n.end_sec??n.start_sec)))+.25};
      if(!confirm('只有听过整首才应使用。确认将首个起音至最后动作结束后的整首范围同时用于采音和配置学习？'))return;
    }
    if(!range||!Number.isFinite(range.start_sec)||!Number.isFinite(range.end_sec)||range.end_sec<=range.start_sec)throw new Error('请在时间轴选择有效片段。');
    const result=await api('/api/quality/source-feedback',{manifest,chart_id:item.chart_id,file_sha256:version.file_sha256,chart_sha256:version.chart_sha256,audio_sha256:version.audio_sha256,audio_path:version.audio_path,...range,label:use==='uncertain'?'uncertain':'good',reason:['timing','layout'].includes(use)?use:'',note:$('candidatePoolReason').value.trim(),learn_entry:use!=='layout'&&use!=='uncertain'&&$('candidatePoolEntry').checked,context_start_sec:Math.max(0,range.start_sec-4),context_end_sec:range.end_sec+4});
    undo=result.undo_token?{manifest,chart_id:item.chart_id,undo_token:result.undo_token}:null;
    window.dispatchEvent(new CustomEvent('training-source-annotated',{detail:{manifest,chart_id:item.chart_id,result}}));
    if(request!==serial)return;
    message('已保存。');$('candidatePoolReason').value='';await refresh();
    if($('candidatePoolAutoNext').checked)await next();
  }
  globalThis.CandidatePool={
    badge(item) {
      if(!item.candidate_status)return '';
      const names={error:'错误',suspicious:'待核对',ok:'低风险',pending:'待扫描'};
      const issues=item.candidate_issues||[];
      return `<br><span class="quality-label">${escapeHtml(names[item.candidate_status]||item.candidate_status)}</span> <span class="muted">已确认 ${escapeHtml(String(item.confirmed_regions??item.reviewed_regions??item.record?.regions?.length??0))} 段</span>${issues.length?`<details class="candidate-issues"><summary>查看 ${issues.length} 项检查提示</summary>${issues.map(issue=>`<p>${escapeHtml(issue.message||issue.code||'待核对')}</p>`).join('')}</details>`:''}`;
    },
    card(item) {
      if(!item.manifest||item.prepared===false)return '';
      if(item.reviewed_regions!=null)return '<p class="muted">人工确认的片段示范，勾选后可加入训练。</p>';
      const pool=isPool(item),info=item.candidate_pool_summary||item.candidate_pool||{},counts=info.counts||info;
      return `<div class="candidate-pool-card"><p class="muted">${pool?`候选库 · 已确认 ${escapeHtml(String(info.confirmed_regions??info.reviewed_regions??0))} 段 · 未检查 ${escapeHtml(String(counts.pending??0))}／错误 ${escapeHtml(String(counts.error??0))}／待核对 ${escapeHtml(String(counts.suspicious??0))}／低风险 ${escapeHtml(String(counts.ok??0))}`:'可独立建立候选库，保留源谱与原标注'}</p>${pool?'':`<button type="button" class="small-button candidate-pool-create" data-manifest="${escapeHtml(item.manifest)}" ${enabled?'':'disabled'}>作为候选库审查</button>`}</div>`;
    },
    capabilities(state) {enabled=state.training_capabilities?.candidate_pools===true||state.capabilities?.candidate_pools===true;this.render();},
    render() {
      $('candidatePoolCapability').textContent=enabled?'候选库已启用：先审阅片段，再加入下一次训练。':'候选库需要重启后台后启用。当前训练可以继续；请在训练妥善结束后再重启。';
      for(const button of document.querySelectorAll('.candidate-pool-create'))button.disabled=!enabled;
      if(selectedDatasetItems().some(isPool)){const input=$('v2TrainQualityEnabled');if(input)input.checked=true;}
      controls();
    },
    activate(item) {++serial;if(scanTimer!==null&&typeof clearTimeout==='function')clearTimeout(scanTimer);scanTimer=null;active=isPool(item)?item:null;current=null;queue=[];seen.clear();if($('candidatePoolStatusFilter'))$('candidatePoolStatusFilter').value='';$('candidatePoolPanel').hidden=!active;controls();if(active)void refresh().catch(error=>{$('candidatePoolSummary').textContent=error.message;});}
  };
  document.addEventListener('click',event=>{
    const create=event.target.closest?.('.candidate-pool-create');if(!create)return;
    void run(async()=>{
      if(!enabled)throw new Error('重启后台后启用候选库。');
      const source=datasetWorkspaceState.items.find(item=>item.manifest===create.dataset.manifest);
      const name=(source?.name||'素材')+' · 候选库';
      const result=await api('/api/candidates/create',{manifest:create.dataset.manifest,name});
      $('candidatePoolCapability').textContent='候选库已建立，正在后台扫描；默认零监督。';
      await loadDatasetCatalog();
      let item=datasetWorkspaceState.items.find(item=>item.manifest===result.manifest);
      if(!item) {
        // The build runs asynchronously; keep its manifest visible before pool.json exists.
        item={name,manifest:result.manifest,path:result.pool_dir||result.manifest,prepared:true,candidate_pool:{pool_id:result.pool_id,scan_state:'queued'}};
        datasetWorkspaceState.items.push(item);renderDatasetWorkspace();
      }
      if(result.task_id)watchTask(result.task_id,async task=>{
        await loadDatasetCatalog();
        if(identity()===result.manifest){if(task.status==='completed'){const built=datasetWorkspaceState.items.find(row=>row.manifest===result.manifest);if(built&&!current&&!busy&&qualityState.annotationManifest!==result.manifest)await openDataset(built);}await refresh();if(task.status!=='completed')message(`扫描${task.status==='cancelled'?'已取消':'失败'}：${task.error||'请检查任务日志'}`);}
      });
      if(item.candidate_pool?.scan_state==='queued'){datasetWorkspaceState.activePath=item.path;datasetWorkspaceState.view='catalog';globalThis.CandidatePool.activate(item);renderDatasetWorkspace();}
      else await openDataset(item);
    });
  });
  function showBudget(data) {
    const result=$('candidatePoolBudgetResult');result.replaceChildren();
    const hint=document.createElement('p');hint.textContent='以下是所选来源的估算学习份额，分别计算采音和配置；百分比不是谱面质量评分，也不是实际训练损失占比。';result.append(hint);
    const share=value=>{const percent=100*Number(value||0);return percent>0&&percent<.1?'<0.1%':percent.toFixed(1)+'%';};
    for(const source of data.sources||[]) {const row=document.createElement('p');const name=datasetWorkspaceState.items.find(item=>item.manifest===source.manifest)?.name||source.manifest;row.textContent=`${name}${source.candidate_pool&&!name.includes('候选库')?'（候选库）':''}：采音 ${share(source.timing_share)} ／ 配置 ${share(source.layout_share)}`;result.append(row);}
    if(!(data.sources||[]).length){const empty=document.createElement('p');empty.textContent='当前没有可计入的学习内容，请先确认候选片段或检查所选来源。';result.append(empty);}
    if(data.warnings?.length){const warning=document.createElement('details'),title=document.createElement('summary');title.textContent=`${data.warnings.length} 项未计入，查看原因`;warning.append(title);for(const item of data.warnings.slice(0,30)){const row=document.createElement('p'),empty=String(item.reason||'').match(/No inote for difficulty (\d+) found/);row.textContent=`${item.source_path||item.manifest||'来源'}：${empty?`难度槽位 ${empty[1]} 为空，未计入估算`:item.reason||'请检查源资料'}`;warning.append(row);}if(data.warnings.length>30){const more=document.createElement('p');more.textContent='其余记录见右侧任务结果中的 result.json。';warning.append(more);}result.append(warning);}
  }
  function budgetProgress(task) {
    if(!['queued','running','cancelling'].includes(task.status))return;
    const progress=task.result?.candidate_budget||{},total=Number(progress.total),done=Number(progress.completed)||0;
    const stages={queued:'等待开始',checking:'检查资料是否变化',loading:'读取来源',scanning:'统计谱面',counting:'统计学习内容',estimating:'估算学习份额',complete:'统计完成',completed:'统计完成'};
    const stage=stages[progress.stage]||progress.stage||'正在估算';
    const sourceName=datasetWorkspaceState.items.find(item=>item.manifest===progress.source)?.name||progress.source;
    $('candidatePoolBudgetResult').textContent=`${task.status==='cancelling'?'正在取消':stage}${total>0?` · ${Math.min(100,Math.round(done/total*100))}%（${done}/${total}）`:''}${sourceName?` · ${sourceName}`:''}${!progress.stage&&task.message?` · ${task.message}`:''}`;
    $('candidatePoolBudgetCancel').hidden=false;
  }
  function finishBudget(task) {
    budgetPending=false;budgetTaskId=null;$('candidatePoolBudgetCancel').hidden=true;controls();
    if(task.status==='completed')showBudget(task.result?.supervision||task.result?.summary?.supervision||task.result?.summary||task.result||{});
    else $('candidatePoolBudgetResult').textContent=task.status==='cancelled'?'估算已取消，可重新选择来源后再估算。':`估算失败：${task.error||task.message||'请重试或查看任务日志'}`;
  }
  $('candidatePoolBudget').addEventListener('click',async()=>{
    if(budgetPending)return;
    const result=$('candidatePoolBudgetResult');result.hidden=false;
    const manifests=selectedDatasetItems().map(item=>item.manifest).filter(Boolean);
    if(!manifests.length){result.textContent='请先勾选要估算的训练来源。';return;}
    budgetPending=true;controls();result.textContent='正在提交估算；接下来会显示阶段和进度，可以继续浏览和审阅素材。';
    try {
      const response=await api('/api/candidates/budget',{manifests});
      if(response.task_id){budgetTaskId=response.task_id;budgetProgress(response);watchTask(response.task_id,finishBudget,budgetProgress);}
      else finishBudget({status:'completed',result:response});
    } catch(error){finishBudget({status:'failed',error:error.message||String(error)});}
  });
  $('candidatePoolBudgetCancel').addEventListener('click',async()=>{
    if(!budgetTaskId)return;
    const button=$('candidatePoolBudgetCancel');button.disabled=true;
    try {await api('/api/cancel',{task_id:budgetTaskId});$('candidatePoolBudgetResult').textContent='已申请取消估算，正在等待后台停止。';}
    catch(error){$('candidatePoolBudgetResult').textContent=`取消失败：${error.message||error}。可重试取消。`;}
    finally{button.disabled=false;}
  });
  $('candidatePoolNext').addEventListener('click',()=>void run(next));
  $('candidatePoolStatusFilter')?.addEventListener('change',()=>void run(async()=>{queue=[];seen.clear();
    if(globalThis.maichartLastStatus&&!globalThis.maichartLastStatus.training_capabilities?.candidate_review_filters){$('candidatePoolStatusFilter').value='';throw new Error('当前后台尚未载入检查状态筛选，请使用已更新的 WebUI 实例。');}
    await loadQualityCatalog(true);
    message($('candidatePoolStatusFilter').value==='error'?'在目录展开检查提示，先修正源资料，再重新扫描。':'已切换筛选；点“继续审阅”试听下一段。');}));
  $('candidatePoolSync').addEventListener('click',()=>void run(async()=>{const manifest=identity(),result=await api('/api/candidates/sync',{manifest});queue=[];seen.clear();message('正在后台重新扫描／同步；源谱变化会使旧确认失效。');if(result.task_id)watchTask(result.task_id,()=>{if(identity()===manifest)void refresh();});await refresh();}));
  $('candidatePoolExcluded').addEventListener('click',()=>void run(async()=>{const manifest=identity(),request=serial,data=await api('/api/candidates/excluded',{manifest});if(request!==serial)return;const list=$('candidatePoolExcludedList');list.hidden=false;list.replaceChildren();for(const item of data.items||[]){const row=document.createElement('p');row.textContent=`${item.title||item.source_path||item.chart_id||item.path||'谱面'}${item.slot!==undefined?` · 槽位${item.slot}`:''}：${describe(item.reason||item.issues||item)}`;list.append(row);}if(!list.children.length)list.textContent='暂无排除项。';}));
  for(const button of document.querySelectorAll('[data-pool-learn]'))button.addEventListener('click',()=>void run(()=>save(button.dataset.poolLearn)));
  $('candidatePoolWhole').addEventListener('click',()=>void run(()=>save('both',true)));
  $('candidatePoolUndo').addEventListener('click',()=>void run(async()=>{const last=undo;if(!last)return;const result=await api('/api/candidates/undo',last);undo=null;if(identity()===last.manifest){queue=[];seen.clear();await refresh();}window.dispatchEvent(new CustomEvent('training-source-annotated',{detail:{...last,result}}));message('已撤销上次保存。');}));
  window.addEventListener('chart-preview-updated',event=>{
    const context=(event?.detail||globalThis.ChartPreview?.getEntry())?.training_context;
    if(active&&context?.manifest===identity()&&context.chart_id&&context.chart_id!==current?.chart_id){
      current={...context};$('candidatePoolWhy').textContent='在时间轴圈选一段完整写法，带前后文试听，再确认值得学习的内容。';
      $('candidatePoolIssues').textContent=(context.issues||[]).map(describe).join('；')||'没有自动提示，请结合歌曲核对采音、衔接和手顺。';
    }
    controls();
  });window.addEventListener('chart-preview-versionchanged',controls);
  globalThis.CandidatePool.render();
  if(globalThis.maichartLastStatus)globalThis.CandidatePool.capabilities(globalThis.maichartLastStatus);
})();

// END CANDIDATE POOL WORKBENCH
