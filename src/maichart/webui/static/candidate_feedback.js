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
  let originPrompt=null,activePrompt=null;
  const chart=()=>entry?.charts[currentIndex];
  const audioKey=version=>version?.audio_sha256||version?.audio_path||version?.audio_url||'';
  const complete=version=>version && !version.partial_preview && version.complete!==false && (!version.content_kind || version.content_kind==='complete_chart');
  const status=text=>{
    node('feedbackStatus').textContent=text;
    const note=node('feedbackAutoChartNote');
    if(note)note.textContent=text;
  };
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
  const noteColor=(kind,isBreak,isEach)=>isBreak?'#e77632':isEach?'#ffd640':colors[kind]||'#747f91';
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
    const autoRegions=entry?.source_kind==='training_source'?(entry?.training_context?.auto_annotation?.regions||[]):[];
    for(const region of autoRegions) {
      const a=Number(region.start_sec),b=Number(region.end_sec);
      if(!Number.isFinite(a)||!Number.isFinite(b)||b<=viewStart||a>=viewEnd)continue;
      const rx=x(Math.max(viewStart,a)),rw=Math.max(3,x(Math.min(viewEnd,b))-rx),good=region.label==='good',review=region.label==='uncertain';
      ctx.save();
      ctx.fillStyle=good?'#2f9e661e':review?'#c591251e':'#d95f491e';
      ctx.fillRect(rx,top,rw,h-top);
      ctx.strokeStyle=good?'#2f9e6699':review?'#c5912599':'#d95f4999';
      ctx.setLineDash([4,3]);
      ctx.strokeRect(rx,top+1,rw,h-top-2);
      ctx.setLineDash([]);
      if(rw>36) {ctx.fillStyle=good?'#1f6f46':review?'#856414':'#a33928';ctx.fillText((good?'好段 ':review?'待复核 ':'自动 ')+(region.reason_label||region.reason||''),rx+4,top+12);}
      ctx.restore();
    }
    if(validRange()) {ctx.fillStyle='#6491e22b';ctx.fillRect(x(range.start_sec),top,x(range.end_sec)-x(range.start_sec),h-top);}
    ctx.save();ctx.beginPath();ctx.rect(left,top,w-left,h-top);ctx.clip();
    for(const note of timeline?.notes||[]) {
      if((note.end_sec??note.start_sec)<viewStart || note.start_sec>viewEnd)continue;
      const px=x(note.start_sec),py=y(noteLane(note));
      const slide=note.kind==='slide';
      const headColor=noteColor(note.kind,note.is_head_break??note.is_break,note.is_each);
      const bodyColor=slide?noteColor(note.kind,note.is_track_break??note.is_break,note.is_track_each):headColor;
      ctx.strokeStyle=bodyColor;ctx.fillStyle=bodyColor;
      if(Number.isFinite(note.end_sec)) {
        if(slide) {
          const trace=Number.isFinite(note.trace_start_sec)?note.trace_start_sec:note.start_sec;
          ctx.strokeStyle=note.has_head===false?bodyColor:headColor;
          ctx.setLineDash([3,3]);ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(px,py);ctx.lineTo(x(trace),py);ctx.stroke();ctx.setLineDash([]);
          ctx.strokeStyle=bodyColor;
          const segments=note.segments?.length?note.segments:[{start_sec:trace,end_sec:note.end_sec,head:note.head,end_head:note.end_head}];
          for(const part of segments) {
            const fromY=y(Math.max(0,Math.min(7,Number(part.head)-1))),toY=y(Math.max(0,Math.min(7,Number(part.end_head)-1))),fromX=x(part.start_sec),toX=x(part.end_sec);
            ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(fromX,fromY);ctx.lineTo(toX,toY);ctx.stroke();const angle=Math.atan2(toY-fromY,toX-fromX);ctx.beginPath();ctx.moveTo(toX,toY);ctx.lineTo(toX-7*Math.cos(angle-.55),toY-7*Math.sin(angle-.55));ctx.lineTo(toX-7*Math.cos(angle+.55),toY-7*Math.sin(angle+.55));ctx.closePath();ctx.fill();
          }
        } else {ctx.globalAlpha=.42;ctx.fillRect(px,py-5,Math.max(2,x(note.end_sec)-px),10);ctx.globalAlpha=1;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x(note.end_sec),py-7);ctx.lineTo(x(note.end_sec),py+7);ctx.stroke();}
      }
      if(note.has_head===false)continue;
      ctx.strokeStyle=headColor;ctx.fillStyle=headColor;
      ctx.lineWidth=2;ctx.beginPath();
      if(slide) {for(let point=0;point<10;point++){const angle=-Math.PI/2+point*Math.PI/5,r=point%2?3.5:8;const hx=px+Math.cos(angle)*r,hy=py+Math.sin(angle)*r;if(!point)ctx.moveTo(hx,hy);else ctx.lineTo(hx,hy);}ctx.closePath();ctx.stroke();}
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
    if(entry?.task_id!==value.task_id || entry?.training_context?.manifest!==value.training_context?.manifest) {
      range=null;originPrompt=null;activePrompt=null;clearLoop();
    }
    collection={items:[],regions:0};++collectionSerial;node('feedbackTrainingList').replaceChildren();
    const previous=chart()?.path,previousAudio=audioKey(chart());entry=value;
    currentIndex=Math.max(0,value.charts.findIndex(version=>version.path===previous));anchorIndex=currentIndex;
    if(previousAudio!==audioKey(chart())) {playhead=0;playing=false;range=null;media=null;mediaAudio=null;clearLoop();++mediaSerial;}
    refreshChoices();updateRange();void loadTimeline();
    const source=entry.source_kind==='training_source';
    node('feedbackTraining').hidden=source;node('feedbackComparison').hidden=source;
    status(entry.training_context?.audit_read_only?'体检清单 · 请在原训练库保存片段标注。':source?'训练素材 · 圈选后保存片段标注。':'生成结果 · 记录评价，或展开下方“收集训练示范”。');
    renderSaved(source?entry.saved_reviews||entry.annotations||[]:[]);
    renderAuto(source?entry.training_context?.auto_annotation:null);
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
  window.addEventListener('chart-preview-annotations-updated',event=>{
    const detail=event.detail;if(!entry||entry.task_id!==detail?.task_id)return;
    entry={...entry,annotations:detail.annotations,saved_reviews:detail.saved_reviews,training_context:detail.training_context};
    if(entry.source_kind==='training_source'){renderSaved(entry.saved_reviews||entry.annotations||[]);renderAuto(entry.training_context?.auto_annotation);}
    else {renderAuto(null);void loadCollection();}
    draw();
  });
  window.addEventListener('training-source-annotated',event=>{
    const detail=event.detail;if(entry?.source_kind!=='training_source'||entry.training_context?.manifest!==detail.manifest||entry.training_context?.chart_id!==detail.chart_id)return;
    if(Array.isArray(detail.result?.records)) {
      const record=detail.result.records.find(row=>row.chart_id===detail.chart_id);
      entry.annotations=record?.regions||[];
      entry.saved_reviews=entry.training_context?.candidate_pool?[...entry.annotations,...(record?.candidate_reviews||[]).filter(row=>row.label==='uncertain')]:entry.annotations;
      renderSaved(entry.saved_reviews);
    }
    if(detail.result?.auto_annotation) entry.training_context.auto_annotation=detail.result.auto_annotation;
    if(detail.result?.auto_annotation || detail.result?.prompt_review) renderAuto(entry.training_context.auto_annotation);
  });
  window.addEventListener('material-origin-prompt',event=>{
    if(entry?.source_kind!=='training_source')return;
    originPrompt=event.detail;activePrompt=event.detail;
    renderAuto(entry?.training_context?.auto_annotation);
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
    const context=entry?.training_context;
    const exclude=node('feedbackExcludeSpan');
    if(exclude) {
      exclude.hidden=!source || !!(context?.zero_unmarked || context?.candidate_pool || context?.audit_read_only);
      exclude.disabled=exclude.hidden || busy || !validRange();
    }
    for(const button of document.querySelectorAll('[data-generation-feedback]'))button.disabled=busy || !!entry?.training_context?.audit_read_only || (source && !validRange());
    for(const button of document.querySelectorAll('[data-feedback-choice]'))button.disabled=busy;
    for(const button of document.querySelectorAll('[data-material-prompt-action]')) {
      button.disabled=busy || (button.dataset.materialPromptAction==='annotate' && !validRange());
    }
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
  function savedSpanEffect(item) {
    if(entry?.source_kind!=='training_source' || !Number.isFinite(item.start_sec) || !Number.isFinite(item.end_sec)
      || typeof qualityRegionDefaults!=='function')return '';
    const context=entry.training_context;
    const zero=!!(context?.zero_unmarked || context?.candidate_pool);
    const [t,c]=qualityRegionDefaults(item.label,item.reason);
    const base=Number(item.weight ?? 1);
    const blocked=zero && (item.label!=='good' || item.training_use==='none');
    const timing=blocked?0:base*Number(item.timing_weight ?? t);
    const components=blocked?0:base*Number(item.component_weight ?? c);
    return ` · 采音 ${timing}× / 配置 ${components}×${timing===0&&components===0?' · 此段不作学习目标':''}`;
  }
  function renderSaved(records) {
    if(entry?.training_context?.candidate_pool)records=candidateSavedReviews(records);
    const list=node('feedbackSavedList');list.replaceChildren();
    const names={good:'好',problem:'有问题',uncertain:'不确定'},reasons={timing:'采音',layout:'配置',comfort:'手感'};
    for(const item of records) {
      const li=document.createElement('li'),button=document.createElement('button');button.type='button';button.className='small-button';
      button.textContent=`${names[item.label]||item.label} · ${Number.isFinite(item.start_sec)?item.start_sec.toFixed(2)+'–'+item.end_sec.toFixed(2)+'秒':'整张谱面'}${item.reason?' · '+(reasons[item.reason]||item.reason):''}${savedSpanEffect(item)}`;
      button.addEventListener('click',()=>void locateReview(item).catch(error=>status(error.message)));
      li.append(button);list.append(li);
    }
    node('feedbackSavedSummary').textContent=`已保存的标注（${records.length}）`;
    node('feedbackSavedEmpty').hidden=!!records.length;
  }
  function renderAuto(autoAnnotation) {
    const panel=node('feedbackAutoRegions'),list=node('feedbackAutoList'),summary=node('feedbackAutoSummary');
    if(!panel || !list || !summary)return;
    list.replaceChildren();
    const regions=Array.isArray(autoAnnotation?.regions)?autoAnnotation.regions:[];
    const excluded=Boolean(autoAnnotation?.excluded || autoAnnotation?.auto_status==='excluded' || autoAnnotation?.chart_label==='excluded');
    if(entry?.source_kind!=='training_source' || (!excluded && !regions.length && !originPrompt && !autoAnnotation?.review_unavailable)) {
      panel.hidden=true;
      return;
    }
    panel.hidden=false;
    if(autoAnnotation?.review_unavailable) {
      const unavailable=document.createElement('li');
      unavailable.textContent='此谱的复核提示需重新体检后核对。';
      list.append(unavailable);
    }
    if(excluded) {
      const li=document.createElement('li'),tag=document.createElement('span');
      tag.className='auto-audit-badge excluded';
      tag.textContent=autoAnnotation.auto_badge||`整谱自动排除：${autoAnnotation.chart_reason_label||autoAnnotation.reason_label||autoAnnotation.chart_reason||autoAnnotation.reason||'问题过多'}`;
      li.append(tag);list.append(li);
    }
    const context=entry.training_context||{};
    const canReview=context.audit_read_only || !!autoAnnotation?.report_version || !!originPrompt;
    const provenance={manual_chart:'有人工整谱决定',bulk_chart:'有批量整谱赋权',legacy:'已有整谱权重（来源未记录）'}[autoAnnotation?.manual_provenance];
    if(context.audit_read_only && context.origin_jump && typeof jumpToMaterialOrigin==='function') {
      const li=document.createElement('li'),button=document.createElement('button');
      button.type='button';button.className='small-button';button.textContent=`打开原库「${context.origin_jump.name}」`;
      button.addEventListener('click',()=>void jumpToMaterialOrigin(context.origin_jump,range,null,null,context).catch(error=>status(error.message)));
      li.append(button);list.append(li);
    }
    const visible=originPrompt?[originPrompt.prompt,...regions.filter(item=>item.prompt_id!==originPrompt.prompt.prompt_id)]:regions;
    for(const item of visible) {
      const li=document.createElement('li'),button=document.createElement('button');
      const imported=item===originPrompt?.prompt;
      const target=imported?originPrompt.target:item.prompt_id && autoAnnotation?.report_version?{
        manifest:context.manifest,chart_id:context.chart_id,auto_scope:autoAnnotation.auto_scope,
        report_version:autoAnnotation.report_version,prompt_id:item.prompt_id,
      }:null;
      button.type='button';
      button.className=`small-button auto-region-jump ${item.label==='good'?'is-good':item.label==='uncertain'?'is-uncertain':'is-problem'}`;
      const spanText=Number.isFinite(item.start_sec)&&Number.isFinite(item.end_sec)?`${Number(item.start_sec).toFixed(2)}–${Number(item.end_sec).toFixed(2)}秒`:'片段';
      const decided={kept:'已复核 · 保留当前标注',annotated:'已复核 · 已保存片段标注',deferred:'稍后处理',needs_recheck:'需重新核对'};
      const prefix=decided[item.review_status]||(item.label==='good'?'★ 优选好段':item.label==='uncertain'?'⚠ 待复核':'⚠ 自动处理');
      button.textContent=`${imported?'组合提示 · ':''}${prefix} · ${spanText} · ${item.reason_label||item.reason||''}${item.effect_label?`（${item.effect_label}）`:''}${item.overlaps_annotation&&!item.reviewed?' · 有重叠的片段标注':''}`;
      if(item.reviewed)button.className='small-button auto-region-jump is-good';
      button.addEventListener('click',()=>void selectPrompt(item,target).catch(error=>status(error.message)));
      li.append(button);
      if(target&&canReview) {
        const actions=document.createElement('div');
        actions.className='material-prompt-actions';
        const choices=context.audit_read_only?(context.origin_jump?[['origin','去原库复核']]:[])
          :item.reviewed?[['reopen','改为待复核']]:[['keep','确认保留'],['defer','稍后处理'],['select','选择此条并标注']];
        for(const [action,label] of choices) {
          const control=document.createElement('button');
          control.type='button';control.className='small-button';control.dataset.materialPromptAction=action;control.textContent=label;
          control.addEventListener('click',()=>{
            if(action==='origin')void jumpToMaterialOrigin(context.origin_jump,item,target.report_version,target.auto_scope,context).catch(error=>status(error.message));
            else if(action==='select')void selectPrompt(item,target).catch(error=>status(error.message));
            else void savePrompt(action,item,target);
          });
          actions.append(control);
        }
        li.append(actions);
      }
      list.append(li);
    }
    const modeHint=autoAnnotation?.auto_scope==='preview'?'抽查预览':provenance|| (autoAnnotation?.auto_override_scope==='region'?'有人工片段标注':'正式体检');
    const progress=autoAnnotation?.auto_review_total ? ` · 已复核 ${autoAnnotation.auto_review_resolved_count}/${autoAnnotation.auto_review_total} 条` : '';
    summary.textContent=`自动体检发现的片段（${visible.length + (excluded ? 1 : 0)} · ${modeHint}${progress}）`;
    updatePromptSelection();
    feedbackButtons();
  }
  async function selectPrompt(item,target) {
    activePrompt=target?{prompt:item,target}:null;
    await locateReview(item);
    if(activePrompt)status('已选择此条提示。试听后可确认保留，或保存评价并完成此条。');
    updatePromptSelection();
  }
  function updatePromptSelection() {
    const control=node('feedbackCompletePrompt');
    if(!control)return;
    const hidden=!activePrompt || activePrompt.prompt.reviewed || entry?.training_context?.audit_read_only;
    const label=node('feedbackCompletePromptLabel');
    if(label)label.hidden=hidden;
    if(hidden || control.dataset.promptId!==activePrompt?.prompt.prompt_id)control.checked=false;
    control.dataset.promptId=hidden?'':activePrompt.prompt.prompt_id;
  }
  async function savePrompt(action,item,target) {
    if(busy)return;
    const context=entry.training_context,task=entry.task_id,version=chart();
    busy=true;feedbackButtons();
    try {
      const result=await trainingRequest('source-feedback',{manifest:context.manifest,chart_id:context.chart_id,
        file_sha256:version.file_sha256,chart_sha256:version.chart_sha256,audio_sha256:version.audio_sha256,
        prompt_action:action,review_target:target});
      showPromptResult(result,context,task);
    } catch(error) {if(entry?.task_id===task)status(error.message);}
    finally {busy=false;feedbackButtons();}
  }
  function showPromptResult(result,context,task,savedText) {
    if(entry?.task_id===task && entry.training_context?.manifest===context.manifest) {
      if(result.auto_annotation)entry.training_context.auto_annotation=result.auto_annotation;
      const updated=result.prompt_review;
      if(updated&&originPrompt?.prompt.prompt_id===updated.prompt_id)originPrompt={...originPrompt,prompt:updated};
      if(updated&&activePrompt?.prompt.prompt_id===updated.prompt_id)activePrompt={...activePrompt,prompt:updated};
      renderAuto(entry.training_context.auto_annotation);
      status(updated?.review_status==='kept'?'已复核 · 保留当前训练标注。':
        updated?.review_status==='annotated'?(savedText||'已保存评价。')+' 此条复核已完成。':
        updated?.review_status==='deferred'?'已标记稍后处理。':'已改为待复核。');
    }
    window.dispatchEvent(new CustomEvent('material-prompt-reviewed',{detail:{manifest:result.review_manifest,result}}));
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
          if(button.id==='feedbackExcludeSpan')Object.assign(payload,{label:'problem',reason:''});
          Object.assign(payload,{manifest:sourceContext.manifest,chart_id:sourceContext.chart_id,
            auto_scope:sourceContext.auto_annotation?.auto_scope||'formal'});url='/api/quality/source-feedback';
          if(activePrompt && node('feedbackCompletePrompt')?.checked) {
            payload.prompt_action='annotate';payload.review_target=activePrompt.target;
          }
        }
      }
      busy=true;feedbackButtons();
      const response=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}),result=await response.json();
      if(!response.ok)throw new Error(result.error||'保存失败');
      if(sourceContext) {
        const record=result.records?.find(row=>row.chart_id===sourceContext.chart_id);
        const saved=record?.regions?.find(item=>item.start_sec===bounds.start_sec && item.end_sec===bounds.end_sec);
        if(chart()?.path===identity && entry.task_id===task && entry.training_context?.manifest===sourceContext.manifest) {
          entry.annotations=record?.regions||[];renderSaved(entry.annotations);
          status(saved?`已保存片段标注${savedSpanEffect(saved)}。`:'已保存质量标注。');
        }
        window.dispatchEvent(new CustomEvent('training-source-annotated',{detail:{...sourceContext,result}}));
        if(result.prompt_review)showPromptResult(result,sourceContext,task,
          saved?`已保存片段标注${savedSpanEffect(saved)}。`:'已保存片段标注。');
      } else {
        if(chart()?.path===identity && entry.task_id===task)status('已记录评价。要学习或导出这段，请展开“收集训练示范”。');
        if(!pair)void loadCollection();
      }
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
  let enabled=false,chartActions=false,autoScreenCapable=false,previewedAutoScreen=false,active=null,current=null,queue=[],seen=new Map(),undo=null,busy=false,scanning=false,serial=0,lastStatus=null;
  let budgetPending=false,budgetTaskId=null;
  let scanTimer=null;
  const isPool=item=>!!(item?.candidate_pool || item?.candidate_pool_summary);
  const identity=()=>active?.manifest;
  const message=text=>{$('candidatePoolReviewStatus').textContent=text;};
  function renderAutoScreenSummary(screen,{preview=false}={}) {
    const box=$('candidateAutoScreenResult');if(!box)return;
    if(!screen||!screen.counts){box.hidden=true;box.replaceChildren();return;}
    const counts=screen.counts||{},slots=screen.approved_by_slot||{};
    const totalIn=screen.total_entering_training??((counts.approved_clean||0)+(counts.approved_masked||0)+(screen.manual_entering_training||0));
    const slotNames={'4':'EXPERT','5':'MASTER','6':'Re:MASTER'};
    const slotParts=['5','6','4'].filter(k=>Number(slots[k]||0)>0).map(k=>`${slotNames[k]||`槽位${k}`} ${slots[k]}`).join(' · ');
    const tag=preview?'【预览（未改标注）】':'【已应用入训标注】';
    const line1=document.createElement('p');
    line1.textContent=`${tag} 实际纳入训练 ${totalIn} 张${slotParts?`（${slotParts}）`:''}：整谱 1× 入训 ${counts.approved_clean??0} 张 · 局部屏蔽硬无理后 1× 入训 ${counts.approved_masked??0} 张（共屏蔽 ${counts.masked_regions??0} 段 0×）${counts.preserved_manual?` · 保留人工决定 ${counts.preserved_manual} 张（入训 ${screen.manual_entering_training??0} 张）`:''}`;
    const line2=document.createElement('p');line2.className='muted';
    line2.textContent=`保持 0× 待人工核对 ${counts.deferred_review??0} 张 ｜ 排除（0×）：非目标难度槽 ${counts.excluded_slot??0} 张 · 定数范围外 ${counts.excluded_level??0} 张 · 扫描/语法异常 ${counts.excluded_error??0} 张 · 整谱无理过多 ${counts.excluded_muri??0} 张`;
    box.replaceChildren(line1,line2);
    box.hidden=false;
  }
  const describe=value=>Array.isArray(value)?value.map(describe).join('；'):typeof value==='string'?
    ({existing_exact_chart:'与已有资料为同一张谱面',duplicate_source_chart:'与已有资料指向同一谱面槽位'}[value]||value):
    value?.message||value?.label||'需要结合音频核对';
  const token=item=>`${item.chart_id}:${item.start_sec}:${item.end_sec}`;
  function scanLabel(data) {
    const counts=data?.counts||{},progress=data?.scan_task?.progress||{};
    const state=data?.scan_state;
    if(['queued','running'].includes(state))return `自动初筛${state==='queued'?'排队中':'进行中'}${progress.total>0?` · ${progress.completed||0}/${progress.total} 张`:''}`;
    if(state==='failed')return '最近一次自动初筛失败';
    if(state==='cancelled'||state==='interrupted')return '最近一次自动初筛中断';
    if(!data?.scan_complete)return '尚无完成的自动初筛';
    const checked=(counts.ok||0)+(counts.suspicious||0)+(counts.error||0);
    const label=data.stale?'上次自动初筛已完成 · 有来源变化，需同步':'自动初筛已完成';
    const date=data.updated_at ? new Date(data.updated_at).toLocaleDateString('zh-CN') : '';
    return `${label}${date?`（${date}）`:''} · ${progress.completed??checked}/${progress.total??counts.charts??checked} 张 · 低风险 ${counts.ok??0} · 待核对 ${counts.suspicious??0} · 错误 ${counts.error??0}${counts.duplicate?` · 扫描前排除 ${counts.duplicate}`:''}`;
  }
  function controls() {
    const source=globalThis.ChartPreview?.getEntry();
    const matches=current && source?.training_context?.manifest===identity() && source?.training_context?.chart_id===current.chart_id;
    $('candidatePoolReview').hidden=!matches;
    const ordinary=$('feedbackStatus')?.closest('.review-tool-group');if(ordinary)ordinary.hidden=!!matches||!!source?.training_context?.audit_read_only;
    const error=['error','stale','pending'].includes(current?.candidate_status)||current?.status==='error';
    for(const button of document.querySelectorAll('[data-pool-learn]'))button.disabled=busy||scanning||!matches||error;
    $('candidatePoolWhole').disabled=busy||scanning||!matches||error;
    $('candidatePoolReject').disabled=busy||scanning||!matches||current?.candidate_status==='stale';
    $('candidatePoolWhole').disabled||=!chartActions;
    $('candidatePoolReject').disabled||=!chartActions;
    $('candidatePoolChartStatus').textContent=!matches?'':current.learning_excluded?'整张谱已设为不学习。重新确认片段或整谱后可恢复。':current.chart_review==='learn_all'?'整张谱已确认学习，不再推荐本谱片段。':'当前按片段学习，未确认部分不参与学习。';
    $('candidatePoolNextChart').disabled=busy||scanning||!matches;
    $('candidatePoolUndo').disabled=busy||!undo;
    for(const id of ['candidatePoolNext','candidatePoolSync','candidatePoolExcluded'])$(id).disabled=busy||!enabled;
    $('candidatePoolBudget').disabled=budgetPending||!enabled;
    if(scanning||$('candidatePoolStatusFilter')?.value==='error')$('candidatePoolNext').disabled=true;
    const chosen=active&&selectedDatasetItems().some(item=>item.manifest===identity());
    $('candidatePoolUse').disabled=busy||scanning||!active||!(lastStatus?.confirmed_regions>0);
    $('candidatePoolUse').textContent=chosen?'已加入下次训练来源':'将已确认内容加入下次训练';
    $('candidatePoolUse').disabled||=!!chosen;
    $('candidatePoolTrainingStatus').textContent=chosen?'已勾选这个库：下次训练会使用已确认内容，当前训练不变。':lastStatus?.confirmed_regions>0?'标注已保存，尚未加入下次训练。':'先确认一段或整张谱，再加入下次训练。';
    $('candidatePoolUsePreview').disabled=$('candidatePoolUse').disabled;
    $('candidatePoolUsePreview').textContent=$('candidatePoolUse').textContent;
    $('candidatePoolPreviewTrainingStatus').textContent=$('candidatePoolTrainingStatus').textContent;
    const canAutoScreen=!busy&&!scanning&&enabled&&autoScreenCapable&&!!active&&!!lastStatus?.scan_complete;
    if($('candidateAutoScreenPreview'))$('candidateAutoScreenPreview').disabled=!canAutoScreen;
    if($('candidateAutoScreenApply'))$('candidateAutoScreenApply').disabled=!canAutoScreen;
    const bulk=$('qualityBulkControls');if(bulk)bulk.hidden=!!active||(typeof activeDatasetItem==='function' && !!activeDatasetItem()?.auditReadOnly);
  }
  function summary(data) {
    const counts={...(data.counts||data)};
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
    lastStatus=data;
    if(active)active.candidate_pool_summary=data;
    const decisions=data.decisions||{};
    const scanCounts=stage==='scanning'&&scanning ? `低风险 ${counts.ok??0} · 待核对 ${counts.suspicious??0} · 错误 ${counts.error??0}` : '';
    $('candidatePoolSummary').textContent=scanMessage ? scanMessage+scanCounts : scanLabel(data);
    $('candidatePoolDecisions').textContent=`你的决定：整谱学习 ${decisions.learn_all??0} 张 · 部分学习 ${decisions.partial??0} 张 · 整谱不学习 ${decisions.exclude??0} 张 · 已确认 ${data.confirmed_regions??data.reviewed_regions??0} 段。`;
    if(!previewedAutoScreen&&data.auto_screen)renderAutoScreenSummary(data.auto_screen,{preview:false});
    controls();
  }
  async function refresh() {
    const manifest=identity(),request=serial;if(!manifest||!enabled)return;
    const data=await api('/api/candidates/status',{manifest,summary_only:true});
    if(request===serial&&manifest===identity()) {
      const completed=Boolean(lastStatus && ['queued','running'].includes(lastStatus.scan_state) && data.scan_state==='completed');
      summary(data);
      if(completed)void loadQualityCatalog(false);
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
  async function next(skipChart=false) {
    const manifest=identity(),request=serial;if(!manifest)return;
    const skipId=skipChart?current?.chart_id:null;
    if(skipId)queue=[];
    if(!queue.length && typeof materialsWorkflowSupported==='function' && materialsWorkflowSupported()) {
      message('正在查找下一段…');
      const data=await api('/api/candidates/queue',{manifest,limit:25,
        candidate_status:$('candidatePoolStatusFilter')?.value||'',
        exclude_chart_id:skipId||null,prefer_chart_id:skipId?null:current?.chart_id,
        seen_segments:[...seen.values()]});
      if(request!==serial)return;
      queue=data.items||[];
    } else if(!queue.length) {
      let offset=0,total=Infinity;
      while(!queue.length&&offset<total) {
        const data=await api('/api/candidates/queue',{manifest,offset,limit:100,candidate_status:$('candidatePoolStatusFilter')?.value||''});
        if(request!==serial)return;
        const items=data.items||[];total=Number(data.total??items.length);
        queue=items.filter(item=>!seen.has(token(item))&&item.chart_id!==skipId);offset=data.next_offset??offset+items.length;
        if(!items.length)break;
      }
    }
    if(!queue.length){message(skipId?'暂时没有其他待审谱面。当前谱面的标注未改变。':$('candidatePoolStatusFilter')?.value==='error'?'错误谱面请先在目录查看原因，修正源资料后再同步。':'本轮暂无其他待审片段；已保存的决定可在目录查看，或打开整谱继续圈选。');return;}
    if(current) {const same=queue.findIndex(item=>item.chart_id===current.chart_id);if(same>0)queue.unshift(...queue.splice(same,1));}
    const item=queue.shift();
    if(current?.chart_id!==item.chart_id||globalThis.ChartPreview?.getEntry()?.training_context?.manifest!==manifest||globalThis.ChartPreview?.getEntry()?.training_context?.chart_id!==item.chart_id) {
      const preview=await api('/api/quality/chart-preview',{manifest,chart_id:item.chart_id});
      if(request!==serial)return;
      globalThis.ChartPreview?.openSource(preview,{scroll:true});
    }
    if(request!==serial)return;
    current=item;seen.set(token(item),[item.chart_id,item.start_sec,item.end_sec]);
    $('feedbackContext').checked=true;
    globalThis.FeedbackRange?.set({start_sec:Number(item.start_sec),end_sec:Number(item.end_sec)});
    $('candidatePoolWhy').textContent=`${Number(item.start_sec).toFixed(2)}–${Number(item.end_sec).toFixed(2)} 秒 · ${describe(item.why||'代表性片段')}。可在时间轴微调，预览始终保留整谱。`;
    $('candidatePoolIssues').textContent=(Array.isArray(item.issues)?item.issues:[item.issues]).filter(Boolean).map(describe).join('；')||'扫描未列出问题；请结合音频与前后文判断。';
    $('candidatePoolEntry').checked=false;message('下面的片段按钮只保存选中范围；整张谱的决定请用“整谱处理”。');controls();
    await globalThis.ChartPreview?.seek?.(Number(item.start_sec));
  }
  async function save(use,whole=false) {
    const manifest=identity(),item=current,request=serial;
    const context=globalThis.ChartPreview?.getEntry()?.training_context;
    if(context?.manifest!==manifest||context?.chart_id!==item?.chart_id)throw new Error('预览已切换，请重新打开候选片段。');
    if(!item)throw new Error('请先打开候选片段。');
    const version=globalThis.ChartPreview?.getEntry()?.charts?.[0];
    if(!version?.file_sha256||!version?.chart_sha256||!version?.audio_sha256||!version?.audio_path)throw new Error('预览缺少谱面或音频身份，请重新打开候选片段后再保存。');
    if(use!=='exclude'&&(item.candidate_status==='error'||item.status==='error'))throw new Error('该谱有阻断错误，请修正源谱并同步后再审阅。');
    let range=globalThis.FeedbackRange?.get();
    if(whole) {
      if(!chartActions)throw new Error('整谱处理需在下次重启 WebUI 后使用。');
      if(!confirm(use==='exclude'?'将这张谱面全部设为不学习？已有确认片段也会移出学习范围，源文件保留，可撤销。':'将这张谱面的采音和配置全部设为学习？这会替换本谱已有的片段选择，并结束本谱审阅。'))return;
    }
    if(!whole&&(!range||!Number.isFinite(range.start_sec)||!Number.isFinite(range.end_sec)||range.end_sec<=range.start_sec))throw new Error('请在时间轴选择有效片段。');
    const selection=whole?{chart_action:use==='exclude'?'exclude':'learn_all'}:{...range,label:use==='uncertain'?'uncertain':'good',reason:['timing','layout'].includes(use)?use:'',learn_entry:use!=='layout'&&use!=='uncertain'&&$('candidatePoolEntry').checked,context_start_sec:Math.max(0,range.start_sec-4),context_end_sec:range.end_sec+4};
    const result=await api('/api/quality/source-feedback',{manifest,chart_id:item.chart_id,file_sha256:version.file_sha256,chart_sha256:version.chart_sha256,audio_sha256:version.audio_sha256,audio_path:version.audio_path,note:$('candidatePoolReason').value.trim(),...selection});
    undo=result.undo_token?{manifest,chart_id:item.chart_id,undo_token:result.undo_token}:null;
    if(manifest===identity())queue=[]; // Never reuse pre-save suggestions.
    const record=result.records?.find(row=>row.chart_id===item.chart_id);
    if(manifest===identity()&&current?.chart_id===item.chart_id)current={...current,chart_review:record?.candidate_chart_review,learning_excluded:record?.label==='excluded'};
    window.dispatchEvent(new CustomEvent('training-source-annotated',{detail:{manifest,chart_id:item.chart_id,result}}));
    if(request!==serial)return;
    const saved=whole?(use==='exclude'?'已保存：整张不学习。':'已保存：整张学习。'):use==='uncertain'?'已保存：这段不学习。':'已保存当前片段。';
    message(saved);$('candidatePoolReason').value='';
    if(result.candidate_pool_summary)summary(result.candidate_pool_summary);
    else void refresh().catch(error=>message(saved+' '+error.message));
    if($('candidatePoolAutoNext').checked&&current?.chart_id===item.chart_id&&globalThis.ChartPreview?.getEntry()?.training_context?.chart_id===item.chart_id){await next(whole);message(saved+' '+$('candidatePoolReviewStatus').textContent);}
  }
  globalThis.CandidatePool={
    scanLabel,
    reviewText(item) {
      const record=item.annotation||item.record||{};
      if(item.label==='excluded'||item.label==='negative'||Number(item.weight??record.weight??1)<=0)return '整张不学习';
      if((item.candidate_chart_review||record.candidate_chart_review)==='learn_all')return '整张学习';
      const count=item.confirmed_regions??record.regions?.length??0;
      return count?`只学已确认的 ${count} 段`:record.candidate_reviews?.length?'已略过片段，未确认学习':'尚未确认学习';
    },
    badge(item) {
      if(!item.candidate_status)return '';
      const names={error:'错误',suspicious:'待核对',ok:'低风险',pending:'待扫描'};
      const issues=item.candidate_issues||[];
      return `<br><span class="quality-label">${escapeHtml(names[item.candidate_status]||item.candidate_status)}</span> <span class="muted">已确认 ${escapeHtml(String(item.confirmed_regions??item.reviewed_regions??item.record?.regions?.length??0))} 段</span>${issues.length?`<details class="candidate-issues"><summary>查看 ${issues.length} 项检查提示</summary>${issues.map(issue=>`<p>${escapeHtml(issue.message||issue.code||'待核对')}</p>`).join('')}</details>`:''}`;
    },
    card(item) {
      if(!item.manifest||item.prepared===false)return '';
      if(item.reviewed_regions!=null&&!isPool(item))return '<p class="muted"><strong>已确认示范库</strong>：只学习保存或导入的示范片段。</p>';
      const pool=isPool(item),info=item.candidate_pool_summary||item.candidate_pool||{};
      return `<div class="candidate-pool-card">${pool?`<p class="dataset-audit-state">${escapeHtml(scanLabel(info))}</p><p class="muted">已确认 ${escapeHtml(String(info.confirmed_regions??info.reviewed_regions??0))} 段 · 整谱学习 ${info.decisions?.learn_all??0} 张 · 部分学习 ${info.decisions?.partial??0} 张 · 整谱不学习 ${info.decisions?.exclude??0} 张</p>`:'<p class="muted">普通训练集 · 默认学习整谱</p>'}${pool?'':`<button type="button" class="small-button candidate-pool-create" data-manifest="${escapeHtml(item.manifest)}" ${enabled?'':'disabled'}>建立待审候选库</button>`}</div>`;
    },
    capabilities(state) {const caps=state.training_capabilities||state.capabilities||{};enabled=caps.candidate_pools===true;chartActions=caps.candidate_chart_review===true;autoScreenCapable=caps.candidate_pool_auto_screen!==undefined?caps.candidate_pool_auto_screen===true:enabled;this.render();},
    render() {
      $('candidatePoolCapability').textContent=enabled?(chartActions?'候选库的扫描只提供检查提示；你保存的学习决定，才决定下次训练学哪些内容。':'片段审阅可用；新增整谱处理需在下次重启 WebUI 后使用。'):'候选库需重启 WebUI 后使用。';
      for(const button of document.querySelectorAll('.candidate-pool-create'))button.disabled=!enabled;
      if(selectedDatasetItems().some(isPool)){const input=$('v2TrainQualityEnabled');if(input)input.checked=true;}
      controls();
    },
    activate(item) {++serial;if(scanTimer!==null&&typeof clearTimeout==='function')clearTimeout(scanTimer);scanTimer=null;active=isPool(item)?item:null;lastStatus=null;current=null;queue=[];seen.clear();previewedAutoScreen=false;renderAutoScreenSummary(null);for(const id of ['candidatePoolStatusFilter','candidatePoolDecisionFilter'])if($(id))$(id).value='';$('candidatePoolPanel').hidden=!active;controls();if(active)void refresh().catch(error=>{$('candidatePoolSummary').textContent=error.message;});}
  };
  document.addEventListener('click',event=>{
    const create=event.target.closest?.('.candidate-pool-create');if(!create)return;
    void run(async()=>{
      if(!enabled)throw new Error('重启后台后启用候选库。');
      const source=datasetWorkspaceState.items.find(item=>item.manifest===create.dataset.manifest);
      const name=(source?.name||'素材')+' · 候选库';
      const result=await api('/api/candidates/create',{manifest:create.dataset.manifest,name});
      $('candidatePoolCapability').textContent='待审候选库已建立，正在扫描。试听确认后才会有可学习的内容。';
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
  async function runAutoScreen(dryRun) {
    const manifest=identity(),request=serial;if(!manifest)return;
    if(!enabled||!autoScreenCapable)throw new Error('请重启 WebUI 后台以启用大数据一键自动筛选。');
    if(!lastStatus?.scan_complete)throw new Error('请先等待候选库自动初筛完成后再执行一键筛选。');
    const includeExpert=!!$('candidateAutoScreenExpert')?.checked;
    const minLevel=Number($('candidateAutoScreenMinLevel')?.value||12);
    const maxLevel=Number($('candidateAutoScreenMaxLevel')?.value||15);
    const maxOffset=Number($('candidateAutoScreenMaxOffset')?.value||35);
    const tapMode=$('candidateAutoScreenTailMode')?.value||'mask_hard';
    const innerMode=$('candidateAutoScreenInnerMode')?.value||'allow';
    const preserveManual=$('candidateAutoScreenPreserveManual')?$('candidateAutoScreenPreserveManual').checked:true;
    if(!dryRun && typeof confirm==='function') {
      const slotDesc=includeExpert?'EXPERT / MASTER / Re:MASTER':'MASTER / Re:MASTER';
      const manualDesc=preserveManual?'保留已有人工审阅决定':'覆盖已有决定';
      if(!confirm(`确认按当前规则（${slotDesc}，定数 ${minLevel}–${maxLevel}，${manualDesc}）生成入训标注？源谱文件绝不修改，之后仍可逐张或逐段微调。`))return;
    }
    const box=$('candidateAutoScreenResult');
    if(box){box.hidden=false;box.textContent=dryRun?'正在预览自动筛选统计…':'正在生成并写入自动筛选入训标注…';}
    const response=await api('/api/candidates/auto-screen',{
      manifest,
      difficulty_slots:includeExpert?[4,5,6]:[5,6],
      min_level:minLevel,
      max_level:maxLevel,
      max_offset_ms:maxOffset,
      tap_on_slide_mode:tapMode,
      inner_muri_mode:innerMode,
      preserve_manual:preserveManual,
      dry_run:Boolean(dryRun),
    });
    if(request!==serial||manifest!==identity())return;
    previewedAutoScreen=Boolean(dryRun);
    renderAutoScreenSummary(response,{preview:Boolean(dryRun)});
    if(!dryRun) {
      queue=[];seen.clear();
      if(response.pool_status)summary(response.pool_status);
      else await refresh();
      if(typeof loadQualityCatalog==='function')await loadQualityCatalog(true);
      if(typeof loadDatasetCatalog==='function')await loadDatasetCatalog();
      window.dispatchEvent(new CustomEvent('training-source-annotated',{detail:{manifest,auto_screen:response}}));
      message(`一键筛选已应用：实际纳入训练 ${response.total_entering_training??0} 张谱面。可点击下方“将已确认内容加入下次训练”。`);
    }
  }
  $('candidateAutoScreenPreview')?.addEventListener('click',()=>void run(()=>runAutoScreen(true)));
  $('candidateAutoScreenApply')?.addEventListener('click',()=>void run(()=>runAutoScreen(false)));
  $('candidatePoolNext').addEventListener('click',()=>void run(next));
  $('candidatePoolDecisionFilter').addEventListener('change',()=>void run(()=>loadQualityCatalog(true)));
  $('candidatePoolNextChart').addEventListener('click',()=>void run(()=>next(true)));
  for(const id of ['candidatePoolUse','candidatePoolUsePreview'])$(id).addEventListener('click',()=>void run(async()=>{
    if(!active||!lastStatus?.confirmed_regions)return;
    const pathKey=value=>String(value||'').replaceAll('\\','/').toLowerCase();
    const original=selectedDatasetItems().find(item=>pathKey(item.manifest)===pathKey(lastStatus.source_manifest));
    if(original&&!confirm('原训练集也已勾选，会提供整谱内容。改为只使用这个候选库的已确认内容，并取消对应原训练集的勾选？其他来源保持不变。'))return;
    if(original)datasetWorkspaceState.selected.delete(datasetKey(original));
    datasetWorkspaceState.selected.add(datasetKey(active));datasetWorkspaceState.selectionTouched=true;datasetWorkspaceState.prepared=null;
    $('v2TrainQualityEnabled').checked=true;renderDatasetWorkspace();saveDatasetWorkspace();saveSettings();
    message('已加入下次训练来源。回到训练页检查来源后开始训练；当前训练不变。');
  }));
  $('candidatePoolStatusFilter')?.addEventListener('change',()=>void run(async()=>{queue=[];seen.clear();
    if(globalThis.maichartLastStatus&&!globalThis.maichartLastStatus.training_capabilities?.candidate_review_filters){$('candidatePoolStatusFilter').value='';throw new Error('当前后台尚未载入检查状态筛选，请使用已更新的 WebUI 实例。');}
    await loadQualityCatalog(true);
    message($('candidatePoolStatusFilter').value==='error'?'在目录展开检查提示，先修正源资料，再重新扫描。':'已切换筛选；点“继续审阅”试听下一段。');}));
  $('candidatePoolSync').addEventListener('click',()=>void run(async()=>{const manifest=identity(),result=await api('/api/candidates/sync',{manifest});queue=[];seen.clear();message('正在后台重新扫描／同步；源谱变化会使旧确认失效。');if(result.task_id)watchTask(result.task_id,()=>{if(identity()===manifest)void refresh();});await refresh();}));
  $('candidatePoolExcluded').addEventListener('click',()=>void run(async()=>{const manifest=identity(),request=serial,data=await api('/api/candidates/excluded',{manifest});if(request!==serial)return;const list=$('candidatePoolExcludedList');list.hidden=false;list.replaceChildren();for(const item of data.items||[]){const row=document.createElement('p');row.textContent=`${item.title||item.source_path||item.chart_id||item.path||'谱面'}${item.slot!==undefined?` · 槽位${item.slot}`:''}：${describe(item.reason||item.issues||item)}`;list.append(row);}if(!list.children.length)list.textContent='暂无排除项。';}));
  for(const button of document.querySelectorAll('[data-pool-learn]'))button.addEventListener('click',()=>void run(()=>save(button.dataset.poolLearn)));
  $('candidatePoolWhole').addEventListener('click',()=>void run(()=>save('both',true)));
  $('candidatePoolReject').addEventListener('click',()=>void run(()=>save('exclude',true)));
  $('candidatePoolUndo').addEventListener('click',()=>void run(async()=>{const last=undo;if(!last)return;const result=await api('/api/candidates/undo',last);undo=null;if(identity()===last.manifest){queue=[];seen.clear();await refresh();}window.dispatchEvent(new CustomEvent('training-source-annotated',{detail:{...last,result}}));message('已撤销上次保存。');}));
  const syncPoolContext=event=>{
    const context=(event?.detail||globalThis.ChartPreview?.getEntry())?.training_context;
    if(active&&context?.manifest===identity()&&context.chart_id){
      if(context.chart_id!==current?.chart_id){
        current={...context};$('candidatePoolWhy').textContent='在时间轴圈选一段完整写法，带前后文试听，再确认值得学习的内容。';
        $('candidatePoolIssues').textContent=(context.issues||[]).map(describe).join('；')||'没有自动提示，请结合歌曲核对采音、衔接和手顺。';
      }else{
        current={...current,...context};
      }
    }
    controls();
  };
  window.addEventListener('chart-preview-updated',syncPoolContext);
  window.addEventListener('chart-preview-annotations-updated',syncPoolContext);
  window.addEventListener('chart-preview-versionchanged',controls);
  window.addEventListener('training-source-annotated',event=>{
    const {manifest,chart_id,result}=event.detail;
    if(manifest!==identity())return;
    queue=[];
    if(current?.chart_id===chart_id){
      const record=result.records?.find(row=>row.chart_id===chart_id);
      const patch={chart_review:record?.candidate_chart_review,learning_excluded:['excluded','negative'].includes(record?.label)||record?.weight===0};
      current={...current,...patch};
      const context=globalThis.ChartPreview?.getEntry()?.training_context;
      if(context?.manifest===manifest&&context.chart_id===chart_id)Object.assign(context,patch);
    }
    controls();
  });
  globalThis.CandidatePool.render();
  if(globalThis.maichartLastStatus)globalThis.CandidatePool.capabilities(globalThis.maichartLastStatus);
})();

// END CANDIDATE POOL WORKBENCH
