"use strict";
// The editor changes learned curve conditions, never note placement rules.
const v2Curve={profile:null,automatic:null,undo:[],source:null,timingStale:false,selected:0,busy:false,
  selection:new Set(),start:0,end:null,cursor:0,media:null,mediaAudio:null,raster:null,mediaSerial:0,mediaBusy:false,drag:null};
function v2CurveTimingContext(){
  return {
    bpm:valueOrDefault('#bpm','').trim(),
    autoFirst:Boolean(element('#v2GenerateAutoFirst')?.checked??true),
    first:element('#v2GenerateAutoFirst')?.checked?'':valueOrDefault('#first','').trim(),
    tempoDetection:valueOrDefault('#v2GenerateTempoDetection','auto'),
    tempoOctavePolicy:valueOrDefault('#v2GenerateTempoOctavePolicy','off'),
    downbeatAlignment:valueOrDefault('#v2GenerateDownbeatAlignment','off'),
    tempoMap:valueOrDefault('#tempoMap','').trim(),
    meterMap:valueOrDefault('#formGenerationMeterMap','').trim(),
  };
}
function v2CurveSource(){
  return {
    audio:valueOrDefault('#audioPath','').trim(),
    checkpoint:valueOrDefault('#checkpointPath','').trim(),
    timing:v2CurveTimingContext(),
  };
}
function v2CurveTimingMatches(){
  return Boolean(v2Curve.source?.timing)&&JSON.stringify(v2Curve.source.timing)===JSON.stringify(v2CurveTimingContext());
}
function v2CurveMatches(){
  const s=v2CurveSource();
  if(v2Curve.source?.audio!==s.audio||v2Curve.source?.checkpoint!==s.checkpoint)return false;
  if(v2Curve.profile?.edited===true)return true;
  if(v2Curve.timingStale)return false;
  if(v2Curve.source?.timing!==undefined||s.timing!==undefined){
    return JSON.stringify(v2Curve.source?.timing)===JSON.stringify(s.timing);
  }
  return true;
}
function v2CurveInvalidateTiming(){
  const s=v2CurveSource();
  if(!v2Curve.profile||v2Curve.source?.audio!==s.audio||v2Curve.source?.checkpoint!==s.checkpoint)return;
  if(JSON.stringify(v2Curve.source?.timing)===JSON.stringify(s.timing)){
    v2Curve.timingStale=false;
    return;
  }
  v2Curve.timingStale=true;
  if(v2Curve.profile.edited===true){
    v2CurveStatus('节拍／拍网设置已变化：已保留手动编辑的按秒控制点；若需基于新拍网重算基底请重新预测，否则可明确沿用当前编辑曲线。',true);
  }else{
    if(element('#v2CurveUse'))element('#v2CurveUse').checked=false;
    v2CurveStatus('节拍／拍网设置已变化，自动预测曲线已过期；直接生成将按新拍网自动预测，如需编辑请重新预测曲线。',true);
  }
  v2CurveDraw();
}
function v2CurvePayload(){
  if(!element('#v2CurveUse')?.checked)return {};
  const s=v2CurveSource(),sameAsset=v2Curve.source?.audio===s.audio&&v2Curve.source?.checkpoint===s.checkpoint;
  if(!v2Curve.profile||!sameAsset)throw new Error('编辑曲线与当前音频／模型不匹配，请重新预测或关闭编辑曲线。');
  if(!v2CurveMatches())throw new Error('节拍／拍网设置已变化，未编辑的自动曲线已过期；请重新预测曲线或取消勾选编辑曲线。');
  return {intensity_profile:v2Curve.profile,intensity_strength:Number(valueOrDefault('#v2CurveStrength',1))};
}
function v2CurveStatus(message,bad=false){const n=element('#v2CurveStatus');n.textContent=message;n.classList.toggle('warn',bad);}
function v2CurveDuration(){return Number(v2Curve.profile?.duration_sec||v2Curve.media?.duration_sec||0);}
function v2CurveView(){const duration=v2CurveDuration()||1,end=Math.min(duration,v2Curve.end??duration),start=Math.max(0,Math.min(v2Curve.start,end-.01));return {start,end,span:end-start,duration};}
function v2CurveTime(t){return `${Math.floor(t/60)}:${(t%60).toFixed(2).padStart(5,'0')}`;}
function v2CurveTrack(){return valueOrDefault('#v2CurveTrack','total');}
function v2CurveValue(p,track=v2CurveTrack()){return track==='events'?p.event_rate:track==='chord'?(p.event_rate>0?p.object_rate/p.event_rate:1):p.object_rate;}
function v2CurveSelection(){return v2Curve.selection||new Set([v2Curve.selected]);}
function v2CurveRemember(){v2Curve.undo.push(structuredClone(v2Curve.profile));if(v2Curve.undo.length>30)v2Curve.undo.shift();}
function v2CurveChanged(){v2Curve.profile.edited=true;element('#v2CurveUse').checked=true;v2CurveDraw();}
function v2CurveSetPoint(p,value,track){
  if(track==='chord')p.object_rate=p.event_rate*Math.max(1,value);
  else if(track==='events'){const ratio=p.event_rate>0?p.object_rate/p.event_rate:1;p.event_rate=value;p.object_rate=value*ratio;}
  else {const fraction=p.object_rate>0?p.event_rate/p.object_rate:1;p.event_rate=value*fraction;p.object_rate=value;}
}
function v2CurveEdit(index,value,remember=true){
  if(!Number.isFinite(value)||value<0)return;
  if(remember)v2CurveRemember();
  v2CurveSetPoint(v2Curve.profile.points[index],value,v2CurveTrack());
  v2CurveChanged();v2CurveStatus('已修改曲线；仍由模型决定具体采音与配置。');
}
function v2CurveBatch(mode,amount){
  const points=v2Curve.profile?.points,selection=[...v2CurveSelection()].filter(i=>points?.[i]);
  if(!selection.length)return v2CurveStatus('请先框选控制点。',true);
  if(!Number.isFinite(amount)||(mode!=='add'&&amount<0))return v2CurveStatus('请输入有效调整数值；比例和统一值不能小于0。',true);
  const track=v2CurveTrack(),floor=track==='chord'?1:0;
  const values=selection.map(i=>Math.max(floor,mode==='set'?amount:mode==='scale'?v2CurveValue(points[i],track)*amount:v2CurveValue(points[i],track)+amount));
  if(values.some(v=>!Number.isFinite(v)||v>1e6))return v2CurveStatus('调整数值超出范围。',true);
  v2CurveRemember();selection.forEach((i,j)=>v2CurveSetPoint(points[i],values[j],track));v2CurveChanged();
  v2CurveStatus(`已调整 ${selection.length} 个点；可一次撤销。`);
}
function v2CurveCanvas(canvas){
  const rect=canvas.getBoundingClientRect(),w=rect.width||960,h=rect.height||200,dpr=window.devicePixelRatio||1;
  if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}
  const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);ctx.fillStyle='#101c2a';ctx.fillRect(0,0,w,h);
  return {ctx,w,h};
}
function v2CurveX(t,w,view){return 48+(w-62)*(t-view.start)/view.span;}
function v2CurveY(value,h,max){return 18+(h-44)*(1-value/max);}
function v2CurveDrawWave(canvas,view,overview=false){
  if(!canvas)return;const {ctx,w,h}=v2CurveCanvas(canvas),wave=v2Curve.media?.waveform;
  if(wave){
    ctx.strokeStyle='#60c7ba';ctx.beginPath();
    for(let x=48;x<w-14;x++){
      const a=Math.max(0,Math.floor((view.start+(x-48)/(w-62)*view.span)/wave.step_sec));
      const b=Math.min(wave.minimum.length,Math.max(a+1,Math.ceil((view.start+(x-47)/(w-62)*view.span)/wave.step_sec)));
      let low=0,high=0;for(let i=a;i<b;i++){low=Math.min(low,wave.minimum[i]);high=Math.max(high,wave.maximum[i]);}
      ctx.moveTo(x,h/2-high*h*.42);ctx.lineTo(x,h/2-low*h*.42);
    }ctx.stroke();
  }else {ctx.fillStyle='#b9c8d8';ctx.fillText('加载音频后显示波形',55,h/2);}
  if(overview){const selected=v2CurveView();ctx.fillStyle='#78b7ff33';ctx.fillRect(v2CurveX(selected.start,w,view),0,(w-62)*selected.span/view.span,h);ctx.strokeStyle='#8bc8ff';ctx.strokeRect(v2CurveX(selected.start,w,view),1,(w-62)*selected.span/view.span,h-2);}
  v2CurveDrawCursor(ctx,w,h,view);
}
function v2CurveDrawCursor(ctx,w,h,view){if(v2Curve.cursor<view.start||v2Curve.cursor>view.end)return;const x=v2CurveX(v2Curve.cursor,w,view);ctx.strokeStyle='#ffe59a';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();}
function v2CurveDraw(){
  const canvas=element('#v2CurveCanvas');if(!canvas)return;
  const {ctx,w,h}=v2CurveCanvas(canvas),view=v2CurveView(),points=v2Curve.profile?.points||[],track=v2CurveTrack(),selection=v2CurveSelection();
  const visible=points.filter(p=>p.time_sec>=view.start&&p.time_sec<=view.end);
  const maximum=v2Curve.drag?.maximum||Math.max(track==='chord'?4:8,...visible.map(p=>v2CurveValue(p)))*1.15;
  canvas.dataset.maximum=String(maximum);
  ctx.strokeStyle='#344658';ctx.fillStyle='#b9c8d8';ctx.font='12px sans-serif';
  for(let i=0;i<=4;i++){const y=18+(h-44)*i/4;ctx.beginPath();ctx.moveTo(48,y);ctx.lineTo(w-14,y);ctx.stroke();ctx.fillText((maximum*(1-i/4)).toFixed(1),5,y+4);}
  for(let i=0;i<=5;i++){const x=48+(w-62)*i/5;ctx.fillText(v2CurveTime(view.start+view.span*i/5),Math.min(w-56,Math.max(48,x-19)),h-6);}
  if(points.length){
    ctx.save();ctx.beginPath();ctx.rect(48,0,w-62,h-25);ctx.clip();ctx.strokeStyle='#69c4ff';ctx.lineWidth=2;ctx.beginPath();
    const line=[{...points[0],time_sec:0},...points,{...points.at(-1),time_sec:view.duration}];
    line.forEach((p,i)=>{const x=v2CurveX(p.time_sec,w,view),y=v2CurveY(v2CurveValue(p),h,maximum);if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y);});ctx.stroke();
    for(let i=0;i<points.length;i++){const p=points[i];if(p.time_sec<view.start||p.time_sec>view.end)continue;ctx.fillStyle=selection.has(i)?'#ffe59a':'#8dd5ff';ctx.beginPath();ctx.arc(v2CurveX(p.time_sec,w,view),v2CurveY(v2CurveValue(p),h,maximum),selection.has(i)?5:3,0,2*Math.PI);ctx.fill();}
    ctx.restore();
  }else {ctx.fillStyle='#b9c8d8';ctx.fillText('先加载音频定位，预测或导入曲线后即可编辑',60,h/2);}
  const drag=v2Curve.drag;
  if(drag?.kind==='box'&&drag.moved){ctx.fillStyle='#8bc8ff22';ctx.strokeStyle='#a5d5ff';ctx.fillRect(drag.x,drag.y,drag.lastX-drag.x,drag.lastY-drag.y);ctx.strokeRect(drag.x,drag.y,drag.lastX-drag.x,drag.lastY-drag.y);}
  v2CurveDrawCursor(ctx,w,h,view);
  v2CurveDrawWave(element('#v2CurveWave'),view);
  const specCanvas=element('#v2CurveSpec');
  if(specCanvas&&!specCanvas.hidden){const s=v2CurveCanvas(specCanvas),spec=v2Curve.media?.spectrogram;if(v2Curve.raster&&spec){const start=view.start/spec.step_sec,stop=Math.min(v2Curve.raster.width,view.end/spec.step_sec);s.ctx.drawImage(v2Curve.raster,start,0,Math.max(.01,stop-start),v2Curve.raster.height,48,0,s.w-62,s.h);s.ctx.fillStyle='#ddd';s.ctx.fillText('高频',6,14);s.ctx.fillText('低频',6,s.h-7);}v2CurveDrawCursor(s.ctx,s.w,s.h,view);}
  v2CurveDrawWave(element('#v2CurveOverview'),{start:0,end:view.duration,span:view.duration,duration:view.duration},true);
  const scroll=element('#v2CurveScroll');scroll.max=String(Math.max(0,view.duration-view.span));scroll.value=String(view.start);scroll.disabled=view.span>=view.duration;
  element('#v2CurveViewLabel').textContent=`${v2CurveTime(view.start)} — ${v2CurveTime(view.end)} / ${v2CurveTime(view.duration)}`;
  element('#v2CurvePosition').textContent=v2CurveTime(v2Curve.cursor);
  const p=points[v2Curve.selected];
  element('#v2CurveSelectedCount').textContent=`已选 ${selection.size} 个点`;
  element('#v2CurvePointValue').disabled=!p||selection.size===0;
  element('#v2CurvePointTimeInput').disabled=!p||selection.size!==1;
  if(p){element('#v2CurvePointTime').textContent=v2CurveTime(p.time_sec);if(document.activeElement!==element('#v2CurvePointTimeInput'))element('#v2CurvePointTimeInput').value=p.time_sec.toFixed(4);if(document.activeElement!==element('#v2CurvePointValue'))element('#v2CurvePointValue').value=v2CurveValue(p).toFixed(4);}
  element('#v2CurveUnits').textContent=track==='chord'?'平均每次音符数':track==='events'?'出音次数／秒':'音符个数／秒';
}
function v2CurveSetView(start,span,syncInputs=true){const duration=v2CurveDuration();if(!duration)return;span=Math.min(duration,Math.max(.5,span));start=Math.max(0,Math.min(duration-span,start));v2Curve.start=start;v2Curve.end=start+span;if(syncInputs){element('#v2CurveViewStart').value=start.toFixed(2);element('#v2CurveViewSpan').value=span.toFixed(2);}v2CurveDraw();}
function v2CurveZoom(factor,anchor){const v=v2CurveView();anchor=anchor??(v2Curve.cursor>=v.start&&v2Curve.cursor<=v.end?v2Curve.cursor:(v.start+v.end)/2);const span=Math.min(v.duration,Math.max(.5,v.span*factor)),fraction=(anchor-v.start)/v.span;v2CurveSetView(anchor-fraction*span,span);}
function v2CurveSeek(time){const audio=element('#v2CurveAudio');v2Curve.cursor=Math.max(0,Math.min(v2CurveDuration(),time));if(audio.src)audio.currentTime=v2Curve.cursor;v2CurveDraw();}
function v2CurveSelect(index,add=false){if(!add)v2Curve.selection.clear();if(add&&v2Curve.selection.has(index))v2Curve.selection.delete(index);else v2Curve.selection.add(index);v2Curve.selected=index;v2CurveDraw();}
function v2CurveAddPoint(time=v2Curve.cursor){
  const points=v2Curve.profile?.points;if(!points)return;
  time=Math.max(0,Math.min(v2CurveDuration(),time));let right=points.findIndex(p=>p.time_sec>=time);if(right<0)right=points.length;
  const same=points.findIndex(p=>Math.abs(p.time_sec-time)<1e-6);if(same>=0){v2CurveSelect(same);return;}
  if(points.length>=2048)return v2CurveStatus('最多2048个控制点；请先删除不需要的点。',true);
  const a=points[Math.max(0,right-1)],b=points[Math.min(points.length-1,right)],mix=a===b?0:(time-a.time_sec)/(b.time_sec-a.time_sec);
  const p={time_sec:time,event_rate:a.event_rate+(b.event_rate-a.event_rate)*mix,object_rate:a.object_rate+(b.object_rate-a.object_rate)*mix};
  v2CurveRemember();points.splice(right,0,p);v2CurveSelect(right);v2CurveChanged();
}
function v2CurveRaster(media){
  const spec=media.spectrogram,canvas=document.createElement('canvas');canvas.width=spec.values.length;canvas.height=spec.values[0]?.length||1;
  const ctx=canvas.getContext('2d'),data=ctx.createImageData(canvas.width,canvas.height);
  for(let x=0;x<canvas.width;x++)for(let band=0;band<canvas.height;band++){
    const v=spec.values[x][band]/255,i=((canvas.height-1-band)*canvas.width+x)*4;
    data.data[i]=Math.min(255,Math.max(0,v-.35)*420);data.data[i+1]=v*180;data.data[i+2]=Math.min(255,35+v*210);data.data[i+3]=255;
  }ctx.putImageData(data,0,0);return canvas;
}
async function v2CurveLoadMedia(){
  const source=v2CurveSource(),button=element('#v2CurveLoadAudio'),status=element('#v2CurveMediaStatus');
  if(v2Curve.mediaAudio===source.audio&&v2Curve.media){if(v2Curve.profile&&v2Curve.media.audio_sha256!==v2Curve.profile.audio_sha256){status.textContent='音频与曲线不匹配，请重新预测曲线。';return;}v2CurveDraw();return;}
  if(v2Curve.mediaBusy===source.audio)return;
  const serial=++v2Curve.mediaSerial;
  try{
    if(!source.audio)throw new Error('请先选择音频。');
    v2Curve.mediaBusy=source.audio;button.disabled=true;
    const backend=await api('/api/status');if(!backend.training_capabilities?.composer_v2_editor_media)throw new Error('音频定位需要新版后台，请在方便时重启WebUI；不会影响训练模型。');
    v2Curve.mediaBusy=source.audio;button.disabled=true;status.textContent='正在准备波形与频谱（仅CPU，复用波形缓存）…';
    const result=await api('/api/composer-v2/media',{model_family:'chart_composer_v2',audio:source.audio,cache_dir:valueOrDefault('#v2GenerateCacheDir','cache/chart_composer_v2')});
    const audio=element('#v2CurveAudio');audio.pause();audio.src=`/api/intensity/audio?id=${encodeURIComponent(result.task_id)}`;audio.playbackRate=Number(valueOrDefault('#v2CurvePlaybackRate',1));
    while(serial===v2Curve.mediaSerial){
      const task=await api(`/api/task?id=${encodeURIComponent(result.task_id)}`);
      if(task.status==='failed'||task.status==='cancelled')throw new Error(task.error||'音频定位已取消。');
      if(task.status==='completed'){
        if(source.audio!==v2CurveSource().audio)return;
        const media=task.result.editor_media;if(!media)throw new Error('没有生成音频定位数据。');
        if(v2Curve.profile&&media.audio_sha256!==v2Curve.profile.audio_sha256)throw new Error('音频与曲线不匹配，请重新预测曲线。');
        v2Curve.media=media;v2Curve.mediaAudio=source.audio;v2Curve.raster=v2CurveRaster(media);
        status.textContent='点击波形定位；Ctrl＋滚轮缩放，Shift＋滚轮平移。频谱仅用于定位音乐，不会强制模型采音。';v2CurveSetView(v2Curve.start,v2CurveView().span);return;
      }
      await new Promise(resolve=>setTimeout(resolve,500));
    }
  }catch(error){if(serial===v2Curve.mediaSerial)status.textContent=error.message;}
  finally{if(serial===v2Curve.mediaSerial){v2Curve.mediaBusy=false;button.disabled=false;}}
}
async function v2CurvePredict(){
  if(v2Curve.busy)return;
  try{
    const payload=composerV2GenerationPayload({omitCurve:true});if(!payload.checkpoint||!payload.audio)throw new Error('请先选择模型和音频。');
    const status=await api('/api/status');if(!status.training_capabilities?.composer_v2_intensity_curve)throw new Error('当前后台未加载新版曲线，请重启WebUI后再使用。');
    if((payload.tempo_octave_policy || payload.downbeat_alignment) && !status.training_capabilities?.composer_v2_timing_calibration)throw new Error('可选半／倍速与强拍对齐校准需要更新后的WebUI后台，请在方便时重启后台后再预测曲线。');
    v2Curve.busy=true;element('#v2CurvePredict').disabled=true;v2CurveStatus('正在预测曲线…');
    const source=v2CurveSource(),result=await api('/api/composer-v2/curve',payload);
    watchTask(result.task_id,task=>{
      if(!['completed','failed','cancelled'].includes(task.status))return;
      v2Curve.busy=false;element('#v2CurvePredict').disabled=false;const profile=task.result?.intensity_profile;
      if(task.status!=='completed'||!profile){v2CurveStatus(task.error||'曲线预测未完成。',true);return;}
      if(JSON.stringify(source)!==JSON.stringify(v2CurveSource())){v2CurveStatus('模型、音频或拍网设置已切换，请重新预测曲线。',true);return;}
      v2Curve.source=source;v2Curve.timingStale=false;v2Curve.profile=structuredClone(profile);v2Curve.automatic=structuredClone(profile);v2Curve.undo=[];v2Curve.selected=0;v2Curve.selection=new Set([0]);
      element('#v2CurveUse').checked=false;v2CurveDraw();v2CurveStatus('曲线已就绪：点击定位、拖动控制点，或在空白处框选多个点。');v2CurveLoadMedia();
    });
  }catch(error){v2Curve.busy=false;element('#v2CurvePredict').disabled=false;v2CurveStatus(error.message,true);}
}
function v2CurvePointer(event,phase){
  if(!v2Curve.profile)return;const canvas=event.currentTarget,box=canvas.getBoundingClientRect(),x=event.clientX-box.left,y=event.clientY-box.top,view=v2CurveView();
  const time=Math.max(0,Math.min(view.duration,view.start+(x-48)/(box.width-62)*view.span)),maximum=Number(canvas.dataset.maximum);
  const value=Math.max(v2CurveTrack()==='chord'?1:0,(1-(y-18)/(box.height-44))*maximum);
  if(phase==='down'){
    event.preventDefault();canvas.setPointerCapture(event.pointerId);
    const hit=v2Curve.profile.points.findIndex(p=>p.time_sec>=view.start&&p.time_sec<=view.end&&Math.hypot(v2CurveX(p.time_sec,box.width,view)-x,v2CurveY(v2CurveValue(p),box.height,maximum)-y)<11);
    const additive=event.ctrlKey||event.metaKey;
    v2Curve.drag={kind:v2CurveTrack()&&valueOrDefault('#v2CurveTool','select')==='draw'?'paint':hit>=0?'point':'box',x,y,lastX:x,lastY:y,index:hit,maximum,additive,before:new Set(v2Curve.selection),moved:false,remembered:false};
    if(hit>=0){if(!additive&&v2Curve.selection.has(hit))v2Curve.selected=hit;else v2CurveSelect(hit,additive);v2Curve.cursor=v2Curve.profile.points[hit].time_sec;v2Curve.drag.baseValues=v2Curve.profile.points.map(p=>v2CurveValue(p));}
  }else if(v2Curve.drag){
    const drag=v2Curve.drag;drag.lastX=x;drag.lastY=y;drag.moved ||= Math.hypot(x-drag.x,y-drag.y)>3;
    if(phase==='move'&&drag.moved&&drag.kind!=='box'){
      if(!drag.remembered){v2CurveRemember();drag.remembered=true;}
      const points=v2Curve.profile.points,index=drag.kind==='point'?drag.index:points.reduce((best,p,i)=>Math.abs(p.time_sec-time)<Math.abs(points[best].time_sec-time)?i:best,0);
      v2Curve.selected=index;if(drag.kind==='paint')v2Curve.selection=new Set([index]);
      const selected=drag.kind==='point'&&v2Curve.selection.size>1?[...v2Curve.selection]:[index];
      for(const i of selected){const adjusted=selected.length>1?Math.max(v2CurveTrack()==='chord'?1:0,drag.baseValues[i]+value-drag.baseValues[drag.index]):value;v2CurveSetPoint(points[i],adjusted,v2CurveTrack());}v2CurveChanged();
    }
    if(phase==='up'){
      if(drag.kind==='paint'&&!drag.moved){const points=v2Curve.profile.points,index=points.reduce((best,p,i)=>Math.abs(p.time_sec-time)<Math.abs(points[best].time_sec-time)?i:best,0);v2Curve.selection=new Set([index]);v2Curve.selected=index;v2CurveEdit(index,value);}
      if(drag.kind==='box'&&drag.moved){
        v2Curve.selection=new Set(drag.additive?drag.before:[]);
        v2Curve.profile.points.forEach((p,i)=>{const px=v2CurveX(p.time_sec,box.width,view),py=v2CurveY(v2CurveValue(p),box.height,maximum);if(px>=Math.min(drag.x,x)&&px<=Math.max(drag.x,x)&&py>=Math.min(drag.y,y)&&py<=Math.max(drag.y,y))v2Curve.selection.add(i);});
        if(v2Curve.selection.size)v2Curve.selected=[...v2Curve.selection][0];
      }else if(drag.kind==='box'&&!drag.moved){v2Curve.selection.clear();v2CurveSeek(time);}
      v2Curve.drag=null;canvas.releasePointerCapture(event.pointerId);
    }
  }v2CurveDraw();
}
function v2CurvePlay(){const audio=element('#v2CurveAudio');if(!audio.src)return v2CurveLoadMedia();return audio.play().catch(e=>{element('#v2CurveMediaStatus').textContent=e.message;});}
element('#v2CurvePredict')?.addEventListener('click',v2CurvePredict);
element('#v2CurveLoadAudio')?.addEventListener('click',v2CurveLoadMedia);
element('#v2CurveTrack')?.addEventListener('change',v2CurveDraw);
element('#v2CurvePointValue')?.addEventListener('change',e=>{if(!v2Curve.profile)return;if(!e.target.value.trim()){v2CurveDraw();return;}const n=Number(e.target.value);if(v2CurveSelection().size>1)v2CurveBatch('set',n);else v2CurveEdit(v2Curve.selected,n);});
element('#v2CurvePointTimeInput')?.addEventListener('change',e=>{
  const points=v2Curve.profile?.points,index=v2Curve.selected,time=Number(e.target.value);if(!points)return;if(!e.target.value.trim()){v2CurveDraw();return;}
  if(!Number.isFinite(time)||time<0||time>v2CurveDuration()||(index>0&&time<=points[index-1].time_sec)||(index+1<points.length&&time>=points[index+1].time_sec)){v2CurveStatus('时间须位于相邻控制点之间。',true);v2CurveDraw();return;}
  v2CurveRemember();points[index].time_sec=time;v2CurveChanged();
});
const v2CurveCanvasElement=element('#v2CurveCanvas');
for(const [event,phase] of [['pointerdown','down'],['pointermove','move'],['pointerup','up']])v2CurveCanvasElement?.addEventListener(event,e=>{if(phase==='down'||v2Curve.drag)v2CurvePointer(e,phase);});
v2CurveCanvasElement?.addEventListener('pointercancel',()=>{v2Curve.drag=null;v2CurveDraw();});
v2CurveCanvasElement?.addEventListener('dblclick',e=>{const box=e.currentTarget.getBoundingClientRect(),v=v2CurveView();v2CurveAddPoint(v.start+(e.clientX-box.left-48)/(box.width-62)*v.span);});
for(const id of ['#v2CurveCanvas','#v2CurveWave','#v2CurveSpec'])element(id)?.addEventListener('wheel',e=>{
  if(!e.ctrlKey&&!e.shiftKey)return;e.preventDefault();const box=e.currentTarget.getBoundingClientRect(),v=v2CurveView();
  if(e.ctrlKey)v2CurveZoom(e.deltaY<0?.7:1/.7,v.start+(e.clientX-box.left-48)/(box.width-62)*v.span);else v2CurveSetView(v.start+Math.sign(e.deltaY)*v.span*.15,v.span);
},{passive:false});
for(const id of ['#v2CurveWave','#v2CurveSpec'])element(id)?.addEventListener('pointerdown',e=>{const box=e.currentTarget.getBoundingClientRect(),v=v2CurveView();v2CurveSeek(v.start+(e.clientX-box.left-48)/(box.width-62)*v.span);});
element('#v2CurveOverview')?.addEventListener('pointerdown',e=>{const box=e.currentTarget.getBoundingClientRect(),v=v2CurveView(),time=(e.clientX-box.left-48)/(box.width-62)*v.duration;v2CurveSetView(time-v.span/2,v.span);v2CurveSeek(time);});
element('#v2CurveZoomIn')?.addEventListener('click',()=>v2CurveZoom(.5));
element('#v2CurveZoomOut')?.addEventListener('click',()=>v2CurveZoom(2));
element('#v2CurveFit')?.addEventListener('click',()=>v2CurveSetView(0,v2CurveDuration()));
element('#v2CurveScroll')?.addEventListener('input',e=>v2CurveSetView(Number(e.target.value),v2CurveView().span));
for(const id of ['#v2CurveViewStart','#v2CurveViewSpan'])element(id)?.addEventListener('change',event=>{const start=Number(element('#v2CurveViewStart').value),span=Number(element('#v2CurveViewSpan').value);if(Number.isFinite(start)&&Number.isFinite(span)&&start>=0&&start<v2CurveDuration()&&span>0){v2CurveSetView(start,Math.min(span,v2CurveDuration()-start),false);event.target.value=(id==='#v2CurveViewStart'?v2Curve.start:v2CurveView().span).toFixed(2);}});
element('#v2CurveSpectrum')?.addEventListener('change',e=>{element('#v2CurveSpec').hidden=!e.target.checked;v2CurveDraw();});
element('#v2CurvePlayView')?.addEventListener('click',()=>{v2CurveSeek(v2CurveView().start);v2CurvePlay();});
element('#v2CurvePlayPoint')?.addEventListener('click',()=>{v2CurveSeek(Math.max(0,(v2Curve.profile?.points[v2Curve.selected]?.time_sec||v2Curve.cursor)-1));v2CurvePlay();});
element('#v2CurveAdd')?.addEventListener('click',()=>v2CurveAddPoint());
element('#v2CurveSelectView')?.addEventListener('click',()=>{const v=v2CurveView();v2Curve.selection=new Set((v2Curve.profile?.points||[]).flatMap((p,i)=>p.time_sec>=v.start&&p.time_sec<=v.end?[i]:[]));v2CurveDraw();});
element('#v2CurveClearSelection')?.addEventListener('click',()=>{v2Curve.selection.clear();v2CurveDraw();});
element('#v2CurveBatchApply')?.addEventListener('click',()=>{const raw=valueOrDefault('#v2CurveBatchAmount','').trim();if(!raw)return v2CurveStatus('请填写批量调整数值。',true);v2CurveBatch(valueOrDefault('#v2CurveBatchMode','scale'),Number(raw));});
element('#v2CurveDelete')?.addEventListener('click',()=>{const points=v2Curve.profile?.points;if(!points||!v2Curve.selection.size)return;if(v2Curve.selection.size>=points.length)return v2CurveStatus('请至少保留一个控制点。',true);v2CurveRemember();v2Curve.profile.points=points.filter((p,i)=>!v2Curve.selection.has(i));v2Curve.selected=0;v2Curve.selection=new Set([0]);v2CurveChanged();});
element('#v2CurveUndo')?.addEventListener('click',()=>{if(v2Curve.undo.length){v2Curve.profile=v2Curve.undo.pop();v2Curve.selected=Math.min(v2Curve.selected,v2Curve.profile.points.length-1);v2Curve.selection=new Set([v2Curve.selected]);v2CurveDraw();}});
element('#v2CurveReset')?.addEventListener('click',()=>{if(v2Curve.automatic){v2CurveRemember();v2Curve.profile=structuredClone(v2Curve.automatic);v2Curve.timingStale=!v2CurveTimingMatches();v2Curve.selected=0;v2Curve.selection=new Set([0]);element('#v2CurveUse').checked=false;v2CurveDraw();if(v2Curve.timingStale)v2CurveStatus('已恢复原自动曲线，但节拍／拍网设置已变化；直接生成将按新拍网自动预测，如需编辑请重新预测曲线。',true);}});
element('#v2CurveExport')?.addEventListener('click',()=>{if(!v2Curve.profile)return;const link=document.createElement('a'),url=URL.createObjectURL(new Blob([JSON.stringify(v2Curve.profile,null,2)],{type:'application/json'}));link.href=url;link.download='composer_v2_intensity.json';link.click();URL.revokeObjectURL(url);});
element('#v2CurveImport')?.addEventListener('change',async e=>{
  try{if(!e.target.files[0])return;const p=JSON.parse(await e.target.files[0].text());
    if(p.schema!=='composer-v2-intensity-v1'||!Number.isFinite(p.duration_sec)||p.duration_sec<=0||!Array.isArray(p.points)||!p.points.length||p.points.length>2048)throw new Error('请选择有效的新版 Composer v2 曲线JSON。');
    let last=-1;for(const q of p.points){if(![q.time_sec,q.event_rate,q.object_rate].every(Number.isFinite)||q.time_sec<=last||q.time_sec>p.duration_sec||q.time_sec<0||q.event_rate<0||q.object_rate<q.event_rate||(q.event_rate===0&&q.object_rate>0))throw new Error('控制点的时间或物量无效。');last=q.time_sec;}
    p.edited=true;v2Curve.profile=p;v2Curve.automatic=null;v2Curve.source=v2CurveSource();v2Curve.timingStale=false;v2Curve.undo=[];v2Curve.selected=0;v2Curve.selection=new Set([0]);v2Curve.start=0;v2Curve.end=null;element('#v2CurveUse').checked=true;v2CurveDraw();v2CurveStatus('已导入，可放大局部、框选或添加控制点。');v2CurveLoadMedia();
  }catch(error){v2CurveStatus(error.message,true);}
});
const v2CurveAudio=element('#v2CurveAudio');
let v2CurveLastPlaybackDraw=0;
function v2CurvePlaybackFrame(stamp=0){
  if(typeof stamp==='number'&&stamp-v2CurveLastPlaybackDraw<33){if(!v2CurveAudio.paused)requestAnimationFrame(v2CurvePlaybackFrame);return;}v2CurveLastPlaybackDraw=typeof stamp==='number'?stamp:0;
  v2Curve.cursor=v2CurveAudio.currentTime;const v=v2CurveView();
  if(element('#v2CurveLoop').checked&&v2Curve.cursor>=v.end){v2CurveAudio.currentTime=v.start;v2Curve.cursor=v.start;}
  else if(element('#v2CurveFollow').checked&&v2Curve.cursor>v.end&&v.span<v.duration)v2CurveSetView(v2Curve.cursor-v.span*.2,v.span);
  v2CurveDraw();if(!v2CurveAudio.paused)requestAnimationFrame(v2CurvePlaybackFrame);
}
v2CurveAudio?.addEventListener('play',v2CurvePlaybackFrame);
v2CurveAudio?.addEventListener('seeked',()=>{v2Curve.cursor=v2CurveAudio.currentTime;v2CurveDraw();});
element('#v2CurvePlaybackRate')?.addEventListener('change',e=>{v2CurveAudio.playbackRate=Number(e.target.value);});
let v2CurveResizeFrame=0;window.addEventListener('resize',()=>{cancelAnimationFrame(v2CurveResizeFrame);v2CurveResizeFrame=requestAnimationFrame(v2CurveDraw);});
function v2CurveAudioChanged(){if((v2Curve.mediaAudio||v2Curve.mediaBusy)&&v2CurveSource().audio!==(v2Curve.mediaAudio||v2Curve.mediaBusy)){v2CurveAudio.pause();v2CurveAudio.removeAttribute('src');v2CurveAudio.load();v2Curve.media=null;v2Curve.mediaAudio=null;v2Curve.raster=null;v2Curve.mediaSerial++;v2Curve.mediaBusy=false;element('#v2CurveLoadAudio').disabled=false;element('#v2CurveMediaStatus').textContent='音频已切换，请重新加载定位；原曲线需确认是否匹配。';v2CurveDraw();}}
element('#audioPath')?.addEventListener('change',v2CurveAudioChanged);
element('#audioPath')?.addEventListener('input',v2CurveAudioChanged);
v2CurveDraw();

element('#v2IntensityControls')?.addEventListener('toggle',()=>{if(element('#v2IntensityControls').open)v2CurveDraw();});
