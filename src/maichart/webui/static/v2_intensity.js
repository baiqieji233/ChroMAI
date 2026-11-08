"use strict";
// A v2 curve is a decoding condition learned during training. The old v1
// post-hoc density controller and its profile cache are deliberately separate.
const v2Curve = {profile:null, automatic:null, undo:[], source:null, selected:0, busy:false};
function v2CurveSource() {
  return {audio:valueOrDefault('#audioPath','').trim(),checkpoint:valueOrDefault('#checkpointPath','').trim()};
}
function v2CurveMatches() {
  const source=v2CurveSource();return v2Curve.source?.audio===source.audio && v2Curve.source?.checkpoint===source.checkpoint;
}
function v2CurvePayload() {
  if(!element('#v2CurveUse')?.checked)return {};
  if(!v2Curve.profile || !v2CurveMatches())throw new Error('编辑曲线与当前音频／模型不匹配，请重新预测或关闭编辑曲线。');
  return {intensity_profile:v2Curve.profile,intensity_strength:Number(valueOrDefault('#v2CurveStrength',1))};
}
function v2CurveStatus(message,bad=false) {
  const status=element('#v2CurveStatus');status.textContent=message;status.classList.toggle('warn',bad);
}
function v2CurveDraw() {
  const profile=v2Curve.profile,canvas=element('#v2CurveCanvas');if(!canvas)return;
  const ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;
  ctx.clearRect(0,0,w,h);ctx.fillStyle='#101c2a';ctx.fillRect(0,0,w,h);
  const track=valueOrDefault('#v2CurveTrack','total'),points=profile?.points||[];
  const value=p=>track==='events'?p.event_rate:track==='chord'?(p.event_rate>0?p.object_rate/p.event_rate:1):p.object_rate;
  const maximum=Math.max(track==='chord'?4:8,...points.map(value))*1.15;
  canvas.dataset.maximum=String(maximum);
  ctx.strokeStyle='#314052';ctx.fillStyle='#b9c8d8';ctx.font='13px sans-serif';
  for(let i=0;i<=4;i++){const y=20+(h-45)*i/4;ctx.beginPath();ctx.moveTo(48,y);ctx.lineTo(w-12,y);ctx.stroke();ctx.fillText((maximum*(1-i/4)).toFixed(1),6,y+4);}
  if(!points.length){ctx.fillText('选择支持强度曲线的模型后，点击「预测并编辑曲线」',65,h/2);return;}
  ctx.strokeStyle='#69c4ff';ctx.lineWidth=2;ctx.beginPath();
  points.forEach((p,i)=>{const x=48+(w-60)*p.time_sec/profile.duration_sec,y=20+(h-45)*(1-value(p)/maximum);if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y);});ctx.stroke();
  const p=points[v2Curve.selected]||points[0];ctx.fillStyle='#ffe59a';ctx.beginPath();ctx.arc(48+(w-60)*p.time_sec/profile.duration_sec,20+(h-45)*(1-value(p)/maximum),4,0,2*Math.PI);ctx.fill();
  ctx.fillStyle='#b9c8d8';ctx.fillText('0秒',48,h-5);ctx.fillText(profile.duration_sec.toFixed(1)+'秒',w-90,h-5);
  element('#v2CurvePointTime').textContent=p.time_sec.toFixed(2)+' 秒';
  element('#v2CurvePointValue').value=value(p).toFixed(3);
  element('#v2CurveUnits').textContent=track==='chord'?'平均每次音符数':track==='events'?'出音次数／秒':'音符个数／秒';
}
function v2CurveEdit(index,value,remember=true) {
  if(!Number.isFinite(value)||value<0)return;
  if(remember){v2Curve.undo.push(structuredClone(v2Curve.profile));if(v2Curve.undo.length>30)v2Curve.undo.shift();}
  const p=v2Curve.profile.points[index],track=valueOrDefault('#v2CurveTrack','total');
  if(track==='chord')p.object_rate=p.event_rate*Math.max(1,value);
  else if(track==='events') {const ratio=p.event_rate>0?p.object_rate/p.event_rate:1;p.event_rate=value;p.object_rate=value*ratio;}
  else {const fraction=p.object_rate>0?p.event_rate/p.object_rate:1;p.event_rate=value*fraction;p.object_rate=value;}
  v2Curve.profile.edited=true;element('#v2CurveUse').checked=true;v2CurveDraw();
  v2CurveStatus('已编辑：生成将使用这条曲线作为学习到的编排倾向，仍保留正常休止与舍音。');
}
async function v2CurvePredict() {
  if(v2Curve.busy)return;
  try {
    const payload=composerV2GenerationPayload({omitCurve:true});
    if(!payload.checkpoint||!payload.audio)throw new Error('请先选择模型和音频。');
    const status=await api('/api/status');if(!status.training_capabilities?.composer_v2_intensity_curve)throw new Error('当前后台未加载新版曲线，请重启 WebUI 后再使用。');
    v2Curve.busy=true;element('#v2CurvePredict').disabled=true;v2CurveStatus('正在用模型预测全曲出音频率与物量…');
    const source=v2CurveSource();const result=await api('/api/composer-v2/curve',payload);
    watchTask(result.task_id,task=>{
      if(!['completed','failed','cancelled'].includes(task.status))return;
      v2Curve.busy=false;element('#v2CurvePredict').disabled=false;
      const profile=task.result?.intensity_profile;
      if(task.status!=='completed'||!profile){v2CurveStatus(task.error||'曲线预测未完成。',true);return;}
      if(JSON.stringify(source)!==JSON.stringify(v2CurveSource())){v2CurveStatus('模型或音频已切换，本次预测未应用，请重新预测。',true);return;}
      v2Curve.source=source;v2Curve.profile=structuredClone(profile);v2Curve.automatic=structuredClone(profile);v2Curve.undo=[];v2Curve.selected=0;
      element('#v2CurveUse').checked=false;v2CurveDraw();v2CurveStatus('自动曲线已就绪。直接生成也会自动预测；拖动曲线或启用编辑曲线后采用本次修改。');
    });
  } catch(error) {v2Curve.busy=false;element('#v2CurvePredict').disabled=false;v2CurveStatus(error.message,true);}
}
element('#v2CurvePredict')?.addEventListener('click',v2CurvePredict);
element('#v2CurveTrack')?.addEventListener('change',v2CurveDraw);
element('#v2CurvePointValue')?.addEventListener('change',event=>{if(v2Curve.profile)v2CurveEdit(v2Curve.selected,Number(event.target.value));});
function v2CurvePointer(event,remember) {
  if(!v2Curve.profile)return;event.preventDefault();
  const canvas=event.currentTarget,box=canvas.getBoundingClientRect(),x=(event.clientX-box.left)*canvas.width/box.width,y=(event.clientY-box.top)*canvas.height/box.height;
  const time=Math.max(0,Math.min(1,(x-48)/(canvas.width-60)))*v2Curve.profile.duration_sec;
  v2Curve.selected=v2Curve.profile.points.reduce((best,p,i,points)=>Math.abs(p.time_sec-time)<Math.abs(points[best].time_sec-time)?i:best,0);
  const value=Math.max(0,(1-(y-20)/(canvas.height-45))*Number(canvas.dataset.maximum));v2CurveEdit(v2Curve.selected,value,remember);
}
element('#v2CurveCanvas')?.addEventListener('pointerdown',event=>{if(!v2Curve.profile)return;event.currentTarget.setPointerCapture(event.pointerId);v2CurvePointer(event,true);});
element('#v2CurveCanvas')?.addEventListener('pointermove',event=>{if(event.currentTarget.hasPointerCapture(event.pointerId))v2CurvePointer(event,false);});
element('#v2CurveCanvas')?.addEventListener('pointerup',event=>{if(event.currentTarget.hasPointerCapture(event.pointerId))event.currentTarget.releasePointerCapture(event.pointerId);});
element('#v2CurveUndo')?.addEventListener('click',()=>{if(v2Curve.undo.length){v2Curve.profile=v2Curve.undo.pop();v2CurveDraw();}});
element('#v2CurveReset')?.addEventListener('click',()=>{if(v2Curve.automatic){v2Curve.profile=structuredClone(v2Curve.automatic);v2Curve.undo=[];element('#v2CurveUse').checked=false;v2CurveDraw();}});
element('#v2CurveExport')?.addEventListener('click',()=>{if(!v2Curve.profile)return;const link=document.createElement('a'),url=URL.createObjectURL(new Blob([JSON.stringify(v2Curve.profile,null,2)],{type:'application/json'}));link.href=url;link.download='composer_v2_intensity.json';link.click();URL.revokeObjectURL(url);});
element('#v2CurveImport')?.addEventListener('change',async event=>{
  try {const profile=JSON.parse(await event.target.files[0].text());
    if(profile.schema!=='composer-v2-intensity-v1'||!Array.isArray(profile.points)||!profile.points.length)throw new Error('请选择新版 Composer v2 曲线 JSON。');
    v2Curve.profile=profile;v2Curve.automatic=null;v2Curve.source=v2CurveSource();v2Curve.undo=[];v2Curve.selected=0;element('#v2CurveUse').checked=true;v2CurveDraw();v2CurveStatus('已导入；生成时会核对音频身份与曲线数值。');
  } catch(error){v2CurveStatus(error.message,true);}
});
v2CurveDraw();
