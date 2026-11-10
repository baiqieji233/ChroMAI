"use strict";
// One exchange flow for confirmed source spans and generated examples.
(() => {
  const $=id=>document.getElementById(id);
  if(!$('reviewBundlePanel'))return;
  let busy=false,preview=null,imported=null;
  const main={status:$('reviewBundleStatus'),download:$('reviewBundleDownload')};
  const tell=(host,text)=>{host.status.textContent=text;};
  function controls(){for(const id of ['reviewBundleExport','reviewBundleChoose','reviewBundleApply'])$(id).disabled=busy;}
  function progress(task,host){
    const value=task.result?.review_bundle_progress;
    if(!value)return;
    const labels={collecting:'整理已确认片段',packing:'打包谱面与音频',verifying:'核对片段包',importing:'保存导入示范'};
    tell(host,`${labels[value.stage]||'正在处理'}${value.total?` ${value.completed||0}/${value.total}`:''}…`);
  }
  async function job(operation,payload,host){
    const capabilities=globalThis.maichartLastStatus?.training_capabilities||(await api('/api/status')).training_capabilities;
    if(capabilities?.review_bundles!==true)throw new Error('当前后台尚未载入片段包功能，请使用已更新的 WebUI 实例；当前训练无需停止。');
    const started=await api('/api/review-bundles/'+operation,payload);
    return await new Promise((resolve,reject)=>watchTask(started.task_id,task=>{
      if(task.status!=='completed')reject(new Error(task.error||'操作已取消'));
      else if(!task.result?.review_bundle)reject(new Error('未收到处理结果，请查看任务日志'));
      else resolve({...task.result.review_bundle,preview_id:started.preview_id});
    },task=>progress(task,host)));
  }
  async function run(action,host=main){
    if(busy){tell(host,'正在处理片段包，请等待当前操作完成。');return;}
    busy=true;controls();
    try{await action();}
    catch(error){tell(host,/Not found|404/i.test(error.message)?'当前后台尚未载入片段包功能，请使用已更新的 WebUI 实例。':error.message);}
    finally{busy=false;controls();}
  }
  async function exportPackage(payload,host=main){
    await run(async()=>{
      if(!payload.candidates&&!payload.manifests?.length)throw new Error('请先勾选要导出的训练来源。');
      host.download.hidden=true;tell(host,'正在核对已确认片段，后台打包中…');
      const result=await job('export',payload,host);
      host.download.href='/api/download?path='+encodeURIComponent(result.path);
      host.download.download='已确认片段.zip';host.download.hidden=false;
      tell(host,`已打包 ${result.charts} 张谱面的 ${result.regions} 段。点击“下载片段包”保存。`);
    },host);
  }
  globalThis.ReviewExchange={export:exportPackage};
  $('reviewBundleExport').addEventListener('click',()=>void exportPackage({manifests:selectedDatasetItems().map(item=>item.manifest).filter(Boolean)}));
  $('reviewBundleChoose').addEventListener('click',()=>$('reviewBundleFile').click());
  function uploadPackage(file){return new Promise((resolve,reject)=>{
    const xhr=new XMLHttpRequest(),form=new FormData();form.append('kind','review_bundle');form.append('file',file,file.name);
    xhr.open('POST','/api/upload');
    xhr.upload.onprogress=event=>tell(main,event.lengthComputable?`正在上传片段包 ${Math.floor(event.loaded/event.total*100)}%…`:'正在上传片段包…');
    xhr.onerror=()=>reject(new Error('上传连接中断，请重试'));
    xhr.onabort=()=>reject(new Error('上传已取消'));
    xhr.onload=()=>{let result;try{result=JSON.parse(xhr.responseText);}catch(_){reject(new Error('上传未返回有效结果'));return;}
      if(xhr.status<200||xhr.status>=300||result.error)reject(new Error(result.error||'上传失败'));else resolve(result);};
    xhr.send(form);
  });}
  $('reviewBundleFile').addEventListener('change',event=>{
    const file=event.target.files?.[0];event.target.value='';if(!file)return;
    void run(async()=>{
      const capabilities=globalThis.maichartLastStatus?.training_capabilities||(await api('/api/status')).training_capabilities;
      if(capabilities?.review_bundles!==true)throw new Error('当前后台尚未载入片段包功能，请使用已更新的 WebUI 实例；当前训练无需停止。');
      preview=null;imported=null;$('reviewBundlePreview').hidden=true;$('reviewBundleImported').hidden=true;
      tell(main,'正在上传片段包…');const uploaded=await uploadPackage(file);
      tell(main,'正在核对谱面、音频与学习范围…');preview=await job('preview',{path:uploaded.path},main);
      $('reviewBundleCounts').textContent=`${preview.charts} 张谱面 · ${preview.regions} 段。确认后保存到示范库，不会开始训练。`;
      const list=$('reviewBundleItems');list.replaceChildren();
      for(const item of preview.items||[]){const li=document.createElement('li'),details=document.createElement('details'),summary=document.createElement('summary');
        summary.textContent=`${item.title} · ${item.level} · ${item.regions.length} 段`;details.append(summary);
        for(const region of item.regions){const row=document.createElement('p');row.textContent=`${region.start_sec.toFixed(2)}–${region.end_sec.toFixed(2)} 秒 · ${(region.timing_weight??1)>0?(region.component_weight??1)>0?'采音和配置':'只学采音':'只学配置'}`;details.append(row);}
        li.append(details);list.append(li);}
      $('reviewBundlePreview').hidden=false;tell(main,'核对完成，请检查清单，再点“确认导入示范库”。');
    });
  });
  $('reviewBundleApply').addEventListener('click',()=>void run(async()=>{
    if(!preview)throw new Error('请先选择并核对片段包');
    tell(main,'正在保存示范，保留完整前后文…');
    imported=await job('import',{preview_id:preview.preview_id},main);
    const registered=await api('/api/datasets/register',{path:imported.manifest,name:imported.name,review_group:imported.review_group});
    await loadDatasetCatalog();
    imported.item=registered.dataset;$('reviewBundlePreview').hidden=true;$('reviewBundleImported').hidden=false;
    tell(main,`已导入 ${imported.imported_charts} 张谱面的 ${imported.imported_regions} 段。已登记示范库，尚未应用训练来源。`);
  }));
  $('reviewBundleOpen').addEventListener('click',()=>void run(async()=>{
    if(!imported?.item)throw new Error('请先导入片段包');
    await openDataset(imported.item);
  }));
  $('reviewBundleAddTraining').addEventListener('click',()=>void run(async()=>{
    if(!imported)throw new Error('请先导入片段包');
    await addReviewedTrainingSource(imported);
    tell(main,'示范库已勾选。检查其他来源后，点“应用勾选的数据集”，再用新目录 finetune。当前训练继续运行。');
  }));
})();
