"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const test = require("node:test");
const source = fs.readFileSync(path.join(__dirname,"../src/maichart/webui/static/app.js"),"utf8");
const begin=source.indexOf("const trainingMetricSupportCache = new Map();");
const end=source.indexOf("\nfunction watchTask(taskId",begin);

function harness(fetch) {
  const renders=[];
  const context={fetch,Date,encodeURIComponent,Promise,taskWatchers:new Map(),activeTaskId:"a",
    renderTaskTrainingMetrics:task=>renders.push(task),renderTask:task=>renders.push(task)};
  vm.createContext(context);
  vm.runInContext(source.slice(begin,end)+"\nglobalThis.api={trainingSupportPaths,hydrateTrainingMetricHistory,parseArchivedFormTrainingLog,viewArchivedTrainingLog};",context);
  return {...context,context,renders,api:context.api};
}

test("support paths recover the checkpoint directory from existing task artifacts",()=>{
  const {api}=harness();
  const paths=api.trainingSupportPaths({kind:"form_train",result:{artifacts:[{path:"E:\\models\\run\\form_last.pt"}]}});
  assert.equal(paths.history,"E:\\models\\run/training_history.json");
  assert.equal(paths.model,"E:\\models\\run/training_model_summary.json");
  assert.equal(api.trainingSupportPaths({kind:"form_generate",result:{}}),null);
});

test("archive parsing retains the current partial phase without claiming it is running",()=>{
  const {api}=harness();
  const text=["# task_id=a","# argv=-m maichart.form.train --checkpoint-dir E:\\models\\full set --epochs 20 --batch-size 2",
    '[11:00:00] form_epoch {"epoch":11,"draft_note_configuration_loss":0.59}',
    '[11:41:25] form_progress {"epoch":12,"phase":"layout_revision","completed":16000,"total":65000,"losses":{"draft_note_configuration_loss":0.57}}'].join("\n");
  const task=api.parseArchivedFormTrainingLog(text,{kind:"form_train",task_id:"a",status:"running",log_path:"log"});
  assert.equal(task.status,"archived");
  assert.equal(task.result.metrics.epoch,11);
  assert.equal(task.result.progress_detail.epoch,12);
  assert.equal(task.result.checkpoint_dir,"E:\\models\\full set");
  assert.match(task.message,/原记录状态：running/);
});

test("history loading supplements live metrics and is throttled across repeated polls",async()=>{
  const calls=[];
  const {api,renders}=harness(async url=>{
    calls.push(url);
    const value=url.includes("training_model_summary")
      ? {schema:"song_form_training_model_summary_v1",parameter_count:48192070}
      : [{epoch:10},{epoch:11}];
    return {ok:true,text:async()=>JSON.stringify(value)};
  });
  const task={task_id:"a",kind:"form_train",result:{checkpoint_dir:"E:/models",progress_detail:{epoch:12}}};
  api.hydrateTrainingMetricHistory(task);
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(calls.length,2);
  assert.equal(task.result.progress_detail.epoch,12);
  assert.equal(task.result.metric_history.length,2);
  assert.equal(task.result.model_summary.parameter_count,48192070);
  assert.equal(renders.length,1);
  const next={...task,result:{checkpoint_dir:"E:/models",progress_detail:{epoch:12}}};
  api.hydrateTrainingMetricHistory(next);
  assert.equal(calls.length,2);
  assert.equal(next.result.metric_history.length,2);
});

test("missing support files leave existing task data usable",async()=>{
  const {api}=harness(async()=>({ok:false,status:404}));
  const task={task_id:"a",kind:"form_train",result:{checkpoint_dir:"E:/models",metrics:{epoch:11}}};
  api.hydrateTrainingMetricHistory(task);
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(task.result.metrics.epoch,11);
  assert.equal(task.result.model_summary,undefined);
});
