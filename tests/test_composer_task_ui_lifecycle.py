"""Exercise the main page's real lifecycle functions with stubbed transport/DOM."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest


def test_reload_multi_task_busy_and_retry_cancel(tmp_path):
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is needed for the main-page lifecycle test')
    source = (Path(__file__).parents[1] / 'src/maichart/webui/static/app.js').read_text(encoding='utf-8')
    names = ['updateDatasetButtons', 'trackComposerTrainingTask', 'watchTask', 'pollTask',
             'hasTaskArtifacts', 'renderTask', 'loadStatus']
    functions = []
    for name in names:
        functions.append(re.search(r'(?ms)^(?:async )?function ' + name
                                  + r'\(.*?(?=^(?:async )?function |\Z)', source).group())
    cancel = source[source.index('on("#cancelButton", "click", async () => {'):]
    cancel = cancel[:cancel.index('\n});') + len('\n});')]
    script = r'''
const assert = require('node:assert/strict');
let activeTaskId=null, datasetUploading=false, preprocessStarting=false, trainingStarting=false,
    trainingPreflight=false;
const liveComposerTrainingTasks=new Set(), pendingTaskCancellations=new Set(), taskWatchers=new Map();
const datasetWorkspaceState={applying:false}, TASK_KIND_LABEL={};
const buttons=new Map(['#trainButton','#cancelButton','#taskMessage'].map(k=>[k,{}]));
const element=k=>buttons.get(k), window={}, currentTrainingMode=()=> 'fresh',
 selectedTrainingManifests=()=>[], trainingEpochStatus=()=>null,
 hydrateTrainingMetricHistory=()=>{},renderTaskTrainingMetrics=()=>{},renderTaskResults=()=>{},
 setStatus=()=>{},populateDatasets=()=>{},loadDatasetCatalog=async()=>{};
let cancelHandler, pendingReply, calls=0;
const on=(selector,event,fn)=>{cancelHandler=fn;};
const timers=[]; const setTimeout=fn=>timers.push(fn);
const task=(id,status)=>({task_id:id,kind:'composer_train',status,result:{},logs:[]});
const tasks=new Map([['a',task('a','cancelling')],['b',task('b','running')]]);
const api=async(url,body)=>{
 if(url==='/api/status') return {tasks:[...tasks.values()]};
 if(url==='/api/cancel') {calls++;return await new Promise(resolve=>pendingReply=resolve);}
 return tasks.get(new URL(url,'http://stub').searchParams.get('id'));
};
'''
    script += '\n'.join(functions) + '\n' + cancel
    script += r'''
(async()=>{
 await loadStatus();
 assert.equal(buttons.get('#trainButton').disabled,true);
 assert.equal(taskWatchers.size,2);
 assert.match(buttons.get('#trainButton').title,/Composer/);
 tasks.set('a',task('a','cancelled')); await pollTask('a');
 assert.equal(buttons.get('#trainButton').disabled,true,'finishing a must not unlock b');
 tasks.set('b',task('b','completed')); await pollTask('b');
 assert.equal(buttons.get('#trainButton').disabled,false);
 activeTaskId='retry';
 const retry={...task('retry','cancelling'),cancel_error:'mock failure',cancel_retryable:true};
 renderTask(retry);
 assert.equal(buttons.get('#cancelButton').disabled,false);
 assert.equal(buttons.get('#cancelButton').textContent,'重试取消');
 assert.match(buttons.get('#taskMessage').textContent,/尚未确认停止/);
 const first=cancelHandler(); const second=cancelHandler();
 assert.equal(calls,1); assert.equal(buttons.get('#cancelButton').disabled,true);
 renderTask(retry); assert.equal(buttons.get('#cancelButton').disabled,true);
 pendingReply(retry); await Promise.all([first,second]);
 assert.equal(buttons.get('#cancelButton').disabled,false);
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
    path = tmp_path / 'lifecycle.cjs'
    path.write_text(script, encoding='utf-8')
    result = subprocess.run([node, str(path)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
