const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync('src/maichart/webui/static/app.js','utf8');
function fn(name){const start=source.indexOf(`function ${name}(`);return source.slice(start,source.indexOf('\n}',start)+2);}
test('training payload submits evaluation zeros, advanced values and fixed seeds',()=>{
  const values={'#composerHoldoutFraction':'0','#composerValidationMaxBatches':'0','#composerValidationBatchSize':'4',
    '#composerGenerationEvalSeeds':'0,17','#composerArchiveCheckpoints':'off','#composerDModel':'512',
    '#composerValidationManifest':' E:/heldout.json '};
  const c=vm.createContext({currentTrainingMode:()=> 'fresh',trainingSourcePaths:()=>({manifest:'train.json'}),
    selectedTrainingManifests:()=>[],selectedDatasetItems:()=>[],valueOrDefault:(id,f)=>values[id]??f,
    checkedOrDefault:(_id,f)=>f,formDevice:()=> 'cpu'});
  vm.runInContext(fn('chartComposerTrainingPayload'),c);
  const p=c.chartComposerTrainingPayload();
  assert.equal(p.validation_manifest,'E:/heldout.json');assert.equal(p.holdout_fraction,0);
  assert.equal(p.validation_max_batches,0);assert.equal(p.validation_batch_size,4);
  assert.equal(p.generation_eval_seeds,'0,17');assert.equal(p.archive_checkpoints,'off');assert.equal(p.d_model,512);
});
