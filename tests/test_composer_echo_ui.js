const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync('src/maichart/webui/static/app.js','utf8');
const html=fs.readFileSync('src/maichart/webui/static/index.html','utf8');
function fn(name){const start=source.indexOf(`function ${name}(`);return source.slice(start,source.indexOf('\n}',start)+2);}

test('main training form forwards the chosen echo policy',()=>{
  const c=vm.createContext({currentTrainingMode:()=> 'finetune',trainingSourcePaths:()=>({manifest:'train.json'}),
    selectedTrainingManifests:()=>[],selectedDatasetItems:()=>[],
    valueOrDefault:(id,f)=>id==='#composerEchoPriorPolicy'?'phase_echo_v1':f,
    checkedOrDefault:(_id,f)=>f,formDevice:()=> 'cpu'});
  vm.runInContext(fn('chartComposerTrainingPayload'),c);
  assert.equal(c.chartComposerTrainingPayload().echo_prior_policy,'phase_echo_v1');
  assert.match(html,/id="composerEchoPriorPolicy"/);
});

test('echo inherits only on resume and remains editable for finetune',()=>{
  const node={value:'auto',disabled:false};
  const c=vm.createContext({element:id=>id==='#composerEchoPriorPolicy'?node:null,
    updateComposerRecoverySamplingControls:()=>{},updateComposerContextControls:()=>{}});
  vm.runInContext(fn('applyChartComposerCheckpointConfig'),c);
  const saved={valid:true,model_config:{architecture_profile:'audio_grounded_v3'},args:{}};
  c.applyChartComposerCheckpointConfig(saved,'resume');
  assert.equal(node.value,'off');assert.equal(node.disabled,true);
  node.value='auto';
  c.applyChartComposerCheckpointConfig(saved,'finetune');
  assert.equal(node.value,'auto');assert.equal(node.disabled,false);
});

test('checkpoint details distinguish old off checkpoints from echo checkpoints',()=>{
  const box={};
  const c=vm.createContext({element:()=>box,setCheckpointStatus:()=>{},currentTrainingMode:()=> 'finetune'});
  vm.runInContext(fn('renderChartComposerCheckpointInspection'),c);
  c.renderChartComposerCheckpointInspection({valid:true,model_config:{}});
  assert.match(box.textContent,/相位呼应先验：off/);
  c.renderChartComposerCheckpointInspection({valid:true,model_config:{echo_prior_policy:'phase_echo_v1'}});
  assert.match(box.textContent,/相位呼应先验：phase_echo_v1/);
});
