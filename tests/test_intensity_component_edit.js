"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const test = require("node:test");
const source = fs.readFileSync(path.join(__dirname, "../src/maichart/webui/static/app.js"), "utf8");
function fn(name) {
  const begin = source.indexOf(`function ${name}(`);
  assert.ok(begin >= 0, name);
  return source.slice(begin, source.indexOf("\n}", begin) + 2);
}
function setup() {
  const profile = {schema:"maichart-intensity-curve-v1",method:"learned_song_form_v1",duration_sec:10,
    audio_sha256:"a".repeat(64),first:0,tempo_map_sha256:null,meter_map:[],
    checkpoint_identity:{sha256:"b".repeat(64)},
    points:[{time_sec:0,intensity:.4},{time_sec:10,intensity:.6}],
    components:{event_density:[.2,.8],tap:[.9,.5],sustain:[.1,.3],slide_touch:[.1,.2]}};
  const fields = {"#intensitySectionStart":"3","#intensitySectionEnd":"7","#intensitySectionValue":".8","#intensitySectionBlend":"1"};
  const context = vm.createContext({intensityCurveState:{profile,history:[],saved:true},intensityView:{selected:new Set()},
    valueOrDefault:(id,fallback)=>fields[id] ?? fallback,
    pushIntensityHistory(){context.intensityCurveState.history.push(JSON.parse(JSON.stringify(context.intensityCurveState.profile)));},
    saveIntensityCurveState(){},drawIntensityCurve(){},updateIntensityButtons(){}});
  vm.runInContext(["resampleIntensityComponents","normalizeIntensityProfile","normalizeRestRegions",
    "intensityProfileValid","applyIntensitySection"].map(fn).join("\n"),context);
  return {context,profile};
}

if (process.argv.includes("--emit-profile")) {
  const {context,profile} = setup();
  context.applyIntensitySection();
  process.stdout.write(JSON.stringify(profile));
} else {
  test("section editing interpolates baseline channels at every inserted point without changing their meaning",()=>{
    const {context,profile} = setup();
    context.applyIntensitySection();
    assert.deepEqual(Array.from(profile.points,p=>p.time_sec),[0,2,3,7,8,10]);
    for (const values of Object.values(profile.components)) assert.equal(values.length,profile.points.length);
    assert.ok(Math.abs(profile.components.event_density[2]-.38)<1e-12);
    assert.equal(profile.points[2].intensity,.8);
    assert.equal(profile.components.tap[0],.9);
    assert.equal(profile.tempo_map_sha256,null);
    assert.deepEqual(profile.meter_map,[]);
    assert.equal(context.intensityProfileValid(profile),true);
    assert.equal(context.intensityCurveState.history[0].points.length,2);
  });
  test("normalization sorts and deduplicates every component with the matching control point",()=>{
    const {context,profile} = setup();
    profile.points = [{time_sec:10,intensity:.9},{time_sec:0,intensity:.1},{time_sec:0,intensity:.2}];
    profile.components = {tap:[.8,.1,.3]};
    const normalized = context.normalizeIntensityProfile(profile);
    assert.deepEqual(Array.from(normalized.points,p=>p.time_sec),[0,10]);
    assert.deepEqual(Array.from(normalized.components.tap),[.3,.8]);
    assert.equal(normalized.points[0].intensity,.2);
    assert.equal(context.intensityProfileValid(normalized),true);
  });
  test("malformed channels cannot be silently saved and editing is transactional",()=>{
    const {context,profile} = setup();
    profile.components.tap = [.5];
    const before = JSON.stringify(profile);
    assert.equal(context.intensityProfileValid(profile),false);
    assert.throws(()=>context.applyIntensitySection(),/基础通道/);
    assert.equal(JSON.stringify(profile),before);
    assert.equal(context.intensityCurveState.history.length,0);
    assert.equal(context.intensityCurveState.saved,true);
  });
}
