// Run with node: verify persisted and older log-only fixed-validation records.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../src/maichart/webui/static/app.js'), 'utf8');
function extract(start, end) { return source.slice(source.indexOf(`function ${start}(`), source.indexOf(`function ${end}(`)); }
const definitions = extract('trainingMetricNumber', 'formatTrainingMetricCount') +
  extract('composerV2TrainingView', 'renderComposerV2TrainingMetrics');
const rows = [500, 1000].map((step, index) => ({stage: 'checkpoint_validation', optimizer_steps: step,
  epoch: 1, batch: step*4, total: 10000, validation_loss: .8+index*.1,
  validation_nonphase_time_loss: .5+index*.2, validation_phase_loss: 1.5-index*.1, validation_sec: 20}));
const training = {stage: 'training', epoch: 1, batch: 4000, total: 10000, optimizer_steps: 1000,
  loss: .8, epoch_processed_batches: 4000, epoch_elapsed_sec: 2000};
function view(task) { return vm.runInNewContext(definitions+'\ncomposerV2TrainingView(task)', {task}); }
function card(result, name) { return result.cards.flatMap(group=>group[1]).find(row=>row[0]===name)?.[1]; }
const persisted = view({status:'running', result:{epoch_target:20, composer_v2_training_metrics:training,
  composer_v2_checkpoint_validation:rows[1], composer_v2_validation_history:rows,
  composer_v2_stage:'training'}});
assert.equal(card(persisted, '最近固定验证步'), '1000');
assert.equal(card(persisted, '固定片段非相位时序 loss'), '0.7000');
assert.equal(persisted.validationHistory.length, 2);
const older = view({status:'running', result:{epoch_target:20}, logs:[training,...rows,rows[1]]
  .map(row=>'[12:00:00] composer_v2_progress '+JSON.stringify(row))});
assert.equal(card(older, '固定片段验证 loss'), '0.9000');
assert.equal(older.validationHistory.length, 2);
const during = view({status:'running', result:{epoch_target:20, composer_v2_training_metrics:training,
  composer_v2_validation_history:rows, composer_v2_stage:'validation',
  composer_v2_validation:{scope:'checkpoint', started_unix:Date.now()/1000}}});
assert.equal(card(during, '本轮预计剩余'), '约 51 分钟');
const empty = view({status:'queued',result:{}});
assert.equal(card(empty, '固定片段非相位时序 loss'), '未记录');
assert.equal(empty.validationHistory.length, 0);
console.log('PASS: persisted/log-only validation, de-duplication, missing metrics and mid-epoch validation ETA.');
