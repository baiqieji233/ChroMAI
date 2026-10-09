const { chromium } = require('playwright-core');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const projectRoot = path.resolve(__dirname, '../..');
process.chdir(projectRoot);
fs.mkdirSync('.scratch', {recursive: true});
(async () => {
  const browser = await chromium.launch({headless: true, channel: 'msedge'});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1050}});
    const errors = []; page.on('pageerror', error => errors.push(error.message));
    await page.goto(process.env.CURVE_QA_URL || 'http://127.0.0.1:7860');
    await page.locator('#generationModeTab').click();
    await page.locator('#audioPath').fill('selection_qa.wav');
    await page.locator('#intensityModeTab').click();
    await page.evaluate(() => showIntensityProfile({schema: 'maichart-intensity-curve-v1', audio_sha256: 'a'.repeat(64), duration_sec: 120,
      points: Array.from({length: 241}, (_, i) => ({time_sec: i / 2, intensity: .4 + .15 * Math.sin(i / 8)})),
      spectrogram: {times: Array.from({length: 121}, (_, i) => i), frequencies_hz: [100, 300, 1000, 3000, 10000],
        values: Array.from({length: 5}, (_, y) => Array.from({length: 121}, (_, x) => Math.round(255 * (1 + Math.sin(x / 3 + y)) / 2)))}}));
    await page.locator('#intensitySectionStart').fill('0'); await page.locator('#intensitySectionEnd').fill('0');
    await page.locator('#markIntensityRestButton').click();
    assert.equal(await page.locator('#intensityRestCount').textContent(), '0');
    assert.match(await page.locator('#intensityCurveStatus').textContent(), /至少两个控制点/);
    await page.locator('#intensityEditMode').selectOption('select');
    await page.locator('#intensityCurveCanvas').scrollIntoViewIfNeeded();
    const geometry = await page.evaluate(() => {
      const canvas = document.querySelector('#intensityCurveCanvas'), rect = canvas.getBoundingClientRect(), geo = curveCanvasGeometry(canvas, intensityCurveState.profile);
      return {x: rect.x + geo.pad.left, y: rect.y + geo.pad.top, w: geo.plotWidth, h: geo.plotHeight};
    });
    await page.mouse.move(geometry.x + geometry.w * 19 / 120, geometry.y + geometry.h * .1);
    await page.mouse.down();
    await page.mouse.move(geometry.x + geometry.w * 31 / 120, geometry.y + geometry.h * .85, {steps: 10});
    await page.mouse.up();
    const selectedTimes = await page.evaluate(() => [...intensityView.selected].map(i => intensityCurveState.profile.points[i].time_sec).sort((a,b) => a-b));
    assert.ok(selectedTimes.length >= 20);
    const start = selectedTimes[0], end = selectedTimes.at(-1);
    assert.equal(Number(await page.locator('#intensitySectionStart').inputValue()), start);
    assert.equal(Number(await page.locator('#intensitySectionEnd').inputValue()), end);
    await page.locator('#markIntensityRestButton').click();
    assert.equal(await page.locator('#intensityRestCount').textContent(), '1');
    assert.equal(await page.locator('.intensity-rest-region').count(), 1);
    assert.match(await page.locator('.intensity-rest-region').textContent(), /休止 1/);
    assert.deepEqual(await page.evaluate(() => intensityCurveState.profile.rest_regions), [{start_sec: start, end_sec: end}]);
    await page.locator('#intensityClearSelection').click();
    const labels = await page.evaluate(() => {
      const ctx = document.querySelector('#intensityCurveCanvas').getContext('2d'), original = ctx.fillText.bind(ctx), seen=[];
      ctx.fillText = (...args) => {seen.push(args[0]); return original(...args);}; drawIntensityCurve(); ctx.fillText = original;
      return seen;
    });
    assert.ok(labels.some(label => label.includes('休止')));
    await page.locator('#intensityUndoButton').click();
    assert.equal(await page.locator('#intensityRestCount').textContent(), '0');
    await page.locator('#markIntensityRestButton').click();
    await page.locator('.intensity-rest-region').click();
    assert.ok(await page.evaluate(() => intensityView.end - intensityView.start < 30));
    await page.locator('#intensityFitAll').click();
    await page.locator('#intensityRestRegions').screenshot({path: '.scratch/rest_selection_ranges.png'});
    await page.locator('.intensity-panel').screenshot({path: '.scratch/rest_selection_desktop.png'});
    await page.locator('#intensitySaveButton').click();
    await page.reload(); await page.locator('#intensityModeTab').click();
    assert.equal(await page.locator('#intensityRestCount').textContent(), '1');
    assert.equal(await page.locator('.intensity-rest-region').count(), 1);
    await page.setViewportSize({width: 390, height: 844});
    assert.ok((await page.locator('.intensity-panel').boundingBox()).width < 390);
    assert.equal(await page.locator('.intensity-panel').evaluate(node => node.scrollWidth > node.clientWidth + 2), false);
    await page.locator('.intensity-panel').screenshot({path: '.scratch/rest_selection_mobile.png'});
    assert.deepEqual(errors, []);
    const result = {passed:true, pointer_selection_points:selectedTimes.length, marked_range:[start,end], automatic_time_fields:true,
      immediate_count:true, local_error_message:true, canvas_rest_label:true, range_list_focus:true, undo_save_reload:true, mobile_no_overflow:true};
    fs.writeFileSync('.scratch/rest_selection_visual_qa_result.json', JSON.stringify(result,null,2)); console.log(JSON.stringify(result));
  } finally {await browser.close();}
})().catch(error => {console.error(error); process.exitCode=1;});
