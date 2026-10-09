const { chromium } = require('playwright-core');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const projectRoot = path.resolve(__dirname, '../..');
process.chdir(projectRoot);
fs.mkdirSync('.scratch', {recursive: true});

(async () => {
  const browser = await chromium.launch({headless: true, channel: 'msedge'});
  const errors = [];
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1050}});
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(process.env.CURVE_QA_URL || 'http://127.0.0.1:7860');
    await page.locator('#generationModeTab').click();
    const modes = await page.locator('#intensityCurveMode option').evaluateAll(nodes => nodes.map(node => [node.value, node.textContent]));
    assert.deepEqual(modes.map(item => item[0]), ['follow', 'hard', 'soft', 'strict']);
    await page.locator('#audioPath').fill('qa.wav');
    await page.locator('#intensityModeTab').click();
    assert.equal(await page.locator('#markIntensityRestButton').isDisabled(), true);
    await page.evaluate(() => showIntensityProfile({schema: 'maichart-intensity-curve-v1', audio_sha256: 'a'.repeat(64), duration_sec: 10, points: [{time_sec: 0, intensity: .5}, {time_sec: 5, intensity: .5}, {time_sec: 10, intensity: .5}], rest_regions: []}));
    await page.evaluate(() => { intensityCurveState.path = 'qa.wav'; intensityCurveState.saved = true; saveIntensityCurveState(); });
    assert.equal(await page.locator('#markIntensityRestButton').isDisabled(), false);
    await page.locator('#intensitySectionStart').fill('2'); await page.locator('#intensitySectionEnd').fill('5');
    await page.locator('#markIntensityRestButton').click();
    await page.locator('#intensitySectionStart').fill('4'); await page.locator('#intensitySectionEnd').fill('7');
    await page.locator('#markIntensityRestButton').click();
    assert.equal(await page.locator('#intensityRestCount').textContent(), '1');
    await page.locator('#intensitySectionStart').fill('3'); await page.locator('#intensitySectionEnd').fill('6');
    await page.locator('#clearIntensityRestButton').click();
    assert.equal(await page.locator('#intensityRestCount').textContent(), '2');
    await page.locator('#intensityUndoButton').click();
    assert.equal(await page.locator('#intensityRestCount').textContent(), '1');
    await page.locator('#intensitySaveButton').click();
    const payload = await page.evaluate(() => intensityCurvePayload());
    assert.deepEqual(payload.rest_regions, [{start_sec: 2, end_sec: 7}]);
    await page.reload();
    await page.locator('#intensityModeTab').click();
    await page.evaluate(() => { document.querySelector('#audioPath').value = 'qa.wav'; intensityCurveState.path = 'qa.wav'; });
    assert.equal(await page.locator('#intensityRestCount').textContent(), '1');
    await page.evaluate(() => { document.querySelector('#intensityCurveEnabled').checked = true; document.querySelector('#intensityCurveMode').value = 'soft'; document.querySelector('#checkpointPath').value = 'qa.pt'; document.querySelector('#audioPath').value = 'qa.wav'; });
    let generated = false;
    await page.route('**/api/generate', route => { generated = true; return route.fulfill({status: 400, contentType: 'application/json', body: JSON.stringify({error: 'captured'})}); });
    await page.route('**/api/status', route => route.fulfill({status: 200, contentType: 'application/json', body: JSON.stringify({intensity_curve_capability: true, music_structure_capability: true, density_feedback_capability: false})}));
    const blocked = await page.evaluate(async () => { try { await startGenerate(); return ''; } catch (error) { return error.message; } });
    await page.waitForTimeout(250);
    assert.equal(generated, false);
    assert.match(blocked, /显式休止区密度反馈能力/);
    await page.evaluate(() => { intensityCurveState.profile.rest_regions = []; });
    for (const mode of ['follow', 'hard']) {
      await page.evaluate(mode => { document.querySelector('#intensityCurveMode').value = mode; }, mode);
      const followBlocked = await page.evaluate(async () => { try { await startGenerate(); return ''; } catch (error) { return error.message; } });
      assert.match(followBlocked, /follow \/ hard 密度反馈能力/);
    }
    await page.evaluate(() => { intensityCurveState.profile.rest_regions = [{start_sec: 2, end_sec: 7}]; });
    await page.locator('#intensityModeTab').click();
    await page.locator('#intensityResetButton').click();
    assert.equal(await page.locator('#intensityRestCount').textContent(), '0');
    await page.locator('#intensityUndoButton').click();
    assert.equal(await page.locator('#intensityRestCount').textContent(), '1');
    await page.locator('#intensitySaveButton').click();
    await page.unroute('**/api/status');
    let captured = null;
    await page.unroute('**/api/generate');
    await page.route('**/api/generate', route => {
      captured = route.request().postDataJSON();
      return route.fulfill({status: 400, contentType: 'application/json', body: JSON.stringify({error: 'QA request captured'})});
    });
    for (const mode of ['follow', 'hard', 'soft', 'strict']) {
      await page.locator('#intensityCurveMode').selectOption(mode);
      await page.locator('#intensityCurveStrength').fill('0');
      const error = await page.evaluate(async () => { try { await startGenerate(); return ''; } catch (error) { return error.message; } });
      assert.equal(error, 'QA request captured');
      assert.equal(captured.intensity_curve_mode, mode);
      assert.equal(captured.intensity_curve_strength, 0);
      assert.deepEqual(captured.intensity_curve.rest_regions, [{start_sec: 2, end_sec: 7}]);
    }
    await page.locator('#intensityModeTab').click();
    await page.locator('.intensity-panel').screenshot({path: '.scratch/curve_density_desktop.png'});
    await page.setViewportSize({width: 390, height: 844});
    assert.equal(await page.locator('.intensity-panel').evaluate(node => node.scrollWidth > node.clientWidth + 2), false);
    assert.ok((await page.locator('.intensity-panel').boundingBox()).width < 390);
    await page.locator('.intensity-panel').screenshot({path: '.scratch/curve_density_mobile.png'});
    assert.deepEqual(errors, []);
    const result = {status: 'passed', modes: modes.map(item => item[0]), rest_merge_split: true, undo_save_reload: true, reset_undo: true, payload_rest_regions: true, all_mode_requests_with_zero_strength: true, capability_guard: true, mobile_no_overflow: true};
    fs.writeFileSync('.scratch/curve_density_ui_qa_result.json', JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
