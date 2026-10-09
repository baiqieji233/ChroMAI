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
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(process.env.CURVE_QA_URL || 'http://127.0.0.1:7860');
    await page.waitForFunction(() => !!levelPresetProfile);
    await page.locator('#generationModeTab').click();
    assert.equal(await page.locator('#note_density').isVisible(), false);
    assert.equal(await page.locator('#preferKeyboard').isVisible(), true);
    const condition = () => page.evaluate(() => collectCondition());
    const baseline = level => page.evaluate(level => composeLevelCondition(levelPresetProfile, level, [], 0), level);
    const setRange = (selector, value) => page.locator(selector).evaluate((node, value) => { node.value = String(value); node.dispatchEvent(new Event('input', {bubbles: true})); }, value);
    for (let index = 0; index < 31; index++) {
      const level = Number((12 + index / 10).toFixed(1));
      await setRange('#levelRange', level);
      const actual = await condition(), expected = await baseline(level);
      assert.equal(actual.level, level);
      assert.deepEqual(actual.radar, expected.radar);
      assert.deepEqual(actual.style, expected.style);
    }
    await page.locator('#level').fill('13.45');
    await page.locator('#level').blur();
    assert.equal((await condition()).level, 13.45);
    assert.deepEqual((await condition()).radar, (await baseline(13.45)).radar);
    await page.locator('#level').fill('13.4'); await page.locator('#level').blur();
    const base = await condition();
    await page.locator('#preferKeyboard').check();
    const keyboard = await condition();
    assert.ok(keyboard.style.tap_bias > base.style.tap_bias);
    assert.ok(keyboard.style.slide_bias < base.style.slide_bias);
    assert.equal(keyboard.radar.note_density, base.radar.note_density);
    await page.locator('#preferStars').check(); await page.locator('#preferTechnical').check();
    const combined = await condition();
    await page.locator('#applyConditionPreset').click();
    assert.deepEqual(await condition(), combined);
    await page.locator('#clearConditionPreferences').click();
    await page.locator('#preferTechnical').check(); await page.locator('#preferStars').check(); await page.locator('#preferKeyboard').check();
    assert.deepEqual(await condition(), combined);
    await setRange('#conditionPreferenceStrength', 0);
    assert.deepEqual((await condition()).radar, base.radar);
    assert.deepEqual((await condition()).style, base.style);
    await setRange('#conditionPreferenceStrength', 0.5);
    await setRange('#levelRange', 14.5);
    assert.equal(await page.locator('#preferStars').isChecked(), true);
    assert.equal((await condition()).level, 14.5);
    assert.equal((await condition()).radar.note_density, (await baseline(14.5)).radar.note_density);

    await page.locator('.condition-advanced').filter({has: page.locator('#radarControls')}).locator('summary').click();
    await page.locator('[data-number-for="touch_bias"]').fill('0.023');
    await page.locator('[data-number-for="touch_bias"]').blur();
    assert.equal((await condition()).style.touch_bias, 0.023);
    await page.evaluate(() => flushSettings());
    await page.reload(); await page.waitForFunction(() => !!levelPresetProfile);
    assert.equal((await condition()).style.touch_bias, 0.023);
    assert.equal(await page.locator('#preferKeyboard').isChecked(), true);
    assert.equal(await page.locator('#preferStars').isChecked(), true);
    await page.locator('#applyConditionPreset').click();
    assert.notEqual((await condition()).style.touch_bias, 0.023);

    await page.locator('#audioPath').fill(path.join(projectRoot, '.scratch/qa.wav'));
    await page.locator('#checkpointPath').fill(path.join(projectRoot, '.scratch/qa.pt'));
    let submitted = null;
    await page.route('**/api/generate', async route => {
      submitted = route.request().postDataJSON();
      await route.fulfill({status: 400, contentType: 'application/json', body: JSON.stringify({error: 'QA request captured; no generation started'})});
    });
    await page.locator('#generateButton').click();
    await page.waitForFunction(() => document.querySelector('#serverStatus').textContent.includes('QA request captured'));
    assert.ok(submitted);
    assert.equal(submitted.condition.level, 14.5);
    assert.ok(submitted.condition.preset_recipe.preferences.includes('keyboard'));
    assert.equal(submitted.condition.style.chord_bias, submitted.condition.radar.chord);
    await page.locator('#trainingModeTab').click(); await page.locator('#batchSize').fill('7');
    await page.locator('#generationModeTab').click(); await page.locator('#resetGenerationButton').click();
    assert.equal(await page.locator('#batchSize').inputValue(), '7');
    assert.equal(await page.locator('#preferKeyboard').isChecked(), false);
    assert.equal(await page.locator('#preferStars').isChecked(), false);
    assert.equal((await condition()).level, 13.3);
    assert.deepEqual((await condition()).radar, (await baseline(13.3)).radar);
    await page.locator('#preferStars').check(); await page.locator('#preferTechnical').check();
    await page.locator('.condition-presets').screenshot({path: '.scratch/continuous_level_desktop.png'});
    await page.setViewportSize({width: 390, height: 844});
    await page.locator('.condition-presets').screenshot({path: '.scratch/continuous_level_mobile.png'});
    const overflow = await page.locator('.condition-presets').evaluate(node => node.scrollWidth > node.clientWidth + 2);
    assert.equal(overflow, false);
    assert.deepEqual(errors, []);
    const result = {status: 'passed', levels: 31, decimal_interpolation: true, preferences_multi_select: true,
      order_independent_and_repeat_safe: true, strength_zero_baseline: true, preserves_density: true,
      manual_edit_and_reload: true, request_captured: true, reset_preserves_training: true, mobile_no_overflow: true};
    fs.writeFileSync('.scratch/continuous_level_browser_qa_result.json', JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
