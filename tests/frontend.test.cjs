// Browser-independent behavior checks. The separate live-browser checks use real API jobs.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'web/index.html'), 'utf8');
const script = fs.readFileSync(path.join(root, 'web/app.js'), 'utf8');

function setup(route = () => ({}), localTest = false) {
  const ids = new Map();
  let now = 1000000;
  let focused = null;
  const intervals = new Map();
  let intervalId = 0;
  class Element {
    constructor(tag = 'div') {
      this.tagName = tag; this.children = []; this.attrs = {}; this.dataset = {};
      this.events = {}; this.hidden = false; this.value = ''; this.disabled = false;
      this.textContent = ''; this.open = false; this.readyState = 0;
      const classes = new Set();
      this.classList = { add: x => classes.add(x), remove: x => classes.delete(x),
        toggle: (x, on) => on ? classes.add(x) : classes.delete(x) };
    }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    get firstElementChild() { return this.children[0]; }
    get lastElementChild() { return this.children[this.children.length - 1]; }
    addEventListener(name, fn) { (this.events[name] ||= []).push(fn); }
    setAttribute(name, value) { this.attrs[name] = value; }
    removeAttribute(name) { delete this.attrs[name]; if (name === 'src') this.src = ''; }
    hasAttribute(name) { return name in this.attrs; }
    focus() { focused = this; }
    scrollIntoView() {}
    pause() {}
    load() {}
    showModal() { this.open = true; }
    close() { this.open = false; this.emit('close'); }
    querySelectorAll(selector) { return this.children.filter(x => selector === 'button' && x.tagName === 'button'); }
    emit(name, extra = {}) { for (const fn of this.events[name] || []) fn({ target: this, preventDefault() {}, ...extra }); }
  }
  for (const match of html.matchAll(/<([a-z][a-z0-9]*)\b([^>]*\bid="([^"]+)"[^>]*)>/g)) {
    const element = new Element(match[1]); element.hidden = /\bhidden\b/.test(match[2]); ids.set(match[3], element);
  }
  ids.get('analyzeButton').append(new Element('span'), new Element('span'));
  ids.get('serviceStatus').append(new Element('i'), new Element('span'));
  const examples = ['normal', 'multiple'].map(name => { const element = new Element('button'); element.dataset.example = name; return element; });
  const requests = [];
  const get = id => ids.get(id) || [...ids.get('eventList').children].find(x => x.id === id);
  const document = { getElementById: get, createElement: tag => new Element(tag),
    querySelectorAll: selector => selector === '[data-example]' ? examples : selector === '.event-card' ? ids.get('eventList').children : [] };
  const testURL = class extends URL {};
  testURL.createObjectURL = () => 'blob:synthetic-test'; testURL.revokeObjectURL = () => {};
  const testDate = class extends Date {}; testDate.now = () => now;
  const context = { document, window: new Element(), URL: testURL, Date: testDate, console,
    matchMedia: () => ({ matches: true }), AbortController,
    setTimeout: callback => { if (!String(callback).includes('abort')) queueMicrotask(callback); return 1; }, clearTimeout() {},
    setInterval: callback => { intervals.set(++intervalId, callback); return intervalId; }, clearInterval: id => intervals.delete(id),
    FormData: class { append(name, value) { this[name] = value; } },
    File: class { constructor(parts, name) { this.name = name; this.size = parts.reduce((sum, p) => sum + (p.size || 1), 0); } },
    fetch: async (url, options) => {
      requests.push({ url, options });
      const value = url === '/health' ? { status: 'ok', model_present: true, ready: true, demo_only: true, local_test: localTest } : await route(url, options);
      if (value instanceof Error) throw value;
      const status = value?.httpStatus || 200;
      return { ok: status < 400, status, headers: { get: () => value?.retryAfter || null },
        json: async () => value, blob: async () => ({ size: 100 }) };
    }
  };
  vm.runInNewContext(script, context, { filename: 'app.js' });
  const settle = async () => { for (let i = 0; i < 30; i++) await new Promise(resolve => setImmediate(resolve)); };
  const key = () => { get('apiKey').value = 'temporary-synthetic-test-key'; };
  const file = (name = 'synthetic.mov', size = 100) => get('videoFile').emit('change', { target: { files: [{ name, size }] } });
  const submit = async () => { get('uploadForm').emit('submit'); await settle(); };
  return { get, key, file, submit, settle, requests, examples, focus: () => focused,
    advance: ms => { now += ms; for (const callback of [...intervals.values()]) callback(); } };
}
const empty = { job_id: 'a'.repeat(16), status: 'done', verdict: 'no_violation', violations: [], meta: { warnings: [] } };

test('local testing hides key field and submits without a key', async () => {
  const ui = setup(url => url === '/api/v1/analyze' ? { status: 'processing', job_id: empty.job_id } : empty, true);
  await ui.settle();
  assert.equal(ui.get('apiKeyField').hidden, true);
  assert.equal(ui.get('apiKey').required, false);
  ui.file(); await ui.submit();
  assert.equal(ui.requests.filter(x => x.url === '/api/v1/analyze').length, 1);
  assert.equal(ui.get('resultSection').hidden, false);
});

test('missing key and missing file are explained without submitting', async () => {
  const ui = setup(); await ui.submit();
  assert.match(ui.get('keyError').textContent, /Введите ключ/);
  ui.key(); await ui.submit(); assert.match(ui.get('fileError').textContent, /Выберите/);
  assert.equal(ui.requests.filter(x => x.url === '/api/v1/analyze').length, 0);
});
test('unsupported, empty and oversized files never reach API', async () => {
  const ui = setup(); ui.key();
  for (const [name, size] of [['bad.exe', 100], ['empty.mov', 0], ['large.mov', 65 * 1024 * 1024]]) {
    ui.file(name, size); await ui.submit(); assert.equal(ui.get('resultSection').hidden, true);
  }
  assert.equal(ui.requests.filter(x => x.url === '/api/v1/analyze').length, 0);
});
test('401 attaches error to API key and never displays a verdict', async () => {
  const ui = setup(() => ({ httpStatus: 401 })); ui.key(); ui.file(); await ui.submit();
  assert.match(ui.get('keyError').textContent, /Неверный ключ/);
  assert.equal(ui.focus(), ui.get('apiKey')); assert.equal(ui.get('resultSection').hidden, true);
});
test('429 respects Retry-After, blocks immediate retry and never resubmits automatically', async () => {
  const ui = setup(() => ({ httpStatus: 429, retryAfter: '7' })); ui.key(); ui.file(); await ui.submit();
  assert.equal(ui.get('analyzeButton').disabled, true); ui.advance(7001);
  assert.equal(ui.get('analyzeButton').disabled, false);
  assert.equal(ui.requests.filter(x => x.url === '/api/v1/analyze').length, 1);
});
test('failed job does not become a clean result', async () => {
  const ui = setup(url => url.includes('/jobs/') ? { status: 'failed', error: 'Synthetic decode failure' } : { status: 'processing', job_id: 'a'.repeat(16) });
  ui.key(); ui.file(); await ui.submit(); assert.equal(ui.get('resultSection').hidden, true);
  assert.match(ui.get('errorText').textContent, /ошибке анализа/); assert.equal(ui.get('analyzeButton').disabled, false);
});
test('negative verdict renders without requiring optional metadata or rule fields', async () => {
  const ui = setup(url => url.includes('/jobs/') ? empty : { status: 'processing', job_id: empty.job_id });
  ui.key(); ui.file(); await ui.submit(); assert.equal(ui.get('resultSection').hidden, false, ui.get('errorText').textContent);
  assert.equal(ui.get('resultTitle').textContent, 'Нарушение не обнаружено');
  assert.match(ui.get('resultNote').textContent, /не подтверждает/); assert.equal(ui.get('errorBox').hidden, true);
});
test('malformed or contradictory result is rejected without a clean verdict', async () => {
  const ui = setup(url => url.includes('/jobs/') ? { ...empty, verdict: 'violation' } : { status: 'processing', job_id: empty.job_id });
  ui.key(); ui.file(); await ui.submit(); assert.equal(ui.get('resultSection').hidden, true);
  assert.match(ui.get('errorText').textContent, /формату результата/);
});
test('network interruption resumes the existing job with no second upload', async () => {
  let polls = 0;
  const ui = setup(url => url.includes('/jobs/') ? (++polls === 1 ? new Error('offline') : empty) : { status: 'processing', job_id: empty.job_id });
  ui.key(); ui.file(); await ui.submit(); assert.equal(ui.get('resumeButton').hidden, false);
  assert.equal(ui.get('analyzeButton').disabled, true);
  ui.get('resumeButton').emit('click'); await ui.settle();
  assert.equal(ui.get('resultSection').hidden, false, ui.get('errorText').textContent);
  assert.equal(ui.requests.filter(x => x.url === '/api/v1/analyze').length, 1);
});
test('five minute budget produces an error with manual status recovery', async () => {
  let ui;
  ui = setup(url => {
    if (url.includes('/jobs/')) { ui.advance(300001); return { status: 'processing' }; }
    return { status: 'processing', job_id: empty.job_id };
  }); ui.key(); ui.file(); await ui.submit();
  assert.match(ui.get('errorText').textContent, /5 минут/);
  assert.equal(ui.get('resultSection').hidden, true); assert.equal(ui.get('resumeButton').hidden, false);
});
test('413, 415, 503 and upload network failures remain recoverable errors', async () => {
  for (const response of [{ httpStatus: 413 }, { httpStatus: 415 }, { httpStatus: 503 }, new Error('network')]) {
    const ui = setup(() => response); ui.key(); ui.file(); await ui.submit();
    assert.equal(ui.get('errorBox').hidden, false); assert.equal(ui.get('resultSection').hidden, true);
    assert.equal(ui.get('analyzeButton').disabled, false);
  }
});
test('source URL protocol is restricted and server observations remain plain text', async () => {
  const event = { type: '<img onerror=alert(1)>', timecode_sec: 2, description: '<script>unsafe()</script>',
    evidence_frames: [{ timestamp_sec: 2, image_base64: '/9j/abc=' }],
    rule: { document: '<b>source</b>', section: 'test', url: 'javascript:alert(1)', summary: 'test' } };
  const ui = setup(url => url.includes('/jobs/') ? { ...empty, verdict: 'violation', violations: [event] } : { status: 'processing', job_id: empty.job_id });
  ui.key(); ui.file(); await ui.submit(); assert.equal(ui.get('resultSection').hidden, false, ui.get('errorText').textContent);
  const card = ui.get('eventList').children[0];
  assert.equal(card.children[1].textContent, event.description);
  assert.equal(card.children.at(-1).children.some(x => x.tagName === 'a'), false);
  card.children[2].children[0].children[0].emit('click');
  assert.equal(ui.get('frameDialog').open, true);
  ui.get('closeDialog').emit('click'); assert.equal(ui.get('frameDialog').open, false);
});
