// k6 load script shared by every stack (KServe v2 JSON, POST /v2/models/<MODEL>/infer).
//
// Env: TARGET (http://host:8000), MODEL (deepfm_small), ROWS (1|50|200|500),
//      MODE (rate|vus), RATE (req/s, rate mode), VUS (closed-loop VUs, vus mode),
//      DURATION (60s), WARMUP (30s), PAYLOAD_DIR (default ./payloads), TIMEOUT (default 10s),
//      MODELS (comma list; with ZIPF_S the model of each request is drawn Zipf(s) over it, first = hottest),
//      SHAPE (constant | step) with STEPS="rate:seconds,rate:seconds,..." for step (open loop).
// Every measured request also carries a `serve_multiplexed_model_id` header = the model name.
// Two scenarios run back to back: `warmup` then `measure`. Only `measure` feeds the custom
// metrics, so the summary excludes warm-up. Final stdout line: K6_SUMMARY_JSON:{...}
import http from 'k6/http';
import exec from 'k6/execution';
import { SharedArray } from 'k6/data';
import { Trend, Counter } from 'k6/metrics';

const TARGET = __ENV.TARGET || 'http://localhost:8000';
const MODEL = __ENV.MODEL || 'deepfm_small';
const ROWS = __ENV.ROWS || '1';
const MODE = __ENV.MODE || 'rate';
const RATE = parseInt(__ENV.RATE || '10');
const VUS = parseInt(__ENV.VUS || '8');
const DURATION = __ENV.DURATION || '60s';
const WARMUP = __ENV.WARMUP || '30s';
const PAYLOAD_DIR = __ENV.PAYLOAD_DIR || './payloads';
const TIMEOUT = __ENV.TIMEOUT || '10s';
const MODELS = (__ENV.MODELS || MODEL).split(',');
const ZIPF_S = parseFloat(__ENV.ZIPF_S || '0');       // 0 = uniform over MODELS
const SHAPE = __ENV.SHAPE || 'constant';
const STEPS = (__ENV.STEPS || '').split(',').filter((x) => x).map((x) => {
  const [r, d] = x.split(':'); return { rate: parseInt(r), seconds: parseInt(d) };
});

// cumulative Zipf weights, drawn with Math.random() (k6 VUs have independent PRNG state)
const cdf = (() => {
  const w = MODELS.map((_, i) => (ZIPF_S > 0 ? 1 / Math.pow(i + 1, ZIPF_S) : 1));
  const tot = w.reduce((a, b) => a + b, 0);
  let acc = 0; return w.map((x) => (acc += x / tot));
})();
function pickModel() {
  if (MODELS.length === 1) return 0;
  const u = Math.random();
  for (let i = 0; i < cdf.length; i++) if (u <= cdf[i]) return i;
  return cdf.length - 1;
}

const bodies = new SharedArray('payloads', () =>
  JSON.parse(open(`${PAYLOAD_DIR}/rows${ROWS}.json`)).map((o) => JSON.stringify(o)));

const latency = new Trend('measure_latency', true);   // ms, measure scenario only
const okCount = new Counter('measure_ok');
const errCount = new Counter('measure_err');
const rowCount = new Counter('measure_rows');

function secs(s) { return parseInt(s) * (s.endsWith('m') ? 60 : 1); }

function measureScenario(startTime, duration) {
  if (SHAPE === 'step' && STEPS.length) {
    const pre = Math.max(20, Math.ceil(Math.max(...STEPS.map((x) => x.rate)) * 0.25));
    const stages = [];
    STEPS.forEach((st, i) => {
      if (i > 0) stages.push({ target: st.rate, duration: '1s' });      // near-instant step
      stages.push({ target: st.rate, duration: `${st.seconds}s` });
    });
    return { executor: 'ramping-arrival-rate', startRate: STEPS[0].rate, timeUnit: '1s', stages,
             startTime, preAllocatedVUs: pre, maxVUs: Math.min(6000, pre * 6), gracefulStop: '5s' };
  }
  return scenario(startTime, duration);
}

function scenario(startTime, duration) {
  if (MODE === 'vus') {
    return { executor: 'constant-vus', vus: VUS, duration, startTime, gracefulStop: '5s' };
  }
  const pre = Math.max(10, Math.ceil(RATE * 0.25));
  return {
    executor: 'constant-arrival-rate', rate: RATE, timeUnit: '1s', duration, startTime,
    preAllocatedVUs: pre, maxVUs: Math.min(4000, Math.max(pre * 4, 50)), gracefulStop: '5s',
  };
}

export const options = {
  discardResponseBodies: true,
  summaryTrendStats: ['avg', 'p(50)', 'p(90)', 'p(99)', 'max', 'count'],
  scenarios: {
    warmup: scenario('0s', WARMUP),
    measure: measureScenario(`${secs(WARMUP)}s`, DURATION),
  },
};

const urls = MODELS.map((m) => `${TARGET}/v2/models/${m}/infer`);
const paramsFor = MODELS.map((m) => ({
  headers: { 'Content-Type': 'application/json', serve_multiplexed_model_id: m }, timeout: TIMEOUT,
  tags: { name: 'infer' },                    // one url tag value, not one per model: keeps csv small
}));

export default function () {
  const body = bodies[Math.floor(Math.random() * bodies.length)];
  const i = pickModel();
  const res = http.post(urls[i], body, paramsFor[i]);
  if (exec.scenario.name !== 'measure') return;
  latency.add(res.timings.duration, { status: String(res.status) });
  if (res.status === 200) { okCount.add(1); rowCount.add(parseInt(ROWS)); } else { errCount.add(1); }
}

export function handleSummary(data) {
  const m = data.metrics;
  const v = (name, key) => (m[name] && m[name].values && m[name].values[key] !== undefined) ? m[name].values[key] : null;
  const out = {
    target: TARGET, model: MODEL, rows: parseInt(ROWS), mode: MODE,
    rate: MODE === 'rate' ? RATE : null, vus: MODE === 'vus' ? VUS : null,
    duration_s: SHAPE === 'step' ? STEPS.reduce((a, x) => a + x.seconds, 0) : secs(DURATION),
    warmup_s: secs(WARMUP), models: MODELS.length, zipf_s: ZIPF_S, shape: SHAPE,
    ok: v('measure_ok', 'count') || 0, err: v('measure_err', 'count') || 0,
    rows_total: v('measure_rows', 'count') || 0,
    dropped_iterations: v('dropped_iterations', 'count') || 0,
    latency_ms: {
      avg: v('measure_latency', 'avg'), p50: v('measure_latency', 'p(50)'),
      p90: v('measure_latency', 'p(90)'), p99: v('measure_latency', 'p(99)'),
      max: v('measure_latency', 'max'),
    },
    http_req_failed_rate: v('http_req_failed', 'rate'),
  };
  out.achieved_rps = out.ok / out.duration_s;
  if (SHAPE === 'step') out.rate = null;
  const line = `${MODEL} rows=${ROWS} ${MODE}=${MODE === 'rate' ? RATE : VUS} ok=${out.ok} err=${out.err} ` +
    `dropped=${out.dropped_iterations} p50=${(out.latency_ms.p50 || 0).toFixed(1)}ms p99=${(out.latency_ms.p99 || 0).toFixed(1)}ms`;
  return { stdout: `${line}\nK6_SUMMARY_JSON:${JSON.stringify(out)}\n` };
}
