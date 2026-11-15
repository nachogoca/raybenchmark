// k6 load script shared by every stack (KServe v2 JSON, POST /v2/models/<MODEL>/infer).
//
// Env: TARGET (http://host:8000), MODEL (deepfm_small), ROWS (1|50|200|500),
//      MODE (rate|vus), RATE (req/s, rate mode), VUS (closed-loop VUs, vus mode),
//      DURATION (60s), WARMUP (30s), PAYLOAD_DIR (default ./payloads), TIMEOUT (default 10s).
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

const bodies = new SharedArray('payloads', () =>
  JSON.parse(open(`${PAYLOAD_DIR}/rows${ROWS}.json`)).map((o) => JSON.stringify(o)));

const latency = new Trend('measure_latency', true);   // ms, measure scenario only
const okCount = new Counter('measure_ok');
const errCount = new Counter('measure_err');
const rowCount = new Counter('measure_rows');

function secs(s) { return parseInt(s) * (s.endsWith('m') ? 60 : 1); }

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
    measure: scenario(`${secs(WARMUP)}s`, DURATION),
  },
};

const url = `${TARGET}/v2/models/${MODEL}/infer`;
const params = { headers: { 'Content-Type': 'application/json' }, timeout: TIMEOUT };

export default function () {
  const body = bodies[Math.floor(Math.random() * bodies.length)];
  const res = http.post(url, body, params);
  if (exec.scenario.name !== 'measure') return;
  latency.add(res.timings.duration);
  if (res.status === 200) { okCount.add(1); rowCount.add(parseInt(ROWS)); } else { errCount.add(1); }
}

export function handleSummary(data) {
  const m = data.metrics;
  const v = (name, key) => (m[name] && m[name].values && m[name].values[key] !== undefined) ? m[name].values[key] : null;
  const out = {
    target: TARGET, model: MODEL, rows: parseInt(ROWS), mode: MODE,
    rate: MODE === 'rate' ? RATE : null, vus: MODE === 'vus' ? VUS : null,
    duration_s: secs(DURATION), warmup_s: secs(WARMUP),
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
  const line = `${MODEL} rows=${ROWS} ${MODE}=${MODE === 'rate' ? RATE : VUS} ok=${out.ok} err=${out.err} ` +
    `dropped=${out.dropped_iterations} p50=${(out.latency_ms.p50 || 0).toFixed(1)}ms p99=${(out.latency_ms.p99 || 0).toFixed(1)}ms`;
  return { stdout: `${line}\nK6_SUMMARY_JSON:${JSON.stringify(out)}\n` };
}
