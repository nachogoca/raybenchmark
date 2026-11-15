// k6 gRPC twin of infer.js: KServe v2 ModelInfer against Triton (:8001) or Ray Serve's gRPC
// proxy (:9000). Same env contract and same `measure_latency` metric; the `status` tag is "200"
// for gRPC OK and "5<code>" otherwise, so bench.collect.k6parse treats both scripts alike.
import grpc from 'k6/net/grpc';
import exec from 'k6/execution';
import { SharedArray } from 'k6/data';
import { Trend, Counter } from 'k6/metrics';

const TARGET = __ENV.TARGET || 'localhost:9000';
const MODEL = __ENV.MODEL || 'deepfm_small';
const ROWS = __ENV.ROWS || '200';
const MODE = __ENV.MODE || 'rate';
const RATE = parseInt(__ENV.RATE || '10');
const VUS = parseInt(__ENV.VUS || '8');
const DURATION = __ENV.DURATION || '60s';
const WARMUP = __ENV.WARMUP || '30s';
const PAYLOAD_DIR = __ENV.PAYLOAD_DIR || './payloads';
const PROTO_DIR = __ENV.PROTO_DIR || '.';
const APP = __ENV.RAY_APP || '';          // Ray gRPC proxy routes on the `application` metadata

const bodies = new SharedArray('payloads', () => JSON.parse(open(`${PAYLOAD_DIR}/rows${ROWS}.json`)));
const latency = new Trend('measure_latency', true);
const okCount = new Counter('measure_ok');
const errCount = new Counter('measure_err');
const rowCount = new Counter('measure_rows');
const client = new grpc.Client();
client.load([PROTO_DIR], 'kserve.proto');

function secs(s) { return parseInt(s) * (s.endsWith('m') ? 60 : 1); }
function scenario(startTime, duration) {
  if (MODE === 'vus') return { executor: 'constant-vus', vus: VUS, duration, startTime, gracefulStop: '5s' };
  const pre = Math.max(10, Math.ceil(RATE * 0.25));
  return { executor: 'constant-arrival-rate', rate: RATE, timeUnit: '1s', duration, startTime,
           preAllocatedVUs: pre, maxVUs: Math.min(4000, Math.max(pre * 4, 50)), gracefulStop: '5s' };
}
export const options = {
  discardResponseBodies: true,
  summaryTrendStats: ['avg', 'p(50)', 'p(90)', 'p(99)', 'max', 'count'],
  scenarios: { warmup: scenario('0s', WARMUP), measure: scenario(`${secs(WARMUP)}s`, DURATION) },
};

function toRequest(b) {
  const [d, s] = b.inputs;
  return { model_name: MODEL, inputs: [
    { name: d.name, datatype: 'FP32', shape: d.shape, contents: { fp32_contents: d.data } },
    { name: s.name, datatype: 'INT64', shape: s.shape, contents: { int64_contents: s.data } } ] };
}
const requests = bodies.map(toRequest);
const params = { metadata: APP ? { application: APP } : {}, timeout: __ENV.TIMEOUT || '10s' };

export default function () {
  if (exec.vu.iterationInScenario === 0) client.connect(TARGET, { plaintext: true });
  const res = client.invoke('inference.GRPCInferenceService/ModelInfer',
                            requests[Math.floor(Math.random() * requests.length)], params);
  if (exec.scenario.name !== 'measure') return;
  const ok = res && res.status === grpc.StatusOK;
  latency.add(res.timings ? res.timings.duration : 0, { status: ok ? '200' : `5${res.status}` });
  if (ok) { okCount.add(1); rowCount.add(parseInt(ROWS)); } else { errCount.add(1); }
}

export function handleSummary(data) {
  const m = data.metrics;
  const v = (n, k) => (m[n] && m[n].values && m[n].values[k] !== undefined) ? m[n].values[k] : null;
  const out = {
    target: TARGET, model: MODEL, rows: parseInt(ROWS), mode: MODE, protocol: 'grpc',
    rate: MODE === 'rate' ? RATE : null, vus: MODE === 'vus' ? VUS : null,
    duration_s: secs(DURATION), warmup_s: secs(WARMUP),
    ok: v('measure_ok', 'count') || 0, err: v('measure_err', 'count') || 0,
    rows_total: v('measure_rows', 'count') || 0, dropped_iterations: v('dropped_iterations', 'count') || 0,
    latency_ms: { avg: v('measure_latency', 'avg'), p50: v('measure_latency', 'p(50)'),
                  p90: v('measure_latency', 'p(90)'), p99: v('measure_latency', 'p(99)'),
                  max: v('measure_latency', 'max') },
    http_req_failed_rate: null,
  };
  out.achieved_rps = out.ok / out.duration_s;
  return { stdout: `${MODEL} grpc rows=${ROWS} ${MODE}=${out.rate || out.vus} ok=${out.ok} err=${out.err}\nK6_SUMMARY_JSON:${JSON.stringify(out)}\n` };
}
