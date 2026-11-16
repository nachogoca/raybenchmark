# Serving benchmark: results

Generated 2026-11-16.

## Scorecard (p99 ≤ 50 ms, DeepFM small, 200 rows per request)

| Stack | Capacity (req/s) | 95 % CI | p99 @ 100 req/s | CPU ms / request | req/s / vCPU | Peak RSS | Cold start | $ / 1M (allocated) | $ / 1M (consumed) |
|---|---|---|---|---|---|---|---|---|---|
| Ray + ORT | 275 | 275-275 | 8.2 ms | 16.35 | 45.8 | 1,025 MB | 10.1 s | 0.192 | 0.145 |
| Ray + FastAPI | 250 | 200-250 | 8.8 ms | 17.89 | 41.7 | 1,029 MB | 10.2 s | 0.211 | 0.159 |
| Ray + in-process Triton | 250 | 250-275 | 8.7 ms | 17.53 | 41.7 | 1,075 MB | 12.3 s | 0.212 | 0.156 |
| Ray → Triton (gRPC) | 250 | 250-275 | 11.2 ms | 18.70 | 41.7 | 1,039 MB | 9.8 s | 0.211 | 0.166 |
| Triton | 250 | 250-300 | 9.5 ms | 22.32 | 41.7 | 41 MB | 0.4 s | 0.207 | 0.192 |
| FastAPI + ORT | 175 | 175-200 | 9.9 ms | 31.23 | 29.2 | 88 MB | 1.2 s | 0.296 | 0.270 |

Prices: gke_n2 $0.031/vCPU-h, $0.0042/GB-h (NOT verified, as of 2026-11-15).

## Ingress (capacity, p99 ≤ 50 ms)

| Config | Capacity (req/s) |
|---|---|
| Ray, gRPC | 112 |
| Ray, HAProxy | 168 |
| Ray, Python proxy | 93 |
| Triton, gRPC | 378 |
| Triton, HTTP | 252 |

## Model family (capacity, p99 ≤ 100 ms)

| Config | Capacity (req/s) |
|---|---|
| Ray, one deployment per model | 42 |
| FastAPI, 16 models | 78 |
| Ray multiplexing, cache of 16 models | 545 |
| Ray multiplexing, 4-model family | 490 |
| Ray multiplexing, 64-model family | 218 |
| Ray multiplexing, cache of 4 models | 145 |
| Triton, 16 models | 118 |

## Overload and backpressure (capacity, p99 ≤ 100 ms)

| Config | Capacity (req/s) |
|---|---|
| FastAPI, limit 50 concurrent | 75 |
| FastAPI, unbounded | 281 |
| Ray, max 50 queued | 338 |
| Ray, unbounded queue | 309 |
| Triton, max 50 queued | 422 |
| Triton, unbounded queue | 422 |
