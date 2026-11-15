# Appendix

## Capacity (all SLOs, all configs)

| env | stack | arch | size | model | rows | batching | replicas | pod_cpu | slo_ms | capacity_rps | confirmed | reps | rows_per_s | p50_ms | p99_ms | cpu_cores | cpu_util_pct | rss_peak_mb | usd_per_1m_req | usd_per_1b_rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| laptop-cpu | fastapi_ort | deepfm | small | deepfm_small | 200 | False | 1 |  | 20 | 168 | False | 1 | 3.36e+04 | 2.95 | 17.38 | 5.45 | 90.8 | 85.5 |  |  |
| laptop-cpu | fastapi_ort | deepfm | small | deepfm_small | 200 | False | 1 |  | 50 | 168 | False | 1 | 3.36e+04 | 2.95 | 17.38 | 5.45 | 90.8 | 85.5 |  |  |
| laptop-cpu | fastapi_ort | deepfm | small | deepfm_small | 200 | False | 1 |  | 100 | 252 | False | 1 | 5.04e+04 | 3.9 | 50.96 | 5.6 | 93.2 | 88.5 |  |  |
| laptop-cpu | triton | deepfm | small | deepfm_small | 200 | True | 1 |  | 20 | 168 | False | 1 | 3.36e+04 | 2.82 | 19.84 | 5.36 | 89.4 | 44.5 |  |  |
| laptop-cpu | triton | deepfm | small | deepfm_small | 200 | True | 1 |  | 50 | 252 | False | 1 | 5.04e+04 | 4.14 | 27.9 | 5.48 | 91.4 | 47 |  |  |
| laptop-cpu | triton | deepfm | small | deepfm_small | 200 | True | 1 |  | 100 | 378 | False | 1 | 7.56e+04 | 18.63 | 94.34 | 5.47 | 91.2 | 53.9 |  |  |
| laptop-cpu-haproxy | ray_fastapi | deepfm | small | deepfm_small | 200 | False | 1 |  | 20 | 112 | False | 1 | 2.24e+04 | 3.8 | 6.06 | 3.73 | 62.2 | 1011 |  |  |
| laptop-cpu-haproxy | ray_fastapi | deepfm | small | deepfm_small | 200 | False | 1 |  | 50 | 252 | False | 1 | 5.04e+04 | 4.5 | 22.18 | 4.41 | 73.4 | 1022 |  |  |
| laptop-cpu-haproxy | ray_fastapi | deepfm | small | deepfm_small | 200 | False | 1 |  | 100 | 252 | False | 1 | 5.04e+04 | 4.5 | 22.18 | 4.41 | 73.4 | 1022 |  |  |
| laptop-cpu-haproxy | ray_ort | deepfm | small | deepfm_small | 200 | False | 1 |  | 20 | 252 | False | 1 | 5.04e+04 | 3.59 | 16.47 | 4.28 | 71.4 | 1019 |  |  |
| laptop-cpu-haproxy | ray_ort | deepfm | small | deepfm_small | 200 | False | 1 |  | 50 | 252 | False | 1 | 5.04e+04 | 3.59 | 16.47 | 4.28 | 71.4 | 1019 |  |  |
| laptop-cpu-haproxy | ray_ort | deepfm | small | deepfm_small | 200 | False | 1 |  | 100 | 378 | False | 1 | 7.56e+04 | 20.39 | 80.3 | 4.8 | 80.1 | 1029 |  |  |
| laptop-cpu-haproxy | ray_triton_grpc | deepfm | small | deepfm_small | 200 | False | 1 |  | 20 | 168 | False | 1 | 3.36e+04 | 4.62 | 19.28 | 4.15 | 69.2 | 1030 |  |  |
| laptop-cpu-haproxy | ray_triton_grpc | deepfm | small | deepfm_small | 200 | False | 1 |  | 50 | 252 | False | 1 | 5.04e+04 | 6.46 | 31.87 | 4.63 | 77.1 | 1028 |  |  |
| laptop-cpu-haproxy | ray_triton_grpc | deepfm | small | deepfm_small | 200 | False | 1 |  | 100 | 252 | False | 1 | 5.04e+04 | 6.46 | 31.87 | 4.63 | 77.1 | 1028 |  |  |
| laptop-cpu-haproxy | ray_triton_inproc | deepfm | small | deepfm_small | 200 | False | 1 |  | 20 | 252 | False | 1 | 5.04e+04 | 3.7 | 12.79 | 4.3 | 71.7 | 1068 |  |  |
| laptop-cpu-haproxy | ray_triton_inproc | deepfm | small | deepfm_small | 200 | False | 1 |  | 50 | 252 | False | 1 | 5.04e+04 | 3.7 | 12.79 | 4.3 | 71.7 | 1068 |  |  |
| laptop-cpu-haproxy | ray_triton_inproc | deepfm | small | deepfm_small | 200 | False | 1 |  | 100 | 252 | False | 1 | 5.04e+04 | 3.7 | 12.79 | 4.3 | 71.7 | 1068 |  |  |
| laptop-cpu-tuned | ray_ort | deepfm | small | deepfm_small | 200 | False | 1 |  | 20 | 75 | False | 1 | 1.5e+04 | 9.27 | 17.31 | 4.32 | 72 | 936 |  |  |
| laptop-cpu-tuned | ray_ort | deepfm | small | deepfm_small | 200 | False | 1 |  | 50 | 75 | False | 1 | 1.5e+04 | 9.27 | 17.31 | 4.32 | 72 | 936 |  |  |
| laptop-cpu-tuned | ray_ort | deepfm | small | deepfm_small | 200 | False | 1 |  | 100 | 112 | False | 1 | 2.24e+04 | 12.18 | 66.81 | 4.88 | 81.3 | 947.6 |  |  |
| laptop-cpu-tuned | ray_triton_grpc | deepfm | small | deepfm_small | 200 | False | 1 |  | 20 | 50 | False | 1 | 1e+04 | 10.01 | 16.11 | 4.03 | 67.1 | 937.3 |  |  |
| laptop-cpu-tuned | ray_triton_grpc | deepfm | small | deepfm_small | 200 | False | 1 |  | 50 | 75 | False | 1 | 1.5e+04 | 10.03 | 21.89 | 4.41 | 73.5 | 945.3 |  |  |
| laptop-cpu-tuned | ray_triton_grpc | deepfm | small | deepfm_small | 200 | False | 1 |  | 100 | 112 | False | 1 | 2.24e+04 | 14.49 | 72.73 | 5.02 | 83.7 | 944.4 |  |  |
| laptop-cpu-tuned | ray_triton_inproc | deepfm | small | deepfm_small | 200 | False | 1 |  | 20 | 75 | False | 1 | 1.5e+04 | 9.17 | 17.72 | 4.28 | 71.4 | 978.1 |  |  |
| laptop-cpu-tuned | ray_triton_inproc | deepfm | small | deepfm_small | 200 | False | 1 |  | 50 | 112 | False | 1 | 2.24e+04 | 11.58 | 45.04 | 4.84 | 80.6 | 984 |  |  |
| laptop-cpu-tuned | ray_triton_inproc | deepfm | small | deepfm_small | 200 | False | 1 |  | 100 | 112 | False | 1 | 2.24e+04 | 11.58 | 45.04 | 4.84 | 80.6 | 984 |  |  |

## Cold start

| env | stack | arch | size | model | replicas | batching | start_to_ready_s | rss_idle_mb | image_size_mb | n |
|---|---|---|---|---|---|---|---|---|---|---|
| laptop-cpu | fastapi_ort | deepfm | small | deepfm_small | 1 | False | 2.28 | 72.6 | 725.3 | 1 |
| laptop-cpu | triton | deepfm | small | deepfm_small | 1 | True | 0.82 | 34.5 | 1.688e+04 | 1 |
| laptop-cpu-haproxy | ray_fastapi | deepfm | small | deepfm_small | 1 | False | 9.69 | 988.5 | 726 | 1 |
| laptop-cpu-haproxy | ray_ort | deepfm | small | deepfm_small | 1 | False | 11.62 | 983.4 | 726 | 1 |
| laptop-cpu-haproxy | ray_triton_grpc | deepfm | small | deepfm_small | 1 | False | 9.49 | 987.9 | 726 | 1 |
| laptop-cpu-haproxy | ray_triton_inproc | deepfm | small | deepfm_small | 1 | False | 12.63 | 1042 | 1.758e+04 | 1 |
| laptop-cpu-tuned | ray_ort | deepfm | small | deepfm_small | 1 | False | 9.7 | 904 | 725.3 | 1 |
| laptop-cpu-tuned | ray_triton_grpc | deepfm | small | deepfm_small | 1 | False | 10.1 | 901.3 | 725.3 | 1 |
| laptop-cpu-tuned | ray_triton_inproc | deepfm | small | deepfm_small | 1 | False | 14.7 | 950.1 | 1.758e+04 | 1 |

## Overload

No data.

## Runs

| env | run_id | steps |
|---|---|---|
| laptop-cpu | 20261115-100813-afb93c | 10 |
| laptop-cpu | 20261115-102644-4e6309 | 11 |
| laptop-cpu-haproxy | 20261115-153938-19ec6d | 41 |
| laptop-cpu-tuned | 20261115-112736-1453ec | 24 |
