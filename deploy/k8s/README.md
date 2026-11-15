# deploy/k8s

Manifests are rendered, not hand-written. Pod sizing (requests == limits) is read from
`config/matrix.yaml` `pod_sizing`; `--cpu` in {1,2,4} is the swept axis.

    kubectl apply -f deploy/k8s/namespace.yaml          # once
    uv run python deploy/k8s/render.py --stack ray_ort --arch deepfm --size small \
        --cpu 2 --replicas 2 --batching 1 --env cpu | kubectl apply -f -
    kubectl -n raybench wait --for=condition=available deploy/fastapi-ort   # or rayservice Ready
    uv run python deploy/k8s/render.py --stack k6 --target http://ray-ort:8000 \
        --arch deepfm --size small --rows 200 --rate 500 --job-name k6-step1 | kubectl apply -f -
    kubectl -n raybench wait --for=condition=complete job/k6-step1 --timeout=10m
    kubectl -n raybench logs job/k6-step1 | grep '^K6_SUMMARY_JSON:'

Python: `from render import render, to_yaml` (list of dicts / YAML). Registry and tag:
`--registry REG/ --tag T`. Delete a stack with `kubectl delete -f -` on the same render.

Names (Service on :8000 each): `fastapi-ort`, `triton` (+8001/8002), `ray-ort`,
`ray-triton-grpc` (+ its backend Deployment/Service `ray-triton-grpc-triton`),
`ray-triton-inproc`; Ray also gets `<name>-metrics` (:8080, head pod).

Labels on every pod: `app`, `raybench/stack`, `raybench/model` (`<arch>_<size>`),
`raybench/component` (server|head|worker|triton|loadgen). E.g.
`kubectl top pod -n raybench -l raybench/stack=ray_ort`.

Node labels required: `raybench/role=serving|loadgen`; GPU env also `raybench/accel=l4`
(GPU pods only: fastapi, Ray worker, Triton) and the `nvidia.com/gpu:NoSchedule` toleration.

Notes: Ray pods have no custom readiness probe (KubeRay injects its own; a probe on :8000
would deadlock RayService bootstrap) - poll `GET /healthz` through the Service or wait on
the RayService status. Ray cost = head (1 cpu) + worker (`--cpu`) [+ Triton pod (`--cpu`)
for ray_triton_grpc]. ORT intra threads default to `max(1, cpu // replicas)`.
