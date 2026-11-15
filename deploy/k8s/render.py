"""Render raybench Kubernetes manifests (multi-document YAML) for one stack.

stdlib + pyyaml only. Pod sizing comes from config/matrix.yaml `pod_sizing`.
Usage: python deploy/k8s/render.py --stack fastapi_ort --arch deepfm --size small --cpu 2 ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "config" / "matrix.yaml"
TRITON_IMAGE = "nvcr.io/nvidia/tritonserver:26.10-py3"
RAY_VERSION = "2.60.0"  # keep in sync with uv.lock
STACKS = ("fastapi_ort", "ray_ort", "ray_triton_grpc", "ray_triton_inproc", "triton", "k6")
NAMES = {
    "fastapi_ort": "fastapi-ort",
    "ray_ort": "ray-ort",
    "ray_triton_grpc": "ray-triton-grpc",
    "ray_triton_inproc": "ray-triton-inproc",
    "triton": "triton",
}
INIT_RES = ("100m", "128Mi")


def _sizing() -> dict:
    with open(MATRIX) as f:
        return yaml.safe_load(f)["pod_sizing"]


def _mem(component: str, size: str) -> str:
    """Memory string from matrix.yaml, normalised to Mi (1.5Gi -> 1536Mi)."""
    raw = str(_sizing()["memory"][component][size])
    if raw.endswith("Gi"):
        return f"{round(float(raw[:-2]) * 1024)}Mi"
    return raw


def _fixed_cpu(component: str) -> int:
    return int(_sizing()["memory"][component]["cpu"])


def _image(name: str, tag: str, registry: str) -> str:
    prefix = registry.rstrip("/") + "/" if registry else ""
    return f"{prefix}{name}:{tag}"


def _res(cpu: str | int, mem: str, gpu: bool = False) -> dict:
    r = {"requests": {"cpu": str(cpu), "memory": mem}, "limits": {"cpu": str(cpu), "memory": mem}}
    if gpu:
        r["limits"]["nvidia.com/gpu"] = 1  # extended resource: request defaults to limit
    return r


def _env(d: dict) -> list[dict]:
    return [{"name": k, "value": str(v)} for k, v in d.items()]


class Ctx:
    """Per-render parameters shared by the builders."""

    def __init__(self, stack, arch, size, cpu, replicas, batching, env, namespace, tag,
                 registry, intra_threads):
        self.stack, self.arch, self.size = stack, arch, size
        self.cpu, self.replicas, self.batching = int(cpu), int(replicas), int(batching)
        self.gpu = env == "gpu"
        self.ns, self.tag, self.registry = namespace, tag, registry
        self.intra = int(intra_threads) if intra_threads else max(1, self.cpu // self.replicas)
        self.model = f"{arch}_{size}"
        self.sizing = _sizing()

    def img(self, name: str) -> str:
        return _image(name, self.tag, self.registry)

    def labels(self, app: str, component: str = "server") -> dict:
        return {"app": app, "raybench/stack": self.stack, "raybench/model": self.model,
                "raybench/component": component}

    def model_env(self) -> dict:
        return {
            "MODEL_ARCH": self.arch, "MODEL_SIZE": self.size, "ONNX_DIR": "/models/onnx",
            "ORT_INTRA_THREADS": self.intra, "ORT_INTER_THREADS": 1,
            "ORT_PROVIDER": "CUDAExecutionProvider" if self.gpu else "CPUExecutionProvider",
            "REPLICAS": self.replicas, "BATCHING": self.batching, "BATCH_MAX": 64,
            "BATCH_WAIT_MS": 5, "PORT": 8000, "MODEL_REPO": "/models/repo",
        }

    def pod_spec_extras(self, gpu_pod: bool) -> dict:
        sel = {"raybench/role": "serving"}
        extra: dict = {}
        if gpu_pod and self.gpu:
            sel["raybench/accel"] = "l4"
            extra["tolerations"] = [
                {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}]
        extra["nodeSelector"] = sel
        return extra

    def models_init(self) -> dict:
        return {
            "name": "copy-models", "image": self.img("raybench-models"),
            "command": ["sh", "-c", "cp -r /models/onnx/. /onnx-out/"],
            "volumeMounts": [{"name": "onnx", "mountPath": "/onnx-out"}],
            "resources": _res(*INIT_RES),
        }

    def repo_init(self) -> dict:
        cmd = ["python", "-m", "bench.stacks.triton.repo", "--out", "/models/repo",
               "--arch", self.arch, "--size", self.size, "--instances", str(self.replicas),
               "--batching", str(self.batching), "--intra-threads", str(self.intra),
               "--onnx-dir", "/models/onnx"]
        if self.gpu:
            cmd.append("--gpu")
        return {
            "name": "render-repo", "image": self.img("raybench-py"), "command": cmd,
            "env": _env(self.model_env()),
            "volumeMounts": [{"name": "onnx", "mountPath": "/models/onnx"},
                             {"name": "repo", "mountPath": "/models/repo"}],
            "resources": _res(*INIT_RES),
        }


def _vols(*names: str) -> list[dict]:
    return [{"name": n, "emptyDir": {}} for n in names]


def _service(ns: str, name: str, selector: dict, ports: list[tuple[str, int]],
             labels: dict | None = None) -> dict:
    return {
        "apiVersion": "v1", "kind": "Service",
        "metadata": {"name": name, "namespace": ns, "labels": labels or {"app": name}},
        "spec": {"selector": selector,
                 "ports": [{"name": n, "port": p, "targetPort": p} for n, p in ports]},
    }


def _fastapi(c: Ctx) -> list[dict]:
    name = NAMES["fastapi_ort"]
    labels = c.labels(name)
    container = {
        "name": "server", "image": c.img("raybench-py"),
        "command": ["python", "-m", "bench.stacks.fastapi_ort.serve"],
        "env": _env(c.model_env()), "ports": [{"containerPort": 8000, "name": "http"}],
        "readinessProbe": {"httpGet": {"path": "/healthz", "port": 8000},
                           "periodSeconds": 2, "failureThreshold": 60},
        "volumeMounts": [{"name": "onnx", "mountPath": "/models/onnx"}],
        "resources": _res(c.cpu, _mem("fastapi_ort", c.size), gpu=c.gpu),
    }
    dep = {
        "apiVersion": "apps/v1", "kind": "Deployment",
        "metadata": {"name": name, "namespace": c.ns, "labels": {"app": name}},
        "spec": {"replicas": 1, "selector": {"matchLabels": {"app": name}},
                 "template": {"metadata": {"labels": labels},
                              "spec": {"initContainers": [c.models_init()],
                                       "containers": [container], "volumes": _vols("onnx"),
                                       **c.pod_spec_extras(True)}}},
    }
    return [dep, _service(c.ns, name, {"app": name}, [("http", 8000)])]


def _triton_deployment(c: Ctx, name: str) -> list[dict]:
    labels = c.labels(name, "triton")
    container = {
        "name": "triton", "image": TRITON_IMAGE,
        "command": ["tritonserver", "--model-repository=/models/repo", "--allow-metrics=true"],
        "ports": [{"containerPort": 8000, "name": "http"}, {"containerPort": 8001, "name": "grpc"},
                  {"containerPort": 8002, "name": "metrics"}],
        "readinessProbe": {"httpGet": {"path": "/v2/health/ready", "port": 8000},
                           "periodSeconds": 2, "failureThreshold": 90},
        "volumeMounts": [{"name": "repo", "mountPath": "/models/repo"},
                         {"name": "onnx", "mountPath": "/models/onnx"}],
        "resources": _res(c.cpu, _mem("triton", c.size), gpu=c.gpu),
    }
    dep = {
        "apiVersion": "apps/v1", "kind": "Deployment",
        "metadata": {"name": name, "namespace": c.ns, "labels": {"app": name}},
        "spec": {"replicas": 1, "selector": {"matchLabels": {"app": name}},
                 "template": {"metadata": {"labels": labels},
                              "spec": {"initContainers": [c.models_init(), c.repo_init()],
                                       "containers": [container],
                                       "volumes": _vols("onnx", "repo"),
                                       **c.pod_spec_extras(True)}}},
    }
    svc = _service(c.ns, name, {"app": name}, [("http", 8000), ("grpc", 8001), ("metrics", 8002)])
    return [dep, svc]


def _ray(c: Ctx) -> list[dict]:
    name = NAMES[c.stack]
    inproc = c.stack == "ray_triton_inproc"
    grpc = c.stack == "ray_triton_grpc"
    ray_img = c.img("raybench-triton-ray" if inproc else "raybench-py")
    worker_mem = _mem("ray_worker_inproc" if inproc else "ray_worker", c.size)
    head_cpu = _fixed_cpu("ray_head")

    args = {"arch": c.arch, "size": c.size, "replicas": c.replicas, "batching": c.batching,
            "batch_max": 64, "batch_wait_ms": 5, "intra_threads": c.intra,
            "num_cpus": c.cpu / c.replicas}
    extra_docs: list[dict] = []
    if grpc:
        triton_name = f"{name}-triton"
        args["triton_url"] = f"{triton_name}:8001"
        extra_docs = _triton_deployment(c, triton_name)
    serve_cfg = {
        "http_options": {"host": "0.0.0.0", "port": 8000},
        "applications": [{
            "name": "app", "route_prefix": "/",
            "import_path": f"bench.stacks.{c.stack}.app:app_builder", "args": args}],
    }
    env = c.model_env()

    def group(role: str, cpu, mem: str, gpu: bool, start: dict) -> dict:
        labels = c.labels(name, role)
        init = [c.models_init()] + ([c.repo_init()] if inproc else [])
        vols = ["onnx"] + (["repo"] if inproc else [])
        mounts = [{"name": "onnx", "mountPath": "/models/onnx"}]
        if inproc:
            mounts.append({"name": "repo", "mountPath": "/models/repo"})
        container = {
            "name": "ray-" + role, "image": ray_img, "env": _env(env),
            "volumeMounts": mounts, "resources": _res(cpu, mem, gpu=gpu),
        }
        if role == "head":
            container["ports"] = [{"containerPort": 8000, "name": "serve"},
                                  {"containerPort": 8265, "name": "dashboard"},
                                  {"containerPort": 8080, "name": "metrics"}]
        # No custom readiness probe on Ray pods: KubeRay injects its own, and a probe on :8000
        # would deadlock because the Serve app is only deployed after the head is Ready.
        return {"rayStartParams": start,
                "template": {"metadata": {"labels": labels},
                             "spec": {"initContainers": init, "containers": [container],
                                      "volumes": _vols(*vols),
                                      **c.pod_spec_extras(gpu)}}}

    head = group("head", head_cpu, _mem("ray_head", c.size), False,
                 {"dashboard-host": "0.0.0.0", "metrics-export-port": "8080", "num-cpus": "0"})
    worker = group("worker", c.cpu, worker_mem, c.gpu and not grpc,
                   {"metrics-export-port": "8080", "num-cpus": str(c.cpu)})
    worker.update({"groupName": "workers", "replicas": 1, "minReplicas": 1, "maxReplicas": 1})

    rs = {
        "apiVersion": "ray.io/v1", "kind": "RayService",
        "metadata": {"name": name, "namespace": c.ns, "labels": {"app": name}},
        "spec": {
            "serveConfigV2": yaml.safe_dump(serve_cfg, sort_keys=False),
            "rayClusterConfig": {"rayVersion": RAY_VERSION, "headGroupSpec": head,
                                 "workerGroupSpecs": [worker]},
        },
    }
    head_sel = {"app": name, "ray.io/node-type": "head"}
    svc = _service(c.ns, name, head_sel, [("http", 8000)])
    msvc = _service(c.ns, f"{name}-metrics", head_sel, [("metrics", 8080)])
    return [rs, svc, msvc] + extra_docs


def _k6(c: Ctx, target: str, rows: int, rate: int, duration: str, warmup: str, mode: str,
        job_name: str) -> list[dict]:
    labels = {"app": job_name, "raybench/stack": "k6", "raybench/model": c.model,
              "raybench/component": "loadgen"}
    envs = {"TARGET": target, "MODEL": c.model, "ROWS": rows, "RATE": rate,
            "DURATION": duration, "WARMUP": warmup, "MODE": mode}
    container = {
        "name": "k6", "image": c.img("raybench-k6"), "command": ["k6", "run", "/load/infer.js"],
        "env": _env(envs), "resources": _res(_fixed_cpu("k6"), _mem("k6", c.size)),
    }
    job = {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"name": job_name, "namespace": c.ns, "labels": {"app": job_name}},
        "spec": {"backoffLimit": 0, "ttlSecondsAfterFinished": 3600,
                 "template": {"metadata": {"labels": labels},
                              "spec": {"restartPolicy": "Never",
                                       "nodeSelector": {"raybench/role": "loadgen"},
                                       "containers": [container]}}},
    }
    return [job]


def render(stack: str, *, arch: str = "deepfm", size: str = "small", cpu: int = 2,
           replicas: int = 1, batching: int = 0, env: str = "cpu", namespace: str = "raybench",
           tag: str = "local", registry: str = "", intra_threads: int | None = None,
           target: str = "", rows: int = 1, rate: int = 10, duration: str = "60s",
           warmup: str = "30s", mode: str = "rate", job_name: str = "") -> list[dict]:
    """Return the list of manifest dicts for one stack (no Namespace)."""
    if stack not in STACKS:
        raise ValueError(f"unknown stack {stack!r}")
    c = Ctx(stack, arch, size, cpu, replicas, batching, env, namespace, tag, registry,
            intra_threads)
    if stack == "fastapi_ort":
        return _fastapi(c)
    if stack == "triton":
        return _triton_deployment(c, NAMES["triton"])
    if stack == "k6":
        return _k6(c, target, rows, rate, duration, warmup, mode,
                   job_name or f"k6-{c.model}-r{rows}".replace("_", "-"))
    return _ray(c)


def to_yaml(docs: list[dict]) -> str:
    return yaml.safe_dump_all(docs, sort_keys=False)


def _pod_templates(doc: dict) -> list[tuple[dict, str]]:
    """(pod template, human path) for every pod template inside a manifest."""
    kind = doc["kind"]
    if kind in ("Deployment", "Job"):
        return [(doc["spec"]["template"], kind)]
    if kind == "RayService":
        cc = doc["spec"]["rayClusterConfig"]
        out = [(cc["headGroupSpec"]["template"], "head")]
        out += [(w["template"], "worker") for w in cc["workerGroupSpecs"]]
        return out
    return []


def selftest() -> None:
    for env in ("cpu", "gpu"):
        for size in ("small", "large"):
            for cpu in (1, 2, 4):
                for stack in STACKS:
                    kw = {"arch": "deepfm", "size": size, "cpu": cpu, "env": env,
                          "replicas": cpu, "batching": 1}
                    if stack == "k6":
                        kw.update(target="http://x:8000")
                    docs = render(stack, **kw)
                    assert len(list(yaml.safe_load_all(to_yaml(docs)))) == len(docs)
                    found = 0
                    for d in docs:
                        for tpl, where in _pod_templates(d):
                            found += 1
                            lab = tpl["metadata"]["labels"]
                            for k in ("app", "raybench/stack", "raybench/model"):
                                assert k in lab, (stack, where, k)
                            assert lab["raybench/stack"] == stack
                            spec = tpl["spec"]
                            for ct in spec.get("initContainers", []) + spec["containers"]:
                                r = ct["resources"]
                                assert r["requests"]["cpu"] == r["limits"]["cpu"], ct["name"]
                                assert r["requests"]["memory"] == r["limits"]["memory"], ct["name"]
                            assert "raybench/role" in spec["nodeSelector"]
                    assert found, stack
    print("ok")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--stack", choices=STACKS)
    p.add_argument("--arch", default="deepfm", choices=("deepfm", "dcnv2"))
    p.add_argument("--size", default="small", choices=("small", "large"))
    p.add_argument("--cpu", type=int, default=2, choices=(1, 2, 4))
    p.add_argument("--replicas", type=int, default=1)
    p.add_argument("--batching", type=int, default=0, choices=(0, 1))
    p.add_argument("--env", default="cpu", choices=("cpu", "gpu"))
    p.add_argument("--namespace", default="raybench")
    p.add_argument("--tag", default="local")
    p.add_argument("--registry", default="")
    p.add_argument("--intra-threads", type=int, default=None,
                   help="default: max(1, cpu // replicas)")
    p.add_argument("--target", default="")
    p.add_argument("--rows", type=int, default=1, choices=(1, 50, 200, 500))
    p.add_argument("--rate", type=int, default=10)
    p.add_argument("--duration", default="60s")
    p.add_argument("--warmup", default="30s")
    p.add_argument("--mode", default="rate", choices=("rate", "vus"))
    p.add_argument("--job-name", default="")
    a = p.parse_args(argv)
    if a.selftest:
        selftest()
        return 0
    if not a.stack:
        p.error("--stack is required")
    if a.stack == "k6" and not a.target:
        p.error("--target is required for k6")
    kw = {k: v for k, v in vars(a).items() if k not in ("selftest", "stack")}
    sys.stdout.write(to_yaml(render(a.stack, **kw)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
