"""Where a stack runs and how k6 reaches it: docker compose on the laptop, kubectl on k8s.

Both backends expose the same four operations the sweep needs:
  up(cfg) -> UpInfo (time to ready, idle memory, image size)
  down()
  run_k6(k6_params) -> summary dict parsed from the K6_SUMMARY_JSON line
  reader() -> a sampler reader for the running stack
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from bench.collect.sampler import Sampler, docker_reader, kubectl_top_reader, nvidia_smi_reader

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "deploy" / "docker" / "compose.yaml"
K6_SCRIPT = ROOT / "bench" / "load" / "infer.js"
PAYLOADS = ROOT / "bench" / "load" / "payloads"
K6_MARK = "K6_SUMMARY_JSON:"


@dataclass
class StackCfg:
    stack: str
    arch: str
    size: str
    batching: bool
    replicas: int
    intra_threads: int
    total_cpus: int
    pod_cpu: int | None = None          # k8s only
    provider: str = "CPUExecutionProvider"
    mem_limit: str = "12g"
    runtime: str = "ort"                # inference runtime: ort | torch

    @property
    def model(self) -> str:
        return f"{self.arch}_{self.size}"

    @property
    def ready_path(self) -> str:
        return "/v2/health/ready" if self.stack == "triton" else "/healthz"

    @property
    def host_port(self) -> int:
        """Compose publishes standalone Triton on 8100 so it cannot collide with Ray on 8000."""
        return 8100 if self.stack == "triton" else 8000


@dataclass
class K6Params:
    model: str
    rows: int
    mode: str            # rate | vus
    rate: int = 0
    vus: int = 0
    duration_s: int = 60
    warmup_s: int = 30
    target: str = "http://localhost:8000"

    def env(self) -> dict[str, str]:
        return {
            "TARGET": self.target, "MODEL": self.model, "ROWS": str(self.rows), "MODE": self.mode,
            "RATE": str(self.rate), "VUS": str(self.vus), "DURATION": f"{self.duration_s}s",
            "WARMUP": f"{self.warmup_s}s",
        }


@dataclass
class UpInfo:
    start_to_ready_s: float
    rss_idle_mb: float | None
    image_size_mb: float | None
    containers: list[str] = field(default_factory=list)


def parse_k6_stdout(text: str) -> dict:
    for line in reversed(text.splitlines()):
        if line.startswith(K6_MARK):
            return json.loads(line[len(K6_MARK):])
    raise RuntimeError("k6 produced no K6_SUMMARY_JSON line:\n" + text[-2000:])


def wait_ready(url: str, timeout_s: float, interval_s: float = 0.2) -> float:
    t0 = time.time()
    with httpx.Client(timeout=2.0) as c:
        while time.time() - t0 < timeout_s:
            try:
                if c.get(url).status_code == 200:
                    return time.time() - t0
            except httpx.HTTPError:
                pass
            time.sleep(interval_s)
    raise TimeoutError(f"{url} not ready after {timeout_s}s")


def _sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, check=True, **kw)


class ComposeBackend:
    def __init__(self, env_cfg: dict, log=print):
        self.serving_cpuset = env_cfg.get("serving_cpuset", "0-5")
        self.loadgen_cpuset = env_cfg.get("loadgen_cpuset", "6-7")
        self.extra_env = {k: str(v) for k, v in env_cfg.get("extra_env", {}).items()}
        self.cfg: StackCfg | None = None
        self.containers: list[str] = []
        self.log = log

    def _compose(self, *args: str) -> list[str]:
        return ["docker", "compose", "-f", str(COMPOSE), "--profile", self.cfg.stack, *args]

    def up(self, cfg: StackCfg) -> UpInfo:
        self.cfg = cfg
        env = {
            "MODEL_ARCH": cfg.arch, "MODEL_SIZE": cfg.size, "REPLICAS": str(cfg.replicas),
            "BATCHING": "1" if cfg.batching else "0", "ORT_INTRA_THREADS": str(cfg.intra_threads),
            "ORT_INTER_THREADS": "1", "ORT_PROVIDER": cfg.provider, "TOTAL_CPUS": str(cfg.total_cpus),
            "SERVING_CPUSET": self.serving_cpuset, "MEM_LIMIT": cfg.mem_limit,
            "ONNX_DIR": "/models/onnx", "MODEL_REPO": "/models/repo", "PORT": "8000",
            "RUNTIME": cfg.runtime,
            # the torch runtime needs torch inside the image (raybench-py has none, on purpose)
            "PY_IMAGE": "raybench-py-torch:local" if cfg.runtime == "torch" else "raybench-py:local",
            **self.extra_env,
        }
        (COMPOSE.parent / ".env").write_text("".join(f"{k}={v}\n" for k, v in env.items()))
        self.down(quiet=True)
        t0 = time.time()
        _sh(self._compose("up", "-d"))
        ready = wait_ready(f"http://localhost:{cfg.host_port}{cfg.ready_path}", timeout_s=900)
        self.containers = _sh(self._compose("ps", "-q")).stdout.split()
        # idle memory once ready
        with Sampler(self.reader(), 0.5) as s:
            time.sleep(5)
        idle = s.summary()
        img = None
        try:
            svc = cfg.stack
            image = _sh(self._compose("config", "--format", "json")).stdout
            name = json.loads(image)["services"][svc]["image"]
            img = int(_sh(["docker", "image", "inspect", "--format", "{{.Size}}", name]).stdout) / 1e6
        except Exception:
            pass
        self.log(f"  up {cfg.stack} {cfg.model} r={cfg.replicas} b={int(cfg.batching)}: "
                 f"ready in {ready:.1f}s, idle rss {idle['rss_peak_mb']} MB, total {time.time()-t0:.0f}s")
        return UpInfo(round(ready, 2), idle["rss_peak_mb"], round(img, 1) if img else None, self.containers)

    def down(self, quiet: bool = False) -> None:
        if self.cfg is None:
            return
        subprocess.run(self._compose("down", "-v", "--remove-orphans", "-t", "10"),
                       capture_output=True, text=True)
        if not quiet:
            self.log(f"  down {self.cfg.stack}")

    def reader(self):
        r = docker_reader(self.containers)
        return nvidia_smi_reader(r) if shutil.which("nvidia-smi") else r

    def limits(self) -> dict:
        return {"limit_cpu": self.cfg.total_cpus, "limit_mem": self.cfg.mem_limit}

    def run_k6(self, p: K6Params, raw_path: Path | None = None) -> dict:
        p.target = f"http://localhost:{self.cfg.host_port}"
        cmd = ["k6", "run", "--quiet", "--no-color", str(K6_SCRIPT)]
        if shutil.which("taskset"):
            cmd = ["taskset", "-c", self.loadgen_cpuset, *cmd]
        env = {**os.environ, **p.env(), "PAYLOAD_DIR": str(PAYLOADS), "K6_NO_USAGE_REPORT": "true"}
        res = subprocess.run(cmd, env=env, text=True, capture_output=True,
                             timeout=p.duration_s + p.warmup_s + 120)
        if raw_path:
            raw_path.write_text(res.stdout + "\n--- stderr ---\n" + res.stderr)
        if res.returncode not in (0, 99):   # 99 = thresholds failed, we set none
            print(res.stderr[-2000:], file=sys.stderr)
            raise RuntimeError(f"k6 exited {res.returncode}")
        return parse_k6_stdout(res.stdout)


class K8sBackend:
    """Renders manifests with deploy/k8s/render.py, applies them, runs k6 as a Job.
    UNTESTED until a cluster exists; the control flow mirrors ComposeBackend."""

    def __init__(self, env_cfg: dict, namespace: str = "raybench", gpu: bool = False,
                 tag: str = "local", registry: str = "", log=print):
        sys.path.insert(0, str(ROOT / "deploy" / "k8s"))
        import render as k8s_render  # type: ignore
        self.render = k8s_render
        self.ns, self.gpu, self.tag, self.registry, self.log = namespace, gpu, tag, registry, log
        self.cfg: StackCfg | None = None
        self.manifest: str = ""

    def _render(self, stack: str, **kw) -> str:
        docs = self.render.render(stack, namespace=self.ns, tag=self.tag, registry=self.registry,
                                  env="gpu" if self.gpu else "cpu", **kw)
        return self.render.to_yaml(docs)

    def up(self, cfg: StackCfg) -> UpInfo:
        self.cfg = cfg
        self.manifest = self._render(cfg.stack, arch=cfg.arch, size=cfg.size, cpu=cfg.pod_cpu,
                                     replicas=cfg.replicas, batching=int(cfg.batching),
                                     intra_threads=cfg.intra_threads)
        t0 = time.time()
        _sh(["kubectl", "apply", "-f", "-"], input=self.manifest)
        svc = self.service_name()
        self._wait_http(f"http://{svc}:8000{cfg.ready_path}", 900)
        ready = time.time() - t0
        with Sampler(self.reader(), 5.0) as s:
            time.sleep(20)
        idle = s.summary()
        self.log(f"  up {cfg.stack} {cfg.model} on k8s: ready in {ready:.1f}s")
        return UpInfo(round(ready, 2), idle["rss_peak_mb"], None, [])

    def service_name(self) -> str:
        return self.cfg.stack.replace("_", "-")

    def _wait_http(self, url: str, timeout_s: float) -> float:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            r = subprocess.run(["kubectl", "run", "-n", self.ns, "readyprobe", "--rm", "-i", "--restart=Never",
                                "--image=curlimages/curl:latest", "--", "curl", "-sf", "-m", "2", url],
                               capture_output=True, text=True)
            if r.returncode == 0:
                return time.time() - t0
            time.sleep(5)
        raise TimeoutError(url)

    def down(self, quiet: bool = False) -> None:
        if self.manifest:
            subprocess.run(["kubectl", "delete", "--wait=true", "--ignore-not-found", "-f", "-"],
                           input=self.manifest, capture_output=True, text=True)
            self.manifest = ""

    def reader(self):
        return kubectl_top_reader(self.ns, f"raybench/stack={self.cfg.stack}")

    def limits(self) -> dict:
        return {"limit_cpu": self.cfg.pod_cpu, "limit_mem": None}

    def run_k6(self, p: K6Params, raw_path: Path | None = None) -> dict:
        name = f"k6-{int(time.time())}"
        p.target = f"http://{self.service_name()}:8000"
        job = self._render("k6", target=p.target, rows=p.rows, rate=p.rate or p.vus, duration=f"{p.duration_s}s",
                           warmup=f"{p.warmup_s}s", mode=p.mode, job_name=name, arch=self.cfg.arch,
                           size=self.cfg.size, cpu=self.cfg.pod_cpu or 1, replicas=1, batching=0)
        _sh(["kubectl", "apply", "-f", "-"], input=job)
        try:
            _sh(["kubectl", "wait", "-n", self.ns, f"--timeout={p.duration_s + p.warmup_s + 300}s",
                 "--for=condition=complete", f"job/{name}"])
            out = _sh(["kubectl", "logs", "-n", self.ns, f"job/{name}"]).stdout
        finally:
            subprocess.run(["kubectl", "delete", "-n", self.ns, "job", name, "--ignore-not-found"],
                           capture_output=True)
        if raw_path:
            raw_path.write_text(out)
        return parse_k6_stdout(out)
