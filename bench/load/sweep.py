"""Benchmark driver: owns the run (ids, resume, jsonl rows) and the one k6 step primitive.

The scenarios themselves (what to run, in which order) live in bench/load/scenarios.py; the
parameters in config/matrix.yaml. Every k6 step is one jsonl row in bench/results/<env>.jsonl;
configs that finished are skipped on re-run (resume).

Usage: uv run python -m bench.load.sweep --profile full|smoke [--env laptop] [--scenarios S1,A1] [--dry-run]
"""
from __future__ import annotations

import argparse
import contextlib
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import orjson
import yaml

from bench.collect.host import HostWatch
from bench.collect.prom import PromSampler
from bench.collect.sampler import Sampler
from bench.load.backends import ComposeBackend, K6Params, K8sBackend, StackCfg

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "config" / "matrix.yaml"
RESULTS = ROOT / "bench" / "results"


class Sweep:
    def __init__(self, a: argparse.Namespace):
        self.m = yaml.safe_load(MATRIX.read_text())
        self.profile_name, self.env = a.profile, a.env
        self.p = {"bisect_steps": self.m["ladder"].get("bisect_steps", 0), **self.m["profiles"][a.profile]}
        if a.scenarios:
            self.p["scenarios"] = a.scenarios.split(",")
        self.env_cfg = self.m["envs"][self.env]
        self.slos = sorted(self.m["slos_ms"])
        self.run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.dry = a.dry_run
        self.resume = not a.no_resume
        (RESULTS / "raw").mkdir(parents=True, exist_ok=True)
        (RESULTS / "runs").mkdir(exist_ok=True)
        self.out = RESULTS / f"{self.env}.jsonl"
        self.done_keys = self._load_done()
        self.log_file = RESULTS / "runs" / f"{self.run_id}.log"
        if self.env.startswith("laptop"):
            self.backend = ComposeBackend(self.env_cfg, log=self.log)
        else:
            self.backend = K8sBackend(self.env_cfg, gpu=bool(self.env_cfg.get("gpu")),
                                      tag=a.tag, registry=a.registry, log=self.log)
        self.total_steps = 0
        self.est_s = 0.0
        self.mem: list[dict] = []     # rows of this run, for scenario logic (refine, base rates)

    # ---------- bookkeeping ----------
    def log(self, msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if not self.dry:
            with self.log_file.open("a") as f:
                f.write(line + "\n")

    def _load_done(self) -> set[str]:
        keys = set()
        if self.out.exists():
            for line in self.out.read_text().splitlines():
                with contextlib.suppress(Exception):
                    r = orjson.loads(line)
                    if r.get("step_kind") == "done":
                        keys.add(r["config_key"])
        return keys

    def write(self, row: dict) -> None:
        if self.dry:
            return
        with self.out.open("ab") as f:
            f.write(orjson.dumps(row) + b"\n")

    def base_row(self, cfg: StackCfg, **kw) -> dict:
        return {
            "run_id": self.run_id, "ts": time.time(), "env": self.env, "profile": self.profile_name,
            "config_key": self.config_key(cfg), "stack": cfg.stack, "arch": cfg.arch,
            "size": cfg.size, "model": cfg.model, "runtime": cfg.runtime, "batching": cfg.batching,
            "replicas": cfg.replicas, "intra_threads": cfg.intra_threads,
            "total_cpus": cfg.total_cpus, "pod_cpu": cfg.pod_cpu,
            "scenario": cfg.scenario, "variant": cfg.variant, **kw,
        }

    def config_key(self, cfg: StackCfg) -> str:
        return (f"{self.env}|{cfg.scenario}|{cfg.stack}|{cfg.variant}|{cfg.replicas}|{int(cfg.batching)}")

    # ---------- one k6 step ----------
    def step(self, cfg: StackCfg, rows: int, *, kind: str, mode: str = "rate", rate: int = 0,
             vus: int = 0, long: bool, rep: int = 0, warm_s: int | None = None,
             meas_s: int | None = None, models: str = "", zipf_s: float = 0.0, shape: str = "constant",
             steps: str = "", grpc: bool = False, hook=None, hook_at_s: float = 0.0, **extra) -> dict:
        warm = warm_s if warm_s is not None else (self.m["warmup_s"] if long else self.m["search"]["warmup_s"])
        meas = meas_s if meas_s is not None else (self.m["measure_s"] if long else self.m["search"]["measure_s"])
        if self.p.get("quick"):                     # pipeline check: never spend more than a few seconds per step
            warm, meas = min(warm, 3), min(meas, 8)
        if shape == "step":
            meas = sum(int(x.split(":")[1]) for x in steps.split(","))
        self.total_steps += 1
        self.est_s += warm + meas + 8
        row = self.base_row(cfg, rows=rows, step_kind=kind, mode=mode, rate=rate, vus=vus,
                            duration_s=meas, warmup_s=warm, rep=rep, shape=shape, steps=steps,
                            n_models=len(models.split(",")) if models else 1, zipf_s=zipf_s,
                            protocol="grpc" if grpc else "http", **extra)
        if self.dry:  # pretend capacity is ~300 req/s so the search terminates realistically
            p99 = 5.0 if rate <= 300 else 500.0
            row = {**row, "valid": True, "k6": {"err": 0, "dropped_iterations": 0, "ok": 1, "latency_ms": {"p99": p99, "p50": 1.0}, "achieved_rps": rate}}
            self.mem.append(row)
            return row
        params = K6Params(model=cfg.model, rows=rows, mode=mode, rate=rate, vus=vus,
                          duration_s=meas, warmup_s=warm, models=models, zipf_s=zipf_s,
                          shape=shape, steps=steps, grpc=grpc)
        tag = f"{cfg.scenario or 'c'}_{cfg.stack}_{cfg.variant or cfg.model}_r{rows}_n{cfg.replicas}_{kind}_{rate or vus}_{rep}"
        raw = RESULTS / "raw" / f"{self.run_id}_{tag}.txt"
        hw = HostWatch(self.env_cfg.get("serving_cpuset", "0-5"))
        stop = threading.Event()

        def watch() -> None:
            while not stop.is_set():
                hw.sample()
                stop.wait(1.0)
        threading.Thread(target=watch, daemon=True).start()
        restarts0 = self.backend.restart_count() if hasattr(self.backend, "restart_count") else 0
        t_start = time.time()
        t_measure = t_start + warm
        if hook is not None:
            row["hook_at_unix"] = t_measure + hook_at_s
            threading.Timer(warm + hook_at_s, hook).start()
        try:
            with Sampler(self.backend.reader(), 1.0) as s, PromSampler(cfg.prom_urls, 1.0) as pr:
                k6 = self.backend.run_k6(params, raw)
        finally:
            stop.set()
        res = {**s.summary(since=t_measure), **self.backend.limits()}
        row.update(k6=k6, res=res, prom=pr.summary(since=t_measure), host=hw.summary(),
                   t_measure_unix=t_measure)
        reasons = []
        if k6.get("k6_cpu_frac", 0) > 0.8:
            reasons.append(f"loadgen_cpu {k6['k6_cpu_frac']:.2f} > 0.8")
        if row["host"].get("throttled"):
            reasons.append("cpu throttled")
        if hook is None and hasattr(self.backend, "restart_count") and self.backend.restart_count() != restarts0:
            reasons.append("container restarted")
        if k6.get("dist_error"):
            reasons.append("dist parse failed")
        row["valid"], row["invalid_reasons"] = not reasons, reasons
        self.write(row)
        self.mem.append(row)
        lat = k6["latency_ms"]
        self.log(f"    {kind:8s} rows={rows:<3d} {mode}={rate or vus:<6d} ok={k6['ok']:<6d} err={k6['err']:<3d} "
                 f"drop={k6['dropped_iterations']:<4d} p50={lat['p50'] or 0:7.1f} p99={lat['p99'] or 0:7.1f} "
                 f"cpu={res.get('cpu_cores_mean')} rss={res.get('rss_peak_mb')}"
                 + ("" if row["valid"] else f"  INVALID {reasons}"))
        return row

    @staticmethod
    def passes(row: dict, slo: float) -> bool:
        """MST rule: no errors, <=0.1 % dropped arrivals, p99 <= SLO."""
        k = row["k6"]
        p99 = k["latency_ms"]["p99"]
        total = max(1, k["ok"] + k["err"] + k["dropped_iterations"])
        return (k["err"] == 0 and k["dropped_iterations"] <= 0.001 * total
                and p99 is not None and p99 <= slo)

    # ---------- capacity search ----------
    def search(self, cfg: StackCfg, rows: int, start: int | None = None, factor: float | None = None,
               **kw) -> dict[int, int]:
        lad = self.m["ladder"]
        rate, factor, loosest = int(start or lad["start_rps"]), float(factor or lad["factor"]), self.slos[-1]
        rows_seen: list[dict] = []
        last_pass, first_fail = 0, None
        while rate <= lad["max_rps"]:
            r = self.step(cfg, rows, kind="search", rate=rate, long=False, **kw)
            rows_seen.append(r)
            if self.passes(r, loosest):
                last_pass = rate
                rate = max(rate + 1, int(round(rate * factor)))
            else:
                first_fail = rate
                break
        for _ in range(int(self.p["bisect_steps"])):
            if first_fail is None or first_fail - last_pass < max(2, last_pass // 10):
                break
            mid = (last_pass + first_fail) // 2
            r = self.step(cfg, rows, kind="search", rate=mid, long=False, **kw)
            rows_seen.append(r)
            if self.passes(r, loosest):
                last_pass = mid
            else:
                first_fail = mid
        caps = {}
        for slo in self.slos:
            ok = [r["rate"] for r in rows_seen if self.passes(r, slo)]
            caps[slo] = max(ok) if ok else 0
        self.log(f"    search done rows={rows}: " + ", ".join(f"@{s}ms={c}" for s, c in caps.items()))
        return caps

    def overload(self, cfg: StackCfg, rows: int, base: int) -> None:
        for f in self.m["overload_factors"]:
            self.step(cfg, rows, kind="overload", rate=max(1, int(base * f)), long=True, factor=f)

    def closed_loop(self, cfg: StackCfg, rows: int) -> None:
        vus = int(self.m["closed_loop_vus_per_replica"]) * cfg.replicas
        self.step(cfg, rows, kind="closed", mode="vus", vus=vus, long=True)

    # ---------- main ----------
    def run(self) -> None:
        from bench.load.scenarios import run_scenarios
        run_scenarios(self)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--env", default="laptop")
    ap.add_argument("--scenarios", help="override the profile's scenario list, e.g. S1,A1")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--tag", default="local")
    ap.add_argument("--registry", default="")
    Sweep(ap.parse_args()).run()


if __name__ == "__main__":
    main()
