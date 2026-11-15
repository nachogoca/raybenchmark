"""Run the benchmark matrix: for every (stack, model, batching, replicas) bring the stack up,
record cold start, then for every rows-per-request value search for capacity with short k6
steps, confirm it with long repeated steps at each SLO, run the overload steps, and tear down.
Every k6 step is one jsonl row in bench/results/<env>.jsonl; completed configs are skipped on
re-run (resume).

Usage: uv run python -m bench.load.sweep --profile smoke --env laptop-cpu [--stacks a,b] [--batching 0,1] [--runtimes ort,torch]
       [--models deepfm] [--rows 1,200] [--dry-run] [--no-resume]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import orjson
import yaml

from bench.collect.sampler import Sampler
from bench.load.backends import ComposeBackend, K6Params, K8sBackend, StackCfg

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "config" / "matrix.yaml"
RESULTS = ROOT / "bench" / "results"


class Sweep:
    def __init__(self, a: argparse.Namespace):
        self.m = yaml.safe_load(MATRIX.read_text())
        self.profile_name, self.env = a.profile, a.env
        self.p = {**{k: self.m[k] for k in ("reps", "measure_s", "warmup_s")},
                  "closed_loop": False, "overload": True, "confirm": True,
                  "bisect_steps": self.m["ladder"].get("bisect_steps", 0),
                  **self.m["profiles"][a.profile]}
        self.env_cfg = self.m["envs"][self.env]
        self.slos = sorted(self.m["slos_ms"])
        self.run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.dry = a.dry_run
        self.resume = not a.no_resume
        self.stacks = a.stacks.split(",") if a.stacks else self.m["stacks"]
        self.models = a.models.split(",") if a.models else self.p["models"]
        self.rows_list = [int(r) for r in a.rows.split(",")] if a.rows else self.p["rows"]
        batching = getattr(a, "batching", None)
        self.batching_opts = [bool(int(b)) for b in batching.split(",")] if batching else None
        runtimes = getattr(a, "runtimes", None)
        self.runtime_opts = runtimes.split(",") if runtimes else None
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "raw").mkdir(exist_ok=True)
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
            "total_cpus": cfg.total_cpus, "pod_cpu": cfg.pod_cpu, **kw,
        }

    def config_key(self, cfg: StackCfg) -> str:
        key = f"{self.env}|{self.profile_name}|{cfg.stack}|{cfg.model}|{int(cfg.batching)}|{cfg.replicas}|{cfg.pod_cpu}"
        return key if cfg.runtime == "ort" else f"{key}|{cfg.runtime}"  # ort keys unchanged (resume)

    # ---------- matrix expansion ----------
    def configs(self) -> list[StackCfg]:
        out = []
        total = int(self.env_cfg["total_cpus"])
        for stack in self.stacks:
            props = self.m["stack_props"][stack]
            reserve = int(props.get("reserve_cores", 0)) if self.env_cfg.get("tuned") else 0
            for arch in self.models:
                for size in self.p["sizes"]:
                    batch_opts = self.batching_opts or [bool(b) for b in self.p["batching"]]
                    if not props["supports_batching"]:
                        batch_opts = [False]           # fastapi: always off, run once
                    wanted = self.runtime_opts or self.p.get("runtimes", ["ort"])
                    runtimes = [r for r in wanted if r in props.get("runtimes", ["ort"])]
                    for runtime in runtimes:
                        for batching in batch_opts:
                            for replicas in self.p["replicas"]:
                                out.append(StackCfg(
                                    stack=stack, arch=arch, size=size, batching=bool(batching),
                                    replicas=int(replicas),
                                    intra_threads=max(1, total // int(replicas) - reserve),
                                    total_cpus=total,
                                    pod_cpu=None if self.env.startswith("laptop") else total,
                                    provider="CUDAExecutionProvider" if self.env_cfg.get("gpu") else "CPUExecutionProvider",
                                    runtime=runtime,
                                ))
        return out

    # ---------- one k6 step ----------
    def step(self, cfg: StackCfg, rows: int, *, kind: str, mode: str = "rate", rate: int = 0,
             vus: int = 0, long: bool, rep: int = 0, **extra) -> dict:
        warm = self.p["warmup_s"] if long else self.m["search"]["warmup_s"]
        meas = self.p["measure_s"] if long else self.m["search"]["measure_s"]
        self.total_steps += 1
        self.est_s += warm + meas + 8
        row = self.base_row(cfg, rows=rows, step_kind=kind, mode=mode, rate=rate, vus=vus,
                            duration_s=meas, warmup_s=warm, rep=rep, **extra)
        if self.dry:  # pretend capacity is ~300 req/s so the search terminates realistically
            p99 = 5.0 if rate <= 300 else 500.0
            return {**row, "k6": {"err": 0, "dropped_iterations": 0, "ok": 1, "latency_ms": {"p99": p99, "p50": 1.0}, "achieved_rps": rate}}
        params = K6Params(model=cfg.model, rows=rows, mode=mode, rate=rate, vus=vus,
                          duration_s=meas, warmup_s=warm)
        raw = RESULTS / "raw" / f"{self.run_id}_{cfg.stack}_{cfg.model}_r{rows}_b{int(cfg.batching)}_n{cfg.replicas}_{kind}_{rate or vus}_{rep}.txt"
        with Sampler(self.backend.reader(), 1.0) as s:
            t_measure = time.time() + warm
            k6 = self.backend.run_k6(params, raw)
        res = {**s.summary(since=t_measure), **self.backend.limits()}
        row.update(k6=k6, res=res)
        self.write(row)
        lat = k6["latency_ms"]
        self.log(f"    {kind:8s} rows={rows:<3d} {mode}={rate or vus:<6d} ok={k6['ok']:<6d} err={k6['err']:<3d} "
                 f"drop={k6['dropped_iterations']:<4d} p50={lat['p50'] or 0:7.1f} p99={lat['p99'] or 0:7.1f} "
                 f"cpu={res.get('cpu_cores_mean')} rss={res.get('rss_peak_mb')}")
        return row

    @staticmethod
    def passes(row: dict, slo: float) -> bool:
        k = row["k6"]
        p99 = k["latency_ms"]["p99"]
        return k["err"] == 0 and k["dropped_iterations"] == 0 and p99 is not None and p99 <= slo

    # ---------- capacity search ----------
    def search(self, cfg: StackCfg, rows: int) -> dict[int, int]:
        lad = self.m["ladder"]
        rate, factor, loosest = int(lad["start_rps"]), float(lad["factor"]), self.slos[-1]
        rows_seen: list[dict] = []
        last_pass, first_fail = 0, None
        while rate <= lad["max_rps"]:
            r = self.step(cfg, rows, kind="search", rate=rate, long=False)
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
            r = self.step(cfg, rows, kind="search", rate=mid, long=False)
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

    def confirm(self, cfg: StackCfg, rows: int, caps: dict[int, int]) -> dict[int, int]:
        confirmed = {}
        factor = float(self.m["ladder"]["factor"])
        for slo, rate in caps.items():
            if rate <= 0:
                confirmed[slo] = 0
                continue
            for attempt in range(2):
                results = [self.step(cfg, rows, kind="confirm", rate=rate, long=True, rep=rep,
                                     slo_target=slo, attempt=attempt)
                           for rep in range(int(self.p["reps"]))]
                p99s = sorted(r["k6"]["latency_ms"]["p99"] or math.inf for r in results)
                med = p99s[len(p99s) // 2]
                clean = all(r["k6"]["err"] == 0 and r["k6"]["dropped_iterations"] == 0 for r in results)
                if clean and med <= slo:
                    confirmed[slo] = rate
                    break
                rate = max(1, int(rate / math.sqrt(factor)))
            else:
                confirmed[slo] = 0
        return confirmed

    def overload(self, cfg: StackCfg, rows: int, base: int) -> None:
        for f in self.m["overload_factors"]:
            self.step(cfg, rows, kind="overload", rate=max(1, int(base * f)), long=True, factor=f)

    def closed_loop(self, cfg: StackCfg, rows: int) -> None:
        vus = int(self.m["closed_loop_vus_per_replica"]) * cfg.replicas
        self.step(cfg, rows, kind="closed", mode="vus", vus=vus, long=True)

    # ---------- main ----------
    def run(self) -> None:
        cfgs = self.configs()
        todo = [c for c in cfgs if not (self.resume and self.config_key(c) in self.done_keys)]
        self.log(f"run {self.run_id}: profile={self.profile_name} env={self.env} "
                 f"configs={len(cfgs)} todo={len(todo)} rows={self.rows_list} slos={self.slos}")
        if not self.dry:
            from bench.versions import collect
            (RESULTS / "runs" / f"{self.run_id}.json").write_text(json.dumps({
                "run_id": self.run_id, "env": self.env, "profile": self.p, "matrix": self.m,
                "versions": collect()}, indent=1, default=str))
        for i, cfg in enumerate(todo, 1):
            self.log(f"[{i}/{len(todo)}] {cfg.stack} {cfg.model} batching={cfg.batching} replicas={cfg.replicas}")
            try:
                if self.dry:
                    self.est_s += 60
                else:
                    info = self.backend.up(cfg)
                    self.write(self.base_row(cfg, step_kind="coldstart", coldstart=info.__dict__))
                for rows in self.rows_list:
                    caps = self.search(cfg, rows)
                    confirmed = self.confirm(cfg, rows, caps) if self.p["confirm"] else dict(caps)
                    self.log(f"    confirmed rows={rows}: " + ", ".join(f"@{s}ms={c}" for s, c in confirmed.items()))
                    base = confirmed[self.slos[-1]] or caps[self.slos[-1]]
                    if self.p["overload"] and base:
                        self.overload(cfg, rows, base)
                    if self.p["closed_loop"]:
                        self.closed_loop(cfg, rows)
                self.write(self.base_row(cfg, step_kind="done"))
            except KeyboardInterrupt:
                self.log("interrupted; tearing down")
                self.backend.down()
                sys.exit(130)
            except Exception as e:
                self.log(f"  ERROR {type(e).__name__}: {e}")
                self.write(self.base_row(cfg, step_kind="error", error=f"{type(e).__name__}: {e}"))
            finally:
                if not self.dry:
                    self.backend.down()
        if self.dry:
            self.log(f"dry run: {self.total_steps} k6 steps, est. {self.est_s/3600:.1f} h "
                     f"(assumes capacity ~300 req/s; the search step count depends on where capacity lands)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--env", default="laptop-cpu")
    ap.add_argument("--stacks")
    ap.add_argument("--models")
    ap.add_argument("--rows")
    ap.add_argument("--batching", help="override the profile's batching list, e.g. 0 or 0,1")
    ap.add_argument("--runtimes", help="override the profile's runtime list, e.g. ort or ort,torch")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--tag", default="local")
    ap.add_argument("--registry", default="")
    Sweep(ap.parse_args()).run()


if __name__ == "__main__":
    main()
