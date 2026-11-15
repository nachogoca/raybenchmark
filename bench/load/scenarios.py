"""Scenario runner: S1, S1_scale, S1_batch, I1, F1, F2, F3, F5, A1, R1, R2 (see docs/BENCHMARK_DESIGN.md).

Every scenario brings stacks up one config at a time, runs k6 steps through `Sweep.step` (so every
step gets the same sampling, validity checks and jsonl row) and marks the config `done` so an
interrupted night resumes. Parameters live in config/matrix.yaml under `scenarios:`; `quick: true`
in a profile shrinks everything to a pipeline check.
"""
from __future__ import annotations

import contextlib
import random
import statistics
import sys
import time

import httpx
import orjson

from bench.load.backends import StackCfg
from bench.models.schema import variant_names

ROOT_PAYLOAD = "bench/load/payloads/rows200.json"


# ---------------------------------------------------------------- helpers
def _threads(sw, stack: str, replicas: int) -> int:
    props = sw.m["stack_props"].get(stack, {})
    reserve = int(props.get("reserve_cores", 0)) if sw.env_cfg.get("tuned") else 0
    return max(1, int(sw.env_cfg["total_cpus"]) // replicas - reserve)


def mk(sw, stack: str, scen: str, variant: str = "", replicas: int = 1, batching: bool = False,
       env: dict | None = None) -> StackCfg:
    return StackCfg(stack=stack, arch="deepfm", size="small", batching=batching, replicas=replicas,
                    intra_threads=_threads(sw, stack, replicas), total_cpus=int(sw.env_cfg["total_cpus"]),
                    scenario=scen, variant=variant, env_extra=dict(env or {}))


def _skip(sw, cfg: StackCfg) -> bool:
    return sw.resume and sw.config_key(cfg) in sw.done_keys


@contextlib.contextmanager
def stack_up(sw, cfg: StackCfg):
    """Bring a config up, write its coldstart row, always tear it down; mark done on success."""
    sw.log(f"[{cfg.scenario}] {cfg.stack} variant={cfg.variant or '-'} replicas={cfg.replicas} "
           f"batching={int(cfg.batching)}")
    try:
        if sw.dry:
            sw.est_s += 60
        else:
            info = sw.backend.up(cfg)
            sw.write(sw.base_row(cfg, step_kind="coldstart", coldstart=info.__dict__))
        yield
        sw.write(sw.base_row(cfg, step_kind="done"))
    except KeyboardInterrupt:
        sw.log("interrupted; tearing down")
        sw.backend.down()
        sys.exit(130)
    except Exception as e:
        sw.log(f"  ERROR {type(e).__name__}: {e}")
        sw.write(sw.base_row(cfg, step_kind="error", error=f"{type(e).__name__}: {e}"))
    finally:
        if not sw.dry:
            sw.backend.down()


def family(n: int = 16) -> list[str]:
    return variant_names("deepfm", "small", n)


def q(sw, seq: list, n: int = 1) -> list:
    """quick mode keeps only the first n entries."""
    return list(seq)[:n] if sw.p.get("quick") else list(seq)


def grid_cfg(sw) -> dict:
    g = dict(sw.m["grid"])
    if sw.p.get("quick"):
        g.update(rps=[50, 150, 400], warmup_s=3, measure_s=6, stop_after_fail=1)
    return g


def grid(sw, cfg, rows, rep=0, rates=None, **kw) -> list[dict]:
    g = grid_cfg(sw)
    out, fails, loosest = [], 0, sw.slos[-1]
    for rate in rates or g["rps"]:
        r = sw.step(cfg, rows, kind="grid", rate=int(rate), long=True, rep=rep,
                    warm_s=g["warmup_s"], meas_s=g["measure_s"], **kw)
        out.append(r)
        fails = 0 if sw.passes(r, loosest) else fails + 1
        if fails >= g["stop_after_fail"]:
            break
    return out


def capacity(rows: list[dict], sw, slo: float) -> int:
    ok = [r["rate"] for r in rows if sw.passes(r, slo)]
    return max(ok) if ok else 0


def search(sw, cfg, rows=200, start=None, **kw) -> dict[int, int]:
    """Coarse capacity search (x1.5 ladder from `start`); the S1 grid is the precise measurement."""
    sw.p["bisect_steps"] = 0 if sw.p.get("quick") else 2
    return sw.search(cfg, rows, start=start or (100 if rows == 1 else 50), factor=1.5, **kw)


def reps(sw, key: str, default: int) -> int:
    return 1 if sw.p.get("quick") else int(sw.m["scenarios"].get(key, {}).get("reps", default))


def sc(sw, name: str) -> dict:
    return sw.m["scenarios"][name]


# ---------------------------------------------------------------- scenarios
def S1(sw) -> None:
    c = sc(sw, "S1")
    stacks = q(sw, c["stacks"], 2)
    n = reps(sw, "S1", 3)
    for rep in range(n):
        order = list(stacks)
        random.Random(1000 + rep).shuffle(order)          # drift (thermals) must not favour a stack
        for stack in order:
            cfg = mk(sw, stack, "S1", variant=f"rep{rep}", batching=False)
            if _skip(sw, cfg):
                continue
            with stack_up(sw, cfg):
                g200 = grid(sw, cfg, 200, rep)
                if rep == 0:
                    grid(sw, cfg, 1, rep, rates=q(sw, c["rows1_rps"], 3))
                    base = capacity(g200, sw, sw.slos[-1])
                    if base and not sw.p.get("quick"):
                        sw.overload(cfg, 200, base)
                    sw.closed_loop(cfg, 200)
    # refine: bisect between last pass and first fail at the loosest SLO, 3 reps
    if sw.p.get("quick") or sw.dry:
        return
    for stack in stacks:
        rows = [r for r in sw.mem if r.get("scenario") == "S1" and r["stack"] == stack
                and r.get("step_kind") == "grid" and r["rows"] == 200]
        if not rows:
            continue
        cfg = mk(sw, stack, "S1", variant="refine")
        if _skip(sw, cfg):
            continue
        with stack_up(sw, cfg):
            for slo in sw.slos:
                passing = [r["rate"] for r in rows if sw.passes(r, slo)]
                failing = [r["rate"] for r in rows if not sw.passes(r, slo)]
                lo = max(passing) if passing else 0
                hi = min([f for f in failing if f > lo] or [0])
                if lo and hi and hi - lo > max(5, lo // 12):
                    mid = (lo + hi) // 2
                    for rep in range(n):
                        sw.step(cfg, 200, kind="refine", rate=mid, long=True, rep=rep, slo_target=slo,
                                warm_s=10, meas_s=30)


def S1_scale(sw) -> None:
    c = sc(sw, "S1_scale")
    for stack in q(sw, c["stacks"], 1):
        for n in q(sw, c["replicas"], 2):
            cfg = mk(sw, stack, "S1_scale", variant=f"n{n}", replicas=n)
            if _skip(sw, cfg):
                continue
            with stack_up(sw, cfg):
                search(sw, cfg, 200)


def S1_batch(sw) -> None:
    c = sc(sw, "S1_batch")
    for rows in q(sw, c["rows"], 1):
        for stack, opts in q(sw, list(c["stacks"].items()), 2):
            for name, (batching, wait_ms) in opts.items():
                cfg = mk(sw, stack, "S1_batch", variant=f"{name}_r{rows}", batching=bool(batching),
                         env={"BATCH_WAIT_MS": wait_ms})
                if _skip(sw, cfg):
                    continue
                with stack_up(sw, cfg):
                    search(sw, cfg, rows)


def I1(sw) -> None:
    c = sc(sw, "I1")
    for name, spec in q(sw, list(c["variants"].items()), 3):
        cfg = mk(sw, spec["stack"], "I1", variant=name, env=spec.get("env"))
        if _skip(sw, cfg):
            continue
        grpc = bool(spec.get("grpc"))
        with stack_up(sw, cfg):
            grid(sw, cfg, 200, 0, rates=q(sw, c["grid_rps"], 3), grpc=grpc)
            search(sw, cfg, 200, grpc=grpc)


def _family_traffic(stack: str, names: list[str], zipf: float, prefix: str = "") -> dict:
    return {"models": ",".join(prefix + n for n in names), "zipf_s": zipf}


def F1(sw) -> None:
    c = sc(sw, "F1")
    zipf = float(c["zipf_s"])
    specs: list[tuple[str, str, int, dict]] = []     # (variant, stack, n_variants, env)
    for per in c["mux_per_replica"]:
        specs.append((f"mux{per}", "ray_family", 16, {"FAMILY_MODE": "mux", "MUX_MODELS_PER_REPLICA": per, "REPLICAS": 2}))
    specs.append(("deployments", "ray_family", 16, {"FAMILY_MODE": "deployments"}))
    specs.append(("triton16", "triton", 16, {"FAMILY_MODE": "mux"}))
    specs.append(("fastapi16", "fastapi_ort", 16, {"FAMILY_MODE": "mux"}))
    for n in c["scale_variants"]:                     # Ray mux with a growing family, cache of 16
        specs.append((f"mux16_v{n}", "ray_family", n, {"FAMILY_MODE": "mux", "MUX_MODELS_PER_REPLICA": 16, "REPLICAS": 2}))
    for variant, stack, n, env in q(sw, specs, 2):
        env = {**env, "VARIANTS": n}
        replicas = int(env.get("REPLICAS", 1))
        cfg = mk(sw, stack, "F1", variant=variant, replicas=replicas, env=env)
        if _skip(sw, cfg):
            continue
        with stack_up(sw, cfg):
            t = _family_traffic(stack, family(n), zipf)
            search(sw, cfg, 200, start=int(c.get("start_rps", 25)), **t)
            sw.step(cfg, 200, kind="family_fixed", rate=int(c["fixed_rps"]), long=True, rep=0,
                    warm_s=10, meas_s=30 if not sw.p.get("quick") else 6, **t)


def _first_cold(url: str, body: bytes, model: str, headers=None) -> tuple[float, float]:
    """(first request ms, median of the next 10 ms) for one model."""
    h = {"content-type": "application/json", "serve_multiplexed_model_id": model, **(headers or {})}
    with httpx.Client(timeout=120) as cl:
        t0 = time.perf_counter()
        r = cl.post(f"{url}/v2/models/{model}/infer", content=body, headers=h)
        first = (time.perf_counter() - t0) * 1e3
        r.raise_for_status()
        steady = []
        for _ in range(10):
            t0 = time.perf_counter()
            cl.post(f"{url}/v2/models/{model}/infer", content=body, headers=h).raise_for_status()
            steady.append((time.perf_counter() - t0) * 1e3)
    return first, statistics.median(steady)


def F2(sw) -> None:
    c = sc(sw, "F2")
    body = orjson.dumps(orjson.loads(open(ROOT_PAYLOAD, "rb").read())[0])
    names = family(16)
    specs = [
        ("ray_mux", "ray_family", {"FAMILY_MODE": "mux", "MUX_MODELS_PER_REPLICA": 4, "VARIANTS": 16}, "http://localhost:8000"),
        ("fastapi_lazy", "fastapi_ort", {"FAMILY_MODE": "mux", "LAZY_LOAD": 1, "VARIANTS": 16}, "http://localhost:8000"),
        ("triton_explicit", "triton", {"FAMILY_MODE": "mux", "VARIANTS": 16,
                                       "TRITON_ARGS": "--model-control-mode=explicit --load-model=" + names[0]},
         "http://localhost:8100"),
    ]
    for variant, stack, env, url in q(sw, specs, 1):
        cfg = mk(sw, stack, "F2", variant=variant, env=env)
        if _skip(sw, cfg) or sw.dry:
            continue
        with stack_up(sw, cfg):
            _first_cold(url, body, names[0])                          # warm the hot model
            trials = 2 if sw.p.get("quick") else int(c["trials"])
            touched = []
            for i in range(trials):
                m = names[8 + i]
                load_ms = None
                if stack == "triton":
                    t0 = time.perf_counter()
                    httpx.post(f"{url}/v2/repository/models/{m}/load", timeout=120).raise_for_status()
                    load_ms = (time.perf_counter() - t0) * 1e3
                first, steady = _first_cold(url, body, m)
                touched.append(m)
                sw.write(sw.base_row(cfg, step_kind="cold_variant", trial=i, model=m, kind="cold",
                                     first_ms=first, steady_ms=steady, load_ms=load_ms))
                sw.log(f"    cold {m}: first {first:.0f} ms (steady {steady:.1f} ms)")
            if variant == "ray_mux":                                  # LRU of 4 is full: m0 was evicted
                first, steady = _first_cold(url, body, names[0])
                sw.write(sw.base_row(cfg, step_kind="cold_variant", trial=trials, model=names[0],
                                     kind="evicted", first_ms=first, steady_ms=steady, load_ms=None))
                sw.log(f"    evicted {names[0]}: first {first:.0f} ms (steady {steady:.1f} ms)")


def F3(sw) -> None:
    c = sc(sw, "F3")
    names = family(16)
    for K in q(sw, c["ks"], 2):
        specs = [
            ("ray_fanout", "ray_family", {"FAMILY_MODE": "fanout", "FANOUT_K": K, "REPLICAS": 2}, ""),
            ("fastapi_fanout", "fastapi_ort", {"FAMILY_MODE": "fanout", "FANOUT_K": K}, ""),
            ("triton_ensemble", "triton", {"FAMILY_MODE": "mux", "ENSEMBLES": 1}, f"fan{K}_" if K > 1 else ""),
        ]
        for variant, stack, env, prefix in q(sw, specs, 2):
            cfg = mk(sw, stack, "F3", variant=f"{variant}_k{K}", replicas=int(env.get("REPLICAS", 1)),
                     env={**env, "VARIANTS": 16})
            if _skip(sw, cfg):
                continue
            with stack_up(sw, cfg):
                t = _family_traffic(stack, names, float(c["zipf_s"]), prefix)
                for rep in range(reps(sw, "F3", 2)):
                    sw.step(cfg, 200, kind="fanout", rate=int(c["rate"]), long=True, rep=rep,
                            warm_s=10, meas_s=30 if not sw.p.get("quick") else 6, k=K, **t)


def F5(sw) -> None:
    c = sc(sw, "F5")
    names = family(16)
    specs = [
        ("ray_pipeline", "ray_family", {"FAMILY_MODE": "pipeline", "REPLICAS": 2}, ""),
        ("fastapi_pipeline", "fastapi_ort", {"FAMILY_MODE": "pipeline"}, ""),
        ("triton_pipeline", "triton", {"FAMILY_MODE": "mux", "ENSEMBLES": 1}, "pipe_"),
    ]
    for variant, stack, env, prefix in q(sw, specs, 2):
        cfg = mk(sw, stack, "F5", variant=variant, replicas=int(env.get("REPLICAS", 1)),
                 env={**env, "VARIANTS": 16})
        if _skip(sw, cfg):
            continue
        with stack_up(sw, cfg):
            t = _family_traffic(stack, names, float(c["zipf_s"]), prefix)
            grid(sw, cfg, 200, 0, rates=q(sw, c["grid_rps"], 3), **t)
            search(sw, cfg, 200, start=int(c["start_rps"]), **t)


def A1(sw) -> None:
    c = sc(sw, "A1")
    base, peak = int(c["base_rps"]), int(c["peak_rps"])
    steps = f"{base}:{c['pre_s']},{peak}:{c['burst_s']},{base}:{c['post_s']}"
    if sw.p.get("quick"):
        steps = f"{base}:5,{peak}:10,{base}:5"
    for name, spec in q(sw, list(c["variants"].items()), 2):
        for rep in range(reps(sw, "A1", 3)):
            cfg = mk(sw, "ray_ort", "A1", variant=f"{name}_rep{rep}", replicas=int(spec.get("replicas", 1)),
                     env=spec["env"])
            if _skip(sw, cfg):
                continue
            with stack_up(sw, cfg):         # fresh stack per rep: autoscaling state must reset
                sw.step(cfg, 200, kind="burst", rate=base, long=True, rep=rep, warm_s=20 if not sw.p.get("quick") else 3,
                        shape="step", steps=steps, policy=name, base_rps=base, peak_rps=peak)


def R1(sw) -> None:
    c = sc(sw, "R1")
    for name, spec in q(sw, list(c["variants"].items()), 2):
        cfg = mk(sw, spec["stack"], "R1", variant=name, batching=bool(spec.get("batching")), env=spec.get("env"))
        if _skip(sw, cfg):
            continue
        with stack_up(sw, cfg):
            caps = search(sw, cfg, 200, start=int(c["start_rps"]))
            base = caps[sw.slos[-1]] or int(c["start_rps"])
            for f in q(sw, c["factors"], 1):
                sw.step(cfg, 200, kind="overload", rate=max(1, int(base * f)), long=True, factor=f,
                        warm_s=10, meas_s=30 if not sw.p.get("quick") else 6, base_capacity=base)


def R2(sw) -> None:
    c = sc(sw, "R2")
    for name, spec in q(sw, list(c["variants"].items()), 1):
        for rep in range(reps(sw, "R2", 3)):
            cfg = mk(sw, spec["stack"], "R2", variant=f"{name}_rep{rep}", replicas=int(spec.get("replicas", 2)),
                     env=spec.get("env"))
            if _skip(sw, cfg):
                continue
            with stack_up(sw, cfg):
                be = sw.backend

                def hook(be=be, spec=spec):
                    if spec.get("kill") == "container":
                        be.kill_container(spec["stack"])
                    else:
                        be.kill_matching(spec["kill"], n=1)
                meas = 20 if sw.p.get("quick") else int(c["measure_s"])
                sw.step(cfg, 200, kind="failure", rate=int(c["rate"]), long=True, rep=rep, warm_s=10,
                        meas_s=meas, hook=None if sw.dry else hook, hook_at_s=meas * 0.35, policy=name)


SCENARIOS = {f.__name__: f for f in (S1, S1_scale, S1_batch, I1, F1, F2, F3, F5, A1, R1, R2)}


def run_scenarios(sw) -> None:
    names = sw.p["scenarios"]
    sw.log(f"run {sw.run_id}: profile={sw.profile_name} env={sw.env} scenarios={names} quick={bool(sw.p.get('quick'))}")
    if not sw.dry:
        import json

        from bench.collect.host import preflight
        from bench.load.sweep import RESULTS
        from bench.versions import collect
        pf = [{"check": a, "ok": b} for a, b, _ in preflight()]
        (RESULTS / "runs" / f"{sw.run_id}.json").write_text(json.dumps({
            "run_id": sw.run_id, "env": sw.env, "profile": sw.p, "matrix": sw.m,
            "versions": collect(), "preflight": pf}, indent=1, default=str))
    for n in names:
        t0 = time.time()
        SCENARIOS[n](sw)
        sw.log(f"== {n} done in {(time.time() - t0) / 60:.1f} min (est. so far {sw.est_s / 3600:.2f} h)")
    if sw.dry:
        sw.log(f"dry run: {sw.total_steps} k6 steps, est. {sw.est_s / 3600:.1f} h "
               "(assumes capacity ~300 req/s; real search lengths vary)")
