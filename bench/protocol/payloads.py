"""Deterministic request bodies for k6 and the correctness gate.

Writes bench/load/payloads/rows{N}.json: a JSON array of `count` KServe v2 request objects.
Ids are < PAYLOAD_VOCAB so the same files serve every model and size.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import orjson

from bench.models.schema import N_DENSE, N_SPARSE, PAYLOAD_VOCAB

from .kserve_v2 import encode_request

PAYLOAD_DIR = Path(__file__).resolve().parents[1] / "load" / "payloads"
DEFAULT_ROWS = (1, 50, 200, 500)
SEED = 42


def make_payload(rows: int, rng: np.random.Generator) -> bytes:
    dense = rng.lognormal(0.0, 1.0, size=(rows, N_DENSE)).astype(np.float32)
    sparse = rng.integers(0, PAYLOAD_VOCAB, size=(rows, N_SPARSE), dtype=np.int64)
    return encode_request(dense, sparse)


def write_all(rows_list=DEFAULT_ROWS, count: int = 64, out_dir: Path = PAYLOAD_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for rows in rows_list:
        rng = np.random.default_rng(SEED + rows)
        bodies = [orjson.loads(make_payload(rows, rng)) for _ in range(count)]
        path = out_dir / f"rows{rows}.json"
        path.write_bytes(orjson.dumps(bodies))
        print(f"{path.name}: {count} bodies, {path.stat().st_size/1e6:.2f} MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=",".join(map(str, DEFAULT_ROWS)))
    ap.add_argument("--count", type=int, default=64)
    a = ap.parse_args()
    write_all(tuple(int(r) for r in a.rows.split(",")), a.count)


if __name__ == "__main__":
    main()
