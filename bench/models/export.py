"""Build each (arch, size) with seeded random weights, export to ONNX, validate against torch.

Usage: uv run python -m bench.models.export [--sizes small,large] [--archs deepfm,dcnv2]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

from .build import build
from .schema import (
    DENSE_NAME,
    MODELS,
    N_DENSE,
    N_SPARSE,
    OUTPUT_NAME,
    PAYLOAD_VOCAB,
    SIZES,
    SPARSE_NAME,
    ModelSpec,
    load_spec,
    onnx_path,
)

OPSET = 18
MAX_BATCH = 4096  # Triton max_batch_size must not exceed this


def sample_inputs(batch: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dense = rng.lognormal(0.0, 1.0, size=(batch, N_DENSE)).astype(np.float32)
    sparse = rng.integers(0, PAYLOAD_VOCAB, size=(batch, N_SPARSE), dtype=np.int64)
    return dense, sparse


def export(spec: ModelSpec) -> None:
    path = onnx_path(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = build(spec)
    dense, sparse = sample_inputs(4)
    t0 = time.time()
    batch = torch.export.Dim("batch", min=1, max=MAX_BATCH)
    prog = torch.onnx.export(
        model,
        (torch.from_numpy(dense), torch.from_numpy(sparse)),
        input_names=[DENSE_NAME, SPARSE_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_shapes={DENSE_NAME: {0: batch}, SPARSE_NAME: {0: batch}},
        opset_version=OPSET,
        dynamo=True,
    )
    prog.optimize()
    prog.save(str(path))
    onnx.checker.check_model(str(path))
    validate(spec, model)
    size_mb = sum(p.stat().st_size for p in path.parent.glob(f"{spec.name}*")) / 1e6
    print(f"{spec.name}: {size_mb:.1f} MB, rows={spec.total_rows:,}, export {time.time()-t0:.1f}s, ok")


def validate(spec: ModelSpec, model: torch.nn.Module) -> None:
    sess = ort.InferenceSession(str(onnx_path(spec)), providers=["CPUExecutionProvider"])
    for batch in (1, 50, 500):
        dense, sparse = sample_inputs(batch, seed=batch)
        with torch.no_grad():
            ref = model(torch.from_numpy(dense), torch.from_numpy(sparse)).numpy()
        out = sess.run([OUTPUT_NAME], {DENSE_NAME: dense, SPARSE_NAME: sparse})[0]
        assert out.shape == (batch, 1), out.shape
        np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-4)
    # batch-invariance: row i of a 500-batch equals the same row alone
    dense, sparse = sample_inputs(500, seed=500)
    full = sess.run([OUTPUT_NAME], {DENSE_NAME: dense, SPARSE_NAME: sparse})[0]
    one = sess.run([OUTPUT_NAME], {DENSE_NAME: dense[7:8], SPARSE_NAME: sparse[7:8]})[0]
    np.testing.assert_allclose(one, full[7:8], atol=1e-6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default=",".join(SIZES))
    ap.add_argument("--archs", default=",".join(MODELS))
    ap.add_argument("--variants", type=int, default=0,
                    help="also export N family members <arch>_<size>_v00.. (seeded weights)")
    ap.add_argument("--only-variants", action="store_true", help="skip the base models")
    a = ap.parse_args()
    for size in a.sizes.split(","):
        for arch in a.archs.split(","):
            if not a.only_variants:
                export(load_spec(arch, size))
            for v in range(a.variants):
                if not onnx_path(load_spec(arch, size, v)).exists():
                    export(load_spec(arch, size, v))


if __name__ == "__main__":
    main()
