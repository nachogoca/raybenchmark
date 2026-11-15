"""Render a Triton model repository for one (arch, size) with the SAME ORT settings as
bench/stacks/common/ort_session.py (threads explicit; graph optimization unspecified = ENABLE_ALL,
which is Triton's default for the onnxruntime backend and matches our Python session).

Usage: python -m bench.stacks.triton.repo --out /models/repo --arch deepfm --size small \
         --instances 2 --batching 1 --intra-threads 3 --onnx-dir /models/onnx [--gpu]
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml

from bench.models.schema import DENSE_NAME, N_DENSE, N_SPARSE, OUTPUT_NAME, SPARSE_NAME

_MATRIX = Path(__file__).resolve().parents[3] / "config" / "matrix.yaml"


def triton_batching_defaults() -> dict:
    cfg = yaml.safe_load(_MATRIX.read_text())["batching"]["triton"]
    return {"preferred": list(cfg["preferred_batch_size"]),
            "delay_us": int(cfg["max_queue_delay_us"]),
            "max_batch": int(cfg["max_batch_size"])}


def config_pbtxt(model_name: str, *, instances: int, batching: bool, intra_threads: int,
                 inter_threads: int = 1, gpu: bool = False, preferred=None, delay_us=None,
                 max_batch=None) -> str:
    d = triton_batching_defaults()
    preferred = preferred or d["preferred"]
    delay_us = delay_us if delay_us is not None else d["delay_us"]
    max_batch = max_batch or d["max_batch"]
    kind = "KIND_GPU" if gpu else "KIND_CPU"
    gpus = "\n    gpus: [ 0 ]" if gpu else ""
    dyn = ""
    if batching:
        pref = ", ".join(str(p) for p in preferred)
        dyn = f"""
dynamic_batching {{
  preferred_batch_size: [ {pref} ]
  max_queue_delay_microseconds: {delay_us}
}}"""
    return f'''name: "{model_name}"
backend: "onnxruntime"
max_batch_size: {max_batch}
input [
  {{ name: "{DENSE_NAME}"  data_type: TYPE_FP32  dims: [ {N_DENSE} ] }},
  {{ name: "{SPARSE_NAME}" data_type: TYPE_INT64 dims: [ {N_SPARSE} ] }}
]
output [
  {{ name: "{OUTPUT_NAME}" data_type: TYPE_FP32 dims: [ 1 ] }}
]
instance_group [
  {{
    count: {instances}
    kind: {kind}{gpus}
  }}
]{dyn}
parameters {{ key: "intra_op_thread_count" value: {{ string_value: "{intra_threads}" }} }}
parameters {{ key: "inter_op_thread_count" value: {{ string_value: "{inter_threads}" }} }}
parameters {{ key: "execution_mode" value: {{ string_value: "0" }} }}
'''


def render_repo(out: Path, arch: str, size: str, *, instances: int, batching: bool,
                intra_threads: int, onnx_dir: Path, gpu: bool = False, link: bool = True,
                **kw) -> Path:
    model_name = f"{arch}_{size}"
    src = Path(onnx_dir) / f"{model_name}.onnx"
    if not src.exists():
        raise FileNotFoundError(src)
    mdir = out / model_name
    vdir = mdir / "1"
    vdir.mkdir(parents=True, exist_ok=True)
    dst = vdir / "model.onnx"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link:
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)
    # external-data sidecar (large models may use one)
    for side in Path(onnx_dir).glob(f"{model_name}.onnx.*"):
        tgt = vdir / side.name.replace(model_name + ".onnx", "model.onnx")
        if tgt.exists() or tgt.is_symlink():
            tgt.unlink()
        os.symlink(side.resolve(), tgt) if link else shutil.copy2(side, tgt)
    (mdir / "config.pbtxt").write_text(config_pbtxt(
        model_name, instances=instances, batching=batching, intra_threads=intra_threads,
        gpu=gpu, **kw))
    return mdir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--size", required=True)
    ap.add_argument("--instances", type=int, default=1)
    ap.add_argument("--batching", type=int, default=0)
    ap.add_argument("--intra-threads", type=int, default=1)
    ap.add_argument("--inter-threads", type=int, default=1)
    ap.add_argument("--onnx-dir", default="/models/onnx")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--copy", action="store_true", help="copy instead of symlink")
    a = ap.parse_args()
    mdir = render_repo(Path(a.out), a.arch, a.size, instances=a.instances, batching=bool(a.batching),
                       intra_threads=a.intra_threads, inter_threads=a.inter_threads,
                       onnx_dir=Path(a.onnx_dir), gpu=a.gpu, link=not a.copy)
    print(f"rendered {mdir}")
    print((mdir / "config.pbtxt").read_text())


if __name__ == "__main__":
    main()
