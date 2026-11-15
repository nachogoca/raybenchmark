"""Triton python-backend models used by the pipeline (F5) and fan-out (F3) ensembles.

`MODEL_PY` files are written verbatim into the model repository. The math mirrors
bench/stacks/common/pipeline.py; tests/test_pipeline_parity.py checks both give equal output.
"""
from __future__ import annotations

PURE_MATH = '''
import numpy as np

_MEAN, _STD, _VOCAB = 1.0, 1.5, 10000
_HASH = np.int64(2654435761)


def featurize(dense, sparse):
    d = ((np.log1p(np.clip(dense, 0.0, None)) - _MEAN) / _STD).astype(np.float32)
    s = ((sparse.astype(np.int64) * _HASH) % _VOCAB).astype(np.int64)
    return d, s


def postprocess(prob):
    p = np.clip(prob.astype(np.float64), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    return (1.0 / (1.0 + np.exp(-(1.1 * logit - 0.05)))).astype(np.float32)


def average(probs):
    return np.mean(np.stack(probs), axis=0).astype(np.float32)
'''

_HEADER = "import triton_python_backend_utils as pb_utils\n" + PURE_MATH

FEATURIZE_PY = _HEADER + '''

class TritonPythonModel:
    def execute(self, requests):
        out = []
        for r in requests:
            dense = pb_utils.get_input_tensor_by_name(r, "dense").as_numpy()
            sparse = pb_utils.get_input_tensor_by_name(r, "sparse").as_numpy()
            d, s = featurize(dense, sparse)
            out.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("dense_o", d), pb_utils.Tensor("sparse_o", s)]))
        return out
'''

POSTPROCESS_PY = _HEADER + '''

class TritonPythonModel:
    def execute(self, requests):
        out = []
        for r in requests:
            p = pb_utils.get_input_tensor_by_name(r, "prob").as_numpy()
            out.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("prob_o", postprocess(p))]))
        return out
'''

AVERAGE_PY = _HEADER + '''

class TritonPythonModel:
    def execute(self, requests):
        out = []
        for r in requests:
            ps, i = [], 0
            while True:
                t = pb_utils.get_input_tensor_by_name(r, f"p{i}")
                if t is None:
                    break
                ps.append(t.as_numpy())
                i += 1
            out.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("prob_o", average(ps))]))
        return out
'''


def _io(name: str, dt: str, dim: int) -> str:
    return f'{{ name: "{name}" data_type: {dt} dims: [ {dim} ] }}'


def py_config(name: str, inputs: list[tuple[str, str, int]], outputs: list[tuple[str, str, int]],
              max_batch: int = 1024) -> str:
    ins = ",\n  ".join(_io(*i) for i in inputs)
    outs = ",\n  ".join(_io(*o) for o in outputs)
    return (f'name: "{name}"\nbackend: "python"\nmax_batch_size: {max_batch}\n'
            f'input [\n  {ins}\n]\noutput [\n  {outs}\n]\n'
            'instance_group [ { count: 1 kind: KIND_CPU } ]\n')


def ensemble_config(name: str, steps: list[dict], max_batch: int = 1024) -> str:
    """steps: [{model, inputs: {model_input: tensor}, outputs: {model_output: tensor}}]."""
    body = []
    for s in steps:
        im = "".join(f'\n      input_map {{ key: "{k}" value: "{v}" }}' for k, v in s["inputs"].items())
        om = "".join(f'\n      output_map {{ key: "{k}" value: "{v}" }}' for k, v in s["outputs"].items())
        body.append(f'    {{\n      model_name: "{s["model"]}"\n      model_version: -1{im}{om}\n    }}')
    return (f'name: "{name}"\nplatform: "ensemble"\nmax_batch_size: {max_batch}\n'
            'input [\n  { name: "dense" data_type: TYPE_FP32 dims: [ 13 ] },\n'
            '  { name: "sparse" data_type: TYPE_INT64 dims: [ 26 ] }\n]\n'
            'output [\n  { name: "prob" data_type: TYPE_FP32 dims: [ 1 ] }\n]\n'
            'ensemble_scheduling {\n  step [\n' + ",\n".join(body) + "\n  ]\n}\n")


def pipeline_ensemble(name: str, model: str) -> str:
    return ensemble_config(name, [
        {"model": "featurize", "inputs": {"dense": "dense", "sparse": "sparse"},
         "outputs": {"dense_o": "d1", "sparse_o": "s1"}},
        {"model": model, "inputs": {"dense": "d1", "sparse": "s1"}, "outputs": {"prob": "p1"}},
        {"model": "postprocess", "inputs": {"prob": "p1"}, "outputs": {"prob_o": "prob"}},
    ])


def fanout_ensemble(name: str, members: list[str]) -> str:
    steps = [{"model": m, "inputs": {"dense": "dense", "sparse": "sparse"}, "outputs": {"prob": f"p{i}"}}
             for i, m in enumerate(members)]
    steps.append({"model": f"average{len(members)}", "inputs": {f"p{i}": f"p{i}" for i in range(len(members))},
                  "outputs": {"prob_o": "prob"}})
    return ensemble_config(name, steps)


def write_support_models(out, ks=(4, 8)) -> None:
    """featurize, postprocess, average{K} python-backend models."""
    from pathlib import Path
    out = Path(out)
    specs = {
        "featurize": (FEATURIZE_PY, py_config("featurize",
            [("dense", "TYPE_FP32", 13), ("sparse", "TYPE_INT64", 26)],
            [("dense_o", "TYPE_FP32", 13), ("sparse_o", "TYPE_INT64", 26)])),
        "postprocess": (POSTPROCESS_PY, py_config("postprocess",
            [("prob", "TYPE_FP32", 1)], [("prob_o", "TYPE_FP32", 1)])),
    }
    for k in ks:
        specs[f"average{k}"] = (AVERAGE_PY, py_config(
            f"average{k}", [(f"p{i}", "TYPE_FP32", 1) for i in range(k)], [("prob_o", "TYPE_FP32", 1)]))
    for name, (code, cfg) in specs.items():
        (out / name / "1").mkdir(parents=True, exist_ok=True)
        (out / name / "1" / "model.py").write_text(code)
        (out / name / "config.pbtxt").write_text(cfg)
