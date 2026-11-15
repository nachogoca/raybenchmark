"""KServe v2 gRPC codec (ModelInferRequest/Response from tritonclient's generated stubs).

Requests may carry tensors as `contents` (what the k6 gRPC script sends) or `raw_input_contents`
(what tritonclient sends); responses always use raw bytes.
"""
from __future__ import annotations

import numpy as np
from tritonclient.grpc import service_pb2 as pb

from bench.models.schema import DENSE_NAME, N_DENSE, N_SPARSE, OUTPUT_NAME, SPARSE_NAME


def decode_infer_request(req: pb.ModelInferRequest) -> tuple[np.ndarray, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for i, t in enumerate(req.inputs):
        shape = list(t.shape)
        if req.raw_input_contents:
            dt = np.float32 if t.datatype == "FP32" else np.int64
            arr = np.frombuffer(req.raw_input_contents[i], dtype=dt)
        elif t.datatype == "FP32":
            arr = np.asarray(t.contents.fp32_contents, dtype=np.float32)
        else:
            arr = np.asarray(t.contents.int64_contents, dtype=np.int64)
        out[t.name] = arr.reshape(shape)
    dense, sparse = out[DENSE_NAME], out[SPARSE_NAME]
    assert dense.shape[1] == N_DENSE and sparse.shape[1] == N_SPARSE
    return dense, sparse


def encode_infer_response(model_name: str, prob: np.ndarray) -> pb.ModelInferResponse:
    prob = np.ascontiguousarray(prob, dtype=np.float32).reshape(-1, 1)
    resp = pb.ModelInferResponse(model_name=model_name)
    o = resp.outputs.add()
    o.name, o.datatype = OUTPUT_NAME, "FP32"
    o.shape.extend(prob.shape)
    resp.raw_output_contents.append(prob.tobytes())
    return resp
