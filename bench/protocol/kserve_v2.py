"""KServe v2 JSON inference protocol, the one wire format every stack speaks.

Request:  {"inputs":[{"name":"dense","shape":[N,13],"datatype":"FP32","data":[...]},
                     {"name":"sparse","shape":[N,26],"datatype":"INT64","data":[...]}]}
Response: {"model_name":"...","outputs":[{"name":"prob","shape":[N,1],"datatype":"FP32","data":[...]}]}
"""
from __future__ import annotations

import numpy as np
import orjson

from bench.models.schema import DENSE_NAME, N_DENSE, N_SPARSE, OUTPUT_NAME, SPARSE_NAME

_DTYPES = {"FP32": np.float32, "INT64": np.int64}


class ProtocolError(ValueError):
    pass


def decode_request(body: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Return (dense float32 [N,13], sparse int64 [N,26])."""
    try:
        req = orjson.loads(body)
        tensors = {t["name"]: t for t in req["inputs"]}
        dense = _tensor(tensors[DENSE_NAME], "FP32", N_DENSE)
        sparse = _tensor(tensors[SPARSE_NAME], "INT64", N_SPARSE)
    except (KeyError, TypeError, orjson.JSONDecodeError) as e:
        raise ProtocolError(f"bad request: {e!r}") from e
    if dense.shape[0] != sparse.shape[0]:
        raise ProtocolError("dense/sparse row count mismatch")
    return dense, sparse


def _tensor(t: dict, datatype: str, width: int) -> np.ndarray:
    if t.get("datatype") != datatype:
        raise ProtocolError(f"{t.get('name')}: expected {datatype}")
    shape = t["shape"]
    if len(shape) != 2 or shape[1] != width:
        raise ProtocolError(f"{t.get('name')}: expected shape [N,{width}], got {shape}")
    arr = np.asarray(t["data"], dtype=_DTYPES[datatype])
    if arr.size != shape[0] * width:
        raise ProtocolError(f"{t.get('name')}: data length != shape")
    return arr.reshape(shape[0], width)


def encode_request(dense: np.ndarray, sparse: np.ndarray) -> bytes:
    return orjson.dumps({
        "inputs": [
            {"name": DENSE_NAME, "shape": list(dense.shape), "datatype": "FP32",
             "data": dense.astype(np.float32).ravel().tolist()},
            {"name": SPARSE_NAME, "shape": list(sparse.shape), "datatype": "INT64",
             "data": sparse.astype(np.int64).ravel().tolist()},
        ]
    })


def encode_response(model_name: str, prob: np.ndarray) -> bytes:
    prob = np.asarray(prob, dtype=np.float32).reshape(-1, 1)
    return orjson.dumps({
        "model_name": model_name,
        "outputs": [{"name": OUTPUT_NAME, "shape": list(prob.shape), "datatype": "FP32",
                     "data": prob.ravel().tolist()}],
    })


def decode_response(body: bytes) -> np.ndarray:
    resp = orjson.loads(body)
    out = next(o for o in resp["outputs"] if o["name"] == OUTPUT_NAME)
    return np.asarray(out["data"], dtype=np.float32).reshape(out["shape"])
