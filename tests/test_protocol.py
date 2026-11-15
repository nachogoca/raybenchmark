import numpy as np
import pytest

from bench.models.schema import N_DENSE, N_SPARSE
from bench.protocol.kserve_v2 import (
    ProtocolError,
    decode_request,
    decode_response,
    encode_request,
    encode_response,
)


def test_request_roundtrip():
    rng = np.random.default_rng(0)
    dense = rng.random((7, N_DENSE), dtype=np.float32)
    sparse = rng.integers(0, 100, (7, N_SPARSE), dtype=np.int64)
    d, s = decode_request(encode_request(dense, sparse))
    assert d.dtype == np.float32 and s.dtype == np.int64
    np.testing.assert_array_equal(d, dense)
    np.testing.assert_array_equal(s, sparse)


def test_response_roundtrip():
    prob = np.linspace(0, 1, 5, dtype=np.float32)
    out = decode_response(encode_response("m", prob))
    assert out.shape == (5, 1)
    np.testing.assert_allclose(out.ravel(), prob)


@pytest.mark.parametrize("bad", [
    b"not json",
    b'{"inputs": []}',
    b'{"inputs": [{"name":"dense","shape":[1,12],"datatype":"FP32","data":[0]*12}]}',
])
def test_bad_requests(bad):
    with pytest.raises(ProtocolError):
        decode_request(bad)


def test_row_mismatch():
    dense = np.zeros((2, N_DENSE), np.float32)
    sparse = np.zeros((3, N_SPARSE), np.int64)
    with pytest.raises(ProtocolError):
        decode_request(encode_request(dense, sparse))
