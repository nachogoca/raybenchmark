"""Alias for tritonclient's KServe protobuf module.

tritonclient's generated classes report their module as `grpc_service_pb2`, which is not importable.
Ray Serve's gRPC proxy pickles the request into the replica, so it needs that name to resolve. Put this
directory on PYTHONPATH (see bench.stacks.common.ray_runner.run_local) and the name maps to the real module.
"""
import sys

import tritonclient.grpc.service_pb2 as _pb

sys.modules[__name__] = _pb
