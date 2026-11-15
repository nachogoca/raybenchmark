# Image raybench-py:prof = raybench-py + py-spy. For profiling only; build with `just setup-profiling`.
# py-spy cannot attach across containers from the host (Yama ptrace_scope=1), so it lives in-image.
FROM raybench-py:local
COPY --from=ghcr.io/astral-sh/uv:python3.12-bookworm-slim /usr/local/bin/uv /usr/local/bin/uv
RUN uv pip install --python /app/.venv/bin/python py-spy
