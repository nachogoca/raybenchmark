tag := "local"
compose := "docker compose -f deploy/docker/compose.yaml"

default:
    @just --list

# Sync deps, print version table, fail loudly if docker daemon / k6 / kind are missing.
setup:
    #!/usr/bin/env bash
    set -u
    uv sync
    uv run python -m bench.versions
    fail=0
    if ! docker info >/dev/null 2>&1; then
      echo "MISSING: docker daemon not reachable. Phase-0 fix:"
      echo "  sudo apt-get remove -y docker-ce docker-ce-cli containerd.io  # old 'cosmic' repo build"
      echo "  install docker-ce + docker-compose-plugin from docs.docker.com/engine/install"
      echo "  sudo usermod -aG docker \$USER && newgrp docker"
      fail=1
    fi
    if ! command -v k6 >/dev/null; then
      echo "MISSING: k6. Phase-0 fix:"
      echo "  curl -LO https://github.com/grafana/k6/releases/download/v2.4.0/k6-v2.4.0-linux-amd64.deb && sudo dpkg -i k6-v2.4.0-linux-amd64.deb"
      fail=1
    fi
    if ! command -v kind >/dev/null; then
      echo "MISSING: kind. Phase-0 fix:"
      echo "  curl -Lo ~/.local/bin/kind https://kind.sigs.k8s.io/dl/v0.33.0/kind-linux-amd64 && chmod +x ~/.local/bin/kind"
      fail=1
    fi
    # Profiling tools are optional: warn, do not fail. `just setup-profiling` installs them.
    for t in perf strace; do
      command -v $t >/dev/null || echo "optional: $t missing (just setup-profiling)"
    done
    [ "$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo 4)" -le 1 ] \
      || echo "optional: kernel.perf_event_paranoid > 1, perf cannot attach (just setup-profiling)"
    docker image inspect raybench-py:prof >/dev/null 2>&1 \
      || echo "optional: profiling image raybench-py:prof not built (just setup-profiling)"
    exit $fail

# Install profiling tools: perf+strace (sudo, interactive), perf_event_paranoid=1, py-spy image. Re-runnable.
setup-profiling:
    #!/usr/bin/env bash
    set -eu
    if ! command -v perf >/dev/null; then
      sudo apt-get install -y linux-tools-common "linux-tools-$(uname -r)" || sudo apt-get install -y linux-tools-generic
    fi
    command -v strace >/dev/null || sudo apt-get install -y strace
    if [ "$(cat /proc/sys/kernel/perf_event_paranoid)" -gt 1 ]; then
      sudo sysctl -w kernel.perf_event_paranoid=1
      echo "kernel.perf_event_paranoid=1" | sudo tee /etc/sysctl.d/99-raybench-perf.conf >/dev/null
    fi
    docker build -f deploy/docker/Dockerfile.profile -t raybench-py:prof .
    echo "profiling ready: perf $(perf --version | cut -d' ' -f3), py-spy image built"

# Bring a stack up for profiling (as host user, live bench/ mounted). E.g. just profile-up ray_ort 0 4
profile-up stack batching="0" intra="4":
    BATCHING={{batching}} ORT_INTRA_THREADS={{intra}} HOST_UID=$(id -u) HOST_GID=$(id -g) \
      docker compose -f deploy/docker/compose.yaml -f deploy/docker/compose.profile.yaml --profile {{stack}} up -d --wait

# Closed-loop latency probe pinned off the serving cores. conc=1 isolates per-request overhead.
probe conc="1" dur="12" rows="200":
    taskset -c 6,7 uv run python -m bench.profile.probe --conc {{conc}} --dur {{dur}} --rows {{rows}}

# Per-thread CPU inside a running container (run while `just probe` is going).
profile-threads container="raybench-ray_ort-1" secs="10":
    docker exec -w /app {{container}} python -m bench.profile.threads --secs {{secs}}

# Syscalls + context switches per request (pass the rps that `just probe` reports).
profile-syscalls rps container="raybench-ray_ort-1" secs="6":
    docker exec -w /app {{container}} python -m bench.profile.syscalls --rps {{rps}} --secs {{secs}}

# Native CPU profile of one process by DSO (proxy by default). Needs `just profile-up` + `just probe`.
perf-top proc="ray::ProxyActor" secs="10":
    #!/usr/bin/env bash
    set -eu
    pid=$(pgrep -f "^{{proc}}" | head -1)
    perf record -q -F 999 -g -o /tmp/raybench-perf.data -p "$pid" -- sleep {{secs}}
    perf report -i /tmp/raybench-perf.data --symfs "/proc/$pid/root" --no-children --sort dso -g none --stdio 2>/dev/null \
      | grep -v '^#' | grep -v '^$' | head -15

# Base DeepFM/DCNv2 models plus the 64-member DeepFM family used by the multiplexing scenarios.
models:
    uv run python -m bench.models.export
    uv run python -m bench.models.export --sizes small --archs deepfm --variants 64 --only-variants

payloads:
    uv run python -m bench.protocol.payloads

# Build all four images from the repo root.
build:
    docker build -f deploy/docker/Dockerfile.python         -t raybench-py:{{tag}} .
    docker build -f deploy/docker/Dockerfile.triton-ray -t raybench-triton-ray:{{tag}} .
    docker build -f deploy/docker/Dockerfile.models     -t raybench-models:{{tag}} .
    docker build -f deploy/docker/Dockerfile.k6         -t raybench-k6:{{tag}} .

# stack: fastapi_ort | ray_ort | ray_fastapi | ray_family | ray_triton_grpc | ray_triton_inproc | triton
# Uses deploy/docker/.env (copy .env.example). Standalone triton listens on host port 8100.
up stack:
    {{compose}} --profile {{stack}} up -d --wait

down stack:
    {{compose}} --profile {{stack}} down -v

logs stack:
    {{compose}} --profile {{stack}} logs -f --tail=100

# Host checks that make results noisy (CPU governor, free RAM, stray containers).
preflight:
    uv run python -c "from bench.collect.host import preflight; [print(('ok  ' if ok else 'FIX ') + c + ('' if ok else '   -> ' + h)) for c, ok, h in preflight()]"

# Servers run from images with a frozen copy of the code: rebuild after any code change.
build-serving:
    docker build -q -f deploy/docker/Dockerfile.python -t raybench-py:{{tag}} .
    docker build -q -f deploy/docker/Dockerfile.triton-ray -t raybench-triton-ray:{{tag}} .

# Every scenario shrunk to a pipeline check (~25 min). Writes bench/results/laptop.jsonl.
smoke: build-serving
    uv run python -m bench.load.sweep --profile smoke

# The full run (~9 h, resume-safe: re-run the same command after an interruption), then the report.
overnight: preflight build-serving
    uv run python -m bench.load.sweep --profile full
    uv run python -m bench.report.report

# Time estimate without starting anything.
estimate:
    uv run python -m bench.load.sweep --profile full --dry-run

# bench/report/report.html (self-contained, open in a browser) and REPORT.md.
report:
    uv run python -m bench.report.report

# Remove leftover Ray session folders (tests clean up after themselves; this catches crashes).
clean-tmp:
    rm -rf /tmp/ray/session_* ~/.rbray/ray/session_* && df -h /tmp | tail -1

test:
    uv run pytest -q

lint:
    uv run ruff check .
