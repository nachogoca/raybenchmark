tag := "local"
compose := "docker compose -f deploy/docker/compose.yaml"
# Results environment: macOS uses its own (Docker VM sizing, unpinned k6); results are not comparable with Linux.
env := if os() == "macos" { "mac" } else { "laptop" }

default:
    @just --list

# Sync deps, print the version table, and check docker / k6 / kind. Prints install commands for this OS (Linux or macOS).
setup:
    #!/usr/bin/env bash
    set -u
    os="{{os()}}"
    uv sync
    uv run python -m bench.versions
    fail=0
    if ! docker info >/dev/null 2>&1; then
      echo "MISSING: docker daemon not reachable."
      if [ "$os" = macos ]; then
        echo "  brew install --cask docker   # then open Docker.app and give it 8+ CPUs and 12+ GB (Settings > Resources)"
        echo "  # or: brew install colima docker docker-compose && colima start --cpu 8 --memory 16"
      else
        echo "  install docker-ce + docker-compose-plugin from docs.docker.com/engine/install"
        echo "  sudo usermod -aG docker \$USER && newgrp docker"
      fi
      fail=1
    fi
    if ! command -v k6 >/dev/null; then
      echo "MISSING: k6."
      if [ "$os" = macos ]; then echo "  brew install k6"
      else echo "  curl -LO https://github.com/grafana/k6/releases/download/v2.4.0/k6-v2.4.0-linux-amd64.deb && sudo dpkg -i k6-v2.4.0-linux-amd64.deb"; fi
      fail=1
    fi
    if ! command -v kind >/dev/null; then
      echo "optional (cluster runs only): kind."
      if [ "$os" = macos ]; then echo "  brew install kind"
      else echo "  curl -Lo ~/.local/bin/kind https://kind.sigs.k8s.io/dl/v0.33.0/kind-linux-amd64 && chmod +x ~/.local/bin/kind"; fi
    fi
    if [ "$os" = macos ]; then
      echo "macOS: results go to the 'mac' environment and are not comparable with Linux (no CPU pinning, k6 shares the host)."
      [ "$(uname -m)" = arm64 ] && echo "macOS arm64: Triton images are x86-64 only, so Triton stacks are skipped (RAYBENCH_EMULATE=1 forces them, numbers meaningless)."
    else
      # Profiling tools are optional on Linux: warn, do not fail. `just setup-profiling` installs them.
      for t in perf strace; do
        command -v $t >/dev/null || echo "optional: $t missing (just setup-profiling)"
      done
      [ "$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo 4)" -le 1 ] \
        || echo "optional: kernel.perf_event_paranoid > 1, perf cannot attach (just setup-profiling)"
      docker image inspect raybench-py:prof >/dev/null 2>&1 \
        || echo "optional: profiling image raybench-py:prof not built (just setup-profiling)"
    fi
    exit $fail

# Install profiling tools (Linux only): perf+strace (sudo), perf_event_paranoid=1, py-spy image. Re-runnable.
setup-profiling:
    #!/usr/bin/env bash
    set -eu
    if [ "{{os()}}" != linux ]; then
      echo "Linux only (perf, strace, taskset). On macOS use Instruments, or py-spy inside a container."
      exit 0
    fi
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
    #!/usr/bin/env bash
    set -eu
    pin=""; command -v taskset >/dev/null && pin="taskset -c 6,7"   # Linux only
    $pin uv run python -m bench.profile.probe --conc {{conc}} --dur {{dur}} --rows {{rows}}

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
    [ "{{os()}}" = linux ] || { echo "perf is Linux only."; exit 0; }
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
    uv run python -m bench.load.sweep --profile smoke --env {{env}}

# The full run (~9 h, resume-safe: re-run the same command after an interruption), then the report.
# A monitor runs beside it and logs to bench/results/runs/monitor.log.
overnight: preflight build-serving
    #!/usr/bin/env bash
    set -u
    uv run python -m bench.monitor --watch 60 >/dev/null 2>&1 &
    MON=$!
    trap 'kill $MON 2>/dev/null' EXIT
    # The sweep is resume-safe: if it crashes, restart it (up to 5 times). Exit code 3 = stopped on purpose (low disk).
    for attempt in 1 2 3 4 5; do
        uv run python -m bench.load.sweep --profile full --env {{env}} && break
        code=$?
        [ "$code" -eq 3 ] && { echo "stopped: disk low"; exit 3; }
        echo "sweep exited with code $code (attempt $attempt); resuming in 60 s"
        sleep 60
    done
    uv run python -m bench.report.report --env {{env}}

# Is the run alive? Process, progress, last log line, disk, containers, alerts. Refreshes every 30 s.
monitor:
    uv run python -m bench.monitor --watch 30

# Time estimate without starting anything.
estimate:
    uv run python -m bench.load.sweep --profile full --env {{env}} --dry-run

# bench/report/report.html (self-contained, open in a browser) and REPORT.md.
report:
    uv run python -m bench.report.report --env {{env}}

# Remove leftover Ray session folders (tests clean up after themselves; this catches crashes).
clean-tmp:
    rm -rf /tmp/ray/session_* ~/.rbray/ray/session_* && df -h /tmp | tail -1

test:
    uv run pytest -q

lint:
    uv run ruff check .
