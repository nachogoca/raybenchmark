# Handoff (2026-11-14)

The harness is built and tested as far as this machine allows. Everything that needs Docker, k6, or a cluster is written but unrun, because Docker cannot start.

**What exists in `~/workspace/raybenchmark/`** (git initialized, nothing committed)
- Models: DeepFM and DCNv2 in PyTorch, exported to ONNX with the new dynamo exporter in two sizes, validated against PyTorch at batch 1, 50 and 500. Large models are ~650 MB each.
- One wire protocol for all five stacks (KServe v2 JSON), a seeded payload generator, and one shared ONNX Runtime settings module that the Triton config renderer mirrors.
- The five stacks: FastAPI, Ray Serve with ONNX Runtime and optional server-side batching, Ray Serve to Triton over gRPC, Ray Serve with in-process Triton, and standalone Triton.
- k6 script with warm-up excluded from the summary, a sweep engine (capacity search, confirmation at each SLO with repeats, overload steps, cold start, resume), a 1 Hz resource sampler, analysis to CSV, three charts, and a one-page report builder. The report generated from synthetic data came out at 51 lines.
- Docker images and compose profiles, Kubernetes manifests rendered from the pod sizing table with requests equal to limits, kind scripts, and GKE and EKS cluster scripts. Written by Sonnet agents and reviewed. One real bug was found and fixed: a host port collision between Ray and Triton.

**Verified**: 20 tests pass, including Ray Serve with batching on and off returning the same probabilities as raw ONNX Runtime. Lint is clean.

**Blocked on sudo work (Phase 0)**. Docker 18.09 is from a 2019 repo and fails on kernel 7.0, k6 and kind are missing, and kubectl is 1.16. The exact commands are in README.md under Phase 0. In short:

```bash
# replace Docker (old engine + stale 'cosmic' apt source), then k6 2.4.0, kind 0.33.0, current kubectl
sudo apt-get remove -y docker-ce docker-ce-cli containerd.io && sudo rm -f /etc/apt/sources.list.d/docker*.list
# ... add Docker's official repo as shown in README.md, then:
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
curl -LO https://github.com/grafana/k6/releases/download/v2.4.0/k6-v2.4.0-linux-amd64.deb && sudo dpkg -i k6-v2.4.0-linux-amd64.deb
docker login nvcr.io   # for the Triton image

```

Then run `just setup`, `just build`, `just bench smoke`, and `just report`. The smoke profile is about 30 minutes for all five stacks.

**Decide before the full laptop run.** Dry-run estimates assume capacity lands near 300 req/s. The `core` profile is about 17 hours, `full` about 100 hours, mostly from 3 repeats of 90-second confirmation steps at each of three SLOs. Dropping `reps` to 2 in `config/matrix.yaml` cuts both by about a third.

**Untested until Docker exists**: the two Triton-based Ray stacks, the Triton Dockerfile built on the NGC image, the compose healthchecks, and the Kubernetes backend. Expect to iterate on those during the first smoke run.
