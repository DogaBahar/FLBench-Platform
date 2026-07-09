# FL Benchmark Platform

A control plane for running standardized federated learning benchmarks across
multiple FL frameworks — **Flower**, **NVIDIA FLARE**, and **FedML** — against
the same datasets and hyperparameters, and comparing them side by side in one
dashboard (accuracy/loss curves, per-client CPU/RAM/compute time, server
aggregation time, TCP connection churn, wall-clock time).

You configure a run once (framework, dataset, number of clients, FL
hyperparameters) and the platform generates that framework's native
server/client scripts on the fly, launches one Docker container per
participant, and normalizes whatever each framework logs into one common
JSON shape.

## Architecture

```
frontend/ (React + Vite, :5173)
  Dashboard.jsx  -- configure & launch a run
  Results.jsx    -- charts for a completed/running run
        |
        | REST (fetch), CORS-enabled
        v
backend/ (Flask API, :5001)
  api/routes.py           -- POST /api/benchmark, GET /api/runs, GET/DELETE /api/runs/<id>
  domain/models.py        -- BenchmarkRun / BenchmarkMetric (Postgres via SQLAlchemy)
  services/tasks.py       -- Celery task: orchestrates one run end-to-end
  services/orchestration.py -- DockerOrchestrator: spins up/tears down containers
  adapters/<fw>_adapter.py  -- one per framework, generates that framework's config/scripts
  datasets/<name>/          -- model.py + dataset.py, shared across all adapters
  telemetry/normalizer.py -- parses each run's metrics.jsonl into a common response shape
        |
        | Celery task queued via Redis, Docker SDK (docker.sock mounted into celery_worker)
        v
Docker (sibling containers on a dedicated bridge network, one per run)
  run_<id>_server, run_<id>_client_1..N   (Flower / FedML)
  run_<id>_server                          (FLARE: runs `nvflare simulator`, hosts all sites in one container)
```

Each run gets its own directory under `$SHARED_RUN_DIR/<run_id>` on the host,
bind-mounted at `/app/workspace` in **every** container belonging to that run.
The adapter writes that framework's server/client scripts there, and those
scripts all append JSON lines to a shared `metrics.jsonl` in the same
directory — that's how server and client containers (different processes,
possibly different containers) end up contributing to one combined
`metrics.jsonl` per run, which `telemetry/normalizer.py` then parses into:

- `global_metrics.losses_distributed` / `metrics_distributed.accuracy` — from `"server_metric"` lines
- `server_logs` — from `"server_sys_metric"` lines (aggregation time, TCP snapshot)
- `client_logs` — from `"client_metric"` lines (CPU/RAM/compute time per client per round)
- `distributions` — from `"distribution"` lines (per-client label counts, for the data-skew chart)
- `global_metrics.wall_clock_time_seconds` — from a single `"wall_clock_time"` line

Round numbers are normalized to be **1-indexed across all three frameworks**
in these logs, even though FedML's own internal round counter is 0-indexed —
adapters are responsible for that shift, not the normalizer.

## Prerequisites

- Docker + Docker Compose
- Node 18+ / npm (for the frontend; it isn't part of `docker-compose.yml`)
- ~a few GB free disk (three separate CPU-only PyTorch images)

## Quickstart (from scratch)

**1. Build the three FL framework images.** These aren't declared in
`docker-compose.yml` — they're built independently and referenced by tag name
(`benchmark-fedml:latest`, `benchmark-flare:latest`, `benchmark-flower:latest`)
from each adapter's `get_docker_commands()`, then launched dynamically by the
orchestrator:

```bash
docker build -f Dockerfile.fedml  -t benchmark-fedml:latest  .
docker build -f Dockerfile.flare  -t benchmark-flare:latest  .
docker build -f Dockerfile.flower -t benchmark-flower:latest .
```

**2. Start the core platform services** (Postgres, Redis, Flask backend, Celery worker):

```bash
docker compose up -d --build
```

`backend/` is bind-mounted into both `backend` and `celery_worker`, so editing
adapter/Python code doesn't require a rebuild — just a restart (see
Troubleshooting). `/var/run/docker.sock` is mounted into `celery_worker` so it
can launch the per-run FL containers as sibling containers on the host's
Docker daemon.

**3. Start the frontend:**

```bash
cd frontend
npm install
npm run dev
```

Open the printed URL (default `http://localhost:5173`).

**4. Launch a run.** From the *Control Panel* tab, pick a framework, dataset,
number of clients, and hyperparameters, then click **Deploy Tasks & Start
Benchmark**. The UI polls `GET /api/runs/<run_id>` every 3s and jumps to the
*Benchmark Results* tab once `status` becomes `COMPLETED`.

Or skip the UI and call the API directly:

```bash
curl -X POST http://localhost:5001/api/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "framework": "flower",
    "clients": 3,
    "config": {
      "federated_settings": {"rounds": 3, "strategy": "FedAvg", "fraction_fit": 1},
      "ml_hyperparameters": {"epochs": 2, "batch_size": 32, "learning_rate": 0.001, "optimizer": "adam"},
      "data_simulation": {"dataset": "cifar100", "samples_per_client": 500, "partition_strategy": "iid"}
    }
  }'
# -> {"message": "Benchmark deployed", "run_id": "flower-cifar100-jul09_2200-a1b2"}

curl http://localhost:5001/api/runs/flower-cifar100-jul09_2200-a1b2
```

`framework` must be one of `flower`, `nvflare`, `fedml` (see `adapters/factory.py`).

## Where things live

- **Run configs, generated scripts, logs, metrics**: `$SHARED_RUN_DIR/<run_id>`
  on the host — default `/tmp/fl_benchmark_runs`, set via the `SHARED_RUN_DIR`
  env var in `docker-compose.yml`. One directory per run; inspecting it
  directly (`config.json`, `metrics.jsonl`, and for FLARE, the full NVFlare
  `simulator_workdir/.../log.txt`) is usually the fastest way to debug a
  failed run — faster than re-running with extra logging.
- **HuggingFace dataset / torch cache**: `$DATA_CACHE_DIR`, default
  `/tmp/fl_benchmark_data` — shared across runs so CIFAR-100/FEMNIST aren't
  re-downloaded every time.
- **Run metadata** (framework, dataset, status, timestamps): Postgres, via
  `backend/domain/models.py`.

## Supported frameworks

| `framework` value | Library                 | Notes |
|---|---|---|
| `flower`  | `flwr~=1.8.0`   | Classic `NumPyClient`/`start_server` API. Pinned below Flower 1.13, where that API is deprecated in favor of the SuperLink/SuperNode architecture the adapter doesn't target. |
| `nvflare` | `nvflare~=2.4.0`| Runs via `nvflare simulator`; **one container** hosts the server and all simulated client sites (unlike the other two, which get one container per participant). |
| `fedml`   | `fedml`         | Cross-silo, horizontal, GRPC backend; one container per participant, IP table built from `RUN_ID` at container start. |

## Supported datasets

| `dataset` value | Task | Source |
|---|---|---|
| `cifar100` | 100-class image classification | `uoft-cs/cifar100` (HuggingFace) |
| `femnist`  | 62-class handwritten character classification | `flwrlabs/femnist` (HuggingFace) |

## Adding a new FL framework

1. **Create `backend/adapters/<name>_adapter.py`** implementing
   `FLFrameworkAdapter` (`backend/adapters/base.py`):
   - `generate_configs(config, output_dir)` — turn the generic config dict
     (`federated_settings`, `ml_hyperparameters`, `data_simulation`) into that
     framework's native server/client scripts and config files, written under
     `output_dir`. Copy the selected dataset's `model.py`/`dataset.py` in
     alongside them (see the existing adapters for the pattern).
   - `get_docker_commands(output_dir, num_clients)` — return
     `[{"name": ..., "image": ..., "command": ..., "env"?: {...}}, ...]`.
     Container `name` becomes `run_<run_id>_<name>` and is how sibling
     containers resolve each other on the run's Docker network — plan your
     server/client addressing around that.
   - `inject_telemetry(output_dir, run_id)` — write any shared
     logging/helper code your generated scripts import. Can be a no-op if you
     inline logging directly in `generate_configs` (FedML/FLARE both do this).
   - Have your generated scripts append JSON lines to
     `/app/workspace/metrics.jsonl` (that's where `/app/workspace` — the per-run
     shared directory — is mounted in every container). `telemetry/normalizer.py`
     only understands these `"type"` values:
     - `"server_metric"` — `{"round", "loss", "accuracy"}` → global accuracy/loss chart
     - `"server_sys_metric"` — `{"round", "agg_time", "tcp_est", "tcp_wait"}` → server telemetry chart
     - `"client_metric"` — `{"client_id", "round", "cpu", "ram", "time", "comm_mb", "iowait"}` → per-client chart
     - `"distribution"` — `{"client_id", "counts": {label: count}}` → data-skew chart
     - `"wall_clock_time"` — `{"time_seconds"}` → total run time card
   - **Round numbers must be 1-indexed** in every line you write, regardless
     of what the underlying framework calls round 0 internally (see FedML's
     `aggregate()`/`train()` for the `+1` shift pattern if the framework you're
     adding is 0-indexed internally).
2. **Register it** in `backend/adapters/factory.py`'s `_adapters` dict.
3. **Write a `Dockerfile.<name>`** in the project root (mirror the existing
   three: `python:3.10-slim`, `git` for HF `datasets`, CPU-only
   torch/torchvision, `psutil`, plus the framework package — pin a version
   deliberately rather than trusting "latest," and verify the exact API you
   generate scripts against actually exists in that version before trusting
   it). Build it as `benchmark-<name>:latest` to match what
   `get_docker_commands()` returns.
4. **Add the option** to the `<select>` in `frontend/src/Dashboard.jsx`.
5. Before considering it done, actually run a benchmark end-to-end and open
   the run's `metrics.jsonl` on disk (see "Where things live" above) — a
   framework integration that merely doesn't crash on round 1 is not the same
   as one that reports correct data for every round.

## Adding a new dataset

Create `backend/datasets/<name>/` with:
- `model.py` — `get_model()` factory, `train(model, loader, epochs, lr, optimizer_name)`,
  `test(model, loader) -> (loss, accuracy)`. Existing datasets use a shared
  `FlexibleCNN(in_channels, num_classes)` architecture — reuse it unless you
  have a reason not to.
- `dataset.py` — `load_data(partition_strategy, samples_per_client, num_clients, client_id, batch_size, alpha, shards_per_client) -> (train_loader, val_loader)`,
  supporting `partition_strategy` values `"iid"`, `"shard"`, `"dirichlet"`.

Then:
- Add the `<option>` to the dataset `<select>` in `frontend/src/Dashboard.jsx`.
- **FLARE-specific**: add an entry to `DATASET_MODEL_SHAPES` in
  `backend/adapters/flare_adapter.py`. NVFlare's persistor instantiates the
  model straight from static JSON config (`in_channels`/`num_classes`) rather
  than calling `get_model()`, so this has to be kept in sync by hand or a
  FLARE run on your new dataset will fail with a shape mismatch between the
  server's persisted model and clients' local models.

## Troubleshooting

- **Backend/adapter code changed, nothing happens**: `backend/` is bind-mounted,
  not baked into the image — Celery doesn't hot-reload modules it already
  imported. Restart: `docker compose up -d --force-recreate backend celery_worker`.
- **`Dockerfile.<framework>` changed** (a new pip package, version bump):
  that *is* baked into the image, so rebuild it:
  `docker build -f Dockerfile.<framework> -t benchmark-<framework>:latest .`
- **A run is stuck in `RUNNING`/silently failed with no clear error in the
  API response**: check `$SHARED_RUN_DIR/<run_id>/metrics.jsonl` directly —
  incomplete rounds usually mean a crash partway through, not a hang. For
  FLARE specifically, also check
  `.../simulator_workdir/simulate_job/log.txt` and
  `.../simulator_workdir/simulate_job/app_site-*/log.txt` — NVFlare's own
  aborts (e.g. a bad task result) show up there, not in `metrics.jsonl`.
- **DB connection errors on backend startup**: `docker-compose.yml`'s `db`
  service has a `pg_isready` healthcheck and `backend`/`celery_worker` wait on
  `condition: service_healthy` — if you've stripped that out, Postgres not
  yet accepting connections on first boot is the usual cause.

## Known limitations

- `DELETE /api/runs/<run_id>` removes the database record, but its on-disk
  cleanup path (`backend/api/routes.py`) is computed relative to the project
  directory rather than from `SHARED_RUN_DIR`, so it doesn't actually match
  where run directories live (`/tmp/fl_benchmark_runs/<run_id>` by default) —
  the filesystem `rmtree` silently no-ops. Run artifacts currently have to be
  cleaned up manually from `$SHARED_RUN_DIR`.
- `tcp_established`/`tcp_time_wait` in `server_logs` are a coarse system-wide
  `psutil.net_connections()` snapshot taken at the moment aggregation
  finishes in that container — a directional signal of connection churn, not
  exact per-connection accounting scoped to FL traffic specifically.
- `iowait_time` is always `0` across all three adapters — the field exists in
  the schema but isn't wired to a real measurement yet.
- FedML prints a non-fatal S3 connectivity diagnostic failure at every server
  startup (dummy AWS credentials). This is expected and harmless — this setup
  uses the GRPC backend, not S3, for actual model transfer.
