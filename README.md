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
  api/routes.py           -- POST /api/benchmark (Pydantic-validated), GET /api/runs,
                             GET/DELETE /api/runs/<id>, GET /api/health
  domain/schemas.py       -- Pydantic request schema, validated before a run is ever queued
  domain/models.py        -- BenchmarkRun / BenchmarkMetric (Postgres via SQLAlchemy)
  migrations/             -- Alembic migrations (schema is no longer created via create_all())
  scripts/migrate.py      -- run at container startup, see "Migrations" below
  services/tasks.py       -- Celery task: orchestrates one run end-to-end, with bounded
                             auto-retry on Docker/connection errors
  services/orchestration.py -- DockerOrchestrator: spins up/tears down containers, idempotently
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

Once a run reaches `COMPLETED`, `services/tasks.py` also archives one
`BenchmarkMetric` row per round into Postgres (accuracy/loss, and the
per-round client average of CPU/RAM/compute time/comm bytes) via this same
telemetry parse. `GET /api/runs/<id>` still reads live from `metrics.jsonl`
rather than this table — the Postgres copy exists so run history survives
even after `$SHARED_RUN_DIR` is cleaned up, not as the API's read path.

## Prerequisites

- Docker 
- Node 18+ / npm (for the frontend)

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

Both containers run `backend/docker-entrypoint.sh` before their actual command,
which applies Alembic migrations first (see "Migrations" below).
`celery_worker` depends on `backend`'s healthcheck (`GET /api/health`) rather
than starting in parallel with it — this is deliberate, so the two containers
never both try to apply the same migration at once on a cold `docker compose up`.

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
A malformed request (missing/invalid field, unsupported `framework` or
`partition_strategy`, etc.) is rejected with `400` and a structured
`{"error": {"message", "details"}}` body (see `domain/schemas.py`) before a
run is ever created — it never reaches Celery/Docker.

## Migrations

Schema is managed by Alembic (`backend/migrations/`), not
`Base.metadata.create_all()`. `backend/scripts/migrate.py` runs automatically
at container startup (via `docker-entrypoint.sh`, before the actual `backend`/
`celery_worker` command) and picks the right action itself:

- Fresh database: `alembic upgrade head`.
- An existing pre-Alembic database (i.e. one created by an older version of
  this platform via `create_all()`): its schema already matches revision
  head, so it's adopted via `alembic stamp head` instead of replaying DDL
  that would fail on tables that already exist. You will *not* lose existing
  run history switching to this version.

To add a schema change: edit `backend/domain/models.py`, then generate a
revision against a real (ideally empty) Postgres —

```bash
docker build -t fl-backend-gen -f backend/Dockerfile backend
docker run --rm --network <a-network-both-containers-share> \
  -e DATABASE_URL=postgresql://postgres:postgres@<db-host>:5432/<db-name> \
  -v "$(pwd)/backend/migrations:/app/migrations" \
  --entrypoint alembic fl-backend-gen revision --autogenerate -m "describe the change"
```

— then review the generated file under `backend/migrations/versions/` before
committing it (autogenerate doesn't always get everything right, e.g. column
renames show up as a drop+add).

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
- **Run metadata** (framework, dataset, status, timestamps) and **per-round
  metrics archive** (`BenchmarkMetric`, written once a run reaches
  `COMPLETED`): Postgres, via `backend/domain/models.py`.

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
     - `"client_metric"` — `{"client_id", "round", "cpu", "ram", "time", "comm_mb", "iowait"}` → per-client chart.
       `comm_mb` should be the actual serialized parameter payload size for that
       round (e.g. `sum(p.nbytes for p in params) / 1024**2` for numpy arrays,
       or the torch tensor equivalent) — not a placeholder constant; this feeds
       the "Network Payload per Round" chart directly.
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
- **Container exits immediately with `alembic.util.exc.CommandError` or a
  "relation already exists" error**: this means `scripts/migrate.py` picked
  the wrong path — most likely a database with tables present but no
  `alembic_version` table, and it *wasn't* created by this platform's old
  `create_all()` path (so adopting it via `stamp head` is wrong). Inspect the
  DB manually (`docker exec -it <db-container> psql -U postgres -d
  fl_benchmark`) before deciding whether to stamp or drop it.
- **A run stays queued and never starts, or a container name collision
  appears in `celery_worker` logs**: `DockerOrchestrator` removes any
  pre-existing container with the same name before creating a new one
  (`run_<run_id>_<name>`), so this should self-heal on the automatic retry
  (`services/tasks.py`'s `run_benchmark_task` auto-retries Docker/connection
  errors up to 3 times) — check `celery_worker` logs for `retry:` lines.

## Known limitations

- `tcp_established`/`tcp_time_wait` in `server_logs` are a coarse system-wide
  `psutil.net_connections()` snapshot taken at the moment aggregation
  finishes in that container — a directional signal of connection churn, not
  exact per-connection accounting scoped to FL traffic specifically.
- FedML prints a non-fatal S3 connectivity diagnostic failure at every server
  startup (dummy AWS credentials). This is expected and harmless — this setup
  uses the GRPC backend, not S3, for actual model transfer.
- `$DATA_CACHE_DIR` (the HuggingFace/torch cache) is shared across all client
  containers of a run with no download coordination between them. On a fully
  cold cache, multiple clients racing to download/prepare the same dataset
  for the first time can corrupt each other's `.incomplete` HuggingFace
  `datasets` cache entry and crash. `run_benchmark_task`'s auto-retry usually
  recovers on the next attempt (the failed download is generally cleaned up
  by the time it retries), but if it doesn't, clear the affected dataset's
  directory under `$DATA_CACHE_DIR/huggingface/` and retry manually. This
  doesn't recur once the cache is warm.
- `container.wait()` in `DockerOrchestrator` has no timeout, so a genuinely
  hung run (rather than a crashed one) will block its Celery worker slot
  indefinitely rather than being marked `FAILED`. There's no principled
  default timeout given `rounds`/`epochs` are user-configurable per run.
