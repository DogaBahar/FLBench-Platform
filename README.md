# FL Benchmark Platform

FL Benchmark Platform is a framework-agnostic benchmarking system for evaluating federated learning frameworks under identical experimental conditions. A control runs standardized federated learning benchmarks across
multiple FL frameworks — **Flower**, **NVIDIA FLARE**, and **FedML** — against
the same datasets and hyperparameters, and comparing them side by side in one
dashboard (accuracy/loss curves, per-client CPU/RAM/compute time, server
aggregation time, TCP connection churn, wall-clock time).

The developer who wishes to use a framework configures a run once (framework, dataset, number of clients, FL
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
  run_<id>_server, run_<id>_client_1..N   (FedML)
  run_<id>_server                          (Flower: Simulation Engine, all clients run in-process via Ray;
                                             FLARE: runs `nvflare simulator`, hosts all sites in one container)
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

Once a run reaches `COMPLETED`, `services/tasks.py` also archives one
`BenchmarkMetric` row per round into Postgres (accuracy/loss, and the
per-round client average of CPU/RAM/compute time/comm bytes) via this same
telemetry parse. `GET /api/runs/<id>` still reads live from `metrics.jsonl`
rather than this table — the Postgres copy exists so run history survives
even after `$SHARED_RUN_DIR` is cleaned up, not as the API's read path.

## Prerequisites

- Docker 
- Node 18+ / npm (for the frontend)

## Quickstart 

**1. Build the three FL framework images.** They're built independently and referenced by tag name
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

Or call the API directly:

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

To add a schema change: edit `backend/domain/models.py`, then generate a
revision against a real (ideally empty) Postgres —

```bash
docker build -t fl-backend-gen -f backend/Dockerfile backend
docker run --rm --network <a-network-both-containers-share> \
  -e DATABASE_URL=postgresql://postgres:postgres@<db-host>:5432/<db-name> \
  -v "$(pwd)/backend/migrations:/app/migrations" \
  --entrypoint alembic fl-backend-gen revision --autogenerate -m "describe the change"
```

## Supported frameworks

| `framework` value | Library                 | Notes |
|---|---|---|
| `flower`  | `flwr[simulation]==1.32.1` | Current `ServerApp`/`ClientApp` message-passing API (`ArrayRecord`/`MetricRecord`/`RecordDict`), run via the **Simulation Engine** (`flwr run . local-simulation`) -- all clients run in-process as Ray actors inside **one container**, not one container per client (see "Known limitations" for why Deployment Engine, which would preserve per-container isolation, isn't used). `pyproject.toml`'s `dependencies` must stay `[]` -- Flower auto-provisions an isolated per-run environment via `uv sync` for anything declared there, which silently tries to reinstall torch etc. from scratch even though the image already has it. |
| `nvflare` | `nvflare~=2.4.0`| Runs via `nvflare simulator`; **one container** hosts the server and all simulated client sites. |
| `fedml`   | `fedml`         | Cross-silo, horizontal, GRPC backend; one container per participant, IP table built from `RUN_ID` at container start. |

## Supported averaging strategies

| `strategy` value | Notes |
|---|---|
| `FedAvg` | Plain weighted-average aggregation (the default). |
| `FedProx` | [Li et al., 2018](https://arxiv.org/abs/1812.06127) — adds a `(proximal_mu / 2) * \|\|w - w_global\|\|^2` penalty to each client's *local* training loss, pulling local updates back toward the global model each round. 

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
   - **For `FedProx` parity**: right after your generated client code applies
     the received global weights to the local model (before calling
     `train(...)`), snapshot `global_params = [p.detach().clone() for p in
     model.parameters()]` and pass it plus `mu=<proximal_mu, or 0.0 for
     FedAvg>` into `train(...)` — see any existing adapter's client code for
     the exact pattern. `mu=0.0`/`global_params=None` is a guaranteed no-op,
     so this is safe to always wire in even if you only support `FedAvg`
     initially.
2. **Register it** in `backend/adapters/factory.py`'s `_adapters` dict.
3. **Write a `Dockerfile.<name>`** in the project root (mirror the existing
   three: a `python:3.1x-slim` base matching whatever your framework version
   requires -- FLARE/FedML use `3.10-slim`, Flower's current version needs
   `3.11-slim` -- `git` for HF `datasets`, CPU-only torch/torchvision,
   `psutil`, plus the framework package — pin a version deliberately rather
   than trusting "latest," and verify the exact API you generate scripts
   against actually exists in that version before trusting it — introspect
   the real installed package (`help()`/`dir()`/`inspect.signature()` inside a
   throwaway container) rather than relying on documentation alone, since it
   can lag or be paraphrased incorrectly by tooling; see the Flower adapter's
   history for how much this mattered in practice). Build it as
   `benchmark-<name>:latest` to match what `get_docker_commands()` returns.
4. **Add the option** to the `<select>` in `frontend/src/Dashboard.jsx`.

## Adding a new dataset

Create `backend/datasets/<name>/` with:
- `model.py` — `get_model()` factory, `train(model, loader, epochs, lr,
  optimizer_name, mu=0.0, global_params=None)`,
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
- `container.wait()` in `DockerOrchestrator` has no timeout, so a genuinely
  hung run (rather than a crashed one) will block its Celery worker slot
  indefinitely rather than being marked `FAILED`. There's no principled
  default timeout given `rounds`/`epochs` are user-configurable per run.
- **Flower runs in one container, not one per client** (unlike FedML). Flower's
  Deployment Engine (persistent `SuperLink` + one `SuperNode` container per
  client) was tried first, to keep true per-container resource isolation
  consistent with FedML -- SuperLink/SuperNode connected and a run submitted
  successfully, but the ClientApp task never actually dispatched (no error,
  indefinite hang) in live testing against `flwr==1.32.1`, for reasons not
  fully diagnosed. The Simulation Engine (this adapter's actual approach) was
  used instead. If you want to revisit Deployment Engine: the hang reproduced
  even in the simplest possible 2-node setup, past node registration and run
  submission, with `flower-superexec`'s `--plugin-type clientapp` subprocess
  confirmed running and listening but never invoking `client_app.py`.
- Also learned the hard way debugging the above: Flower now auto-migrates
  legacy `[tool.flwr.federations]` `pyproject.toml` entries into a separate
  `~/.flwr/config.toml` on first `flwr run` in a given container, *mutating
  the source file in place* to comment that section out. Harmless here since
  every run gets a freshly generated `pyproject.toml`, but confusing if you're
  ever inspecting `$SHARED_RUN_DIR/<run_id>/pyproject.toml` after the fact and
  wondering why the federation section is commented out.
