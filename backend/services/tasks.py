import logging
import os
from datetime import datetime

import docker
from celery import Celery

from core.config import config
from core.database import db_session
from domain.models import BenchmarkRun, BenchmarkMetric
from adapters.factory import AdapterFactory
from services.orchestration import DockerOrchestrator
from telemetry.normalizer import get_run_telemetry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

# Initialize Celery
celery_app = Celery(
    'benchmark_tasks',
    broker=config.CELERY_BROKER_URL,
    backend=config.CELERY_RESULT_BACKEND
)

# A killed/OOM'd worker should redeliver its in-flight task instead of losing
# it silently. Paired with prefetch_multiplier=1 so a worker doesn't hold
# several long-running benchmark tasks hostage at once.
celery_app.conf.task_acks_late = True
celery_app.conf.worker_prefetch_multiplier = 1


def _persist_round_metrics(run_id: str, telemetry_data: dict) -> None:
    """
    Archives the per-round telemetry (already parsed from metrics.jsonl by
    telemetry/normalizer.py) into Postgres, so run history survives even if
    SHARED_RUN_DIR is cleaned up later. Mirrors the per-round client average
    Results.jsx already computes client-side.
    """
    global_metrics = telemetry_data.get("global_metrics", {})
    acc_by_round = dict(global_metrics.get("metrics_distributed", {}).get("accuracy", []))
    loss_by_round = dict(global_metrics.get("losses_distributed", []))

    rounds_seen = set(acc_by_round) | set(loss_by_round)
    per_round = {}
    for client in telemetry_data.get("client_logs", []):
        for log in client.get("logs", []):
            if log.get("action") != "fit":
                continue
            r = log.get("round", 1)
            rounds_seen.add(r)
            bucket = per_round.setdefault(r, {"cpu": 0.0, "ram": 0.0, "time": 0.0, "comm_mb": 0.0, "count": 0})
            bucket["cpu"] += log.get("cpu_usage_percent", 0.0)
            bucket["ram"] += log.get("peak_memory_mb", 0.0)
            bucket["time"] += log.get("compute_time_seconds", 0.0)
            bucket["comm_mb"] += log.get("comm_size_mb", 0.0)
            bucket["count"] += 1

    for r in sorted(rounds_seen):
        bucket = per_round.get(r, {"count": 0})
        count = bucket.get("count", 0)
        db_session.add(BenchmarkMetric(
            run_id=run_id,
            round_number=r,
            accuracy=acc_by_round.get(r),
            loss=loss_by_round.get(r),
            # communication_time_ms / network_bytes_received have no measured
            # signal in any adapter today -- left NULL rather than fabricated.
            training_time_ms=int((bucket["time"] / count) * 1000) if count else None,
            cpu_usage_pct=(bucket["cpu"] / count) if count else None,
            memory_usage_mb=(bucket["ram"] / count) if count else None,
            network_bytes_sent=int((bucket["comm_mb"] / count) * 1024 * 1024) if count else None,
        ))


@celery_app.task(
    bind=True,
    autoretry_for=(docker.errors.APIError, ConnectionError),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def run_benchmark_task(self, run_id: str, payload: dict):
    """Background job that orchestrates the FL execution."""
    run_record = db_session.query(BenchmarkRun).filter(BenchmarkRun.id == run_id).first()
    if not run_record:
        logger.warning("Run %s not found in DB, skipping.", run_id)
        return "Run not found."

    if run_record.status == "COMPLETED":
        # Guards against a redelivered task (task_acks_late) re-running an
        # already-successful benchmark and double-launching its containers.
        logger.info("Run %s already COMPLETED, skipping redelivered task.", run_id)
        return "Already completed."

    run_record.status = "RUNNING"
    run_record.completed_at = None  # clear any stale timestamp from a prior failed attempt being retried
    db_session.commit()

    try:
        framework = payload.get("framework")
        adapter = AdapterFactory.get_adapter(framework)

        # 1. Setup Shared Directory
        run_dir = os.path.join(config.SHARED_RUN_DIR, run_id)
        os.makedirs(run_dir, exist_ok=True)

        # 2. Generate Python files using the Adaptive Layer
        adapter.generate_configs(payload, run_dir)
        adapter.inject_telemetry(run_dir, run_id)

        # 3. Get Docker commands
        docker_specs = adapter.get_docker_commands(run_dir, payload.get("clients", 3))

        # 4. Execute via Docker Orchestrator
        orchestrator = DockerOrchestrator()
        orchestrator.run_containers(run_id, run_dir, docker_specs)

        # 5. Archive per-round metrics into Postgres, then mark complete
        telemetry_data = get_run_telemetry(run_id)
        _persist_round_metrics(run_id, telemetry_data)

        run_record.status = "COMPLETED"
        run_record.completed_at = datetime.utcnow()
        db_session.commit()

    except Exception as e:
        db_session.rollback()
        run_record.status = "FAILED"
        run_record.completed_at = datetime.utcnow()
        db_session.commit()
        logger.exception("Benchmark run %s failed", run_id)
        raise e
