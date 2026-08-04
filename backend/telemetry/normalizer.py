import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from core.config import config
from core.database import db_session

EXPORT_SCHEMA_VERSION = 1

def get_run_telemetry(run_id: str) -> Dict[str, Any]:
    run_dir = os.path.join(config.SHARED_RUN_DIR, run_id)
    metrics_file = os.path.join(run_dir, "metrics.jsonl")
    
    telemetry = {
        "global_metrics": {
            "metrics_distributed": {"accuracy": []},
            "losses_distributed": [],
            "wall_clock_time_seconds": 0
        },
        "client_logs": [],
        "server_logs": [],
        "distributions": {},
        "full_config": {} 
    }

    config_file = os.path.join(run_dir, "config.json")

    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                telemetry["full_config"] = json.load(f)
        except Exception:
            pass

    if not os.path.exists(metrics_file):
        return telemetry

    clients_data = {}

    with open(metrics_file, "r") as f:
        for line in f:
            try:
                log = json.loads(line.strip())
                log_type = log.get("type")

                if log_type == "server_metric":
                    rnd = log.get("round", 1)
                    telemetry["global_metrics"]["metrics_distributed"]["accuracy"].append([rnd, log.get("accuracy", 0.0)])
                    telemetry["global_metrics"]["losses_distributed"].append([rnd, log.get("loss", 0.0)])
                
                # --- NEW: Parse Server TCP and Aggregation Time ---
                elif log_type == "server_sys_metric":
                    telemetry["server_logs"].append({
                        "round": log.get("round", 1),
                        "tcp_established": log.get("tcp_est", 0),
                        "tcp_time_wait": log.get("tcp_wait", 0),
                        "aggregation_time_sec": log.get("agg_time", 0.0),
                        "iowait_time": 0.0
                    })
                
                elif log_type == "client_metric":
                    client_id = log.get("client_id")
                    if client_id not in clients_data:
                        clients_data[client_id] = []
                    
                    clients_data[client_id].append({
                        "action": "fit",
                        "round": log.get("round", 1),
                        "cpu_usage_percent": log.get("cpu", 0.0),
                        "peak_memory_mb": log.get("ram", 0.0),
                        "compute_time_seconds": log.get("time", 0.0),
                        "comm_size_mb": log.get("comm_mb", 0.0),
                        "iowait": log.get("iowait", 0.0)
                    })
                
                elif log_type == "distribution":
                    telemetry["distributions"][log.get("client_id")] = log.get("counts", {})

                elif log_type == "wall_clock_time":
                    telemetry["global_metrics"]["wall_clock_time_seconds"] = log.get("time_seconds", 0.0)

            except json.JSONDecodeError:
                continue

    for c_id, logs in clients_data.items():
        telemetry["client_logs"].append({
            "client_id": c_id,
            "logs": logs
        })

    return telemetry


def get_archived_telemetry(run) -> Dict[str, Any]:
    
    from domain.models import BenchmarkMetric  # local: keeps this module DB-free on the (usual) live-file path

    rows = (
        db_session.query(BenchmarkMetric)
        .filter(BenchmarkMetric.run_id == run.id)
        .order_by(BenchmarkMetric.round_number)
        .all()
    )

    wall_clock = 0.0
    if run.started_at and run.completed_at:
        wall_clock = (run.completed_at - run.started_at).total_seconds()

    telemetry = {
        "global_metrics": {
            "metrics_distributed": {"accuracy": []},
            "losses_distributed": [],
            "wall_clock_time_seconds": round(wall_clock, 2)
        },
        "client_logs": [],
        "server_logs": [],
        "distributions": {},
        "full_config": {},
        "telemetry_source": "archived"
    }

    archived_logs = []
    for row in rows:
        if row.accuracy is not None:
            telemetry["global_metrics"]["metrics_distributed"]["accuracy"].append([row.round_number, row.accuracy])
        if row.loss is not None:
            telemetry["global_metrics"]["losses_distributed"].append([row.round_number, row.loss])
        archived_logs.append({
            "action": "fit",
            "round": row.round_number,
            "cpu_usage_percent": row.cpu_usage_pct or 0.0,
            "peak_memory_mb": row.memory_usage_mb or 0.0,
            "compute_time_seconds": (row.training_time_ms or 0) / 1000,
            "comm_size_mb": (row.network_bytes_sent or 0) / (1024 * 1024),
            "iowait": 0.0
        })

    if archived_logs:
        telemetry["client_logs"] = [{"client_id": "archived_avg", "logs": archived_logs}]

    return telemetry


def build_run_export(run, submitted_by: Optional[str] = None) -> Dict[str, Any]:
    """
    Packages a BenchmarkRun into the envelope shape documented by
    results/schema.json at the repo root -- shared by scripts/export_run.py
    (CLI) and GET /api/runs/<id>/export (frontend "Export for results/" button).
    """
    telemetry = get_run_telemetry(run.id)

    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "run_id": run.id,
        "framework": run.framework,
        "dataset": run.dataset,
        "strategy": run.strategy,
        "status": run.status,
        "config": {
            "rounds": run.rounds,
            "epochs": run.epochs,
            "batch_size": run.batch_size,
            "full_config": telemetry.get("full_config", {}),
        },
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "submitted_by": submitted_by,
        "results": {
            "global_metrics": telemetry["global_metrics"],
            "client_logs": telemetry["client_logs"],
            "server_logs": telemetry["server_logs"],
            "distributions": telemetry["distributions"],
        },
    }