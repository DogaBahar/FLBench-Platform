"""
Packages a completed run into a single self-contained JSON file, in the
shape documented by results/schema.json at the repo root. Run this inside
the backend/celery_worker container (it needs the DB and SHARED_RUN_DIR),
then copy the resulting file into a fork of this repo's results/ folder and
open a PR -- see results/README.md for the full contribution flow.

Usage:
    docker compose exec backend python -m scripts.export_run <run_id> [--submitted-by NAME] [--out-dir DIR]

Written under SHARED_RUN_DIR by default (which docker-compose already bind-mounts
to the host at /tmp/fl_benchmark_runs), so the output is visible on the host
without adding a new mount.
"""
import argparse
import json
import os
import sys

from core.config import config
from core.database import db_session
from domain.models import BenchmarkRun
from telemetry.normalizer import build_run_export


def export_run(run_id: str, submitted_by: str | None, out_dir: str) -> str:
    run = db_session.query(BenchmarkRun).filter(BenchmarkRun.id == run_id).first()
    if run is None:
        raise SystemExit(f"No run found with id '{run_id}'")

    envelope = build_run_export(run, submitted_by)

    os.makedirs(out_dir, exist_ok=True)
    filename = f"{run.framework}-{run.dataset}-{run.id}.json"
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w") as f:
        json.dump(envelope, f, indent=2)

    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--submitted-by", default=None, help="Optional name/handle to credit in the export")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write the export into (default: $SHARED_RUN_DIR/<run_id>/export)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(config.SHARED_RUN_DIR, args.run_id, "export")
    out_path = export_run(args.run_id, args.submitted_by, out_dir)

    print(f"Wrote {out_path}")
    print("Copy this file into results/ in a fork of the platform repo and open a PR.")


if __name__ == "__main__":
    sys.exit(main())
