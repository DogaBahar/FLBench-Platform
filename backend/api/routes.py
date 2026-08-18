import json
import logging
from flask import Blueprint, request, jsonify
from pydantic import ValidationError
from core.config import config
from core.database import db_session
from domain.models import BenchmarkRun
from domain.schemas import BenchmarkRequestSchema
from services.tasks import run_benchmark_task
from telemetry.normalizer import get_run_telemetry, get_archived_telemetry, build_run_export
from datetime import datetime
import os
import shutil
import uuid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

bp = Blueprint('api', __name__)


@bp.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200


def error_response(message, status=400, details=None):
    body = {"error": {"message": message}}
    if details is not None:
        body["error"]["details"] = details
    return jsonify(body), status


@bp.route('/benchmark', methods=['POST'])
def start_benchmark():
    raw_data = request.get_json(silent=True)
    if raw_data is None:
        return error_response("Request body must be valid JSON", 400)

    try:
        validated = BenchmarkRequestSchema(**raw_data)
    except ValidationError as e:
        return error_response("Invalid benchmark configuration", 400, details=json.loads(e.json()))

    try:
        framework = validated.framework
        num_clients = validated.clients
        config_dict = validated.config.model_dump(exclude_none=True)
        payload = {
            "framework": framework,
            "clients": num_clients,
            "config": config_dict,
        }

        # --- Generate Semantic Run ID ---
        dataset = config_dict["data_simulation"]["dataset"]
        timestamp = datetime.now().strftime("%b%d_%H%M").lower()
        short_hash = str(uuid.uuid4())[:4]
        custom_run_id = f"{framework}-{dataset}-{timestamp}-{short_hash}"

        new_run = BenchmarkRun(
            id=custom_run_id,  # Override the default UUID here
            framework=framework,
            dataset=dataset,
            strategy=config_dict["federated_settings"]["strategy"],
            rounds=config_dict["federated_settings"]["rounds"],
            epochs=config_dict["ml_hyperparameters"]["epochs"],
            batch_size=config_dict["ml_hyperparameters"]["batch_size"],
            status="PENDING"
        )

        db_session.add(new_run)
        db_session.commit()

        # Dispatch the Celery task for background execution
        run_benchmark_task.delay(new_run.id, payload)

        return jsonify({
            "message": "Benchmark deployed",
            "run_id": new_run.id
        }), 201

    except Exception as e:
        db_session.rollback()
        logger.exception("Failed to start benchmark")
        return error_response("Internal server error", 500, details=str(e))

@bp.route('/runs', methods=['GET'])
def get_all_runs():
    runs = db_session.query(BenchmarkRun).order_by(BenchmarkRun.started_at.desc()).all()
    return jsonify([run.id for run in runs]), 200

@bp.route('/runs/<run_id>', methods=['GET', 'DELETE'])
def handle_specific_run(run_id):

    # --- HANDLE DELETE REQUEST ---
    if request.method == 'DELETE':
        # 1. Delete from the Database
        run = db_session.query(BenchmarkRun).filter(BenchmarkRun.id == run_id).first()
        if run:
            db_session.delete(run)
            db_session.commit()

        # 2. Delete from the File System (the same directory the adapter/orchestrator
        # write to and mount into every container -- see core/config.py SHARED_RUN_DIR)
        run_dir = os.path.join(config.SHARED_RUN_DIR, run_id)

        if os.path.exists(run_dir):
            try:
                shutil.rmtree(run_dir)
            except Exception as e:
                logger.exception("Failed to delete run directory %s", run_dir)
                return error_response(f"Failed to delete run directory: {str(e)}", 500)

        return jsonify({"message": f"Run {run_id} successfully deleted from DB and filesystem."}), 200


    # --- HANDLE GET REQUEST ---
    if request.method == 'GET':
        run = db_session.query(BenchmarkRun).filter(BenchmarkRun.id == run_id).first()
        if not run:
            return error_response("Run not found", 404)

        response_data = {
            "run_id": run.id,
            "framework": run.framework,
            "status": run.status,
            "error_message": run.error_message,
            "config": {
                "dataset": run.dataset,
                "strategy": run.strategy,
                "rounds": run.rounds
            }
        }

        # --- Always fetch telemetry regardless of run status ---
        telemetry_data = get_run_telemetry(run_id)
        
        if run.status == "COMPLETED" and not telemetry_data["global_metrics"]["metrics_distributed"]["accuracy"]:
            telemetry_data = get_archived_telemetry(run)

        response_data.update(telemetry_data)

        return jsonify(response_data), 200


@bp.route('/runs/<run_id>/export', methods=['GET'])
def export_run(run_id):
    run = db_session.query(BenchmarkRun).filter(BenchmarkRun.id == run_id).first()
    if not run:
        return error_response("Run not found", 404)

    submitted_by = request.args.get('submitted_by')
    envelope = build_run_export(run, submitted_by)
    filename = f"{run.framework}-{run.dataset}-{run.id}.json"

    response = jsonify(envelope)
    response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response, 200
