from flask import Blueprint, request, jsonify
from core.database import db_session
from domain.models import BenchmarkRun
from adapters.factory import AdapterFactory
from services.tasks import run_benchmark_task
from telemetry.normalizer import get_run_telemetry
from datetime import datetime
import os
import shutil
import uuid

bp = Blueprint('api', __name__)

@bp.route('/benchmark', methods=['POST'])
def start_benchmark():
    try:
        raw_data = request.get_json()
        
        framework = raw_data.get('framework', 'flower')
        num_clients = raw_data.get('clients', 3)
        config = raw_data.get('config', {})
        
        try:
            AdapterFactory.get_adapter(framework)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        # --- Generate Semantic Run ID ---
        dataset = config.get('data_simulation', {}).get('dataset', 'cifar100')
        timestamp = datetime.now().strftime("%b%d_%H%M").lower()
        short_hash = str(uuid.uuid4())[:4]
        custom_run_id = f"{framework}-{dataset}-{timestamp}-{short_hash}"

        new_run = BenchmarkRun(
            id=custom_run_id,  # Override the default UUID here
            framework=framework,
            dataset=dataset,
            strategy=config.get('federated_settings', {}).get('strategy', 'FedAvg'),
            rounds=config.get('federated_settings', {}).get('rounds', 3),
            epochs=config.get('ml_hyperparameters', {}).get('epochs', 2),
            batch_size=config.get('ml_hyperparameters', {}).get('batch_size', 32),
            status="PENDING"
        )
        
        db_session.add(new_run)
        db_session.commit()

        # Dispatch the Celery task for background execution
        run_benchmark_task.delay(new_run.id, raw_data)

        return jsonify({
            "message": "Benchmark deployed",
            "run_id": new_run.id
        }), 201

    except Exception as e:
        db_session.rollback()
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@bp.route('/runs', methods=['GET'])
def get_all_runs():
    runs = db_session.query(BenchmarkRun).order_by(BenchmarkRun.started_at.desc()).all()
    return jsonify([run.id for run in runs]), 200

# --- FIX: Combined GET and DELETE into a single, clean route ---
@bp.route('/runs/<run_id>', methods=['GET', 'DELETE'])
def handle_specific_run(run_id):
    
    # --- HANDLE DELETE REQUEST ---
    if request.method == 'DELETE':
        # 1. Delete from the Database
        run = db_session.query(BenchmarkRun).filter(BenchmarkRun.id == run_id).first()
        if run:
            db_session.delete(run)
            db_session.commit()
            
        # 2. Delete from the File System
        workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'workspace'))
        run_dir = os.path.join(workspace_dir, run_id)
        
        if os.path.exists(run_dir):
            try:
                shutil.rmtree(run_dir)
            except Exception as e:
                return jsonify({"error": f"Failed to delete run directory: {str(e)}"}), 500
                
        return jsonify({"message": f"Run {run_id} successfully deleted from DB and filesystem."}), 200


    # --- HANDLE GET REQUEST ---
    if request.method == 'GET':
        run = db_session.query(BenchmarkRun).filter(BenchmarkRun.id == run_id).first()
        if not run:
            return jsonify({"error": "Run not found"}), 404
            
        response_data = {
            "run_id": run.id,
            "framework": run.framework,
            "status": run.status,
            "config": {
                "dataset": run.dataset,
                "strategy": run.strategy,
                "rounds": run.rounds
            }
        }
        
        # --- Always fetch telemetry regardless of run status ---
        telemetry_data = get_run_telemetry(run_id)
        response_data.update(telemetry_data)
            
        return jsonify(response_data), 200