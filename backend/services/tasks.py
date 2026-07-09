import os
import time
from celery import Celery
from core.config import config
from core.database import db_session
from domain.models import BenchmarkRun
from adapters.factory import AdapterFactory
from services.orchestration import DockerOrchestrator

# Initialize Celery
celery_app = Celery(
    'benchmark_tasks',
    broker=config.CELERY_BROKER_URL,
    backend=config.CELERY_RESULT_BACKEND
)

@celery_app.task(bind=True)
def run_benchmark_task(self, run_id: str, payload: dict):
    """Background job that orchestrates the FL execution."""
    run_record = db_session.query(BenchmarkRun).filter(BenchmarkRun.id == run_id).first()
    if not run_record:
        return "Run not found."
        
    run_record.status = "RUNNING"
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
        
        # 5. Mark as Complete
        run_record.status = "COMPLETED"
        db_session.commit()
        
    except Exception as e:
        run_record.status = "FAILED"
        db_session.commit()
        raise e