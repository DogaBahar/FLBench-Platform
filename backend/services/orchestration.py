import os
import docker
import time
from typing import List, Dict
from core.config import config

class DockerOrchestrator:
    def __init__(self):
        self.client = docker.from_env()
        self.network_name = "fl_benchmark_net"
        self._ensure_network()

    def _ensure_network(self):
        try:
            self.client.networks.get(self.network_name)
        except docker.errors.NotFound:
            self.client.networks.create(self.network_name, driver="bridge")

    def run_containers(self, run_id: str, volume_dir: str, specs: List[Dict[str, str]]):
        # Ensure the cache directory exists on the host machine before mounting
        os.makedirs(config.DATA_CACHE_DIR, exist_ok=True)
        
        containers = []
        try:
            for spec in specs:
                # Inject Cache Environment Variables alongside any existing ones
                env_vars = {
                    "RUN_ID": run_id,
                    "HF_DATASETS_CACHE": "/app/data/huggingface",
                    "TORCH_HOME": "/app/data/torch",
                    # Without this, torch/OpenMP defaults to one thread per
                    # visible host core -- on a many-core host that's
                    # thousands of threads per container, and several
                    # concurrent runs (e.g. a redelivered/retried task
                    # racing the original, see services/tasks.py's
                    # task_acks_late) turns into severe oversubscription
                    # that starves everything, including the Docker daemon
                    # itself, making the APIError/ConnectionError retries
                    # that cause that race more likely in the first place.
                    "OMP_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4",
                }
                if "env" in spec:
                    env_vars.update(spec["env"])

                container_name = f"run_{run_id}_{spec['name']}"
                # Make launches idempotent: a retried/redelivered task (see
                # task_acks_late in services/tasks.py) could be re-running a
                # run_id whose containers from a prior, uncleanly-killed
                # attempt are still sitting around under this same name.
                try:
                    self.client.containers.get(container_name).remove(force=True)
                except docker.errors.NotFound:
                    pass

                run_kwargs = dict(
                    image=spec["image"],
                    command=spec["command"],
                    name=container_name,
                    network=self.network_name,
                    volumes={
                        volume_dir: {'bind': '/app/workspace', 'mode': 'rw'},
                        config.DATA_CACHE_DIR: {'bind': '/app/data', 'mode': 'rw'} # New Mount!
                    },
                    working_dir="/app/workspace",
                    detach=True,
                    environment=env_vars
                )
                if config.USE_GPU:
                    # count=-1 requests all visible GPUs rather than pinning
                    # to one -- fine at this model size (a few hundred MB of
                    # VRAM per client against 46GB/GPU), and lets each
                    # framework's own scheduling (Ray for Flower, one process
                    # per participant for FedML) place work across whichever
                    # GPUs are free rather than this orchestrator having to
                    # track per-run GPU assignment itself.
                    run_kwargs["device_requests"] = [
                        docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
                    ]

                container = self.client.containers.run(**run_kwargs)
                containers.append(container)
                
                if spec['name'] == 'server':
                    time.sleep(3)

            for container in containers:
                container.wait()

        finally:
            for container in containers:
                try:
                    container.remove(force=True)
                except Exception:
                    pass