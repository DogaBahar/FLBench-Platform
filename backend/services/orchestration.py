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
                    "TORCH_HOME": "/app/data/torch"
                }
                if "env" in spec:
                    env_vars.update(spec["env"])

                container = self.client.containers.run(
                    image=spec["image"],
                    command=spec["command"],
                    name=f"run_{run_id}_{spec['name']}",
                    network=self.network_name,
                    volumes={
                        volume_dir: {'bind': '/app/workspace', 'mode': 'rw'},
                        config.DATA_CACHE_DIR: {'bind': '/app/data', 'mode': 'rw'} # New Mount!
                    },
                    working_dir="/app/workspace",
                    detach=True,
                    environment=env_vars
                )
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