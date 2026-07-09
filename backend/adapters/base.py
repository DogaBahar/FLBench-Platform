from abc import ABC, abstractmethod
from typing import Dict, Any, List

class FLFrameworkAdapter(ABC):
    
    @abstractmethod
    def generate_configs(self, config: Dict[str, Any], output_dir: str) -> None:
        """
        Translates the generic JSON configuration into framework-specific 
        files (e.g., server.py, client.py) and saves them to output_dir.
        """
        pass

    @abstractmethod
    def get_docker_commands(self, output_dir: str) -> List[Dict[str, str]]:
        """
        Returns a list of container specs to be executed by the Docker SDK.
        Example: [{"name": "server", "command": "python server.py"}, ...]
        """
        pass
        
    @abstractmethod
    def inject_telemetry(self, output_dir: str, run_id: str) -> None:
        """
        Injects standard logging callbacks into the framework's workspace
        so metrics are saved uniformly to a shared .jsonl file.
        """
        pass