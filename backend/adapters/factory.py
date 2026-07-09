from typing import Dict, Type
from adapters.base import FLFrameworkAdapter

# We will import actual adapters as we build them
from adapters.flower_adapter import FlowerAdapter
from adapters.flare_adapter import FlareAdapter
from adapters.fedml_adapter import FedMLAdapter  # Uncomment when implemented

class AdapterFactory:
    _adapters: Dict[str, Type[FLFrameworkAdapter]] = {
        "flower": FlowerAdapter,
        "nvflare": FlareAdapter,   # Uncomment when implemented
        "fedml": FedMLAdapter  # Uncomment when implemented
    }

    @classmethod
    def get_adapter(cls, framework_name: str) -> FLFrameworkAdapter:
        adapter_class = cls._adapters.get(framework_name.lower())
        if not adapter_class:
            raise ValueError(f"Unsupported framework: '{framework_name}'. "
                             f"Available: {list(cls._adapters.keys())}")
        return adapter_class()