from typing import Dict, List, Type
from adapters.base import FLFrameworkAdapter

# import adapters as we build them
from adapters.flower_adapter import FlowerAdapter
from adapters.flare_adapter import FlareAdapter
from adapters.fedml_adapter import FedMLAdapter

class AdapterFactory:
    _adapters: Dict[str, Type[FLFrameworkAdapter]] = {
        "flower": FlowerAdapter,
        "nvflare": FlareAdapter,
        "fedml": FedMLAdapter
    }

    @classmethod
    def get_adapter(cls, framework_name: str) -> FLFrameworkAdapter:
        adapter_class = cls._adapters.get(framework_name.lower())
        if not adapter_class:
            raise ValueError(f"Unsupported framework: '{framework_name}'. "
                             f"Available: {list(cls._adapters.keys())}")
        return adapter_class()

    @classmethod
    def available_frameworks(cls) -> List[str]:
        return list(cls._adapters.keys())