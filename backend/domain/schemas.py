from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator
from adapters.factory import AdapterFactory


class MLHyperparameters(BaseModel):
    epochs: int = Field(default=2, gt=0)
    batch_size: int = Field(default=32, gt=0)
    learning_rate: float = Field(default=0.001, gt=0)
    optimizer: Literal["adam", "sgd"] = "adam"


class FederatedSettings(BaseModel):
    rounds: int = Field(default=3, gt=0)
    strategy: Literal["FedAvg", "FedProx"] = "FedAvg"
    fraction_fit: float = Field(default=1.0, gt=0, le=1.0)
    proximal_mu: float = Field(default=0.01, ge=0)


class DataSimulation(BaseModel):
    dataset: Literal["cifar100", "femnist"] = "cifar100"
    samples_per_client: int = Field(default=500, gt=0)
    partition_strategy: Literal["iid", "shard", "dirichlet"] = "iid"
    alpha: Optional[float] = Field(default=None, gt=0)
    shards_per_client: Optional[int] = Field(default=None, gt=0)


class BenchmarkConfigSchema(BaseModel):
    ml_hyperparameters: MLHyperparameters = MLHyperparameters()
    federated_settings: FederatedSettings = FederatedSettings()
    data_simulation: DataSimulation = DataSimulation()


class BenchmarkRequestSchema(BaseModel):
    framework: str = Field(..., description="One of AdapterFactory.available_frameworks()")
    clients: int = Field(default=3, gt=0, le=10)
    config: BenchmarkConfigSchema = BenchmarkConfigSchema()

    @model_validator(mode="after")
    def check_framework_supported(self):
        available = AdapterFactory.available_frameworks()
        if self.framework.lower() not in available:
            raise ValueError(
                f"Unsupported framework: '{self.framework}'. Available: {available}"
            )
        self.framework = self.framework.lower()
        return self
