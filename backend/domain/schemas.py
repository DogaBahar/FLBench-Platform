from pydantic import BaseModel, Field

class InfrastructureConfig(BaseModel):
    framework: str = Field(..., description="flower, flare, or openfl")
    target_clients: int

class MLHyperparameters(BaseModel):
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str

class FederatedSettings(BaseModel):
    rounds: int
    strategy: str
    fraction_fit: float
    target_clients: int

class DataSimulation(BaseModel):
    dataset: str = Field(..., description="cifar100 or femnist")
    samples_per_client: int
    partition_strategy: str

class BenchmarkConfigSchema(BaseModel):
    infrastructure: InfrastructureConfig
    ml_hyperparameters: MLHyperparameters
    federated_settings: FederatedSettings
    data_simulation: DataSimulation