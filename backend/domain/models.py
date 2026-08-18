import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, BigInteger, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class BenchmarkRun(Base):
    __tablename__ = 'benchmark_runs'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    framework = Column(String(50), nullable=False)    # 'flower', 'nvflare', 'fedml'
    dataset = Column(String(50), nullable=False)      # 'cifar100', 'femnist'
    strategy = Column(String(50), nullable=False)     # 'FedAvg', etc.
    rounds = Column(Integer, nullable=False)
    epochs = Column(Integer, nullable=False)
    batch_size = Column(Integer, nullable=False)
    
    # State tracking: PENDING, RUNNING, COMPLETED, FAILED
    status = Column(String(20), default="PENDING")
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)

    metrics = relationship("BenchmarkMetric", back_populates="run", cascade="all, delete")

    def __repr__(self):
        return f"<BenchmarkRun {self.id} ({self.framework} - {self.status})>"

class BenchmarkMetric(Base):
    __tablename__ = 'benchmark_metrics'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id = Column(String(36), ForeignKey('benchmark_runs.id'), nullable=False)
    
    round_number = Column(Integer, nullable=False)
    accuracy = Column(Float, nullable=True)
    loss = Column(Float, nullable=True)

    # Fairness under non-IID splits: spread of per-client accuracy evaluating
    # the same round's global model on each client's own held-out data.
    accuracy_std = Column(Float, nullable=True)
    worst_client_accuracy = Column(Float, nullable=True)

    # Telemetry
    training_time_ms = Column(Integer, nullable=True)
    communication_time_ms = Column(Integer, nullable=True)
    cpu_usage_pct = Column(Float, nullable=True)
    memory_usage_mb = Column(Float, nullable=True)
    network_bytes_sent = Column(BigInteger, nullable=True)
    network_bytes_received = Column(BigInteger, nullable=True)
    
    recorded_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("BenchmarkRun", back_populates="metrics")