import os
import shutil
import json
from typing import Dict, Any, List
from adapters.base import FLFrameworkAdapter

class FlowerAdapter(FLFrameworkAdapter):

    def generate_configs(self, config: Dict[str, Any], output_dir: str) -> None:
        ui_config = config.get("config", {})
        fed_settings = ui_config.get("federated_settings", {})
        ml_settings = ui_config.get("ml_hyperparameters", {})
        data_settings = ui_config.get("data_simulation", {})

        num_rounds = fed_settings.get("rounds", 3)
        fraction_fit = fed_settings.get("fraction_fit", 1.0)
        strategy_name = fed_settings.get("strategy", "FedAvg")
        proximal_mu = fed_settings.get("proximal_mu", 0.01) if strategy_name == "FedProx" else 0.0
        epochs = ml_settings.get("epochs", 2)
        batch_size = ml_settings.get("batch_size", 32)
        learning_rate = ml_settings.get("learning_rate", 0.001)
        optimizer_name = ml_settings.get("optimizer", "adam")

        dataset = data_settings.get("dataset", "cifar100")
        samples_per_client = data_settings.get("samples_per_client", 500)
        partition_strategy = data_settings.get("partition_strategy", "iid")
        alpha = data_settings.get("alpha", 0.5)
        shards_per_client = data_settings.get("shards_per_client", 2)
        num_clients = config.get("clients", 3)

        source_dataset_dir = os.path.join("datasets", dataset)
        shutil.copy(os.path.join(source_dataset_dir, "model.py"), os.path.join(output_dir, "model.py"))
        shutil.copy(os.path.join(source_dataset_dir, "dataset.py"), os.path.join(output_dir, "dataset.py"))

        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        pyproject_toml = f"""[project]
name = "benchmark-app"
version = "0.1.0"
description = "FL benchmark run"
license = "MIT"
dependencies = []

[tool.flwr.app]
publisher = "benchmark"

[tool.flwr.app.components]
serverapp = "server_app:app"
clientapp = "client_app:app"

[tool.flwr.app.config]
num-server-rounds = {num_rounds}
fraction-fit = {fraction_fit}
strategy-name = "{strategy_name}"
proximal-mu = {proximal_mu}
local-epochs = {epochs}
batch-size = {batch_size}
learning-rate = {learning_rate}
optimizer-name = "{optimizer_name}"
dataset = "{dataset}"
samples-per-client = {samples_per_client}
partition-strategy = "{partition_strategy}"
alpha = {alpha}
shards-per-client = {shards_per_client}
num-clients = {num_clients}

[tool.flwr.federations]
default = "local-simulation"

[tool.flwr.federations.local-simulation]
options.num-supernodes = {num_clients}
"""
        with open(os.path.join(output_dir, "pyproject.toml"), "w") as f:
            f.write(pyproject_toml)

        # --- server_app.py ---
        # FedProx only differs from FedAvg by carrying a proximal_mu value --
        # the client (below) is what actually applies the proximal term. The
        # class name still reflects the chosen strategy for readability.
        strategy_base_class = "FedProx" if strategy_name == "FedProx" else "FedAvg"
        proximal_mu_kwarg = f",\n        proximal_mu={proximal_mu}" if strategy_name == "FedProx" else ""

        server_app_code = f"""
import json
import time
import psutil
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx
from model import get_model
from dataset import warm_cache

app = ServerApp()


class CustomStrategy(FedProx if "{strategy_base_class}" == "FedProx" else FedAvg):
    def aggregate_train(self, server_round, replies):
        start_time = time.time()
        arrays, metrics = super().aggregate_train(server_round, replies)
        agg_time = time.time() - start_time

        conns = psutil.net_connections(kind='tcp')
        tcp_est = sum(1 for c in conns if c.status == 'ESTABLISHED')
        tcp_wait = sum(1 for c in conns if c.status == 'TIME_WAIT')

        with open("/app/workspace/metrics.jsonl", "a") as f:
            f.write(json.dumps({{
                "type": "server_sys_metric",
                "round": server_round,
                "agg_time": agg_time,
                "tcp_est": tcp_est,
                "tcp_wait": tcp_wait
            }}) + "\\n")

        return arrays, metrics

    def aggregate_evaluate(self, server_round, replies):
        # Computed directly from raw replies (rather than via
        # evaluate_metrics_aggr_fn) so this stays in one place alongside the
        # telemetry write below.
        total_examples = 0
        weighted_loss = 0.0
        weighted_acc = 0.0
        per_client_lines = []
        for msg in replies:
            try:
                m = msg.content["metrics"]
                n = m["num-examples"]
                weighted_loss += m["loss"] * n
                weighted_acc += m["accuracy"] * n
                total_examples += n
                # Each client's own eval of this round's global model on its
                # local held-out split -- fairness across clients (accuracy
                # std, worst-client accuracy) is derived from these below,
                # not just the weighted average.
                per_client_lines.append(json.dumps({{
                    "type": "client_eval_metric",
                    "client_id": f"client_{{m.get('partition-id', 'unknown')}}",
                    "round": server_round,
                    "loss": m["loss"],
                    "accuracy": m["accuracy"],
                    "num_samples": n
                }}))
            except Exception:
                continue

        if total_examples == 0:
            return None

        avg_loss = weighted_loss / total_examples
        avg_acc = weighted_acc / total_examples

        with open("/app/workspace/metrics.jsonl", "a") as f:
            for line in per_client_lines:
                f.write(line + "\\n")
            f.write(json.dumps({{
                "type": "server_metric",
                "round": server_round,
                "loss": avg_loss,
                "accuracy": avg_acc
            }}) + "\\n")

        from flwr.app import MetricRecord
        return MetricRecord({{"loss": avg_loss, "accuracy": avg_acc}})


@app.main()
def main(grid, context: Context) -> None:
    run_config = context.run_config
    num_rounds = int(run_config["num-server-rounds"])
    fraction_fit = float(run_config["fraction-fit"])
    num_clients = int(run_config["num-clients"])

    # Downloads/builds the HF dataset cache once, serially, here in the
    # driver process -- before strategy.start() spawns concurrent client
    # actors that would otherwise race to build the same cold cache.
    warm_cache()

    model = get_model()
    arrays = ArrayRecord(model.state_dict())

    strategy = CustomStrategy(
        fraction_train=fraction_fit,
        fraction_evaluate=1.0,
        min_train_nodes=num_clients,
        min_evaluate_nodes=num_clients,
        min_available_nodes=num_clients{proximal_mu_kwarg}
    )

    run_start = time.time()
    strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
        train_config=ConfigRecord(dict(run_config)),
    )
    total_time = time.time() - run_start

    with open("/app/workspace/metrics.jsonl", "a") as f:
        f.write(json.dumps({{
            "type": "wall_clock_time",
            "time_seconds": round(total_time, 2)
        }}) + "\\n")
"""
        with open(os.path.join(output_dir, "server_app.py"), "w") as f:
            f.write(server_app_code)

        client_app_code = """
import time
import torch
from flwr.app import ArrayRecord, MetricRecord, RecordDict, Message, Context
from flwr.clientapp import ClientApp
from model import get_model, train, test
from dataset import load_data
from telemetry import log_client_metrics, log_distribution

app = ClientApp()

_distribution_logged = set()


def _load_partition(context: Context):
    run_config = context.run_config
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    trainloader, testloader = load_data(
        partition_strategy=run_config["partition-strategy"],
        samples_per_client=int(run_config["samples-per-client"]),
        num_clients=num_partitions,
        client_id=partition_id,
        batch_size=int(run_config["batch-size"]),
        alpha=float(run_config["alpha"]),
        shards_per_client=int(run_config["shards-per-client"]),
    )
    return partition_id, trainloader, testloader


@app.train()
def train_fn(msg: Message, context: Context):
    run_config = context.run_config
    partition_id, trainloader, _ = _load_partition(context)

    if partition_id not in _distribution_logged:
        counts = {}
        for _, labels in trainloader:
            for label in labels.numpy():
                lbl_str = str(label)
                counts[lbl_str] = counts.get(lbl_str, 0) + 1
        log_distribution(f"client_{partition_id}", counts)
        _distribution_logged.add(partition_id)

    model = get_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    mu = float(run_config["proximal-mu"])
    global_params = [p.detach().clone() for p in model.parameters()] if mu > 0 else None

    start_time = time.time()
    train(
        model,
        trainloader,
        int(run_config["local-epochs"]),
        float(run_config["learning-rate"]),
        run_config["optimizer-name"],
        mu=mu,
        global_params=global_params,
    )
    compute_time = time.time() - start_time

    # FedAvg.configure_train() (base class) always injects "server-round" into
    # the outgoing config -- reliable regardless of which Ray actor handled
    # this call (a local counter would silently reset to 1 whenever Ray
    # schedules a round on a fresh actor process).
    rnd = int(msg.content["config"]["server-round"])

    state_dict = model.state_dict()
    comm_mb = sum(v.element_size() * v.nelement() for v in state_dict.values()) / (1024 * 1024)
    log_client_metrics(f"client_{partition_id}", rnd, compute_time, comm_mb)

    model_record = ArrayRecord(state_dict)
    metrics = MetricRecord({"num-examples": len(trainloader.dataset)})
    content = RecordDict({"arrays": model_record, "metrics": metrics})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate_fn(msg: Message, context: Context):
    partition_id, _, testloader = _load_partition(context)

    model = get_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    loss, accuracy = test(model, testloader)
    metrics = MetricRecord({
        "loss": float(loss),
        "accuracy": float(accuracy),
        "num-examples": len(testloader.dataset),
        "partition-id": partition_id,
    })
    content = RecordDict({"metrics": metrics})
    return Message(content=content, reply_to=msg)
"""
        with open(os.path.join(output_dir, "client_app.py"), "w") as f:
            f.write(client_app_code)

        # --- run_flwr.py: thin wrapper the container actually executes ---
        run_flwr_code = """
import subprocess
import sys

result = subprocess.run(["flwr", "run", ".", "local-simulation", "--stream"])
sys.exit(result.returncode)
"""
        with open(os.path.join(output_dir, "run_flwr.py"), "w") as f:
            f.write(run_flwr_code)

    def get_docker_commands(self, output_dir: str, num_clients: int) -> List[Dict[str, str]]:
        # Simulation Engine: all `num_clients` participants are simulated
        # in-process (via Ray) within this single container, driven by
        # `options.num-supernodes` in pyproject.toml -- matching how the FLARE
        # adapter's `nvflare simulator` already works in this platform.
        return [{
            "name": "server",
            "image": "benchmark-flower:latest",
            "command": "python run_flwr.py"
        }]

    def inject_telemetry(self, output_dir: str, run_id: str) -> None:
        telemetry_code = """
import json
import psutil

def log_client_metrics(client_id, round_num, compute_time, comm_mb):
    # psutil.cpu_percent() with no Process target is system-wide (whatever
    # else happens to be running on the host), not this process's own usage.
    # Process().cpu_percent(interval=None) would be process-scoped but needs
    # a prior call on the *same* Process object to baseline against -- a
    # fresh object here every call would just always read 0.0 -- so block
    # briefly instead to measure synchronously regardless of prior state.
    cpu_usage = psutil.Process().cpu_percent(interval=0.1)
    ram_usage = psutil.Process().memory_info().rss / (1024 * 1024)

    metric = {
        "type": "client_metric",
        "client_id": client_id,
        "round": round_num,
        "cpu": cpu_usage,
        "ram": ram_usage,
        "time": compute_time,
        "comm_mb": comm_mb,
        "iowait": 0.0
    }
    # Absolute path: ClientApp train/evaluate calls run inside the
    # Simulation Engine's Ray actor processes, which do not inherit the
    # driver's /app/workspace working directory, so a relative path here
    # silently writes into whatever directory Ray happened to give the
    # actor instead of the run's real metrics.jsonl.
    with open("/app/workspace/metrics.jsonl", "a") as f:
        f.write(json.dumps(metric) + "\\n")

def log_distribution(client_id, counts):
    metric = {
        "type": "distribution",
        "client_id": client_id,
        "counts": counts
    }
    with open("/app/workspace/metrics.jsonl", "a") as f:
        f.write(json.dumps(metric) + "\\n")
"""
        with open(os.path.join(output_dir, "telemetry.py"), "w") as f:
            f.write(telemetry_code)
