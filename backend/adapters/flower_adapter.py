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

        server_code = f"""
import flwr as fl
import json
import time
import psutil

def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {{"accuracy": sum(accuracies) / sum(examples)}}

class CustomFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        start_time = time.time()
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        agg_time = time.time() - start_time
        
        conns = psutil.net_connections(kind='tcp')
        tcp_est = sum(1 for c in conns if c.status == 'ESTABLISHED')
        tcp_wait = sum(1 for c in conns if c.status == 'TIME_WAIT')
        
        with open("metrics.jsonl", "a") as f:
            f.write(json.dumps({{
                "type": "server_sys_metric",
                "round": rnd,
                "agg_time": agg_time,
                "tcp_est": tcp_est,
                "tcp_wait": tcp_wait
            }}) + "\\n")
            
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, rnd, results, failures):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(rnd, results, failures)
        
        if aggregated_metrics and "accuracy" in aggregated_metrics:
            acc = aggregated_metrics["accuracy"]
            with open("metrics.jsonl", "a") as f:
                f.write(json.dumps({{
                    "type": "server_metric",
                    "round": rnd,
                    "loss": float(aggregated_loss) if aggregated_loss else 0.0,
                    "accuracy": float(acc)
                }}) + "\\n")
                
        return aggregated_loss, aggregated_metrics

if __name__ == "__main__":
    strategy = CustomFedAvg(
        fraction_fit={fraction_fit},
        min_fit_clients={num_clients},
        min_available_clients={num_clients},
        evaluate_metrics_aggregation_fn=weighted_average
    )

    run_start = time.time()

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds={num_rounds}),
        strategy=strategy
    )

    total_time = time.time() - run_start
    with open("metrics.jsonl", "a") as f:
        f.write(json.dumps({{
            "type": "wall_clock_time",
            "time_seconds": round(total_time, 2)
        }}) + "\\n")
"""
        with open(os.path.join(output_dir, "server.py"), "w") as f:
            f.write(server_code)

        client_code = f"""
import os
import time
import flwr as fl
import torch
from model import get_model, train, test
from dataset import load_data
from telemetry import log_client_metrics, log_distribution

CLIENT_ID = int(os.environ.get("CLIENT_INDEX", 0))
RUN_ID = os.environ.get("RUN_ID", "default")
SERVER_HOSTNAME = f"run_{{RUN_ID}}_server:8080"

class FLClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = get_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.trainloader, self.testloader = load_data(
            partition_strategy="{partition_strategy}",
            samples_per_client={samples_per_client},
            num_clients={num_clients},
            client_id=CLIENT_ID,
            batch_size={batch_size},
            alpha={alpha},
            shards_per_client={shards_per_client}
        )
        
        # --- NEW: Calculate and log label distribution once on boot ---
        counts = {{}}
        for _, labels in self.trainloader:
            for label in labels.numpy():
                lbl_str = str(label)
                counts[lbl_str] = counts.get(lbl_str, 0) + 1
        log_distribution(f"client_{{CLIENT_ID}}", counts)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = dict({{k: torch.tensor(v) for k, v in params_dict}})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        start_time = time.time()
        
        train(self.model, self.trainloader, {epochs}, {learning_rate}, "{optimizer_name}")
        
        compute_time = time.time() - start_time
        rnd = config.get("server_round", getattr(self, "_round_counter", 0) + 1)
        self._round_counter = rnd 
        
        updated_parameters = self.get_parameters(config)
        comm_mb = sum(p.nbytes for p in updated_parameters) / (1024 * 1024)
        log_client_metrics(f"client_{{CLIENT_ID}}", rnd, compute_time, comm_mb)
        return updated_parameters, len(self.trainloader.dataset), {{}}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader)
        return float(loss), len(self.testloader.dataset), {{"accuracy": float(accuracy)}}

if __name__ == "__main__":
    fl.client.start_client(server_address=SERVER_HOSTNAME, client=FLClient().to_client())
"""
        with open(os.path.join(output_dir, "client.py"), "w") as f:
            f.write(client_code)

    def get_docker_commands(self, output_dir: str, num_clients: int) -> List[Dict[str, str]]:
        base_image = "benchmark-flower:latest" 
        specs = [{
            "name": "server",
            "image": base_image,
            "command": "python server.py"
        }]
        for i in range(num_clients):
            specs.append({
                "name": f"client_{i}",
                "image": base_image,
                "command": "python client.py",
                "env": {"CLIENT_INDEX": str(i)}
            })
        return specs

    def inject_telemetry(self, output_dir: str, run_id: str) -> None:
        telemetry_code = """
import json
import psutil

def log_client_metrics(client_id, round_num, compute_time, comm_mb):
    cpu_usage = psutil.cpu_percent(interval=None)
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
    with open("metrics.jsonl", "a") as f:
        f.write(json.dumps(metric) + "\\n")

def log_distribution(client_id, counts):
    metric = {
        "type": "distribution",
        "client_id": client_id,
        "counts": counts
    }
    with open("metrics.jsonl", "a") as f:
        f.write(json.dumps(metric) + "\\n")
"""
        with open(os.path.join(output_dir, "telemetry.py"), "w") as f:
            f.write(telemetry_code)