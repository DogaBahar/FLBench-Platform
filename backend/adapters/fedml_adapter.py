import os
import json
import shutil
from typing import Dict, Any, List
from adapters.base import FLFrameworkAdapter

class FedMLAdapter(FLFrameworkAdapter):
    
    def generate_configs(self, config: Dict[str, Any], output_dir: str) -> None:
        ui_config = config.get("config", {})
        fed_settings = ui_config.get("federated_settings", {})
        ml_settings = ui_config.get("ml_hyperparameters", {})
        data_settings = ui_config.get("data_simulation", {})
        
        num_rounds = fed_settings.get("rounds", 3)
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

        # 1. Save Configs & Copy ML Stack
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        source_dataset_dir = os.path.join("datasets", dataset)
        shutil.copy(os.path.join(source_dataset_dir, "model.py"), os.path.join(output_dir, "model.py"))
        shutil.copy(os.path.join(source_dataset_dir, "dataset.py"), os.path.join(output_dir, "dataset.py"))

        # 2. Generate FedML YAML Configuration
        fedml_config = f"""
common_args:
  training_type: "cross_silo"
  scenario: "horizontal"
  random_seed: 0

data_args:
  dataset: "{dataset}"

model_args:
  model: "FlexibleCNN"

train_args:
  federated_optimizer: "{strategy_name}"
  client_num_in_total: {num_clients}
  client_num_per_round: {num_clients}
  comm_round: {num_rounds}
  epochs: {epochs}
  batch_size: {batch_size}
  client_optimizer: "{optimizer_name}"
  learning_rate: {learning_rate}
  fedprox_mu: {proximal_mu}

validation_args:
  frequency_of_the_test: 1

comm_args:
  backend: "GRPC"
  grpc_ipconfig_path: "/app/workspace/grpc_ipconfig.csv"
"""
        with open(os.path.join(output_dir, "fedml_config.yaml"), "w") as f:
            f.write(fedml_config)

        # 3. Generate FedML Python Script with Subclassed Telemetry
        run_script = f"""
import os
import time
import json
import psutil
import torch
import fedml
from fedml import FedMLRunner
from fedml.core import ClientTrainer, ServerAggregator
from model import get_model, train, test
from dataset import load_data

class DashboardClientTrainer(ClientTrainer):
    def __init__(self, model, args, client_id):
        super().__init__(model, args)
        self.client_id = client_id
        self.trainloader, self.valloader = load_data(
            partition_strategy="{partition_strategy}",
            samples_per_client={samples_per_client},
            num_clients={num_clients},
            client_id=client_id - 1,
            batch_size={batch_size},
            alpha={alpha},
            shards_per_client={shards_per_client}
        )

        # __init__ runs exactly once per client process (unlike train(),
        # which fires every round), so this is the one natural place to
        # log the data distribution -- matching what the Flower/FLARE
        # adapters already do client-side.
        counts = {{}}
        for _, labels in self.trainloader:
            for label in labels.numpy():
                lbl_str = str(label)
                counts[lbl_str] = counts.get(lbl_str, 0) + 1
        with open("/app/workspace/metrics.jsonl", "a") as f:
            f.write(json.dumps({{
                "type": "distribution",
                "client_id": f"client_{{self.client_id}}",
                "counts": counts
            }}) + "\\n")

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        self.model.to(device)

        # Snapshot the global weights set_model_params() just loaded, before
        # local training mutates them, for the FedProx proximal term below.
        mu = {proximal_mu}
        global_params = [p.detach().clone() for p in self.model.parameters()] if mu > 0 else None

        start_time = time.time()

        train(self.model, self.trainloader, {epochs}, {learning_rate}, "{optimizer_name}", mu=mu, global_params=global_params)

        compute_time = time.time() - start_time
        cpu_usage = psutil.cpu_percent(interval=None)
        ram_usage = psutil.Process().memory_info().rss / (1024 * 1024)
        comm_mb = sum(v.element_size() * v.nelement() for v in self.model.state_dict().values()) / (1024 * 1024)

        # Post-training local validation, reported so the server can compute
        # a sample-weighted global loss/accuracy for this round once all
        # clients have checked in (see DashboardServerAggregator.aggregate).
        val_loss, val_accuracy = test(self.model, self.valloader)

        # FedML's own round_idx is 0-indexed; Flower and FLARE both report
        # rounds starting at 1. Shift by 1 here (and in aggregate() below)
        # so round numbers line up across frameworks in the results UI.
        display_round = args.round_idx + 1

        with open("/app/workspace/metrics.jsonl", "a") as f:
            f.write(json.dumps({{
                "type": "client_metric",
                "client_id": f"client_{{self.client_id}}",
                "round": display_round,
                "cpu": cpu_usage,
                "ram": ram_usage,
                "time": compute_time,
                "comm_mb": comm_mb,
                "iowait": 0.0
            }}) + "\\n")
            f.write(json.dumps({{
                "type": "client_eval_metric",
                "client_id": f"client_{{self.client_id}}",
                "round": display_round,
                "loss": val_loss,
                "accuracy": val_accuracy,
                "num_samples": len(self.valloader.dataset)
            }}) + "\\n")

class DashboardServerAggregator(ServerAggregator):
    def __init__(self, model, args):
        super().__init__(model, args)

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def test(self, test_data, device, args):
        pass # Standardized benchmark leaves testing to evaluation hooks

    def aggregate(self, *args, **kwargs):
        start = time.time()
        res = super().aggregate(*args, **kwargs)
        agg_time = time.time() - start

        # Matches the same +1 shift applied client-side (see train() above),
        # so this lines up with the "round" values clients already wrote.
        round_idx = self.args.round_idx + 1

        # Clients already wrote their post-training "client_eval_metric" for
        # this round before sending their model update, so those lines are
        # in the shared workspace file by the time all updates are in hand.
        total_samples, weighted_loss, weighted_accuracy = 0, 0.0, 0.0
        try:
            with open("/app/workspace/metrics.jsonl", "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if entry.get("type") == "client_eval_metric" and entry.get("round") == round_idx:
                        n = entry.get("num_samples", 0)
                        total_samples += n
                        weighted_loss += entry.get("loss", 0.0) * n
                        weighted_accuracy += entry.get("accuracy", 0.0) * n
        except FileNotFoundError:
            pass

        try:
            conns = psutil.net_connections(kind='tcp')
            tcp_est = sum(1 for c in conns if c.status == 'ESTABLISHED')
            tcp_wait = sum(1 for c in conns if c.status == 'TIME_WAIT')
        except Exception:
            tcp_est, tcp_wait = 0, 0

        with open("/app/workspace/metrics.jsonl", "a") as f:
            f.write(json.dumps({{
                "type": "server_sys_metric",
                "round": round_idx,
                "agg_time": round(agg_time, 2),
                "tcp_est": tcp_est,
                "tcp_wait": tcp_wait
            }}) + "\\n")
            if total_samples > 0:
                f.write(json.dumps({{
                    "type": "server_metric",
                    "round": round_idx,
                    "loss": weighted_loss / total_samples,
                    "accuracy": weighted_accuracy / total_samples
                }}) + "\\n")
        return res

if __name__ == "__main__":
    # The docker network only resolves the run-scoped container names
    # (run_<RUN_ID>_server, run_<RUN_ID>_client_i), so the gRPC ip
    # table has to be built at container start, once RUN_ID is known.
    run_id = os.environ.get("RUN_ID", "default")
    ip_config = "receiver_id,ip\\n"
    ip_config += f"0,run_{{run_id}}_server\\n"
    for i in range(1, {num_clients} + 1):
        ip_config += f"{{i}},run_{{run_id}}_client_{{i}}\\n"
    with open("/app/workspace/grpc_ipconfig.csv", "w") as f:
        f.write(ip_config)

    args = fedml.init()
    model = get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # FedML expects a dataset array [train_data_num, test_data_num, train_data_global, test_data_global, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    # We bypass this by injecting our DataLoaders directly into our custom Trainer!
    if args.role == "server":
        dummy_dataset = [1, 1, None, None, None, None, None, 100]
        aggregator = DashboardServerAggregator(model, args)

        # In FedML cross-silo, the server's whole blocking training loop
        # runs inside the FedMLRunner() constructor itself (Server.__init__
        # -> server_initializer.init_server() -> server_manager.run()).
        # Server.run() is a no-op (`pass`), so timing has to wrap the
        # constructor call, not the .run() call below it.
        run_start = time.time()
        runner = FedMLRunner(args, device, dummy_dataset, model, server_aggregator=aggregator)
        total_time = time.time() - run_start

        runner.run()

        with open("/app/workspace/metrics.jsonl", "a") as f:
            f.write(json.dumps({{
                "type": "wall_clock_time",
                "time_seconds": round(total_time, 2)
            }}) + "\\n")
    else:
        trainer = DashboardClientTrainer(model, args, args.rank)
        # FedMLTrainer.update_dataset() reads local_sample_number from
        # train_data_local_num_dict[client_index], not from our custom
        # trainer, and that value is what's reported to the server for
        # weighted aggregation. Leaving it None makes every client report
        # 0 samples, so the server divides by zero when averaging.
        client_index = args.rank - 1
        num_local_samples = len(trainer.trainloader.dataset)
        train_data_local_num_dict = {{client_index: num_local_samples}}
        dummy_dataset = [num_local_samples, 1, None, None, train_data_local_num_dict, None, None, 100]
        runner = FedMLRunner(args, device, dummy_dataset, model, client_trainer=trainer)
        runner.run()
"""
        with open(os.path.join(output_dir, "run_fedml.py"), "w") as f:
            f.write(run_script)

    def get_docker_commands(self, output_dir: str, num_clients: int) -> List[Dict[str, str]]:
        commands = [{
            "name": "server",
            "image": "benchmark-fedml:latest",
            "command": "python run_fedml.py --cf fedml_config.yaml --role server --rank 0 --run_id fedml_benchmark"
        }]
        
        for i in range(1, num_clients + 1):
            commands.append({
                "name": f"client_{i}",
                "image": "benchmark-fedml:latest",
                "command": f"python run_fedml.py --cf fedml_config.yaml --role client --rank {i} --run_id fedml_benchmark"
            })
            
        return commands

    def inject_telemetry(self, output_dir: str, run_id: str) -> None:
        pass # Telemetry is safely subclassed inside the run script above!