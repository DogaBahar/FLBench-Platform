import os
import json
import shutil
from typing import Dict, Any, List
from adapters.base import FLFrameworkAdapter

# NVFlare's PTFileModelPersistor builds the global FlexibleCNN from this JSON
# config directly (it can't call each dataset's get_model() factory), so its
# shape has to be kept in sync with datasets/<dataset>/model.py by hand.
DATASET_MODEL_SHAPES = {
    "cifar100": {"in_channels": 3, "num_classes": 100},
    "femnist": {"in_channels": 1, "num_classes": 62},
}

class FlareAdapter(FLFrameworkAdapter):
    
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

        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        job_dir = os.path.join(output_dir, "job")
        app_dir = os.path.join(job_dir, "app")
        config_dir = os.path.join(app_dir, "config")
        custom_dir = os.path.join(app_dir, "custom")
        
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(custom_dir, exist_ok=True)

        source_dataset_dir = os.path.join("datasets", dataset)
        shutil.copy(os.path.join(source_dataset_dir, "model.py"), os.path.join(custom_dir, "model.py"))
        shutil.copy(os.path.join(source_dataset_dir, "dataset.py"), os.path.join(custom_dir, "dataset.py"))

        meta_config = {
            "name": "benchmark_job",
            "resource_spec": {"site-1": {"num_gpus": 0}},
            "deploy_map": {"app": ["@ALL"]}
        }
        with open(os.path.join(job_dir, "meta.json"), "w") as f:
            json.dump(meta_config, f, indent=2)

        server_config = {
            "format_version": 2,
            "min_clients": num_clients,
            "num_rounds": num_rounds,
            "workflows": [
                {
                    "id": "scatter_and_gather",
                    "name": "ScatterAndGather",
                    "args": {
                        "min_clients": num_clients,
                        "num_rounds": num_rounds,
                        "start_round": 1,
                        "wait_time_after_min_received": 0,
                        "aggregator_id": "aggregator",
                        "persistor_id": "persistor",
                        "shareable_generator_id": "shareable_generator",
                        "train_task_name": "train"
                    }
                }
            ],
            "components": [
                {
                    "id": "persistor",
                    "name": "PTFileModelPersistor",
                    "args": {
                        "model": {
                            "path": "model.FlexibleCNN",
                            "args": DATASET_MODEL_SHAPES.get(dataset, DATASET_MODEL_SHAPES["cifar100"])
                        }
                    }
                },
                {
                    "id": "shareable_generator",
                    "name": "FullModelShareableGenerator",
                    "args": {}
                },
                {
                    "id": "aggregator",
                    "name": "InTimeAccumulateWeightedAggregator",
                    "args": {"expected_data_kind": "WEIGHTS"}
                }
            ]
        }
        with open(os.path.join(config_dir, "config_fed_server.json"), "w") as f:
            json.dump(server_config, f, indent=2)

        client_config = {
            "format_version": 2,
            "executors": [
                {
                    "tasks": ["train"],
                    "executor": {
                        "id": "executor",
                        "path": "nvflare.app_opt.pt.client_api_launcher_executor.PTClientAPILauncherExecutor",
                        "args": {
                            "launcher_id": "launcher",
                            "pipe_id": "pipe",
                            "train_with_evaluation": False,
                            "external_pre_init_timeout": 300.0
                        }
                    }
                }
            ],
            "components": [
                {
                    "id": "launcher",
                    "path": "nvflare.app_common.launchers.subprocess_launcher.SubprocessLauncher",
                    "args": {
                        "script": "python custom/train_script.py" 
                    }
                },
                {
                    "id": "pipe",
                    "path": "nvflare.fuel.utils.pipe.file_pipe.FilePipe",
                    "args": {
                        "mode": "PASSIVE",
                        "root_path": "{WORKSPACE}/{JOB_ID}/{SITE_NAME}"
                    }
                }
            ]
        }
        with open(os.path.join(config_dir, "config_fed_client.json"), "w") as f:
            json.dump(client_config, f, indent=2)

        train_script = f"""
import os
import time
import json
import torch
import nvflare.client as flare
from model import get_model, train, test
from dataset import load_data
from telemetry import log_client_metrics, log_distribution

def main():
    flare.init()
    
    site_name = flare.get_site_name()
    client_id = int(site_name.replace("site-", "")) - 1 if "site-" in site_name else 0
    
    model = get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    trainloader, testloader = load_data(
        partition_strategy="{partition_strategy}",
        samples_per_client={samples_per_client},
        num_clients={num_clients},
        client_id=client_id,
        batch_size={batch_size},
        alpha={alpha},
        shards_per_client={shards_per_client}
    )
    
    counts = {{}}
    for _, labels in trainloader:
        for label in labels.numpy():
            lbl_str = str(label)
            counts[lbl_str] = counts.get(lbl_str, 0) + 1
    log_distribution(f"client_{{client_id}}", counts)

    # Track server aggregation time and TCP data inside the training flow safely
    round_start_time = time.time()

    while flare.is_running():
        input_model = flare.receive()
        if not input_model:
            break
            
        rnd = input_model.current_round
        
        # PTClientAPILauncherExecutor's NumpyToPTParamsConverter already
        # converts incoming weights to torch tensors before we see them here.
        for k, v in input_model.params.items():
            if k in model.state_dict():
                target_dtype = model.state_dict()[k].dtype
                model.state_dict()[k].copy_(v.to(dtype=target_dtype))

        # Snapshot the just-received global weights (before local training
        # mutates them in place) for the FedProx proximal term below.
        mu = {proximal_mu}
        global_params = [p.detach().clone() for p in model.parameters()] if mu > 0 else None

        # Every client evaluates this round's just-received global model on its
        # own held-out split (not just site-1) so per-round fairness across
        # clients (accuracy std, worst-client accuracy) can be measured, not
        # just one client's local test accuracy.
        loss, accuracy = test(model, testloader)
        with open("/app/workspace/metrics.jsonl", "a") as f:
            f.write(json.dumps({{
                "type": "client_eval_metric",
                "client_id": f"client_{{client_id}}",
                "round": rnd,
                "loss": float(loss),
                "accuracy": float(accuracy),
                "num_samples": len(testloader.dataset)
            }}) + "\\n")

        if client_id == 0:
            # Extract System Server Telemetry right here from the training side safely
            import psutil
            try:
                conns = psutil.net_connections(kind='tcp')
                tcp_est = sum(1 for c in conns if c.status == 'ESTABLISHED')
                tcp_wait = sum(1 for c in conns if c.status == 'TIME_WAIT')
            except Exception:
                tcp_est, tcp_wait = 0, 0

            agg_time = time.time() - round_start_time

            with open("/app/workspace/metrics.jsonl", "a") as f:
                f.write(json.dumps({{
                    "type": "server_metric",
                    "round": rnd,
                    "loss": float(loss),
                    "accuracy": float(accuracy)
                }}) + "\\n")

                f.write(json.dumps({{
                    "type": "server_sys_metric",
                    "round": rnd,
                    "agg_time": round(agg_time, 2),
                    "tcp_est": tcp_est,
                    "tcp_wait": tcp_wait
                }}) + "\\n")

        start_time = time.time()
        train(model, trainloader, {epochs}, {learning_rate}, "{optimizer_name}", mu=mu, global_params=global_params)
        compute_time = time.time() - start_time

        state_dict = model.state_dict()
        comm_mb = sum(v.element_size() * v.nelement() for v in state_dict.values()) / (1024 * 1024)
        log_client_metrics(f"client_{{client_id}}", rnd, compute_time, comm_mb)

        # Send raw torch tensors: PTClientAPILauncherExecutor's own
        # PTToNumpyParamsConverter converts them to numpy for the wire, and
        # calls v.cpu().numpy() itself -- pre-converting here would make it
        # crash with AttributeError on a plain numpy.ndarray.
        output_model = flare.FLModel(
            params={{k: v.detach().cpu() for k, v in state_dict.items()}},
            meta={{"NUM_STEPS_CURRENT_ROUND": len(trainloader.dataset)}}
        )
        round_start_time = time.time() # Reset for next round tracking
        flare.send(output_model)

if __name__ == "__main__":
    main()
"""
        with open(os.path.join(custom_dir, "train_script.py"), "w") as f:
            f.write(train_script)

        self.inject_telemetry(custom_dir, "default")

        wrapper_script = f"""
import os
import time
import json
import subprocess

start_time = time.time()

subprocess.run([
    "nvflare", "simulator", "job", 
    "-w", "/app/workspace/simulator_workdir", 
    "-n", "{num_clients}", 
    "-t", "{num_clients}"
])

total_time = time.time() - start_time

with open("/app/workspace/metrics.jsonl", "a") as f:
    f.write(json.dumps({{
        "type": "wall_clock_time",
        "time_seconds": round(total_time, 2)
    }}) + "\\n")
"""
        with open(os.path.join(output_dir, "run_simulator.py"), "w") as f:
            f.write(wrapper_script)

    def get_docker_commands(self, output_dir: str, num_clients: int) -> List[Dict[str, str]]:
        return [{
            "name": "server",
            "image": "benchmark-flare:latest",
            "command": "python run_simulator.py"
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