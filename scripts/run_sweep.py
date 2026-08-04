"""
Runs a batch of benchmarks against a live platform instance (docker compose
up, as usual) and reports a summary at the end. Stdlib only -- no pip
install needed to use it.

Each entry in the configs file is exactly the body POST /api/benchmark
expects (see README's Quickstart curl example / domain/schemas.py), so any
existing single-run curl payload can be dropped straight into the list.
Runs execute one at a time, deliberately: the orchestrator spins up real
Docker containers per run, and running several concurrently competes for
host CPU/RAM and (for FedML) per-run container networking.

Usage:
    python3 scripts/run_sweep.py --configs scripts/sweep_example.json
    python3 scripts/run_sweep.py --configs my_sweep.json --export-dir results/ --submitted-by "your-name"

Config file format: a JSON list of objects, each shaped like:
    {
      "framework": "flower",
      "clients": 3,
      "config": {
        "federated_settings": {"rounds": 3, "strategy": "FedAvg", "fraction_fit": 1},
        "ml_hyperparameters": {"epochs": 2, "batch_size": 32, "learning_rate": 0.001, "optimizer": "adam"},
        "data_simulation": {"dataset": "cifar100", "samples_per_client": 500, "partition_strategy": "iid"}
      }
    }
See scripts/sweep_example.json for a working starting point covering all
three frameworks and both strategies -- edit/duplicate entries to build
whatever grid you want (there's no separate "grid mode": a Python list
comprehension over frameworks/datasets/strategies, dumped to JSON, is the
same thing and easier to reason about than a DSL for it).
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def api_call(base_url, method, path, body=None):
    url = f"{base_url}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def start_run(base_url, run_config):
    status, body = api_call(base_url, "POST", "/api/benchmark", run_config)
    if status != 201:
        raise RuntimeError(f"Failed to start run: HTTP {status} {body}")
    return body["run_id"]


def force_cleanup(run_id):
    """Best-effort: docker rm -f any container this run spawned, so a hung
    run doesn't tie up host resources for the rest of the sweep. Container
    names follow run_<run_id>_<name> (see services/orchestration.py)."""
    ps = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"name=run_{run_id}_"],
        capture_output=True, text=True,
    )
    container_ids = ps.stdout.split()
    if container_ids:
        subprocess.run(["docker", "rm", "-f", *container_ids], capture_output=True)
    return container_ids


def wait_for_run(base_url, run_id, poll_interval, timeout):
    start = time.time()
    while True:
        status, body = api_call(base_url, "GET", f"/api/runs/{run_id}")
        if status != 200:
            return {"status": "ERROR", "detail": body}

        run_status = body.get("status")
        if run_status in ("COMPLETED", "FAILED"):
            return body

        if time.time() - start > timeout:
            killed = force_cleanup(run_id)
            return {"status": "TIMEOUT", "killed_containers": killed}

        time.sleep(poll_interval)


def export_run(base_url, run_id, export_dir, submitted_by):
    params = f"?submitted_by={urllib.parse.quote(submitted_by)}" if submitted_by else ""
    status, body = api_call(base_url, "GET", f"/api/runs/{run_id}/export{params}")
    if status != 200:
        print(f"  (export failed: HTTP {status})")
        return
    os.makedirs(export_dir, exist_ok=True)
    filename = f"{body['framework']}-{body['dataset']}-{body['run_id']}.json"
    out_path = os.path.join(export_dir, filename)
    with open(out_path, "w") as f:
        json.dump(body, f, indent=2)
    print(f"  exported -> {out_path}")


def summarize(result):
    if result.get("status") != "COMPLETED":
        return result.get("status")
    acc = result.get("global_metrics", {}).get("metrics_distributed", {}).get("accuracy", [])
    final_acc = acc[-1][1] if acc else None
    return f"COMPLETED (final accuracy={final_acc})"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", required=True, help="Path to a JSON file with a list of run configs")
    parser.add_argument("--base-url", default="http://localhost:5001", help="Backend base URL")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Seconds between status polls")
    parser.add_argument("--timeout", type=float, default=1800.0, help="Max seconds to wait for a single run")
    parser.add_argument("--export-dir", default=None, help="If set, export each COMPLETED run here (results/ shape)")
    parser.add_argument("--submitted-by", default=None, help="Credit name/handle for --export-dir exports")
    args = parser.parse_args()

    with open(args.configs) as f:
        run_configs = json.load(f)

    print(f"Loaded {len(run_configs)} run config(s) from {args.configs}\n")

    results = []
    for i, run_config in enumerate(run_configs, start=1):
        label = f"[{i}/{len(run_configs)}] {run_config.get('framework')}/{run_config['config']['data_simulation']['dataset']}/{run_config['config']['federated_settings'].get('strategy', 'FedAvg')}"
        print(label)
        try:
            run_id = start_run(args.base_url, run_config)
        except Exception as e:
            print(f"  FAILED TO START: {e}\n")
            results.append({"config": run_config, "run_id": None, "status": f"START_ERROR: {e}"})
            continue

        print(f"  run_id={run_id}, waiting (poll every {args.poll_interval}s, timeout {args.timeout}s)...")
        result = wait_for_run(args.base_url, run_id, args.poll_interval, args.timeout)
        summary = summarize(result)
        print(f"  {summary}")

        if result.get("status") == "COMPLETED" and args.export_dir:
            export_run(args.base_url, run_id, args.export_dir, args.submitted_by)

        results.append({"config": run_config, "run_id": run_id, "status": summary})
        print()

    print("=== Sweep summary ===")
    for r in results:
        cfg = r["config"]
        print(f"{r['run_id'] or '(not started)':40s} {cfg.get('framework'):10s} {cfg['config']['data_simulation']['dataset']:10s} {r['status']}")

    failures = [r for r in results if r["run_id"] is None or "COMPLETED" not in r["status"]]
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
