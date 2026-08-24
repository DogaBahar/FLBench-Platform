"""
Compares an intended sweep configs file (e.g. scripts/flare_sweep.json) against
what's actually validly present in results/, and writes out a new configs file
containing only what's still missing -- whether because it was never attempted,
or because the existing result is corrupted (wrong round count, a known failure
mode of the Celery task-redelivery race described in
scripts/generate_thesis_analysis.py's module docstring).

Matches by configuration content (framework/dataset/clients/rounds/samples_per_client/
partition_strategy/strategy/alpha/shards_per_client), not run_id, since every
(re)submission gets a fresh run_id.

Usage:
    python3 scripts/reconcile_sweep.py --configs scripts/flare_sweep.json --results-dir results
    # writes scripts/flare_sweep.remaining.json
"""
import argparse
import glob
import json
import os


def config_key(cfg: dict) -> tuple:
    """A hashable fingerprint of the parts of a run config that determine
    what it measures -- excludes anything incidental like run_id/timestamps."""
    ds = cfg["config"]["data_simulation"]
    fs = cfg["config"]["federated_settings"]
    return (
        cfg["framework"],
        cfg["clients"],
        ds["dataset"],
        ds["samples_per_client"],
        ds["partition_strategy"],
        ds.get("alpha"),
        ds.get("shards_per_client"),
        fs["rounds"],
        fs["strategy"],
    )


def result_key(d: dict) -> tuple:
    fc = d["config"]["full_config"]
    ds = fc["config"]["data_simulation"]
    fs = fc["config"]["federated_settings"]
    return (
        fc["framework"],
        fc["clients"],
        ds["dataset"],
        ds["samples_per_client"],
        ds["partition_strategy"],
        ds.get("alpha"),
        ds.get("shards_per_client"),
        fs["rounds"],
        fs["strategy"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", required=True, help="Original sweep configs JSON (the intended full list)")
    parser.add_argument("--results-dir", default="results", help="Directory of exported run JSON files")
    parser.add_argument("--out", default=None, help="Output path (default: <configs>.remaining.json)")
    args = parser.parse_args()

    with open(args.configs) as f:
        intended = json.load(f)

    valid_keys = set()
    corrupted_keys = set()
    for fp in glob.glob(os.path.join(args.results_dir, "*.json")):
        if fp.endswith("schema.json"):
            continue
        with open(fp) as f:
            d = json.load(f)
        if d.get("status") != "COMPLETED":
            continue
        fs = d["config"]["full_config"]["config"]["federated_settings"]
        n_logged = len(d["results"]["global_metrics"]["metrics_distributed"]["accuracy"])
        key = result_key(d)
        if n_logged == fs["rounds"]:
            valid_keys.add(key)
        else:
            corrupted_keys.add(key)

    remaining = [cfg for cfg in intended if config_key(cfg) not in valid_keys]

    out_path = args.out or args.configs.replace(".json", ".remaining.json")
    with open(out_path, "w") as f:
        json.dump(remaining, f, indent=2)
        f.write("\n")

    n_never_attempted = sum(1 for cfg in remaining if config_key(cfg) not in corrupted_keys)
    n_corrupted = sum(1 for cfg in remaining if config_key(cfg) in corrupted_keys)
    print(f"Intended configs: {len(intended)}")
    print(f"Already valid: {len(intended) - len(remaining)}")
    print(f"Remaining: {len(remaining)} ({n_never_attempted} never attempted, {n_corrupted} corrupted/needs rerun)")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
