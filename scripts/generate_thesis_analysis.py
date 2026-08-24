"""
Builds the Flower/NVFlare/FedML comparative analysis (accuracy distribution,
scalability trends, matched-config paired comparison, preliminary
heterogeneity observations, and a resource/data-engineering comparison of
RAM and server aggregation time) from results/*.json, and writes figures +
data tables to docs/analysis/.

Requires: pandas, numpy, matplotlib, seaborn, scipy (not stdlib-only, unlike
the other scripts/ tools -- install with:
    python3 -m pip install pandas numpy matplotlib seaborn scipy

Usage:
    python3 scripts/generate_thesis_analysis.py
    python3 scripts/generate_thesis_analysis.py --results-dir results --out-dir docs/analysis

What this script does NOT do: it does not regenerate analysis.md's prose. The
written analysis is interpretation, not a template -- rerun this script to
refresh figures/CSVs as more results land, then revisit the numbers quoted in
analysis.md by hand (or ask again) since new data can shift which findings
are significant.

Known data-quality issue this script guards against: a Celery task-redelivery
race (see backend/services/tasks.py's task_acks_late + autoretry_for, and
backend/services/orchestration.py's container-idempotency guard) can mark a
run COMPLETED after only partial rounds actually logged, if a redelivered
task's container recreation unblocks the original task's container.wait()
early. Such runs always have wall_clock_time_seconds == 0 (that telemetry
line is only written on genuine completion), which this script uses as the
exclusion filter -- see build_dataframe()'s `truncated` mask.

Known data-quality issue #2: NVFlare runs collected before the per-container
thread cap fix (OMP_NUM_THREADS/MKL_NUM_THREADS in orchestration.py) have
inflated, highly variable wall-clock/compute/aggregation timing from the same
contention that caused issue #1 above, even when their round count is valid.
See NVFLARE_TIMING_CLEAN_CUTOFF -- applied only to timing figures/stats, not
accuracy (which isn't wall-clock-dependent).

Known data-quality issue #3: cpu_usage_percent was measured via bare
psutil.cpu_percent() (system-wide host load, not this process's own usage)
in all three adapters until this was caught and fixed to
Process().cpu_percent(). No run collected before the fix has valid data for
this field -- it's excluded from all figures/stats here, not just filtered.
"""
import argparse
import glob
import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

PALETTE = {"flower": "#3b82f6", "nvflare": "#f59e0b", "fedml": "#10b981"}
FW_LABEL = {"flower": "Flower", "nvflare": "NVFlare", "fedml": "FedML"}
DS_LABEL = {"cifar100": "CIFAR-100", "femnist": "FEMNIST"}

# Representative configuration used for the (currently Flower-only, n=1/condition)
# heterogeneity/FedProx sweep -- matches the fixed point used when that sweep was
# designed (see scripts/flower_sweep.json's generator). Edit if you redesign it.
HETEROGENEITY_REP_CONFIG = dict(clients=5, rounds=10, samples_per_client=2000, dataset="cifar100")

# NVFlare runs started before this timestamp predate the per-container thread
# cap fix (OMP_NUM_THREADS/MKL_NUM_THREADS in orchestration.py) and were
# collected while a container-orphaning contention bug was active (see
# module docstring's item (a)) -- their wall-clock/compute-time/aggregation-
# time numbers are inflated and highly variable as a result (observed: pre
# ~15.1s +/- 15.6s vs post ~3.7s +/- 0.75s mean aggregation time on the same
# workload), even for runs whose round count is otherwise valid. Accuracy is
# unaffected (that's a property of the training itself, not wall-clock), so
# this filter is applied only for timing/resource figures and stats, never
# for accuracy. Adjust this if you know the actual fix-deployment time on
# your own re-run.
NVFLARE_TIMING_CLEAN_CUTOFF = pd.Timestamp("2026-08-24 19:51:00")


def clean_timing_subset(df: pd.DataFrame) -> pd.DataFrame:
    is_stale_nvflare = (df["framework"] == "nvflare") & (df["started_at"] < NVFLARE_TIMING_CLEAN_CUTOFF)
    dropped = is_stale_nvflare.sum()
    if dropped:
        print(f"(timing figures/stats only) excluding {dropped} pre-fix NVFlare run(s) "
              f"with contaminated timing data, started before {NVFLARE_TIMING_CLEAN_CUTOFF}")
    return df[~is_stale_nvflare].copy()


def load_results(results_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    files = [f for f in files if not f.endswith("schema.json")]

    rows = []
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        if d.get("status") != "COMPLETED":
            continue

        fc = d["config"]["full_config"]
        ds = fc["config"]["data_simulation"]
        fs = fc["config"]["federated_settings"]
        ml = fc["config"]["ml_hyperparameters"]
        gm = d["results"]["global_metrics"]
        acc = gm["metrics_distributed"]["accuracy"]
        loss = gm["losses_distributed"]
        std = gm["fairness_metrics"]["accuracy_std"]
        worst = gm["fairness_metrics"]["worst_client_accuracy"]

        final_acc = acc[-1][1] if acc else None
        final_worst = worst[-1][1] if worst else None

        cpu_vals, ram_vals, time_vals, comm_vals = [], [], [], []
        for client in d["results"]["client_logs"]:
            for log in client.get("logs", []):
                if log.get("action") != "fit":
                    continue
                cpu_vals.append(log.get("cpu_usage_percent"))
                ram_vals.append(log.get("peak_memory_mb"))
                time_vals.append(log.get("compute_time_seconds"))
                comm_vals.append(log.get("comm_size_mb"))

        server_logs = d["results"]["server_logs"]
        agg_vals = [s["aggregation_time_sec"] for s in server_logs]

        def avg(lst):
            vals = [v for v in lst if v is not None]
            return sum(vals) / len(vals) if vals else None

        rows.append({
            "run_id": d["run_id"],
            "framework": d["framework"],
            "dataset": d["dataset"],
            "strategy": fs["strategy"],
            "partition_strategy": ds.get("partition_strategy"),
            "clients": fc["clients"],
            "rounds": fs["rounds"],
            "samples_per_client": ds.get("samples_per_client"),
            "alpha": ds.get("alpha"),
            "shards_per_client": ds.get("shards_per_client"),
            "epochs": ml["epochs"],
            "final_accuracy": final_acc,
            "final_loss": loss[-1][1] if loss else None,
            "final_accuracy_std": std[-1][1] if std else None,
            "final_worst_client_accuracy": final_worst,
            "fairness_gap": (final_acc - final_worst) if (final_acc is not None and final_worst is not None) else None,
            "wall_clock_time_seconds": gm.get("wall_clock_time_seconds"),
            "avg_cpu_usage_percent": avg(cpu_vals),
            "avg_peak_memory_mb": avg(ram_vals),
            "avg_compute_time_seconds": avg(time_vals),
            "avg_comm_size_mb": avg(comm_vals),
            "avg_aggregation_time_sec": avg(agg_vals),
            "started_at": d["started_at"],
            "completed_at": d["completed_at"],
            "n_rounds_logged": len(acc),
        })

    df = pd.DataFrame(rows)
    df["started_at"] = pd.to_datetime(df["started_at"])
    df["completed_at"] = pd.to_datetime(df["completed_at"])
    return df


def exclude_truncated(df: pd.DataFrame) -> pd.DataFrame:
    # A redelivered/overlapping task execution (see module docstring) can either
    # truncate a run (fewer rounds logged than configured -- container replaced
    # mid-run) or, if two executions' containers both append to the same shared
    # metrics.jsonl concurrently, inflate it past the configured round count with
    # interleaved/duplicate round numbers. Either shape means the reported
    # "final" accuracy doesn't reliably correspond to the configured round count,
    # so both directions are excluded, not just under-counting.
    bad = df["n_rounds_logged"] != df["rounds"]
    if bad.any():
        print(f"Excluding {bad.sum()} truncated/corrupted run(s) "
              f"(logged round count != configured rounds -- see module docstring):")
        for _, row in df[bad].iterrows():
            direction = "under" if row["n_rounds_logged"] < row["rounds"] else "OVER"
            print(f"  - {row['run_id']} (configured {row['rounds']}, logged {row['n_rounds_logged']}, {direction}-counted)")
    return df[~bad].copy()


def fig_accuracy_by_framework(base: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    frameworks = [f for f in ["flower", "nvflare", "fedml"] if f in base["framework"].unique()]
    pal = {FW_LABEL[f]: PALETTE[f] for f in frameworks}
    sns.boxplot(data=base, x="dataset_label", y="final_accuracy", hue="framework_label",
                palette=pal, ax=ax, showfliers=False, width=0.6)
    sns.stripplot(data=base, x="dataset_label", y="final_accuracy", hue="framework_label",
                  dodge=True, palette=pal, ax=ax, alpha=0.5, size=4, legend=False,
                  edgecolor="white", linewidth=0.3)
    ax.set_xlabel("")
    ax.set_ylabel("Final accuracy")
    ax.set_title("Final accuracy distribution by framework (base grid, IID/FedAvg)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:len(frameworks)], labels[:len(frameworks)], title="Framework",
              loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig1_accuracy_by_framework.png", dpi=200)
    fig.savefig(f"{out_dir}/fig1_accuracy_by_framework.pdf")
    plt.close(fig)


def fig_scalability(base: pd.DataFrame, xvar: str, xlabel: str, fname: str, title: str, out_dir: str):
    frameworks = [f for f in ["flower", "nvflare", "fedml"] if f in base["framework"].unique()]
    agg = (base.groupby(["framework", "dataset", xvar])["final_accuracy"]
           .agg(["mean", "std", "count"]).reset_index())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, dsname in zip(axes, ["cifar100", "femnist"]):
        sub = agg[agg["dataset"] == dsname]
        for fw in frameworks:
            s = sub[sub["framework"] == fw].sort_values(xvar)
            if s.empty:
                continue
            ax.errorbar(s[xvar], s["mean"], yerr=s["std"], marker="o", capsize=3,
                        label=FW_LABEL[fw], color=PALETTE[fw], linewidth=2)
        ax.set_title(DS_LABEL.get(dsname, dsname))
        ax.set_xlabel(xlabel)
        xt = sorted(base.loc[base["dataset"] == dsname, xvar].unique())
        if xt:
            ax.set_xticks(xt)
    axes[0].set_ylabel("Final accuracy (mean ± SD across other grid dims)")
    axes[1].legend(title="Framework", loc="best", frameon=True)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/{fname}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out_dir}/{fname}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_convergence_curves(clean: pd.DataFrame, results_dir: str, out_dir: str):
    base = clean[(clean["partition_strategy"] == "iid") & (clean["strategy"] == "FedAvg")]
    pivot = base.pivot_table(index=["dataset", "clients", "rounds", "samples_per_client"],
                              columns="framework", values="run_id", aggfunc="first")
    frameworks = [f for f in ["flower", "nvflare", "fedml"] if f in base["framework"].unique()]
    matched = pivot.dropna(subset=frameworks) if len(frameworks) > 1 else pivot.dropna()
    if matched.empty:
        print("No matched configs across frameworks yet -- skipping convergence-curve figure.")
        return

    def load_curve(run_id):
        matches = glob.glob(os.path.join(results_dir, f"*{run_id}.json"))
        with open(matches[0]) as f:
            d = json.load(f)
        return d["results"]["global_metrics"]["metrics_distributed"]["accuracy"]

    datasets = [d for d in ["cifar100", "femnist"] if d in matched.index.get_level_values("dataset")]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.2))
    if len(datasets) == 1:
        axes = [axes]
    for ax, dsname in zip(axes, datasets):
        # pick the largest (clients, rounds) matched config available for this dataset
        sub = matched.loc[dsname].reset_index().sort_values(["clients", "rounds"], ascending=False)
        row = sub.iloc[0]
        for fw in frameworks:
            run_id = row.get(fw)
            if pd.isna(run_id):
                continue
            curve = load_curve(run_id)
            r, a = zip(*curve)
            ax.plot(r, a, marker="o", color=PALETTE[fw], label=FW_LABEL[fw], linewidth=2)
        ax.set_title(f"{DS_LABEL.get(dsname, dsname)}\n"
                     f"{int(row['clients'])} clients, {int(row['rounds'])} rounds, "
                     f"{int(row['samples_per_client'])} samples/client", fontsize=10)
        ax.set_xlabel("Round")
    axes[0].set_ylabel("Accuracy")
    axes[-1].legend(title="Framework", loc="best", frameon=True)
    fig.suptitle("Convergence trajectories at matched configuration", y=1.03)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig5_convergence_curves.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out_dir}/fig5_convergence_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_heterogeneity(clean: pd.DataFrame, out_dir: str):
    cfg = HETEROGENEITY_REP_CONFIG
    het = clean[(clean["clients"] == cfg["clients"]) & (clean["rounds"] == cfg["rounds"]) &
                (clean["samples_per_client"] == cfg["samples_per_client"]) &
                (clean["dataset"] == cfg["dataset"])].copy()
    if het.empty:
        print("No runs found at the heterogeneity representative config -- skipping fig6.")
        return

    het["condition"] = (het["framework"].map(FW_LABEL) + ": " +
                         het["partition_strategy"].str.capitalize() + " / " + het["strategy"])
    het = het.sort_values(["framework", "partition_strategy", "strategy"])

    fig, axes = plt.subplots(1, 2, figsize=(max(11, 1.8 * len(het)), 4.5))
    colors = ["#3b82f6" if "FedAvg" in c else "#8b5cf6" for c in het["condition"]]

    axes[0].bar(het["condition"], het["final_accuracy"], color=colors)
    axes[0].set_ylabel("Final accuracy")
    axes[0].set_title("Final accuracy by partition × averaging strategy")
    axes[0].tick_params(axis="x", rotation=35)
    for tick in axes[0].get_xticklabels():
        tick.set_ha("right")

    axes[1].bar(het["condition"], het["fairness_gap"], color=colors)
    axes[1].set_ylabel("Fairness gap (mean acc. − worst-client acc.)")
    axes[1].set_title("Client fairness gap by condition")
    axes[1].tick_params(axis="x", rotation=35)
    for tick in axes[1].get_xticklabels():
        tick.set_ha("right")

    n_per_condition = het.groupby(["framework", "partition_strategy", "strategy"]).size().max()
    fig.suptitle(
        f"Heterogeneity & FedProx effect (n={n_per_condition}/condition, "
        f"{cfg['clients']} clients/{cfg['rounds']} rounds/{cfg['samples_per_client']} samples, "
        f"{DS_LABEL.get(cfg['dataset'], cfg['dataset'])})"
        + (" -- PRELIMINARY, single replicate" if n_per_condition == 1 else ""),
        fontsize=10, y=1.04)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig6_heterogeneity_preliminary.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out_dir}/fig6_heterogeneity_preliminary.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_ram_by_framework(base: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    frameworks = [f for f in ["flower", "nvflare", "fedml"] if f in base["framework"].unique()]
    pal = {FW_LABEL[f]: PALETTE[f] for f in frameworks}
    sns.boxplot(data=base, x="dataset_label", y="avg_peak_memory_mb", hue="framework_label",
                palette=pal, ax=ax, showfliers=False, width=0.6)
    sns.stripplot(data=base, x="dataset_label", y="avg_peak_memory_mb", hue="framework_label",
                  dodge=True, palette=pal, ax=ax, alpha=0.5, size=4, legend=False,
                  edgecolor="white", linewidth=0.3)
    ax.set_xlabel("")
    ax.set_ylabel("Avg. peak client RAM (MB)")
    ax.set_title("Peak client memory usage by framework (base grid, IID/FedAvg)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:len(frameworks)], labels[:len(frameworks)], title="Framework",
              loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig7_ram_by_framework.png", dpi=200)
    fig.savefig(f"{out_dir}/fig7_ram_by_framework.pdf")
    plt.close(fig)


def fig_aggregation_time_by_framework(clean: pd.DataFrame, out_dir: str):
    timing = clean_timing_subset(clean)
    base = timing[(timing["partition_strategy"] == "iid") & (timing["strategy"] == "FedAvg")]
    frameworks = [f for f in ["flower", "nvflare", "fedml"] if f in base["framework"].unique()]
    pal = {FW_LABEL[f]: PALETTE[f] for f in frameworks}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(data=base, x="dataset_label", y="avg_aggregation_time_sec", hue="framework_label",
                palette=pal, ax=ax, showfliers=False, width=0.6)
    sns.stripplot(data=base, x="dataset_label", y="avg_aggregation_time_sec", hue="framework_label",
                  dodge=True, palette=pal, ax=ax, alpha=0.5, size=4, legend=False,
                  edgecolor="white", linewidth=0.3)
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_xlabel("")
    ax.set_ylabel("Avg. server aggregation time (s, log scale)")
    ax.set_title("Server aggregation time by framework\n(NVFlare restricted to post-fix runs -- see module docstring)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:len(frameworks)], labels[:len(frameworks)], title="Framework",
              loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig8_aggregation_time_by_framework.png", dpi=200)
    fig.savefig(f"{out_dir}/fig8_aggregation_time_by_framework.pdf")
    plt.close(fig)


def print_stats(clean: pd.DataFrame):
    base = clean[(clean["partition_strategy"] == "iid") & (clean["strategy"] == "FedAvg")]
    frameworks = [f for f in ["flower", "nvflare", "fedml"] if f in base["framework"].unique()]

    print("\n=== Summary stats (base grid, IID/FedAvg) ===")
    print(base.groupby(["framework", "dataset"])["final_accuracy"]
          .agg(["mean", "std", "min", "max", "count"]).round(4))

    if len(frameworks) >= 2:
        pivot = base.pivot_table(index=["dataset", "clients", "rounds", "samples_per_client"],
                                  columns="framework", values="final_accuracy", aggfunc="first")
        for fw_a, fw_b in itertools.combinations(frameworks, 2):
            pair = pivot[[fw_a, fw_b]].dropna()
            if len(pair) < 2:
                continue
            diff = pair[fw_a] - pair[fw_b]
            t, p = stats.ttest_rel(pair[fw_a], pair[fw_b])
            w, pw = stats.wilcoxon(pair[fw_a], pair[fw_b])
            print(f"\n=== Paired comparison: {FW_LABEL[fw_a]} vs {FW_LABEL[fw_b]} (n={len(pair)}) ===")
            print(f"Mean({FW_LABEL[fw_a]} - {FW_LABEL[fw_b]}): {diff.mean():.4f}  SD: {diff.std():.4f}")
            print(f"Paired t-test: t={t:.3f}, p={p:.4f}")
            print(f"Wilcoxon signed-rank: W={w:.3f}, p={pw:.4f}")
            print(f"{FW_LABEL[fw_a]} higher in {(diff > 0).sum()}/{len(diff)} matched configs")

    print("\n=== Trend correlations (Pearson r, accuracy vs X) ===")
    for xvar in ["rounds", "clients", "samples_per_client"]:
        print(f"--- {xvar} ---")
        for fw in frameworks:
            for dsname in sorted(base["dataset"].unique()):
                sub = base[(base["framework"] == fw) & (base["dataset"] == dsname)]
                if len(sub) < 3:
                    continue
                r, p = stats.pearsonr(sub[xvar], sub["final_accuracy"])
                print(f"  {fw:8s} {dsname:10s}: r={r:+.3f} p={p:.4f} n={len(sub)}")

    print("\n=== RAM: avg peak client memory (MB) -- valid across all runs ===")
    print(base.groupby(["framework", "dataset"])["avg_peak_memory_mb"]
          .agg(["mean", "std", "min", "max", "count"]).round(1))

    timing = clean_timing_subset(clean)
    timing_base = timing[(timing["partition_strategy"] == "iid") & (timing["strategy"] == "FedAvg")]
    print("\n=== Server aggregation time (s) -- NVFlare restricted to post-fix runs ===")
    print(timing_base.groupby(["framework", "dataset"])["avg_aggregation_time_sec"]
          .agg(["mean", "std", "min", "max", "count"]).round(3))

    print("\nNote: avg_comm_size_mb is identical across all frameworks (raw serialized parameter "
          "size of the same shared model architecture, not actual wire-protocol overhead) -- not "
          "a differentiating metric, omitted from figures.")
    print("Note: cpu_usage_percent is excluded entirely. Historically measured via bare "
          "psutil.cpu_percent() (system-wide load, not this process's own usage) in all three "
          "adapters -- fixed to Process().cpu_percent() going forward, but no existing run has "
          "valid data for this field.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results", help="Directory of exported run JSON files")
    parser.add_argument("--out-dir", default="docs/analysis", help="Where to write figures/CSVs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)

    df = load_results(args.results_dir)
    print(f"Loaded {len(df)} completed runs from {args.results_dir}/")
    clean = exclude_truncated(df)
    print(f"Clean dataset: {len(clean)} runs")

    clean["framework_label"] = clean["framework"].map(FW_LABEL)
    clean["dataset_label"] = clean["dataset"].map(DS_LABEL)

    df.to_csv(f"{args.out_dir}/runs_all.csv", index=False)
    clean.to_csv(f"{args.out_dir}/runs_clean.csv", index=False)

    base = clean[(clean["partition_strategy"] == "iid") & (clean["strategy"] == "FedAvg")]
    base.groupby(["framework", "dataset"])["final_accuracy"].agg(
        ["mean", "std", "min", "max", "count"]).round(4).to_csv(f"{args.out_dir}/summary_stats.csv")

    frameworks = [f for f in ["flower", "nvflare", "fedml"] if f in base["framework"].unique()]
    if len(frameworks) >= 2:
        pivot = base.pivot_table(index=["dataset", "clients", "rounds", "samples_per_client"],
                                  columns="framework", values="final_accuracy", aggfunc="first").dropna()
        pivot.to_csv(f"{args.out_dir}/matched_configs.csv")

    fig_accuracy_by_framework(base, args.out_dir)
    fig_scalability(base, "rounds", "Communication rounds", "fig2_accuracy_vs_rounds",
                     "Accuracy vs. number of federated rounds", args.out_dir)
    fig_scalability(base, "clients", "Number of clients", "fig3_accuracy_vs_clients",
                     "Accuracy vs. number of clients", args.out_dir)
    fig_scalability(base, "samples_per_client", "Samples per client", "fig4_accuracy_vs_samples",
                     "Accuracy vs. local data volume per client", args.out_dir)
    fig_convergence_curves(clean, args.results_dir, args.out_dir)
    fig_heterogeneity(clean, args.out_dir)
    fig_ram_by_framework(base, args.out_dir)
    fig_aggregation_time_by_framework(clean, args.out_dir)
    print(f"\nFigures + CSVs written to {args.out_dir}/")

    print_stats(clean)


if __name__ == "__main__":
    main()
