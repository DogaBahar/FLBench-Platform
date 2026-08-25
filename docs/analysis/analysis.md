# FL Benchmark Platform — Comparative Analysis (Flower vs. NVFlare vs. FedML)

*Full three-framework analysis (accuracy + resource/data-engineering comparison), complete
including CPU, generated from `results/*.json`, 2026-08-25 (post Flower/NVFlare/FedML base-grid
reruns). Figures: `fig1`–`fig9` (PNG for preview, PDF for LaTeX embedding). Regenerate anytime with
`python3 scripts/generate_thesis_analysis.py`.*

## 1. Experimental setup

All runs share a common generic configuration (`epochs=3`, `batch_size=32`, `optimizer=adam`,
`learning_rate=0.001`, `fraction_fit=1.0`), translated by the platform into each framework's
native scripts, so the only variables are the ones deliberately swept:

- **Framework**: Flower (Simulation Engine), NVFlare (`nvflare simulator`), FedML (cross-silo, one
  container per participant)
- **Dataset**: CIFAR-100 (100-class), FEMNIST (62-class)
- **Clients**: 2, 5, 8
- **Rounds**: 5, 10, 15
- **Samples per client**: 500†, 1000, 2000, 4000
- **Data partitioning**: IID (primary grid), plus a Shard/Dirichlet/FedProx sweep at one
  representative configuration (5 clients, 10 rounds, 2000 samples/client, CIFAR-100)
- **Averaging strategy**: FedAvg (primary), FedProx (heterogeneity sweep)

† 500 samples/client only exists for Flower (an earlier pilot sweep predating the grid's
extension to NVFlare/FedML/FEMNIST). Doesn't affect the paired framework comparisons (§3, which
restrict to configurations present across all three frameworks) or the samples-per-client trend
(§4.3, a valid x-axis point on its own), but Flower's marginal accuracy distribution in Figures
2–3 includes 9 extra, lower-volume data points the other two frameworks lack.

## 2. Data quality: exclusions and fixes applied mid-study

Three distinct data-integrity issues were caught and resolved during data collection. All are
worth documenting since they affect what can be trusted in the dataset, and two are genuine
software bugs rather than infrastructure artifacts.

**(a) Task-redelivery race corrupting NVFlare results.** Celery's `task_acks_late=True` combined
with `autoretry_for=(docker.errors.APIError, ConnectionError)` allowed a benchmark task to be
redelivered while its original execution was still mid-run — typically triggered by restarting
the `celery_worker`/`backend` containers while a run was still in flight. The redelivered
execution's container-idempotency guard would force-remove and recreate the original's container,
which could unblock the original task's `container.wait()` early (reporting completion on
partial data) or, worse, leave two executions appending to the same shared `metrics.jsonl`
concurrently, producing a *higher* round count than configured with interleaved/duplicate round
numbers. **13 NVFlare runs total** were excluded on this basis across the study (9 from the
original sweep, 4 more surfacing during the Aug 25 base-grid rerun itself — the race is
probabilistic, not fully eliminated, just far less likely with the thread cap in place). All are
reliably identified by `n_rounds_logged != rounds` (checked in both directions — under- and
over-counting). `scripts/reconcile_sweep.py` diffs an intended sweep against `results/` and reruns
exactly what's missing or corrupted; every config affected has a valid replacement in the final
dataset (either a clean rerun or, where a rerun attempt was itself corrupted, the dedup logic in
§2c falls back to an earlier valid run of that same config).

**(b) FedML crashing on every FedProx configuration.** All FedML+FedProx runs (3/3 in the initial
sweep, reproducibly) failed with a generic "no round metrics produced" error. Root-caused via a
local repro (bypassing the platform to run the generated FedML server script directly, in the
foreground) to `fedml_adapter.py` passing the benchmark's `strategy` value straight through to
FedML's own `federated_optimizer` config field. The installed FedML version's cross-silo
`Server.__init__` only recognizes `"FedAvg"`/`"LSA"`/`"SA"` for a custom `ServerAggregator` and
raises a bare, unhelpful `Exception("Exception")` for anything else — including `"FedProx"` —
before any round can run. This is a genuine adapter bug, not an infrastructure issue: the
proximal term was already being applied correctly and independently in the generated client
code via `mu`/`global_params`, so FedML's own optimizer label never needed to reflect the actual
strategy. Fixed by hardcoding `federated_optimizer: "FedAvg"` regardless of the configured
strategy (`backend/adapters/fedml_adapter.py`); confirmed via the same local repro that the
server initializes correctly post-fix. All 3 FedProx configs were rerun successfully afterward.

**(c) CPU measurement bug, and rerun/deduplication.** All three adapters measured CPU via bare
`psutil.cpu_percent()`, which returns system-wide host load rather than the training process's
own usage — see §6.3 for the full story. After fixing this (and, separately, applying a
per-container thread cap that fixed a related resource-contention issue affecting NVFlare's
timing data), the Flower and NVFlare base grids (54 configs each) were rerun from scratch on
2026-08-25 to get valid CPU data and, for NVFlare, a larger clean-timing sample. FedML's base grid
was rerun the same way shortly after, completing the three-framework CPU comparison in §6.3.

Since reruns always get fresh `run_id`s, `results/` now has old and new entries for the same 54
configs side by side, for all three frameworks. `generate_thesis_analysis.py`'s
`dedupe_latest_per_config()` keeps only the most recent valid run per unique config (matched on
framework/dataset/clients/rounds/samples/partition/strategy, not `run_id`) — applied *after*
excluding corrupted runs (§2a), so a corrupted rerun correctly falls back to an earlier valid
attempt rather than leaving that config missing entirely.

**(d) Base-grid contamination from a hyperparameter-mismatched side-study.** Separately from the
main sweep, two runs were collected to compare against an external paper (UniFed) using
deliberately different hyperparameters to match that paper's setup (`epochs=1`, `batch_size=8`,
`optimizer=sgd`, `clients=10`, `rounds=25`, `samples_per_client=226` — vs. the main study's fixed
`epochs=3`/`batch_size=32`/`optimizer=adam` and `clients∈{2,5,8}`/`rounds∈{5,10,15}`). Because
these runs also happen to use `partition_strategy=iid`/`strategy=FedAvg`, an early version of
`generate_thesis_analysis.py`'s base-grid filter (which only checked those two fields) silently
included them as if they were two more uncontrolled points on the same scalability grid. The
effect was severe for how small the contamination was: two runs (one Flower, one FedML) were
enough to drop FEMNIST accuracy minimums to 5–13% (vs. a normal 65–87% range) and flip two
rounds/clients correlations from positive to significantly negative. Fixed by adding
`epochs == 3` to the base-grid filter (factored into a single `base_grid()` helper used
everywhere in the script, rather than the same filter condition duplicated seven times, which is
how the first version of the bug happened to only partially get noticed). The UniFed-comparison
runs remain in `results/` and are still available for their intended purpose — comparing against
the paper directly — just correctly excluded from this study's own internal grid.

The clean dataset used throughout this analysis is **188 runs** (368 completed − 13 excluded −
167 superseded-by-rerun duplicates), of which the base IID/FedAvg grid proper (§3–§4, §6) is
**171 runs**: 54 each for NVFlare and FedML (27 configs × 2 datasets), 63 for Flower (54 + 9 extra
pilot-phase points at 500 samples/client, CIFAR-100 only).

## 3. Framework comparison

**Figure 1** (`fig1_accuracy_by_framework`) shows the overall accuracy distribution per
framework, split by dataset, over the base IID/FedAvg grid. Raw distributions overlap
considerably — the real signal only emerges from a **paired comparison** across the 54
configurations present for all three frameworks (identical dataset/clients/rounds/samples_per_client,
IID/FedAvg — see `matched_configs.csv`), which controls for the exact configuration mix.

| Framework | Dataset  | Mean accuracy | SD     | Min   | Max    | n  |
|-----------|----------|---------------|--------|-------|--------|----|
| Flower    | CIFAR-100| 0.2170        | 0.0742 | 0.050 | 0.3588 | 36 |
| NVFlare   | CIFAR-100| 0.2444        | 0.0731 | 0.100 | 0.3800 | 27 |
| FedML     | CIFAR-100| 0.2107        | 0.0614 | 0.095 | 0.3272 | 27 |
| Flower    | FEMNIST  | 0.7878        | 0.0444 | 0.690 | 0.8466 | 27 |
| NVFlare   | FEMNIST  | 0.7732        | 0.0592 | 0.640 | 0.8575 | 27 |
| FedML     | FEMNIST  | 0.7671        | 0.0397 | 0.680 | 0.8203 | 27 |

**Paired comparison, overall (n=54):**

| Comparison | Mean diff | t | p | Winner in n/54 |
|---|---|---|---|---|
| Flower vs. NVFlare | +0.0072 | 1.657 | 0.104 (n.s.) | Flower 31/54 |
| Flower vs. FedML | +0.0270 | 8.257 | **<0.0001** | Flower 50/54 |
| NVFlare vs. FedML | +0.0199 | 5.257 | **<0.0001** | NVFlare 43/54 |

**Per-dataset breakdown:**

| Comparison | CIFAR-100 (n=27) | FEMNIST (n=27) |
|---|---|---|
| Flower vs. NVFlare | p=0.964 (n.s.) | p=**0.029** (Flower higher 18/27) |
| Flower vs. FedML | p<**0.0001** (Flower higher 26/27) | p=**0.0002** (Flower higher 24/27) |
| NVFlare vs. FedML | p<**0.0001** (NVFlare higher 24/27) | p=0.252 (n.s.) |

**Finding**: the core result from the earlier draft holds up after the rerun — pooled,
Flower/NVFlare are statistically indistinguishable while both beat FedML decisively (p<0.0001).
The per-dataset picture is more nuanced with the larger, cleaner dataset: on CIFAR-100, NVFlare
is essentially tied with Flower (p=0.96) and both clearly beat FedML; on FEMNIST, Flower now
edges out NVFlare significantly (p=0.029, a new finding not present in the earlier draft) while
NVFlare and FedML are statistically tied there. In other words, **FedML is reliably behind on
CIFAR-100 specifically, and reliably behind Flower (but not NVFlare) on FEMNIST** — a more precise
claim than "FedML is worse than both, always." This survived a full FedProx bug-fix, a corruption
cleanup, and a from-scratch rerun of two of the three frameworks (§2), so it isn't an artifact of
any single data snapshot. Worth noting as a discussion point rather than a definitive framework
indictment: FedML's cross-silo architecture (real GRPC across separate containers per participant,
vs. in-process simulation for Flower/NVFlare) introduces genuine implementation differences —
serialization, weight synchronization timing, effective batching — any of which could explain a
consistent accuracy gap without indicating a deeper correctness problem.

## 4. Scalability

### 4.1 Communication rounds (Figure 2, `fig2_accuracy_vs_rounds`)

| Framework | Dataset  | r      | p      |
|-----------|----------|--------|--------|
| Flower    | CIFAR-100| +0.212 | 0.214  |
| Flower    | FEMNIST  | +0.083 | 0.683  |
| NVFlare   | CIFAR-100| +0.281 | 0.155  |
| NVFlare   | FEMNIST  | +0.182 | 0.365  |
| FedML     | CIFAR-100| +0.326 | 0.097 (marginal) |
| FedML     | FEMNIST  | +0.294 | 0.136  |

**Finding**: unchanged from the earlier draft — no significant relationship between round count
and final accuracy in the 5–15 range, for any framework or dataset. FEMNIST is already near its
ceiling by round 5 for all three frameworks (Figure 5); CIFAR-100 doesn't improve enough within 15
rounds for round count to be a significant driver either. 5 rounds captures most of the achievable
accuracy at this model/dataset scale and CPU-only training budget — additional rounds mainly add
wall-clock cost.

### 4.2 Number of clients (Figure 3, `fig3_accuracy_vs_clients`)

| Framework | Dataset  | r      | p      |
|-----------|----------|--------|--------|
| Flower    | CIFAR-100| +0.369 | **0.0270** |
| Flower    | FEMNIST  | +0.429 | **0.0255** |
| NVFlare   | CIFAR-100| +0.269 | 0.175  |
| NVFlare   | FEMNIST  | +0.057 | 0.779  |
| FedML     | CIFAR-100| +0.340 | 0.083 (marginal) |
| FedML     | FEMNIST  | +0.129 | 0.522  |

**Finding**: Flower remains the only framework with a statistically significant positive
relationship between client count and accuracy, on both datasets — consistent with more clients
meaning more total training data seen per round at fixed `samples_per_client`. NVFlare's FEMNIST
correlation moved from the earlier draft's r=−0.029 to +0.057 with the rerun's cleaner data —
still nowhere near significant either way, reinforcing (not reversing) the earlier conclusion that
there's no real clients/accuracy relationship for NVFlare on FEMNIST. This is now the second time
this specific cell has been re-measured with different data and landed on the same "no effect"
conclusion, which is about as much confidence as this dataset can offer on that point.

### 4.3 Samples per client (Figure 4, `fig4_accuracy_vs_samples`)

| Framework | Dataset  | r      | p        |
|-----------|----------|--------|----------|
| Flower    | CIFAR-100| +0.846 | <0.0001  |
| Flower    | FEMNIST  | +0.739 | <0.0001  |
| NVFlare   | CIFAR-100| +0.851 | <0.0001  |
| NVFlare   | FEMNIST  | +0.749 | <0.0001  |
| FedML     | CIFAR-100| +0.823 | <0.0001  |
| FedML     | FEMNIST  | +0.715 | <0.0001  |

**Finding**: unchanged and, if anything, strengthened — this remains the dominant, most robust
driver of accuracy across the entire dataset. Strong, highly significant positive correlation for
every framework/dataset combination (r=0.72–0.85), clearly stronger than either the rounds or
clients effect, and the *only* trend that's consistent across all three frameworks without
exception, across two independent data collections. This is the single best-supported
quantitative claim in the dataset and should anchor whatever scalability narrative the thesis
builds.

### 4.4 Convergence trajectories (Figure 5, `fig5_convergence_curves`)

Overlays per-round accuracy for the largest matched configuration available per dataset across
frameworks (8 clients, 15 rounds, 1000 samples/client, both datasets). Complements §3's aggregate
statistics by showing *how* each framework arrives at its final accuracy, not just where it ends
up. NVFlare's CIFAR-100 curve is still the most oscillatory of the three (a dip around round 6),
though less extreme than in the pre-rerun data; Flower's curve remains the smoothest on both
datasets.

## 5. Data heterogeneity & FedProx (Figure 6)

Covers all three frameworks at the fixed representative configuration (5 clients, 10 rounds,
2000 samples/client, CIFAR-100) — 18 runs total (3 frameworks × 3 partition strategies × 2
averaging strategies), still **one replicate per condition** (no repeated seeds), so treat
magnitudes as illustrative rather than statistically established. Note: only the IID/FedAvg row
per framework was refreshed by the Aug 25 rerun (it's part of the base grid); the other five
conditions per framework are unchanged from the original sweep.

| Framework | Partition | Averaging | Accuracy | Fairness gap |
|---|---|---|---|---|
| Flower  | IID       | FedAvg  | 0.258 | 0.018 |
| Flower  | IID       | FedProx | 0.246 | 0.041 |
| Flower  | Shard     | FedAvg  | 0.226 | 0.051 |
| Flower  | Shard     | FedProx | 0.235 | 0.025 |
| Flower  | Dirichlet | FedAvg  | 0.254 | 0.039 |
| Flower  | Dirichlet | FedProx | 0.235 | 0.030 |
| NVFlare | IID       | FedAvg  | 0.270 | 0.030 |
| NVFlare | IID       | FedProx | 0.210 | 0.005 |
| NVFlare | Shard     | FedAvg  | 0.135 | **0.000** |
| NVFlare | Shard     | FedProx | 0.165 | **0.000** |
| NVFlare | Dirichlet | FedAvg  | 0.255 | 0.060 |
| NVFlare | Dirichlet | FedProx | 0.240 | 0.035 |
| FedML   | IID       | FedAvg  | 0.236 | 0.036 |
| FedML   | IID       | FedProx | 0.198 | 0.043 |
| FedML   | Shard     | FedAvg  | 0.469 | 0.069 |
| FedML   | Shard     | FedProx | 0.468 | 0.098 |
| FedML   | Dirichlet | FedAvg  | 0.307 | 0.027 |
| FedML   | Dirichlet | FedProx | 0.289 | 0.059 |

**Notable, unexplained pattern worth flagging rather than over-interpreting**: NVFlare's Shard
partitioning still produces both the lowest accuracy in the entire table (0.135–0.165) *and* an
exact 0.000 fairness gap — meaning every client converged equally, rather than unevenly. That's
unusual: low accuracy under severe non-IID shard partitioning normally comes *with* a large
fairness gap, not instead of one. Not a data-corruption artifact (round counts match configured
values). Two plausible non-exclusive explanations: (a) NVFlare's `worst_client_accuracy` tracking
may behave differently under shard partitioning specifically, or (b) the model may have collapsed
to a degenerate, class-agnostic solution all clients score equally poorly on. Recommend a small
follow-up (repeat this one condition with different seeds) before writing anything definitive.

FedML's Shard results remain the most striking directionally — noticeably *higher* accuracy
(0.468–0.469) than its own IID condition, counter to what non-IID partitioning usually does to
accuracy. Given FedML's established overall accuracy deficit (§3), this warrants the same caution
as the NVFlare pattern: interesting, single-replicate, not yet a thesis claim without repeats.

## 6. Resource / data-engineering comparison

Beyond accuracy, the platform captures per-client RAM, CPU, communication payload, and
server-side aggregation time. All four metrics are now usable in some form — the earlier draft
had to exclude CPU entirely; that's fixed as of this rerun (§6.3).

### 6.1 RAM (Figure 7, `fig7_ram_by_framework`)

`psutil.Process().memory_info().rss` was already correctly process-scoped in all three adapters,
so this metric is trustworthy across the full clean dataset with no contamination caveat.

| Framework | Dataset  | Mean RAM (MB) | SD   | n  |
|-----------|----------|---------------|------|----|
| NVFlare   | CIFAR-100| 618.9         | 19.9 | 27 |
| Flower    | CIFAR-100| 700.6         | 26.1 | 36 |
| FedML     | CIFAR-100| 764.0         | 17.8 | 27 |
| NVFlare   | FEMNIST  | 815.9         | 8.5  | 27 |
| Flower    | FEMNIST  | 924.4         | 14.0 | 27 |
| FedML     | FEMNIST  | 957.5         | 11.6 | 27 |

**Finding**: unchanged, and reinforced — the same clean ordering (NVFlare lowest, Flower middle,
FedML highest) holds after a complete rerun of two of the three frameworks, with the numbers
shifting by only single-digit MB. This is now confirmed as a robust, low-noise finding rather than
a snapshot artifact — the ~130–190MB gap between NVFlare and FedML per client remains the most
crisply-separated resource metric in the whole dataset. Maps sensibly onto architecture: FedML's
genuinely separate OS processes per participant likely carry more baseline Python/torch/GRPC
overhead per client than NVFlare's or Flower's in-process simulation approaches.

### 6.2 Server aggregation time (Figure 8, `fig8_aggregation_time_by_framework`)

**NVFlare figures here are restricted to runs started after the thread-cap fix
(2026-08-24 19:51) — see §2 and `NVFLARE_TIMING_CLEAN_CUTOFF`.** The Aug 25 rerun means this
restriction now excludes only 1 run instead of the 30+ excluded in the earlier draft — NVFlare's
timing sample size is essentially back to full (n=27/26 vs. the full dataset's 27/27), resolving
the "smaller n" limitation flagged previously.

| Framework | Dataset  | Mean (s) | SD    | n  |
|-----------|----------|----------|-------|----|
| FedML     | CIFAR-100| 0.000    | 0.001 | 27 |
| Flower    | CIFAR-100| 0.014    | 0.006 | 36 |
| NVFlare   | CIFAR-100| 3.625    | 0.759 | 27 |
| FedML     | FEMNIST  | 0.000    | 0.000 | 27 |
| Flower    | FEMNIST  | 0.012    | 0.004 | 27 |
| NVFlare   | FEMNIST  | 3.538    | 0.713 | 26 |

**Finding**: unchanged in substance — NVFlare's aggregation time (~3.5–3.6s) is roughly 250–300×
Flower's and ~1000× FedML's, now backed by a full sample rather than n=8/16. This is architectural,
not a bug: NVFlare's simulator persists model checkpoints to disk between rounds, where Flower's
and FedML's aggregation is a pure in-memory weighted average. Worth stating explicitly as a real
trade-off — NVFlare's persistence buys fault-tolerance/resume capability the other two don't have,
at a real, measured, now well-confirmed wall-clock cost per round.

### 6.3 CPU usage (Figure 9, `fig9_cpu_by_framework`) — now complete for all three frameworks

**The earlier draft of this analysis excluded CPU entirely.** All three adapters measured it via
bare `psutil.cpu_percent()`, which returns **system-wide** host load (whatever else happens to be
running on the host at that instant) rather than the training process's own usage — a different
function from the correctly-scoped `psutil.Process().memory_info()` used for RAM two lines below
it in the same generated code. This explained why Flower's numbers looked implausibly tiny
(1.8–3.3%) originally: Flower ran alone on a 256-core host, so its own container's CPU use barely
registered against the whole machine's utilization.

Fixed to `Process().cpu_percent(interval=0.1)` (a short blocking measurement, chosen because these
adapters create a fresh `Process()` object each call rather than holding a persistent handle, so
an unprimed `interval=None` call would always read `0.0`). The fix is empirically confirmed active
from **2026-08-24 23:26** onward (`CPU_CLEAN_CUTOFF`) — CPU values jump from a uniformly
implausible <3% before that instant to a consistent, plausible 10–20% range after it, across every
framework, at the exact same timestamp. All three frameworks' base grids were rerun post-fix
(§2c), so this table is now on the same full footing as RAM and aggregation time.

| Framework | Dataset  | Mean CPU % | SD   | n  |
|-----------|----------|------------|------|----|
| Flower    | CIFAR-100| 14.04      | 1.86 | 27 |
| FedML     | CIFAR-100| 15.10      | 0.86 | 27 |
| NVFlare   | CIFAR-100| 16.36      | 1.43 | 27 |
| Flower    | FEMNIST  | 12.20      | 1.07 | 27 |
| FedML     | FEMNIST  | 14.57      | 1.16 | 27 |
| NVFlare   | FEMNIST  | 15.84      | 1.23 | 23 |

**Finding**: a clean, consistent three-way ordering on *both* datasets — Flower lowest, FedML in
the middle, NVFlare highest — with tight within-framework variance throughout (SD under 2
percentage points in every group). This is the inverse of the RAM ordering (§6.1: NVFlare lowest,
Flower middle, FedML highest), which is the more interesting result: no framework is simply
"more efficient" across the board, each sits at a different point on a CPU/RAM trade-off curve.
Combined with §6.2's aggregation-time result (NVFlare ~250–300× Flower's, ~1000× FedML's), a
coherent systems picture emerges: NVFlare's checkpoint-persistence design spends more CPU time and
far more wall-clock time per round on serialization/disk I/O, in exchange for the smallest memory
footprint; FedML's fully-distributed, one-process-per-participant architecture sits in the middle
on CPU and aggregation time but costs the most RAM, plausibly from per-process Python/torch/GRPC
overhead multiplied across separate containers; Flower's in-process Ray-actor simulation is
lightest on CPU and aggregation time at a middling RAM cost. None of the three metrics alone tells
this story — it only emerges from having all three (plus accuracy) side by side, which is the
core argument for why this platform's four-metric comparison is more informative than any single
number a framework's own README might quote.

### 6.4 What still isn't usable

**Communication payload** (`avg_comm_size_mb`) is identical across all three frameworks (6.397MB
CIFAR-100, 6.320MB FEMNIST, zero variance) — it's computed as the raw serialized size of the
shared `FlexibleCNN` model's parameters, not actual wire-protocol overhead (framing, headers,
compression). Real, but not differentiating, so no figure was built for it.

## 7. Limitations & recommended next steps

1. **All data-quality issues identified during this study are resolved** (§2) — NVFlare
   corruption is rerun clean (with dedup correctly falling back to valid earlier runs where a
   rerun attempt was itself corrupted), the FedML FedProx crash has a confirmed root cause and
   fix, the CPU measurement bug is fixed with an empirically-verified cutoff and rerun for all
   three frameworks, and a base-grid contamination bug (a hyperparameter-mismatched side-study
   leaking into the controlled grid) was caught and fixed the same day it was introduced.
2. **The resource comparison (§6) is now complete** — RAM, aggregation time, and CPU all have
   full or near-full sample sizes (n=27, occasionally 23-26) across all three frameworks. This
   was the main open item from the previous draft and is no longer outstanding.
3. **Heterogeneity sweep is still single-replicate** (§5) — 3 frameworks, but n=1/condition. The
   two "notable patterns" flagged there need repeat runs (different seeds, same config) before
   they can support a thesis claim rather than just motivate one.
4. **Accuracy magnitudes are low in absolute terms** (CIFAR-100 tops out around 30–38%, FEMNIST
   around 64–87%) — expected given short local training (3 epochs), no LR schedule, and this being
   a *relative* framework/scalability comparison rather than an attempt at state-of-the-art
   accuracy. Caveat any absolute number quoted out of context.
5. **The FedML accuracy gap (§3) and the RAM/CPU/aggregation-time trade-offs (§6) are documented
   but not fully explained mechanistically.** Worth a short follow-up investigation (e.g. comparing
   per-round loss curves, or profiling where NVFlare's extra CPU/aggregation time actually goes)
   if the thesis wants to make a causal claim about *why*, not just *that*.
6. **Watch for the same class of bug that caused §2(d)** if more side-studies get added
   alongside the main sweep (e.g. more paper comparisons) — anything sharing `partition_strategy`/
   `strategy` with the main grid but different `epochs`/`batch_size`/`optimizer` needs either a
   distinguishing marker in `base_grid()`'s filter or to live in a clearly separate results
   directory, or it risks silently re-contaminating the controlled comparison the same way.

## Figure index

| File | Content |
|---|---|
| `fig1_accuracy_by_framework.png/pdf` | Accuracy distribution, all 3 frameworks, by dataset |
| `fig2_accuracy_vs_rounds.png/pdf` | Accuracy vs. rounds |
| `fig3_accuracy_vs_clients.png/pdf` | Accuracy vs. client count |
| `fig4_accuracy_vs_samples.png/pdf` | Accuracy vs. samples per client |
| `fig5_convergence_curves.png/pdf` | Per-round convergence, matched configuration |
| `fig6_heterogeneity_preliminary.png/pdf` | Partition strategy × FedProx, all 3 frameworks |
| `fig7_ram_by_framework.png/pdf` | Peak client RAM by framework |
| `fig8_aggregation_time_by_framework.png/pdf` | Server aggregation time (NVFlare post-fix only) |
| `fig9_cpu_by_framework.png/pdf` | Client CPU usage, all 3 frameworks (post-fix) |

Data files: `runs_clean.csv` (188-run clean dataset), `runs_all.csv` (all 368 before
exclusion/dedup), `matched_configs.csv` (54 configs matched across all 3 frameworks),
`summary_stats.csv`.
