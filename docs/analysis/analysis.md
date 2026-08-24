# FL Benchmark Platform — Comparative Analysis (Flower vs. NVFlare vs. FedML)

*Full three-framework analysis (accuracy + resource/data-engineering comparison) generated from
`results/*.json`, 2026-08-25. Figures: `fig1`–`fig8` (PNG for preview, PDF for LaTeX embedding).
Regenerate anytime with `python3 scripts/generate_thesis_analysis.py`.*

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
2–3 includes one extra, lower-volume condition the other two frameworks lack.

## 2. Data quality: exclusions and fixes applied mid-study

Two distinct data-integrity issues were caught and resolved during data collection — both are
worth documenting since they affect what can be trusted in the dataset, and the second is a
genuine software bug rather than an infrastructure artifact.

**(a) Task-redelivery race corrupting NVFlare results.** Celery's `task_acks_late=True` combined
with `autoretry_for=(docker.errors.APIError, ConnectionError)` allowed a benchmark task to be
redelivered while its original execution was still mid-run — typically triggered by restarting
the `celery_worker`/`backend` containers while a run was still in flight. The redelivered
execution's container-idempotency guard would force-remove and recreate the original's container,
which could unblock the original task's `container.wait()` early (reporting completion on
partial data) or, worse, leave two executions appending to the same shared `metrics.jsonl`
concurrently, producing a *higher* round count than configured with interleaved/duplicate round
numbers. **9 NVFlare runs** were excluded on this basis (all reliably identified by
`wall_clock_time_seconds == 0`, a telemetry line only written on genuine completion, or by
`n_rounds_logged != rounds` more generally — the exclusion filter checks both directions). All 9
were subsequently rerun cleanly once the platform was left idle through a full restart cycle
(see `scripts/reconcile_sweep.py`, which diffs an intended sweep against `results/` and reruns
exactly what's missing or corrupted).

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

The clean dataset used throughout this analysis is **195 runs** (204 completed − 9 excluded).

## 3. Framework comparison

**Figure 1** (`fig1_accuracy_by_framework`) shows the overall accuracy distribution per
framework, split by dataset, over the base IID/FedAvg grid. Raw distributions overlap
considerably — the real signal only emerges from a **paired comparison** across the 54
configurations present for all three frameworks (identical dataset/clients/rounds/samples_per_client,
IID/FedAvg — see `matched_configs.csv`), which controls for the exact configuration mix.

| Framework | Dataset  | Mean accuracy | SD     | Min   | Max    | n  |
|-----------|----------|---------------|--------|-------|--------|----|
| Flower    | CIFAR-100| 0.2061        | 0.0696 | 0.050 | 0.3559 | 45 |
| NVFlare   | CIFAR-100| 0.2429        | 0.0657 | 0.120 | 0.3575 | 27 |
| FedML     | CIFAR-100| 0.2107        | 0.0614 | 0.095 | 0.3272 | 27 |
| Flower    | FEMNIST  | 0.7913        | 0.0407 | 0.700 | 0.8450 | 27 |
| NVFlare   | FEMNIST  | 0.7791        | 0.0555 | 0.650 | 0.8650 | 27 |
| FedML     | FEMNIST  | 0.7671        | 0.0397 | 0.680 | 0.8203 | 27 |

**Paired comparison, overall (n=54):**

| Comparison | Mean diff | t | p | Winner in n/54 |
|---|---|---|---|---|
| Flower vs. NVFlare | +0.0041 | 0.864 | 0.392 (n.s.) | Flower 26/54 |
| Flower vs. FedML | +0.0262 | 7.836 | **<0.0001** | Flower 46/54 |
| NVFlare vs. FedML | +0.0221 | 5.740 | **<0.0001** | NVFlare 42/54 |

**Per-dataset breakdown** (both directions consistent with the pooled result):

| Comparison | CIFAR-100 p | FEMNIST p |
|---|---|---|
| Flower vs. NVFlare | 0.539 (n.s.) | 0.077 (n.s., marginal) |
| Flower vs. FedML | **<0.0001** | **<0.0001** |
| NVFlare vs. FedML | **<0.0001** | 0.0239 |

**Finding**: Flower and NVFlare are statistically indistinguishable in accuracy at matched
configurations — consistent with both being correct, comparable FedAvg implementations. FedML,
however, is significantly lower than *both* other frameworks, on both datasets, with a fairly
consistent effect size (~0.02–0.03 accuracy points). This survived a full FedProx bug-fix and
rerun (§2b), so it isn't an artifact of the corrupted/failed runs — it's a genuine, reproducible
gap. The magnitude is modest but the statistical support is strong (n=54, p<0.0001 pooled). Worth
noting as a discussion point rather than a definitive framework indictment: FedML's cross-silo
architecture (real GRPC across separate containers per participant, vs. in-process simulation for
Flower/NVFlare) introduces genuine implementation differences — e.g. serialization, weight
synchronization timing, or subtly different effective batching — any of which could explain a
small but consistent accuracy gap without indicating a deeper correctness problem. This is a
natural angle for further investigation if the thesis wants to explain *why*, not just *that*.

## 4. Scalability

### 4.1 Communication rounds (Figure 2, `fig2_accuracy_vs_rounds`)

| Framework | Dataset  | r      | p      |
|-----------|----------|--------|--------|
| Flower    | CIFAR-100| +0.187 | 0.219  |
| Flower    | FEMNIST  | +0.081 | 0.686  |
| NVFlare   | CIFAR-100| +0.236 | 0.237  |
| NVFlare   | FEMNIST  | +0.135 | 0.501  |
| FedML     | CIFAR-100| +0.326 | 0.097 (marginal) |
| FedML     | FEMNIST  | +0.294 | 0.136  |

**Finding**: no significant relationship between round count and final accuracy in the 5–15
range, for any framework or dataset (FedML/CIFAR-100 is the closest to significance but still
short of p<0.05). FEMNIST is already near its ceiling (~0.77–0.85) by round 5 for all three
frameworks (visible directly in Figure 5's convergence curves); CIFAR-100 doesn't improve enough
within 15 rounds for round count to be a significant driver either. Practically: 5 rounds
captures most of the achievable accuracy at this model/dataset scale and CPU-only training
budget — additional rounds mainly add wall-clock cost.

### 4.2 Number of clients (Figure 3, `fig3_accuracy_vs_clients`)

| Framework | Dataset  | r      | p      |
|-----------|----------|--------|--------|
| Flower    | CIFAR-100| +0.440 | **0.0025** |
| Flower    | FEMNIST  | +0.397 | **0.0405** |
| NVFlare   | CIFAR-100| +0.272 | 0.169  |
| NVFlare   | FEMNIST  | −0.029 | 0.885  |
| FedML     | CIFAR-100| +0.340 | 0.083 (marginal) |
| FedML     | FEMNIST  | +0.129 | 0.522  |

**Finding**: Flower is the only framework with a statistically significant positive relationship
between client count and accuracy, on both datasets — consistent with more clients meaning more
total training data seen per round at fixed `samples_per_client`. NVFlare and FedML both trend
positive on CIFAR-100 but don't reach significance, and NVFlare/FEMNIST is essentially flat.
**Correction from an earlier draft of this analysis**: with only 10 NVFlare/FEMNIST points, an
earlier pass found a significant *negative* client/accuracy relationship there (r=−0.715,
p=0.02) — with the full 27-point grid now available, that relationship is gone (r=−0.029,
p=0.885). That earlier result was flagged as provisional at the time specifically because of the
small n, and this is exactly the outcome that caution was for — a reminder that small-n
subgroup findings in this dataset should be treated skeptically until the full grid confirms
them.

### 4.3 Samples per client (Figure 4, `fig4_accuracy_vs_samples`)

| Framework | Dataset  | r      | p        |
|-----------|----------|--------|----------|
| Flower    | CIFAR-100| +0.798 | <0.0001  |
| Flower    | FEMNIST  | +0.754 | <0.0001  |
| NVFlare   | CIFAR-100| +0.837 | <0.0001  |
| NVFlare   | FEMNIST  | +0.774 | <0.0001  |
| FedML     | CIFAR-100| +0.823 | <0.0001  |
| FedML     | FEMNIST  | +0.715 | <0.0001  |

**Finding**: unchanged from the two-framework analysis, and now confirmed a third time — this is
the dominant, most robust driver of accuracy across the entire dataset. Strong, highly
significant positive correlation for every framework/dataset combination (r=0.72–0.84), clearly
stronger than either the rounds or clients effect, and the *only* trend that's consistent across
all three frameworks without exception. This is the single best-supported quantitative claim in
the dataset and should anchor whatever scalability narrative the thesis builds.

### 4.4 Convergence trajectories (Figure 5, `fig5_convergence_curves`)

Overlays per-round accuracy for the largest matched configuration available per dataset across
frameworks. Complements §3's aggregate statistics by showing *how* each framework arrives at its
final accuracy, not just where it ends up. At this configuration (8 clients, 15 rounds, 1000
samples/client), NVFlare's curve is visibly noisier round-to-round than Flower's or FedML's —
on CIFAR-100 it dips at round 8 and round 13 rather than climbing monotonically — while Flower's
curve is the smoothest of the three on both datasets. Worth checking whether this is a consistent
pattern across other configurations or specific to this one before treating it as a general
claim about NVFlare's stability.

## 5. Data heterogeneity & FedProx (Figure 6)

Now covers all three frameworks at the fixed representative configuration (5 clients, 10 rounds,
2000 samples/client, CIFAR-100) — 18 runs total (3 frameworks × 3 partition strategies × 2
averaging strategies), still **one replicate per condition** (no repeated seeds), so treat
magnitudes as illustrative rather than statistically established.

| Framework | Partition | Averaging | Accuracy | Fairness gap |
|---|---|---|---|---|
| Flower  | IID       | FedAvg  | 0.252 | 0.022 |
| Flower  | IID       | FedProx | 0.246 | 0.041 |
| Flower  | Shard     | FedAvg  | 0.226 | 0.051 |
| Flower  | Shard     | FedProx | 0.235 | 0.025 |
| Flower  | Dirichlet | FedAvg  | 0.254 | 0.039 |
| Flower  | Dirichlet | FedProx | 0.235 | 0.030 |
| NVFlare | IID       | FedAvg  | 0.230 | 0.025 |
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
partitioning produces both the lowest accuracy in the entire table (0.135–0.165) *and* an exact
0.000 fairness gap — meaning every client converged equally, rather than unevenly. That's
unusual: low accuracy under severe non-IID shard partitioning normally comes *with* a large
fairness gap (some clients doing much worse than others), not instead of one. Verified this isn't
a data-corruption artifact (round counts match configured values exactly). Two plausible
non-exclusive explanations: (a) NVFlare's `worst_client_accuracy` tracking may behave differently
under shard partitioning specifically — worth checking the FLARE adapter's per-client validation
code, or (b) the model may have collapsed to a degenerate, class-agnostic solution that all
clients score equally poorly on. This is exactly the kind of finding that's easy to over-claim
without more data — recommend a small follow-up (repeat this one condition a few times with
different seeds) before writing anything definitive about it.

FedML's Shard results are the most striking directionally — noticeably *higher* accuracy (0.468–
0.469) than its own IID condition (0.198–0.236), which is counter to what non-IID partitioning
usually does to accuracy. Given FedML's already-established overall accuracy deficit (§3), this
warrants the same caution as the NVFlare pattern: interesting, single-replicate, not yet
something to build a thesis claim on without repeats.

## 6. Resource / data-engineering comparison

Beyond accuracy, the platform captures per-client RAM, CPU, communication payload, and
server-side aggregation time — a "systems efficiency" angle complementing the ML-accuracy
analysis above. Three metrics turned out to be usable; two did not, for reasons worth documenting
precisely since they reflect genuine measurement issues caught during this analysis, not just
data gaps.

### 6.1 RAM (Figure 7, `fig7_ram_by_framework`)

`psutil.Process().memory_info().rss` was already correctly process-scoped in all three adapters,
so this metric is trustworthy across the full 195-run clean dataset with no timing-contamination
caveat.

| Framework | Dataset  | Mean RAM (MB) | SD   | n  |
|-----------|----------|---------------|------|----|
| NVFlare   | CIFAR-100| 631.4         | 10.0 | 27 |
| Flower    | CIFAR-100| 695.4         | 23.0 | 45 |
| FedML     | CIFAR-100| 764.0         | 17.8 | 27 |
| NVFlare   | FEMNIST  | 827.4         | 15.4 | 27 |
| Flower    | FEMNIST  | 927.7         | 11.6 | 27 |
| FedML     | FEMNIST  | 957.5         | 11.6 | 27 |

**Finding**: a clean, consistent ordering on *both* datasets — NVFlare lowest, Flower middle,
FedML highest — with tight within-framework variance (SD is 1–3% of the mean in every group,
meaning this is a real, low-noise signal, not measurement scatter). The ~130–190MB gap between
NVFlare and FedML per client is the most crisply-separated resource metric in the whole dataset,
more so even than the accuracy differences in §3. This maps sensibly onto architecture: FedML's
genuinely separate OS processes per participant likely carry more baseline Python/torch/GRPC
overhead per client than NVFlare's or Flower's in-process simulation approaches.

### 6.2 Server aggregation time (Figure 8, `fig8_aggregation_time_by_framework`)

**NVFlare figures here are restricted to the 24 runs started after the thread-cap fix
(2026-08-24 19:51) — see §2 and the script's `NVFLARE_TIMING_CLEAN_CUTOFF`.** The pre-fix subset
showed both a much higher mean (15.1s vs. 3.7s) and far higher variance (SD 15.6 vs. 0.75) on the
same workload — the classic signature of contention noise, not a real measurement. Flower and
FedML's full sweeps ran without this contamination (Flower ran alone before NVFlare/FedML sweeps
existed; FedML's sweep ran entirely after the fix was deployed and confirmed active), so their
numbers use the complete dataset.

| Framework | Dataset  | Mean (s) | SD    | n  |
|-----------|----------|----------|-------|----|
| FedML     | CIFAR-100| 0.000    | 0.001 | 27 |
| Flower    | CIFAR-100| 0.022    | 0.007 | 45 |
| NVFlare   | CIFAR-100| 3.863    | 0.712 | 8  |
| FedML     | FEMNIST  | 0.000    | 0.000 | 27 |
| Flower    | FEMNIST  | 0.024    | 0.008 | 27 |
| NVFlare   | FEMNIST  | 3.714    | 0.840 | 16 |

**Finding**: a roughly 150–200× gap between NVFlare and Flower, and NVFlare's aggregation is
essentially always the ~2–3 orders of magnitude that Figure 8's log scale makes necessary just to
show all three on one axis. This is architectural, not a bug: NVFlare's simulator persists model
checkpoints to disk between rounds (see the README's `DATASET_MODEL_SHAPES`/persistor notes),
where Flower's and FedML's aggregation is a pure in-memory weighted average. Worth stating
explicitly in the thesis as a real trade-off — NVFlare's persistence buys fault-tolerance/resume
capability that the other two don't have, at a real, measured wall-clock cost per round. The
NVFlare sample size here (n=8/16) is smaller than the other two because most of its sweep predates
the clean-timing cutoff; a dedicated rerun of the full NVFlare grid post-fix would tighten this
if the thesis wants a larger n here specifically.

### 6.3 What didn't make it in

- **Communication payload** (`avg_comm_size_mb`) is identical across all three frameworks
  (6.397MB CIFAR-100, 6.320MB FEMNIST, zero variance) — it's computed as the raw serialized size
  of the shared `FlexibleCNN` model's parameters, not actual wire-protocol overhead (framing,
  headers, compression). Real, but not differentiating, so no figure was built for it.
- **CPU utilization** (`cpu_usage_percent`) is excluded entirely, not just filtered. All three
  adapters measured it via bare `psutil.cpu_percent(interval=None)`, which returns **system-wide**
  CPU load (whatever else happens to be running on the host at that instant), not the training
  process's own usage — a different function from the correctly-scoped
  `psutil.Process().memory_info()` used for RAM two lines below it in the same generated code.
  This explains why Flower's numbers looked implausibly tiny (1.8–3.3%) in an earlier pass of
  this analysis: Flower ran alone on a 256-core host, so its own container's CPU use barely
  registered against the whole machine's utilization. Fixed going forward
  (`Process().cpu_percent(interval=0.1)`, a short blocking measurement chosen because these
  adapters create a fresh `Process()` object each call rather than holding a persistent handle, so
  an unprimed `interval=None` call would always read `0.0`) — but no run collected before the fix
  has valid data for this field, and it can't be recovered retroactively. A clean CPU-comparison
  pass is a good candidate for a small follow-up sweep if the thesis wants a complete
  RAM+CPU+timing resource chapter.

## 7. Limitations & recommended next steps

1. **Both major accuracy-data-quality issues from the two-framework draft are resolved** (§2) —
   the NVFlare corruption is fully rerun clean, and the FedML FedProx crash is fixed with a
   confirmed root cause, not just a symptom mitigation.
2. **NVFlare's resource comparison runs on a smaller n** (8/16 vs. 27 for the other two — §6.2)
   because most of its sweep predates the clean-timing cutoff. RAM is unaffected (full n=27).
3. **CPU utilization has no valid data at all yet** (§6.3) — the measurement bug is fixed in code,
   but a fresh sweep is needed before it can be reported.
4. **Heterogeneity sweep is broader but still single-replicate** (§5) — now 3 frameworks instead
   of 1, but still n=1/condition. The "notable patterns" flagged in §5 are exactly the kind of
   result that needs repeat runs (different random seeds, same config) before they can support a
   thesis claim rather than just motivate one.
5. **Accuracy magnitudes are low in absolute terms** (CIFAR-100 tops out around 30–47%,
   FEMNIST around 65–87%) — expected given short local training (3 epochs), no LR schedule, and
   this being a *relative* framework/scalability comparison rather than an attempt at
   state-of-the-art accuracy. Caveat any absolute number quoted out of context.
6. **The FedML accuracy gap (§3) and RAM gap (§6.1) are documented but not fully explained.**
   Worth a short follow-up investigation (e.g., comparing per-round loss curves, checking for
   subtle differences in data loading/batching or per-process overhead between the FedML adapter
   and the other two) if the thesis wants to make a causal claim about *why*, not just *that*.

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

Data files: `runs_clean.csv` (195-run clean dataset), `runs_all.csv` (all 204 before exclusion),
`matched_configs.csv` (54 configs matched across all 3 frameworks), `summary_stats.csv`.
