# Newest Matched-Configuration Comparison (Flower vs. NVFlare vs. FedML)

*A focused, single-configuration case study — not an aggregate statistic. Complements the full
grid analysis in `analysis.md` with one concrete, fully apples-to-apples example: the single most
recently completed configuration for which all three frameworks have a base-grid run.*

## Configuration

**FEMNIST, 8 clients, 15 rounds, 4000 samples/client, IID partitioning, FedAvg** — identical
hyperparameters across all three (`epochs=3`, `batch_size=32`, `optimizer=adam`,
`learning_rate=0.001`). This is the most recently completed matched configuration in the entire
dataset (FedML's run finished 2026-08-25 19:39, the latest of any base-grid run across all three
frameworks).

| Framework | Run ID | Started |
|---|---|---|
| Flower  | `flower-femnist-aug25_0303-e6da`  | 2026-08-25 03:03 |
| NVFlare | `nvflare-femnist-aug25_1542-3466` | 2026-08-25 15:42 |
| FedML   | `fedml-femnist-aug25_1939-1645`   | 2026-08-25 19:39 |

## Per-round accuracy and loss

See `fig10_newest_matched_comparison.png/pdf` for the accuracy/loss curves plotted directly.

| Round | Flower Acc. | Flower Loss | NVFlare Acc. | NVFlare Loss | FedML Acc. | FedML Loss |
|---|---|---|---|---|---|---|
| 1  | 0.5919 | 1.9271 | 0.0050 | 4.1439 | 0.7031 | 1.0159 |
| 2  | 0.7859 | 0.7041 | 0.6325 | 1.7341 | 0.7500 | 0.8175 |
| 3  | 0.8156 | 0.5905 | 0.7850 | 0.7237 | 0.7672 | 0.7464 |
| 4  | 0.8241 | 0.5449 | 0.8200 | 0.6198 | 0.7909 | 0.6827 |
| 5  | 0.8347 | 0.5103 | 0.8325 | 0.5609 | 0.7922 | 0.6800 |
| 6  | 0.8409 | 0.5047 | 0.8475 | 0.5364 | 0.8025 | 0.6657 |
| 7  | 0.8369 | 0.4994 | 0.8525 | 0.5125 | 0.8131 | 0.6632 |
| 8  | 0.8447 | 0.4984 | 0.8550 | 0.5222 | 0.8103 | 0.6686 |
| 9  | 0.8462 | 0.5066 | 0.8525 | 0.5186 | 0.8131 | 0.6957 |
| 10 | 0.8447 | 0.5098 | 0.8525 | 0.5513 | 0.8131 | 0.6846 |
| 11 | 0.8441 | 0.5269 | **0.8600** | 0.5317 | 0.8063 | 0.7301 |
| 12 | 0.8422 | 0.5396 | 0.8500 | 0.5760 | 0.8122 | 0.7314 |
| 13 | 0.8447 | 0.5508 | 0.8475 | 0.5868 | 0.8137 | 0.7465 |
| 14 | **0.8469** | 0.5625 | 0.8550 | 0.6074 | 0.8203 | 0.7527 |
| 15 | 0.8447 | 0.5714 | 0.8450 | 0.6264 | **0.8203** | 0.7929 |

**Peak accuracy**: NVFlare 0.8600 (round 11), Flower 0.8469 (round 14), FedML 0.8203 (round 14-15,
plateaued). All three lose a little accuracy from their peak by round 15 except FedML, which is
still at its peak — consistent with §4.1's finding that additional rounds beyond ~5-8 mainly add
noise/cost rather than accuracy for FEMNIST.

**Round 1 is the most dramatic point of divergence**: NVFlare starts at essentially random-guess
accuracy (0.005, 62-class FEMNIST), while FedML (0.703) and Flower (0.592) are already most of the
way to their final accuracy after just one round. NVFlare catches up entirely by round 4. This
single-run pattern matches what Figure 5 (`fig5_convergence_curves`) shows in aggregate — NVFlare
consistently has the slowest first-round ramp-up of the three frameworks.

## Resource usage

| Metric | Flower | NVFlare | FedML |
|---|---|---|---|
| Final accuracy | 0.8447 | 0.8450 | 0.8203 |
| Peak RAM (MB) | 952.3 | 812.8 | 973.0 |
| CPU usage (%) | 14.25 | 15.91 | 15.08 |
| Server aggregation time (s) | 0.017 | 3.582 | 0.001 |
| Avg. client compute time (s) | 7.41 | 6.05 | 6.16 |

**This single matched triplet reproduces every ordering found in the full aggregate analysis**
(§6 of `analysis.md`), which is a meaningful cross-check — a concrete example, not just averages
across 27 configurations, shows the same pattern:
- **RAM**: NVFlare lowest, Flower middle, FedML highest — matches §6.1 exactly.
- **CPU**: Flower lowest, FedML middle, NVFlare highest — matches §6.3 exactly.
- **Aggregation time**: NVFlare ~200-3500× the other two — matches §6.2's architectural
  explanation (checkpoint persistence vs. in-memory aggregation).
- **Accuracy**: Flower and NVFlare are close (0.8447 vs. 0.8450, a 0.03-point difference — well
  within the "statistically indistinguishable" finding of §3), FedML trails both by ~2-2.5 points,
  consistent with §3's finding that Flower/NVFlare beat FedML on FEMNIST specifically.

The one metric that *doesn't* show a clean ranking here is average client compute time (Flower
highest at 7.41s, NVFlare and FedML close together around 6.1s) — this metric wasn't singled out
for aggregate comparison in `analysis.md` (aggregation time and CPU were used instead as the
primary timing/compute signals), so there's no aggregate baseline to check this single-run number
against; worth treating as a data point rather than a confirmed pattern.

## Caveat

This is **one run per framework**, not an aggregate — useful as a concrete illustration of the
grid-level findings, and reassuring that they hold up at the level of an individual matched
comparison, not just as statistical artifacts of averaging. It should not be read as adding
independent statistical weight beyond what's already in `analysis.md`'s full-grid analysis (which
this configuration is already one data point within).
