# Full Metrics Comparison: Beyond Accuracy (Flower vs. NVFlare vs. FedML)

*Extends `all_matched_configs_comparison.md` (which covers only final accuracy) with RAM,
aggregation time, CPU, per-round convergence behavior, and per-client resource spread — all
computed from the same 54 matched configurations, latest run per config. Every per-config table
below is generated directly from `runs_clean.csv`/raw result JSON, not hand-transcribed.*

## 1. Peak client RAM (MB), all 54 matched configs

| Dataset | Clients | Rounds | Samples/Client | Flower | NVFlare | FedML |
|---|---|---|---|---|---|---|
| cifar100 | 2 | 5 | 1000 | 679.7 | 687.2 | 732.0 |
| cifar100 | 2 | 5 | 2000 | 689.6 | 614.6 | 755.8 |
| cifar100 | 2 | 5 | 4000 | 717.7 | 635.0 | 760.9 |
| cifar100 | 2 | 10 | 1000 | 663.4 | 598.9 | 738.8 |
| cifar100 | 2 | 10 | 2000 | 687.0 | 621.3 | 763.5 |
| cifar100 | 2 | 10 | 4000 | 719.7 | 636.1 | 785.5 |
| cifar100 | 2 | 15 | 1000 | 693.8 | 603.9 | 753.2 |
| cifar100 | 2 | 15 | 2000 | 684.6 | 615.6 | 776.6 |
| cifar100 | 2 | 15 | 4000 | 732.6 | 632.2 | 792.2 |
| cifar100 | 5 | 5 | 1000 | 688.4 | 597.1 | 734.2 |
| cifar100 | 5 | 5 | 2000 | 707.5 | 608.9 | 757.0 |
| cifar100 | 5 | 5 | 4000 | 728.0 | 629.6 | 768.5 |
| cifar100 | 5 | 10 | 1000 | 691.1 | 593.2 | 747.3 |
| cifar100 | 5 | 10 | 2000 | 704.6 | 616.6 | 768.0 |
| cifar100 | 5 | 10 | 4000 | 737.6 | 633.8 | 781.9 |
| cifar100 | 5 | 15 | 1000 | 697.5 | 600.4 | 754.5 |
| cifar100 | 5 | 15 | 2000 | 716.5 | 617.2 | 780.9 |
| cifar100 | 5 | 15 | 4000 | 744.8 | 636.8 | 793.9 |
| cifar100 | 8 | 5 | 1000 | 688.3 | 596.0 | 731.0 |
| cifar100 | 8 | 5 | 2000 | 704.9 | 611.3 | 755.3 |
| cifar100 | 8 | 5 | 4000 | 721.3 | 629.9 | 768.6 |
| cifar100 | 8 | 10 | 1000 | 706.9 | 597.5 | 747.3 |
| cifar100 | 8 | 10 | 2000 | 723.3 | 615.8 | 773.0 |
| cifar100 | 8 | 10 | 4000 | 743.2 | 633.8 | 783.6 |
| cifar100 | 8 | 15 | 1000 | 712.7 | 598.8 | 750.7 |
| cifar100 | 8 | 15 | 2000 | 730.0 | 617.0 | 780.4 |
| cifar100 | 8 | 15 | 4000 | 752.6 | 632.8 | 789.9 |
| femnist | 2 | 5 | 1000 | 902.2 | 840.4 | 961.0 |
| femnist | 2 | 5 | 2000 | 906.7 | 809.5 | 943.8 |
| femnist | 2 | 5 | 4000 | 911.7 | 808.1 | 945.5 |
| femnist | 2 | 10 | 1000 | 905.6 | 816.9 | 955.0 |
| femnist | 2 | 10 | 2000 | 906.1 | 808.8 | 955.0 |
| femnist | 2 | 10 | 4000 | 923.4 | 807.3 | 959.7 |
| femnist | 2 | 15 | 1000 | 922.5 | 816.5 | 966.4 |
| femnist | 2 | 15 | 2000 | 923.2 | 812.3 | 968.0 |
| femnist | 2 | 15 | 4000 | 933.6 | 821.6 | 966.8 |
| femnist | 5 | 5 | 1000 | 906.6 | 812.2 | 942.8 |
| femnist | 5 | 5 | 2000 | 911.8 | 812.3 | 943.0 |
| femnist | 5 | 5 | 4000 | 924.4 | 814.4 | 942.8 |
| femnist | 5 | 10 | 1000 | 927.1 | 809.6 | 958.8 |
| femnist | 5 | 10 | 2000 | 926.2 | 813.4 | 953.8 |
| femnist | 5 | 10 | 4000 | 933.0 | 814.0 | 953.6 |
| femnist | 5 | 15 | 1000 | 930.4 | 813.0 | 966.3 |
| femnist | 5 | 15 | 2000 | 933.4 | 820.0 | 966.9 |
| femnist | 5 | 15 | 4000 | 940.4 | 817.1 | 970.6 |
| femnist | 8 | 5 | 1000 | 912.4 | 844.9 | 942.9 |
| femnist | 8 | 5 | 2000 | 910.8 | 810.8 | 943.8 |
| femnist | 8 | 5 | 4000 | 918.3 | 815.9 | 946.6 |
| femnist | 8 | 10 | 1000 | 929.3 | 814.6 | 958.3 |
| femnist | 8 | 10 | 2000 | 938.6 | 815.6 | 953.3 |
| femnist | 8 | 10 | 4000 | 948.1 | 818.9 | 964.4 |
| femnist | 8 | 15 | 1000 | 936.0 | 814.4 | 970.6 |
| femnist | 8 | 15 | 2000 | 943.7 | 813.9 | 967.5 |
| femnist | 8 | 15 | 4000 | 952.3 | 812.8 | 973.0 |

## 2. Server aggregation time (s), 53/54 matched configs

Missing 1 config (`femnist, 8 clients, 5 rounds, 1000 samples`) — its NVFlare run predates the
thread-cap fix and is excluded from timing figures throughout this analysis (see `analysis.md`
§2).

| Dataset | Clients | Rounds | Samples/Client | Flower | NVFlare | FedML |
|---|---|---|---|---|---|---|
| cifar100 | 2 | 5 | 1000 | 0.009 | 2.532 | 0.000 |
| cifar100 | 2 | 5 | 2000 | 0.008 | 2.756 | 0.000 |
| cifar100 | 2 | 5 | 4000 | 0.010 | 3.384 | 0.000 |
| cifar100 | 2 | 10 | 1000 | 0.009 | 2.840 | 0.000 |
| cifar100 | 2 | 10 | 2000 | 0.007 | 2.922 | 0.000 |
| cifar100 | 2 | 10 | 4000 | 0.007 | 3.065 | 0.001 |
| cifar100 | 2 | 15 | 1000 | 0.007 | 3.185 | 0.000 |
| cifar100 | 2 | 15 | 2000 | 0.007 | 2.979 | 0.001 |
| cifar100 | 2 | 15 | 4000 | 0.007 | 3.003 | 0.001 |
| cifar100 | 5 | 5 | 1000 | 0.012 | 3.168 | 0.002 |
| cifar100 | 5 | 5 | 2000 | 0.012 | 4.042 | 0.000 |
| cifar100 | 5 | 5 | 4000 | 0.013 | 3.776 | 0.002 |
| cifar100 | 5 | 10 | 1000 | 0.012 | 3.372 | 0.001 |
| cifar100 | 5 | 10 | 2000 | 0.012 | 3.928 | 0.001 |
| cifar100 | 5 | 10 | 4000 | 0.011 | 4.426 | 0.001 |
| cifar100 | 5 | 15 | 1000 | 0.011 | 3.445 | 0.001 |
| cifar100 | 5 | 15 | 2000 | 0.011 | 4.240 | 0.001 |
| cifar100 | 5 | 15 | 4000 | 0.013 | 5.108 | 0.001 |
| cifar100 | 8 | 5 | 1000 | 0.017 | 2.456 | 0.002 |
| cifar100 | 8 | 5 | 2000 | 0.016 | 5.160 | 0.002 |
| cifar100 | 8 | 5 | 4000 | 0.020 | 3.444 | 0.000 |
| cifar100 | 8 | 10 | 1000 | 0.016 | 3.582 | 0.000 |
| cifar100 | 8 | 10 | 2000 | 0.016 | 3.786 | 0.001 |
| cifar100 | 8 | 10 | 4000 | 0.016 | 4.985 | 0.002 |
| cifar100 | 8 | 15 | 1000 | 0.016 | 3.597 | 0.001 |
| cifar100 | 8 | 15 | 2000 | 0.018 | 4.135 | 0.001 |
| cifar100 | 8 | 15 | 4000 | 0.016 | 4.544 | 0.001 |
| femnist | 2 | 5 | 1000 | 0.008 | 3.282 | 0.002 |
| femnist | 2 | 5 | 2000 | 0.008 | 2.990 | 0.000 |
| femnist | 2 | 5 | 4000 | 0.009 | 3.084 | 0.002 |
| femnist | 2 | 10 | 1000 | 0.008 | 2.942 | 0.000 |
| femnist | 2 | 10 | 2000 | 0.007 | 2.862 | 0.000 |
| femnist | 2 | 10 | 4000 | 0.009 | 2.881 | 0.000 |
| femnist | 2 | 15 | 1000 | 0.008 | 3.059 | 0.000 |
| femnist | 2 | 15 | 2000 | 0.008 | 3.073 | 0.000 |
| femnist | 2 | 15 | 4000 | 0.008 | 3.031 | 0.001 |
| femnist | 5 | 5 | 1000 | 0.013 | 3.220 | 0.000 |
| femnist | 5 | 5 | 2000 | 0.013 | 4.560 | 0.002 |
| femnist | 5 | 5 | 4000 | 0.013 | 3.568 | 0.000 |
| femnist | 5 | 10 | 1000 | 0.013 | 3.484 | 0.000 |
| femnist | 5 | 10 | 2000 | 0.013 | 3.845 | 0.000 |
| femnist | 5 | 10 | 4000 | 0.011 | 4.121 | 0.001 |
| femnist | 5 | 15 | 1000 | 0.013 | 3.269 | 0.000 |
| femnist | 5 | 15 | 2000 | 0.012 | 4.051 | 0.002 |
| femnist | 5 | 15 | 4000 | 0.011 | 5.251 | 0.000 |
| femnist | 8 | 5 | 2000 | 0.015 | 2.580 | 0.002 |
| femnist | 8 | 5 | 4000 | 0.019 | 2.564 | 0.000 |
| femnist | 8 | 10 | 1000 | 0.014 | 3.532 | 0.001 |
| femnist | 8 | 10 | 2000 | 0.018 | 4.515 | 0.001 |
| femnist | 8 | 10 | 4000 | 0.018 | 5.078 | 0.001 |
| femnist | 8 | 15 | 1000 | 0.015 | 3.771 | 0.001 |
| femnist | 8 | 15 | 2000 | 0.014 | 3.802 | 0.001 |
| femnist | 8 | 15 | 4000 | 0.017 | 3.582 | 0.001 |

## 3. Client CPU usage (%), 50/54 matched configs

Missing 4 configs (all FEMNIST) where at least one framework's run predates the CPU-measurement
fix: `(2,10,4000)`, `(5,5,4000)`, `(8,5,1000)`, `(8,10,2000)` (clients, rounds, samples/client).

| Dataset | Clients | Rounds | Samples/Client | Flower | NVFlare | FedML |
|---|---|---|---|---|---|---|
| cifar100 | 2 | 5 | 1000 | 16.00 | 17.99 | 13.00 |
| cifar100 | 2 | 5 | 2000 | 14.00 | 16.98 | 15.00 |
| cifar100 | 2 | 5 | 4000 | 16.00 | 19.97 | 15.99 |
| cifar100 | 2 | 10 | 1000 | 11.50 | 17.98 | 16.50 |
| cifar100 | 2 | 10 | 2000 | 11.00 | 16.49 | 15.00 |
| cifar100 | 2 | 10 | 4000 | 12.00 | 11.99 | 14.50 |
| cifar100 | 2 | 15 | 1000 | 16.33 | 17.32 | 12.67 |
| cifar100 | 2 | 15 | 2000 | 9.33 | 16.64 | 15.66 |
| cifar100 | 2 | 15 | 4000 | 15.67 | 16.31 | 14.33 |
| cifar100 | 5 | 5 | 1000 | 16.40 | 14.80 | 15.20 |
| cifar100 | 5 | 5 | 2000 | 13.60 | 15.60 | 14.40 |
| cifar100 | 5 | 5 | 4000 | 16.00 | 17.99 | 14.80 |
| cifar100 | 5 | 10 | 1000 | 13.80 | 15.99 | 15.20 |
| cifar100 | 5 | 10 | 2000 | 11.20 | 17.60 | 15.40 |
| cifar100 | 5 | 10 | 4000 | 13.40 | 15.40 | 14.60 |
| cifar100 | 5 | 15 | 1000 | 13.73 | 16.13 | 16.00 |
| cifar100 | 5 | 15 | 2000 | 13.20 | 16.26 | 15.33 |
| cifar100 | 5 | 15 | 4000 | 16.00 | 17.19 | 15.46 |
| cifar100 | 8 | 5 | 1000 | 15.00 | 16.74 | 15.75 |
| cifar100 | 8 | 5 | 2000 | 13.50 | 16.50 | 16.25 |
| cifar100 | 8 | 5 | 4000 | 12.75 | 15.00 | 15.75 |
| cifar100 | 8 | 10 | 1000 | 14.88 | 16.87 | 14.62 |
| cifar100 | 8 | 10 | 2000 | 13.62 | 15.37 | 15.50 |
| cifar100 | 8 | 10 | 4000 | 14.25 | 15.87 | 14.87 |
| cifar100 | 8 | 15 | 1000 | 15.00 | 15.83 | 14.92 |
| cifar100 | 8 | 15 | 2000 | 14.67 | 15.83 | 15.25 |
| cifar100 | 8 | 15 | 4000 | 16.17 | 15.16 | 15.75 |
| femnist | 2 | 5 | 1000 | 12.00 | 17.97 | 18.00 |
| femnist | 2 | 5 | 2000 | 12.00 | 15.96 | 14.00 |
| femnist | 2 | 5 | 4000 | 12.99 | 11.99 | 12.00 |
| femnist | 2 | 10 | 1000 | 10.50 | 17.47 | 15.49 |
| femnist | 2 | 10 | 2000 | 10.50 | 16.48 | 15.49 |
| femnist | 2 | 15 | 1000 | 12.33 | 16.98 | 15.00 |
| femnist | 2 | 15 | 2000 | 13.33 | 15.31 | 13.33 |
| femnist | 2 | 15 | 4000 | 14.00 | 17.31 | 12.67 |
| femnist | 5 | 5 | 1000 | 9.60 | 15.20 | 13.20 |
| femnist | 5 | 5 | 2000 | 11.60 | 16.40 | 14.00 |
| femnist | 5 | 10 | 1000 | 12.60 | 14.99 | 14.80 |
| femnist | 5 | 10 | 2000 | 12.20 | 16.39 | 14.60 |
| femnist | 5 | 10 | 4000 | 12.60 | 14.80 | 13.40 |
| femnist | 5 | 15 | 1000 | 11.87 | 15.47 | 14.00 |
| femnist | 5 | 15 | 2000 | 12.40 | 16.39 | 14.93 |
| femnist | 5 | 15 | 4000 | 12.67 | 16.40 | 15.73 |
| femnist | 8 | 5 | 2000 | 11.25 | 14.49 | 15.25 |
| femnist | 8 | 5 | 4000 | 11.75 | 15.25 | 14.25 |
| femnist | 8 | 10 | 1000 | 11.88 | 15.25 | 14.12 |
| femnist | 8 | 10 | 4000 | 13.38 | 16.50 | 14.38 |
| femnist | 8 | 15 | 1000 | 11.33 | 16.08 | 15.91 |
| femnist | 8 | 15 | 2000 | 12.25 | 15.33 | 15.33 |
| femnist | 8 | 15 | 4000 | 14.25 | 15.91 | 15.08 |

## 4. Convergence speed: rounds to reach 95% of final accuracy

Computed per matched run, then summarized by framework/dataset (54 individual data points each,
27 per dataset).

| Framework | Dataset | Mean rounds | Median rounds | SD | As % of total rounds (mean) |
|---|---|---|---|---|---|
| Flower  | CIFAR-100 | 5.93 | 5.0 | 2.15 | 67.2% |
| NVFlare | CIFAR-100 | 7.22 | 6.0 | 3.13 | 77.5% |
| FedML   | CIFAR-100 | 7.11 | 7.0 | 2.99 | 74.3% |
| Flower  | FEMNIST   | 3.37 | 3.0 | 1.28 | 40.1% |
| NVFlare | FEMNIST   | 4.63 | 4.0 | 1.69 | 54.2% |
| FedML   | FEMNIST   | 4.04 | 4.0 | 1.40 | 46.3% |

**Finding**: Flower converges fastest to within 95% of its own final accuracy on *both* datasets —
using roughly two-thirds of its configured rounds on CIFAR-100 and only ~40% on FEMNIST, versus
NVFlare needing ~75-78% of its rounds on both. This is a new finding not previously surfaced in
`analysis.md`'s round-count correlation analysis (§4.1), which looked at whether *more configured
rounds* correlates with *higher final accuracy* (it doesn't, significantly, for any framework) —
a different question from *how many of the rounds a framework is actually given does it need*.
Both are true simultaneously: extra configured rounds don't buy much extra accuracy for any
framework, but among the rounds each framework *is* given, Flower reaches its ceiling soonest.

## 5. Cold-start behavior: round-1 accuracy as a fraction of final accuracy

| Framework | Dataset | Mean round-1 acc. (% of final) | SD |
|---|---|---|---|
| Flower  | CIFAR-100 | 29.4% | 9.1pp |
| NVFlare | CIFAR-100 | **3.4%** | 2.7pp |
| FedML   | CIFAR-100 | 38.5% | 7.3pp |
| Flower  | FEMNIST   | 64.8% | 11.9pp |
| NVFlare | FEMNIST   | **2.1%** | 2.4pp |
| FedML   | FEMNIST   | 75.3% | 13.5pp |

**Finding**: NVFlare's round-1 accuracy is dramatically, consistently near-zero relative to its
own final accuracy — 2-3%, versus 29-39% for Flower and 39-75% for FedML — and this holds with
very low variance (SD 2.4-2.7pp) across all 27 configs per dataset, so it's a systematic
architectural property, not an occasional fluke. This exactly matches and statistically confirms
the anecdotal pattern from `newest_matched_comparison.md`'s single-run case study (NVFlare:
0.005 → Flower: 0.592 → FedML: 0.703 at round 1, for that one config). A plausible mechanism:
NVFlare's simulator may not apply the initial global model broadcast as effectively before the
first local training pass as the other two frameworks' initialization paths — this is a specific,
testable hypothesis for follow-up code inspection, not confirmed here.

## 6. Per-client resource spread (within-run range, max − min across clients)

| Framework | Dataset | Compute time range (s), median | CPU range (pp), median |
|---|---|---|---|
| Flower  | CIFAR-100 | 0.762 | 9.4 |
| NVFlare | CIFAR-100 | 0.929 | 9.5 |
| FedML   | CIFAR-100 | 0.749 | 9.2 |
| Flower  | FEMNIST   | 0.731 | 8.8 |
| NVFlare | FEMNIST   | 0.847 | 7.6 |
| FedML   | FEMNIST   | 0.894 | 8.4 |

**Finding**: median client-to-client resource spread is similar across all three frameworks (no
clear winner/loser) — clients within a single run take roughly 0.7-0.9s different compute time and
7-10 percentage points different CPU usage from each other, regardless of framework. **One
mean-distorting outlier is worth flagging explicitly**: NVFlare's FEMNIST compute-time range has a
mean of 2.14s (vs. median 0.85s) driven entirely by a single run (`8 clients, 5 rounds, 1000
samples/client`) with a 34.85s range — one client took dramatically longer than the others in that
specific run. This is reported as median above specifically to avoid that single run distorting
the framework-level comparison; the raw mean is available in `/tmp`-generated intermediate data if
the outlier run itself is worth investigating separately (possible resource contention specific to
that run, not a general NVFlare property, since no other config shows anything close to this).

## Source data

- `ram_matched.csv`, `aggtime_matched.csv`, `cpu_matched.csv` — per-config pivot tables (this
  document's tables 1-3, machine-generated).
- Convergence and client-spread analyses (tables 4-6) read every matched run's full per-round
  `metrics_distributed.accuracy` and per-client `client_logs` directly from `results/*.json` —
  not currently wired into `generate_thesis_analysis.py`; ask to have this promoted into the
  script (with `--out-dir` CSVs) if you'll want to regenerate this after future reruns.
