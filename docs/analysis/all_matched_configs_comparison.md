# All Matched-Configuration Comparison (Flower vs. NVFlare vs. FedML)

*Every configuration for which all three frameworks have a valid base-grid (IID/FedAvg) run —
54 configurations total, one row per unique combination of dataset/clients/rounds/samples_per_client.
Where a configuration was rerun (see `analysis.md` §2), only the latest valid run per framework is
used, matching the deduplication already applied throughout the rest of the analysis. This is the
full per-configuration data underlying the aggregate paired-comparison statistics in `analysis.md`
§3 — here nothing is averaged away, every individual matchup is shown.*

## Win-rate summary

| Framework | Wins | Win rate |
|---|---|---|
| Flower  | 31 / 54 | 57.4% |
| NVFlare | 22 / 54 | 40.7% |
| FedML   | 1 / 54  | 1.9%  |

**By dataset:**

| Dataset | Flower wins | NVFlare wins | FedML wins |
|---|---|---|---|
| CIFAR-100 (27 configs) | 13 | 14 | 0 |
| FEMNIST (27 configs)   | 18 | 8  | 1 |

**Average winning margin** (accuracy points, winner minus runner-up): FedML 0.040 (n=1, the one
config it wins), Flower 0.0206, NVFlare 0.0178. FedML's single win has a larger margin than
Flower's or NVFlare's *typical* winning margin, but it's one data point — not a pattern.

**Closest calls** (margin < 0.002, essentially ties): 5 of 54 configs, spanning both datasets and
all three frameworks as the nominal "winner" — a reminder that a razor-thin win in an individual
config carries much less weight than the aggregate paired t-test in `analysis.md` §3, which is
the statistically appropriate way to read this table rather than counting wins config-by-config.

**Biggest margins** (top 5, all favor Flower or FedML, none favor NVFlare): the largest gaps are
concentrated in `samples_per_client=1000` configs (4 of the top 5), consistent with §4.3's finding
that low sample counts amplify accuracy variance between frameworks — more data per client narrows
the gap between frameworks, not just raising everyone's accuracy uniformly.

**Reading caution**: raw win-counts (31 vs. 22 vs. 1) look more decisive than the paired
statistical tests in `analysis.md` §3 actually support — Flower vs. NVFlare is *not* statistically
significant overall (p=0.10), despite Flower winning more often numerically, because the wins are
by small, inconsistent margins in both directions. Win-counting is intuitive but is not a
substitute for the paired t-test; both are presented here for that reason — see them together, not
either in isolation.

## Full comparison table

| Dataset | Clients | Rounds | Samples/Client | Flower | NVFlare | FedML | Winner |
|---|---|---|---|---|---|---|---|
| cifar100 | 2 | 5  | 1000 | 0.1600 | 0.1000 | 0.0950 | Flower |
| cifar100 | 2 | 5  | 2000 | 0.1950 | 0.2150 | 0.1825 | NVFlare |
| cifar100 | 2 | 5  | 4000 | 0.2450 | 0.2850 | 0.2325 | NVFlare |
| cifar100 | 2 | 10 | 1000 | 0.1400 | 0.1700 | 0.1150 | NVFlare |
| cifar100 | 2 | 10 | 2000 | 0.1850 | 0.2100 | 0.1825 | NVFlare |
| cifar100 | 2 | 10 | 4000 | 0.2463 | 0.2750 | 0.2425 | NVFlare |
| cifar100 | 2 | 15 | 1000 | 0.1700 | 0.1400 | 0.1400 | Flower |
| cifar100 | 2 | 15 | 2000 | 0.1925 | 0.2400 | 0.2100 | NVFlare |
| cifar100 | 2 | 15 | 4000 | 0.2625 | 0.2775 | 0.2313 | NVFlare |
| cifar100 | 5 | 5  | 1000 | 0.1720 | 0.1400 | 0.1160 | Flower |
| cifar100 | 5 | 5  | 2000 | 0.2320 | 0.2100 | 0.1970 | Flower |
| cifar100 | 5 | 5  | 4000 | 0.3105 | 0.3100 | 0.2575 | Flower |
| cifar100 | 5 | 10 | 1000 | 0.2040 | 0.1900 | 0.1420 | Flower |
| cifar100 | 5 | 10 | 2000 | 0.2580 | 0.2700 | 0.2360 | NVFlare |
| cifar100 | 5 | 10 | 4000 | 0.3430 | 0.3800 | 0.2865 | NVFlare |
| cifar100 | 5 | 15 | 1000 | 0.1900 | 0.2300 | 0.1840 | NVFlare |
| cifar100 | 5 | 15 | 2000 | 0.2630 | 0.2750 | 0.2470 | NVFlare |
| cifar100 | 5 | 15 | 4000 | 0.3330 | 0.3425 | 0.3095 | NVFlare |
| cifar100 | 8 | 5  | 1000 | 0.1800 | 0.1300 | 0.1313 | Flower |
| cifar100 | 8 | 5  | 2000 | 0.2350 | 0.2200 | 0.1850 | Flower |
| cifar100 | 8 | 5  | 4000 | 0.3331 | 0.3025 | 0.2650 | Flower |
| cifar100 | 8 | 10 | 1000 | 0.2325 | 0.2200 | 0.2063 | Flower |
| cifar100 | 8 | 10 | 2000 | 0.2775 | 0.2600 | 0.2181 | Flower |
| cifar100 | 8 | 10 | 4000 | 0.3375 | 0.3525 | 0.3038 | NVFlare |
| cifar100 | 8 | 15 | 1000 | 0.2513 | 0.2100 | 0.2163 | Flower |
| cifar100 | 8 | 15 | 2000 | 0.2831 | 0.2750 | 0.2300 | Flower |
| cifar100 | 8 | 15 | 4000 | 0.3588 | 0.3675 | 0.3272 | NVFlare |
| femnist  | 2 | 5  | 1000 | 0.6900 | 0.6800 | 0.6900 | Flower\* |
| femnist  | 2 | 5  | 2000 | 0.7900 | 0.8250 | 0.8175 | NVFlare |
| femnist  | 2 | 5  | 4000 | 0.8175 | 0.8150 | 0.7838 | Flower |
| femnist  | 2 | 10 | 1000 | 0.7250 | 0.7500 | 0.7200 | NVFlare |
| femnist  | 2 | 10 | 2000 | 0.8025 | 0.8100 | 0.7800 | NVFlare |
| femnist  | 2 | 10 | 4000 | 0.8113 | 0.8100 | 0.7813 | Flower |
| femnist  | 2 | 15 | 1000 | 0.6950 | 0.7200 | 0.7600 | FedML |
| femnist  | 2 | 15 | 2000 | 0.7950 | 0.8000 | 0.7800 | NVFlare |
| femnist  | 2 | 15 | 4000 | 0.7850 | 0.7950 | 0.7775 | NVFlare |
| femnist  | 5 | 5  | 1000 | 0.7120 | 0.6400 | 0.6800 | Flower |
| femnist  | 5 | 5  | 2000 | 0.7880 | 0.7800 | 0.7520 | Flower |
| femnist  | 5 | 5  | 4000 | 0.8245 | 0.8075 | 0.7845 | Flower |
| femnist  | 5 | 10 | 1000 | 0.7280 | 0.7100 | 0.7140 | Flower |
| femnist  | 5 | 10 | 2000 | 0.7910 | 0.7900 | 0.7700 | Flower |
| femnist  | 5 | 10 | 4000 | 0.8300 | 0.8125 | 0.8035 | Flower |
| femnist  | 5 | 15 | 1000 | 0.7560 | 0.6800 | 0.7200 | Flower |
| femnist  | 5 | 15 | 2000 | 0.7890 | 0.7550 | 0.7860 | Flower |
| femnist  | 5 | 15 | 4000 | 0.8175 | 0.8200 | 0.8115 | NVFlare |
| femnist  | 8 | 5  | 1000 | 0.7550 | 0.6500 | 0.7013 | Flower |
| femnist  | 8 | 5  | 2000 | 0.8050 | 0.7450 | 0.7713 | Flower |
| femnist  | 8 | 5  | 4000 | 0.8353 | 0.8300 | 0.7922 | Flower |
| femnist  | 8 | 10 | 1000 | 0.7988 | 0.7500 | 0.7438 | Flower |
| femnist  | 8 | 10 | 2000 | 0.8238 | 0.8100 | 0.7888 | Flower |
| femnist  | 8 | 10 | 4000 | 0.8466 | 0.8575 | 0.8131 | NVFlare |
| femnist  | 8 | 15 | 1000 | 0.7900 | 0.7800 | 0.7638 | Flower |
| femnist  | 8 | 15 | 2000 | 0.8244 | 0.8100 | 0.8063 | Flower |
| femnist  | 8 | 15 | 4000 | 0.8447 | 0.8450 | 0.8203 | NVFlare |

\* `femnist, 2 clients, 5 rounds, 1000 samples`: Flower (0.6900) and FedML (0.6900) are exactly
tied to 4 decimal places; listed as Flower per `pandas.idxmax`'s first-occurrence tie-break, not a
genuine win.

## Source

`docs/analysis/matched_configs.csv` (regenerate with `python3 scripts/generate_thesis_analysis.py`,
which rebuilds it from `results/*.json` using the same deduplication logic — latest valid run per
config — used everywhere else in this analysis).
