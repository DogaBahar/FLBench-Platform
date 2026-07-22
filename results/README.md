# Community benchmark results

This folder is the shared results pool for the FL Benchmark Platform. There's
no hosted service behind it — a submission is a JSON file added to this
folder through a normal pull request. Every file here is validated by CI on
the PR that adds it (see `.github/workflows/validate-results.yml`), and the
combined set is rendered into a static leaderboard published via GitHub
Pages on merge (see `.github/workflows/publish-leaderboard.yml`).

## Contributing a run

1. Run your benchmark as usual (see the root README's Quickstart).
2. Once it reaches `COMPLETED`, export it — either way produces the same file:
   - **From the UI**: open the run in the *Benchmark Results* tab and click
     **Export for results/** — downloads `<framework>-<dataset>-<run_id>.json`
     straight from your browser (`GET /api/runs/<run_id>/export`).
   - **From the CLI**:
     ```bash
     docker compose exec backend python -m scripts.export_run <run_id> --submitted-by "your-name-or-handle"
     ```
     This writes the same file under `$SHARED_RUN_DIR/<run_id>/export/` on
     your host (default `/tmp/fl_benchmark_runs/<run_id>/export/`), which is
     where `docker-compose.yml` already bind-mounts that directory.
3. Fork this repository, copy that file into `results/`, and open a pull
   request. Keep the filename as generated — the validation workflow checks
   that it matches `<framework>-<dataset>-<run_id>.json` and that the
   `run_id` inside the file agrees with it.
4. CI validates the file against `results/schema.json`. Fix anything it
   flags and push again — the check reruns automatically.
5. Once merged, the leaderboard at the repo's GitHub Pages URL rebuilds
   automatically to include your run.

## What's in a submission file

Each file is a self-contained snapshot of one run: its config (framework,
dataset, strategy, rounds/epochs/batch size, full hyperparameters) and its
normalized results (accuracy/loss per round, per-client CPU/RAM/compute
time, server aggregation time, data distribution, wall-clock time) — the
same shape the platform's own `GET /api/runs/<id>` returns. See
`schema.json` for the exact structure, or any existing file in this folder
for a live example.

## Why this instead of a hosted endpoint

Nobody has to run or trust a server for this to work: submissions are
ordinary git history, reviewable in a PR diff, and a bad submission is a
`git revert` away. The trade-off is that it's PR-speed, not real-time —
that's an intentional fit for a research benchmark archive, not a live
telemetry stream.
