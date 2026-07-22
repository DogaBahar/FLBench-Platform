"""
Reads every submission in results/ and renders docs/index.html, a static
leaderboard comparing frameworks/strategies per dataset. Run by
.github/workflows/publish-leaderboard.yml on every push to main that touches
results/**; GitHub Pages serves docs/ as-is, no build step on the Pages side.

Usage: python scripts/generate_leaderboard.py [--repo owner/name]
"""
import argparse
import html
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OUT_PATH = REPO_ROOT / "docs" / "index.html"


def final_value(pairs):
    if not pairs:
        return None
    return max(pairs, key=lambda p: p[0])[1]


def load_rows(repo: str | None):
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        if path.name == "schema.json":
            continue
        data = json.loads(path.read_text())
        if data.get("status") != "COMPLETED":
            continue

        gm = data["results"]["global_metrics"]
        accuracy = final_value(gm["metrics_distributed"]["accuracy"])
        loss = final_value(gm["losses_distributed"])

        rows.append({
            "dataset": data["dataset"],
            "framework": data["framework"],
            "strategy": data["strategy"],
            "rounds": data["config"]["rounds"],
            "accuracy": accuracy,
            "loss": loss,
            "wall_clock_time_seconds": gm.get("wall_clock_time_seconds"),
            "submitted_by": data.get("submitted_by") or "anonymous",
            "run_id": data["run_id"],
            "filename": path.name,
            "url": f"https://github.com/{repo}/blob/main/results/{path.name}" if repo else None,
        })

    rows.sort(key=lambda r: (r["dataset"], -(r["accuracy"] or -1)))
    return rows


def fmt(value, digits=4):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "–"


def render(rows) -> str:
    body_rows = []
    for r in rows:
        run_cell = (
            f'<a href="{html.escape(r["url"])}">{html.escape(r["run_id"])}</a>'
            if r["url"] else html.escape(r["run_id"])
        )
        body_rows.append(f"""
        <tr>
          <td>{html.escape(r["dataset"])}</td>
          <td>{html.escape(r["framework"])}</td>
          <td>{html.escape(r["strategy"])}</td>
          <td>{r["rounds"]}</td>
          <td>{fmt(r["accuracy"])}</td>
          <td>{fmt(r["loss"])}</td>
          <td>{fmt(r["wall_clock_time_seconds"], 1)}</td>
          <td>{html.escape(r["submitted_by"])}</td>
          <td>{run_cell}</td>
        </tr>""")

    rows_html = "".join(body_rows) if body_rows else (
        '<tr><td colspan="9">No submissions yet — see results/README.md to contribute one.</td></tr>'
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FL Benchmark Platform - Community Leaderboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; }}
  h1 {{ font-size: 1.4rem; }}
  p.sub {{ opacity: 0.7; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
  th, td {{ text-align: left; padding: 0.5rem 0.6rem; border-bottom: 1px solid rgba(128,128,128,0.3); font-size: 0.9rem; }}
  th {{ position: sticky; top: 0; background: Canvas; }}
  tr:hover td {{ background: rgba(128,128,128,0.08); }}
</style>
</head>
<body>
<h1>FL Benchmark Platform &mdash; Community Leaderboard</h1>
<p class="sub">Generated from community-submitted runs in <code>results/</code>. Best accuracy per dataset first. See the repo's <code>results/README.md</code> to submit your own run.</p>
<table>
  <thead>
    <tr>
      <th>Dataset</th><th>Framework</th><th>Strategy</th><th>Rounds</th>
      <th>Final accuracy</th><th>Final loss</th><th>Wall clock (s)</th>
      <th>Submitted by</th><th>Run</th>
    </tr>
  </thead>
  <tbody>{rows_html}
  </tbody>
</table>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=None, help="owner/name, used to link runs back to their JSON on GitHub")
    args = parser.parse_args()

    rows = load_rows(args.repo)
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(render(rows))
    print(f"Wrote {OUT_PATH} ({len(rows)} completed submissions)")


if __name__ == "__main__":
    main()
