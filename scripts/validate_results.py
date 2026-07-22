"""
Validates every submission in results/ against results/schema.json, plus a
filename/run_id consistency check the JSON Schema itself can't express. Run
by .github/workflows/validate-results.yml on every PR touching results/**.

Usage: python scripts/validate_results.py
"""
import json
import re
import sys
from pathlib import Path

import jsonschema

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
SCHEMA_PATH = RESULTS_DIR / "schema.json"

FILENAME_RE = re.compile(r"^(?P<framework>[a-z0-9]+)-(?P<dataset>[a-z0-9]+)-(?P<run_id>.+)\.json$")


def validate_file(path: Path, schema: dict) -> list[str]:
    errors = []

    m = FILENAME_RE.match(path.name)
    if not m:
        errors.append(
            f"filename '{path.name}' doesn't match the required "
            "'<framework>-<dataset>-<run_id>.json' pattern"
        )

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        errors.append(f"invalid JSON: {e}")
        return errors

    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as e:
        errors.append(f"schema violation at {list(e.absolute_path)}: {e.message}")
        return errors

    if m:
        if data.get("framework") != m.group("framework"):
            errors.append(f"filename says framework='{m.group('framework')}' but file has '{data.get('framework')}'")
        if data.get("dataset") != m.group("dataset"):
            errors.append(f"filename says dataset='{m.group('dataset')}' but file has '{data.get('dataset')}'")
        if data.get("run_id") != m.group("run_id"):
            errors.append(f"filename run_id '{m.group('run_id')}' doesn't match file's run_id '{data.get('run_id')}'")

    return errors


def main() -> int:
    schema = json.loads(SCHEMA_PATH.read_text())
    result_files = sorted(RESULTS_DIR.glob("*.json"))
    result_files = [p for p in result_files if p.name != "schema.json"]

    if not result_files:
        print("No result files to validate.")
        return 0

    had_errors = False
    for path in result_files:
        errors = validate_file(path, schema)
        if errors:
            had_errors = True
            print(f"FAIL {path.relative_to(REPO_ROOT)}")
            for err in errors:
                print(f"  - {err}")
        else:
            print(f"OK   {path.relative_to(REPO_ROOT)}")

    return 1 if had_errors else 0


if __name__ == "__main__":
    sys.exit(main())
