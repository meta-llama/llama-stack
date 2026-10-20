#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Clean up unused test recordings based on CI test collection.

This script:
1. Reads CI matrix definitions from tests/integration/ci_matrix.json (default + scheduled overrides)
2. Uses pytest --collect-only with --json-report to gather all test IDs that run in CI,
   running the per-config collections concurrently
3. Compares against existing recordings to identify unused ones
4. Optionally deletes unused recordings

Usage:
    # Dry run - see what would be deleted
    ./scripts/cleanup_recordings.py

    # Save manifest of CI test IDs for inspection
    ./scripts/cleanup_recordings.py --manifest ci_tests.txt

    # Actually delete unused recordings
    ./scripts/cleanup_recordings.py --delete

    # Control how many configs collect concurrently (default: min(cpu_count, 8))
    ./scripts/cleanup_recordings.py --jobs 4
"""

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# Load CI matrix from JSON file
CI_MATRIX_FILE = REPO_ROOT / "tests/integration/ci_matrix.json"
with open(CI_MATRIX_FILE) as f:
    _matrix_config = json.load(f)

DEFAULT_CI_MATRIX: list[dict[str, str]] = _matrix_config["default"]
SCHEDULED_MATRICES: dict[str, list[dict[str, str]]] = _matrix_config.get("schedules", {})

# Top-level ``test_id`` is always written as the first key (see store_recording),
# so a short file prefix is enough to read it without parsing the response body.
_TEST_ID_RE = re.compile(rb'"test_id"\s*:\s*("(?:[^"\\]|\\.)*"|null)')


def _unique_configs(entries):
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        suite = entry["suite"]
        setup = entry["setup"]
        key = (suite, setup)
        if key in seen:
            continue
        seen.add(key)
        yield {"suite": suite, "setup": setup}


def iter_all_ci_configs() -> list[dict[str, str]]:
    """Return unique CI configs across default and scheduled matrices."""
    combined = list(DEFAULT_CI_MATRIX)
    for configs in SCHEDULED_MATRICES.values():
        combined.extend(configs)
    return list(_unique_configs(combined))


def _collect_config(config: dict[str, str]) -> tuple[set[str], int, str]:
    """Run one ``pytest --collect-only`` subprocess and parse its test node IDs.

    Returns ``(node_ids, returncode, message)``. ``message`` holds the failure
    detail when ``returncode != 0``, or a parse warning otherwise.
    """
    # Each config gets its own report file so concurrent runs never collide.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json_report_file = f.name

    try:
        env = os.environ.copy()
        env["PYTEST_ADDOPTS"] = f"--json-report --json-report-file={json_report_file}"
        repo_path = str(REPO_ROOT)
        existing_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{repo_path}{os.pathsep}{existing_path}" if existing_path else repo_path

        result = subprocess.run(
            [
                "./scripts/integration-tests.sh",
                "--collect-only",
                "--suite",
                config["suite"],
                "--setup",
                config["setup"],
            ],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env=env,
        )

        if result.returncode != 0:
            return (
                set(),
                result.returncode,
                "Test collection failed.\n"
                f"Command: {' '.join(result.args)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}",
            )

        try:
            with open(json_report_file) as f:
                report = json.load(f)

            node_ids: set[str] = set()
            # The "collectors" field contains collected test items; each collector
            # has a "result" array with test node IDs.
            for collector in report.get("collectors", []):
                for item in collector.get("result", []):
                    if "nodeid" in item:
                        node_ids.add(item["nodeid"])
            return node_ids, 0, ""
        except (json.JSONDecodeError, FileNotFoundError) as e:
            return set(), 0, f"Failed to parse JSON report: {e}"
    finally:
        if os.path.exists(json_report_file):
            os.unlink(json_report_file)


def collect_ci_tests(jobs: int) -> tuple[set[str], list[dict[str, str]]]:
    """Collect all test IDs that would run in CI using --collect-only with JSON output.

    Each config is a separate pytest subprocess; the collection work is CPU-bound,
    so running ``jobs`` of them concurrently overlaps the per-config cost.
    """
    all_test_ids: set[str] = set()
    configs = iter_all_ci_configs()
    print(f"Collecting tests for {len(configs)} CI configs with {jobs} concurrent job(s)...")

    hard_failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as executor:
        futures = {executor.submit(_collect_config, config): config for config in configs}
        for future in concurrent.futures.as_completed(futures):
            config = futures[future]
            node_ids, returncode, message = future.result()
            all_test_ids |= node_ids
            if returncode != 0:
                hard_failures.append(f"suite={config['suite']}, setup={config['setup']}\n{message}")
            elif message:
                print(f"  Warning (suite={config['suite']}, setup={config['setup']}): {message}")
            print(
                f"  suite={config['suite']}, setup={config['setup']}: {len(node_ids)} IDs (total {len(all_test_ids)})"
            )

    if hard_failures:
        raise RuntimeError("Test collection failed for:\n\n" + "\n\n".join(hard_failures))

    print(f"\nTotal unique test IDs collected: {len(all_test_ids)}")
    return all_test_ids, configs


def get_base_test_id(test_id: str) -> str:
    """Extract base test ID without parameterization.

    Example:
      'tests/integration/inference/test_foo.py::test_bar[param1-param2]'
      -> 'tests/integration/inference/test_foo.py::test_bar'
    """
    return test_id.split("[")[0] if "[" in test_id else test_id


def find_all_recordings():
    """Find all recording JSON files."""
    return list((REPO_ROOT / "tests/integration").rglob("recordings/*.json"))


def _read_test_id(json_file: Path) -> str | None:
    """Return a recording's top-level ``test_id`` without parsing its body.

    Only a short prefix is read, which matters for large streaming recordings.
    Falls back to a full parse if the key is not found in the prefix.
    """
    with open(json_file, "rb") as f:
        match = _TEST_ID_RE.search(f.read(65536))
    if match is not None:
        try:
            value: object = json.loads(match.group(1))
        except ValueError:
            value = None
        return value if isinstance(value, str) else None
    with open(json_file) as f:
        value = json.load(f).get("test_id", "")
    return value if isinstance(value, str) else None


def analyze_recordings(ci_test_ids, dry_run=True):
    """Analyze recordings and identify unused ones."""

    # Use full test IDs with parameterization for exact matching
    all_recordings = find_all_recordings()
    print(f"\nTotal recording files: {len(all_recordings)}")

    # Categorize recordings
    used_recordings = []
    unused_recordings = []
    shared_recordings = []  # model-list endpoints without test_id
    parse_errors = []

    for json_file in all_recordings:
        try:
            test_id = _read_test_id(json_file)

            if not test_id:
                # Shared/infrastructure recordings (model lists, etc)
                shared_recordings.append(json_file)
                continue

            # Match exact test_id (with full parameterization)
            if test_id in ci_test_ids:
                used_recordings.append(json_file)
            else:
                unused_recordings.append((json_file, test_id))

        except Exception as e:
            parse_errors.append((json_file, str(e)))

    # Print summary
    print("\nRecording Analysis:")
    print(f"  Used in CI:     {len(used_recordings)}")
    print(f"  Shared (no ID): {len(shared_recordings)}")
    print(f"  UNUSED:         {len(unused_recordings)}")
    print(f"  Parse errors:   {len(parse_errors)}")

    if unused_recordings:
        print("\nUnused recordings by test:")

        # Group by base test ID
        by_test = defaultdict(list)
        for file, test_id in unused_recordings:
            base = get_base_test_id(test_id)
            by_test[base].append(file)

        for base_test, files in sorted(by_test.items()):
            print(f"\n  {base_test}")
            print(f"    ({len(files)} recording(s))")
            for f in files[:3]:
                print(f"      - {f.relative_to(REPO_ROOT / 'tests/integration')}")
            if len(files) > 3:
                print(f"      ... and {len(files) - 3} more")

    if parse_errors:
        print("\nParse errors:")
        for file, error in parse_errors[:5]:
            print(f"  {file.relative_to(REPO_ROOT)}: {error}")
        if len(parse_errors) > 5:
            print(f"  ... and {len(parse_errors) - 5} more")

    # Perform cleanup
    if not dry_run:
        print(f"\nDeleting {len(unused_recordings)} unused recordings...")
        for file, _ in unused_recordings:
            file.unlink()
            print(f"  Deleted: {file.relative_to(REPO_ROOT / 'tests/integration')}")
        print("✅ Cleanup complete")
    else:
        print("\n(Dry run - no files deleted)")
        print("\nTo delete these files, run with --delete")

    return len(unused_recordings)


def main():
    parser = argparse.ArgumentParser(
        description="Clean up unused test recordings based on CI test collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--delete", action="store_true", help="Actually delete unused recordings (default is dry-run)")
    parser.add_argument("--manifest", help="Save collected test IDs to file (optional)")
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(os.cpu_count() or 4, 8),
        help="Number of configs to collect concurrently (default: min(cpu_count, 8))",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Recording Cleanup Utility")
    print("=" * 60)

    ci_configs = iter_all_ci_configs()

    print(f"\nDetected CI configurations: {len(ci_configs)}")
    for config in ci_configs:
        print(f"  - suite={config['suite']}, setup={config['setup']}")

    # Collect test IDs from CI configurations
    ci_test_ids, _ = collect_ci_tests(args.jobs)

    if args.manifest:
        with open(args.manifest, "w") as f:
            for test_id in sorted(ci_test_ids):
                f.write(f"{test_id}\n")
        print(f"\nSaved test IDs to: {args.manifest}")

    # Analyze and cleanup
    unused_count = analyze_recordings(ci_test_ids, dry_run=not args.delete)

    print("\n" + "=" * 60)
    if unused_count > 0 and not args.delete:
        print("Run with --delete to remove unused recordings")


if __name__ == "__main__":
    main()
