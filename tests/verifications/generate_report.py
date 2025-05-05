#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""
Test Report Generator

Description:
    This script runs pytest tests (specifically designed for OpenAI API compatibility checks)
    for different providers, aggregates the results from JSON reports, and generates
    a markdown summary report (REPORT.md).

    It automatically cleans up old test result files, keeping only the latest
    per provider.


Configuration:
    - Provider details (models, display names) are loaded from `tests/verifications/conf/*.yaml`.
    - Test cases are defined in YAML files within `tests/verifications/openai_api/fixtures/test_cases/`.
    - Test results are stored in `tests/verifications/test_results/`.

Usage:
    # Generate a report using the latest existing test results
    python tests/verifications/generate_report.py

    # Run tests for all configured providers and generate a report
    python tests/verifications/generate_report.py --run-tests

    # Run tests only for specific providers (space-separated)
    python tests/verifications/generate_report.py --run-tests --providers fireworks openai

    # Run tests matching a keyword expression (uses pytest -k)
    python tests/verifications/generate_report.py --run-tests --providers fireworks --k "streaming"

    # Run a specific test case for a provider
    python tests/verifications/generate_report.py --run-tests --providers fireworks --k "test_chat_streaming_basic and basic_earth"

    # Save the report to a custom location
    python tests/verifications/generate_report.py --output custom_report.md
"""

import argparse
import json
import os
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from tests.verifications.openai_api.fixtures.fixtures import _load_all_verification_configs

# Define the root directory for test results
RESULTS_DIR = Path(__file__).parent / "test_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Maximum number of test result files to keep per provider
MAX_RESULTS_PER_PROVIDER = 1

DEFAULT_PROVIDERS = [
    "meta_reference",
    "together",
    "fireworks",
    "openai",
]

VERIFICATION_CONFIG = _load_all_verification_configs()


def run_tests(provider, keyword=None):
    """Run pytest for a specific provider and save results"""
    print(f"Running tests for provider: {provider}")

    timestamp = int(time.time())
    # Use a constant filename for the final result and temp file
    result_file = RESULTS_DIR / f"{provider}.json"
    temp_json_file = RESULTS_DIR / f"temp_{provider}.json"

    # Determine project root directory relative to this script
    project_root = Path(__file__).parent.parent.parent

    # Run pytest with JSON output
    cmd = [
        "python",
        "-m",
        "pytest",
        "tests/verifications/openai_api/test_chat_completion.py",
        f"--provider={provider}",
        "-v",
        "--json-report",
        f"--json-report-file={temp_json_file}",
    ]

    # Append -k argument if provided
    if keyword:
        cmd.extend(["-k", keyword])

    try:
        # Run subprocess with cwd set to project root
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
        print(f"Pytest exit code: {result.returncode}")

        # Check if the JSON file was created
        if temp_json_file.exists():
            with open(temp_json_file) as f:
                test_results = json.load(f)

            test_results["run_timestamp"] = timestamp

            # Save results to the final (overwritten) file
            with open(result_file, "w") as f:
                json.dump(test_results, f, indent=2)
                f.write("\n")  # Add a trailing newline for precommit

            # Clean up temp file
            temp_json_file.unlink()

            print(f"Test results saved to {result_file}")
            return result_file
        else:
            print(f"Error: JSON report file not created for {provider}")
            print(f"Command stdout: {result.stdout}")
            print(f"Command stderr: {result.stderr}")
            return None
    except Exception as e:
        print(f"Error running tests for {provider}: {e}")
        return None


def run_multiple_tests(providers_to_run: list[str], keyword: str | None):
    """Runs tests for a list of providers."""
    print(f"Running tests for providers: {', '.join(providers_to_run)}")
    for provider in providers_to_run:
        run_tests(provider.strip(), keyword=keyword)
    print("Finished running tests.")


def parse_results(
    result_file,
) -> tuple[defaultdict[str, defaultdict[str, dict[str, bool]]], defaultdict[str, set[str]], set[str], str]:
    """Parse a single test results file.

    Returns:
        Tuple containing:
        - parsed_results: DefaultDict[provider, DefaultDict[model, Dict[test_name, pass_status]]]
        - providers_in_file: DefaultDict[provider, Set[model]] found in this file.
        - tests_in_file: Set[test_name] found in this file.
        - run_timestamp: Timestamp when the test was run
    """
    if not os.path.exists(result_file):
        print(f"Results file does not exist: {result_file}")
        # Return empty defaultdicts/set matching the type hint
        return defaultdict(lambda: defaultdict(dict)), defaultdict(set), set(), ""

    with open(result_file) as f:
        results = json.load(f)

    # Initialize results dictionary with specific types
    parsed_results: defaultdict[str, defaultdict[str, dict[str, bool]]] = defaultdict(lambda: defaultdict(dict))
    providers_in_file: defaultdict[str, set[str]] = defaultdict(set)
    tests_in_file: set[str] = set()
    # Extract provider from filename (e.g., "openai.json" -> "openai")
    provider: str = result_file.stem

    # Extract run timestamp from the JSON data
    run_timestamp_unix = results.get("run_timestamp")
    run_timestamp_str = (
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run_timestamp_unix))
        if run_timestamp_unix is not None
        else "Unknown"
    )

    # Debug: Print summary of test results
    print(f"Test results summary for {provider}:")
    print(f"Total tests: {results.get('summary', {}).get('total', 0)}")
    print(f"Passed: {results.get('summary', {}).get('passed', 0)}")
    print(f"Failed: {results.get('summary', {}).get('failed', 0)}")
    print(f"Error: {results.get('summary', {}).get('error', 0)}")
    print(f"Skipped: {results.get('summary', {}).get('skipped', 0)}")

    # Extract test results
    if "tests" not in results or not results["tests"]:
        print(f"No test results found in {result_file}")
        # Return empty defaultdicts/set matching the type hint
        return defaultdict(lambda: defaultdict(dict)), defaultdict(set), set(), ""

    # Process the tests
    for test in results["tests"]:
        test_id = test.get("nodeid", "")

        if not (call_phase := test.get("call")):
            continue
        call_outcome = call_phase.get("outcome")
        if call_outcome not in ("passed", "failed"):
            continue

        # --- Extract data from metadata ---
        metadata = test.get("metadata", {})
        model = metadata.get("model")
        case_id = metadata.get("case_id")  # String ID (if provided)
        case_index = metadata.get("case_index")  # Integer index (if no ID provided)

        # Check if we have a model and at least one case identifier
        if not model or (case_id is None and case_index is None):
            print(
                f"Warning: Missing 'model' or case identifier ('case_id'/'case_index') metadata for test: {test_id}. Skipping."
            )
            continue

        try:
            test_name_base = test_id.split("::")[1].split("[")[0]
        except (IndexError, ValueError) as e:
            print(f"Warning: Could not parse base test name for {test_id}. Error: {e}. Skipping.")
            continue

        # Construct detailed test name using ID or index
        if case_id is not None:
            detailed_test_name = f"{test_name_base} ({case_id})"
        elif case_index == 0:
            # If case_id is missing and index is 0, assume single case, use base name only
            detailed_test_name = test_name_base
        elif case_index is not None:  # case_index > 0
            # Use case_index for naming if case_id wasn't provided and index > 0
            detailed_test_name = f"{test_name_base} (case{case_index})"
        else:
            # This case should be prevented by the earlier check, but handle defensively
            print(f"Error: No case identifier found for test {test_id} after initial check. Skipping.")
            continue

        # Populate collections for this file
        tests_in_file.add(detailed_test_name)
        providers_in_file[provider].add(model)

        if call_outcome == "passed":
            parsed_results[provider][model][detailed_test_name] = True
        elif call_outcome == "failed":
            parsed_results[provider][model][detailed_test_name] = False

    # Final Summary Warning (Optional)
    if not parsed_results.get(provider):
        print(f"Warning: No valid test results parsed for provider {provider} from file {result_file}")

    return parsed_results, providers_in_file, tests_in_file, run_timestamp_str


def generate_report(
    results_dict: dict[str, Any],
    providers: dict[str, set[str]],
    all_tests: set[str],
    provider_timestamps: dict[str, str],
    output_file=None,
):
    """Generate the markdown report.

    Args:
        results_dict: Aggregated results [provider][model][test_name] -> status.
        providers: Dict of all providers and their models {provider: {models}}.
                   The order of keys in this dict determines the report order.
        all_tests: Set of all test names found.
        provider_timestamps: Dict of provider to timestamp when tests were run
        output_file: Optional path to save the report.
    """
    if output_file is None:
        # Default to creating the report in the same directory as this script
        output_file = Path(__file__).parent / "REPORT.md"
    else:
        output_file = Path(output_file)

    # Convert provider model sets to sorted lists (use passed-in providers dict)
    providers_sorted = {prov: sorted(models) for prov, models in providers.items()}

    # Sort tests alphabetically (use passed-in all_tests set)
    sorted_tests = sorted(all_tests)

    # Calculate counts for each base test name
    base_test_case_counts: defaultdict[str, int] = defaultdict(int)
    base_test_name_map: dict[str, str] = {}
    for test_name in sorted_tests:
        match = re.match(r"^(.*?)( \([^)]+\))?$", test_name)
        if match:
            base_name = match.group(1).strip()
            base_test_case_counts[base_name] += 1
            base_test_name_map[test_name] = base_name
        else:
            # Should not happen with current naming, but handle defensively
            base_test_case_counts[test_name] += 1
            base_test_name_map[test_name] = test_name

    if not sorted_tests:
        print("Warning: No test results found to generate a report.")
        # Optionally create an empty report or return early
        with open(output_file, "w") as f:
            f.write("# Test Results Report\n\nNo test results found.\n")
        print(f"Generated empty report: {output_file}")
        return

    report = ["# Test Results Report\n"]
    report.append(f"*Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")
    report.append("*This report was generated by running `python tests/verifications/generate_report.py`*\n")

    # Icons for pass/fail
    pass_icon = "✅"
    fail_icon = "❌"
    na_icon = "⚪"

    # Add emoji legend
    report.append("## Legend\n")
    report.append(f"- {pass_icon} - Test passed")
    report.append(f"- {fail_icon} - Test failed")
    report.append(f"- {na_icon} - Test not applicable or not run for this model")
    report.append("\n")

    # Add a summary section
    report.append("## Summary\n")

    # Count total tests and passes (use passed-in providers and all_tests)
    total_tests = 0
    passed_tests = 0
    provider_totals = {}
    for provider, models in providers_sorted.items():
        provider_passed = 0
        provider_total = 0
        if provider in results_dict:
            for model in models:
                if model in results_dict[provider]:
                    model_results = results_dict[provider][model]
                    for test in sorted_tests:
                        if test in model_results:
                            provider_total += 1
                            total_tests += 1
                            if model_results[test]:
                                provider_passed += 1
                                passed_tests += 1
        provider_totals[provider] = (provider_passed, provider_total)

    # Add summary table (use the order from the providers dict keys)
    report.append("| Provider | Pass Rate | Tests Passed | Total Tests |")
    report.append("| --- | --- | --- | --- |")
    # Iterate through providers in the order they appear in the input dict
    for provider in providers_sorted.keys():
        passed, total = provider_totals.get(provider, (0, 0))
        pass_rate = f"{(passed / total * 100):.1f}%" if total > 0 else "N/A"
        report.append(f"| {provider.capitalize()} | {pass_rate} | {passed} | {total} |")
    report.append("\n")

    for provider in providers_sorted.keys():
        provider_models = providers_sorted[provider]  # Use sorted models
        if not provider_models:
            continue

        report.append(f"\n## {provider.capitalize()}\n")

        # Add timestamp when test was run
        if provider in provider_timestamps:
            report.append(f"*Tests run on: {provider_timestamps[provider]}*\n")

        # Add test command for reproducing results
        test_cmd_all = f"pytest tests/verifications/openai_api/test_chat_completion.py --provider={provider} -v"
        report.append(f"```bash\n# Run all tests for this provider:\n{test_cmd_all}\n")

        # Find an example test with a case ID
        example_base_test_name = None
        example_case_id = None
        # Get first test as fallback base, handle empty list
        first_test_name = sorted_tests[0] if sorted_tests else "unknown_test"

        match = re.match(r"^(.*?) \((.*?)\)$", first_test_name)
        if match:
            example_base_test_name = match.group(1).strip()
            example_case_id = match.group(2).strip()
        else:
            example_base_test_name = first_test_name

        base_name = base_test_name_map.get(first_test_name, first_test_name)  # Get base name
        case_count = base_test_case_counts.get(base_name, 1)  # Get count
        filter_str = f"{example_base_test_name} and {example_case_id}" if case_count > 1 else example_base_test_name

        test_cmd_specific_case = (
            f'pytest tests/verifications/openai_api/test_chat_completion.py --provider={provider} -k "{filter_str}"'
        )
        report.append(
            f"# Example: Run only the '{example_case_id}' case of {example_base_test_name}:\n{test_cmd_specific_case}\n```\n"
        )

        # Get display names (use passed-in providers dict)
        provider_config = VERIFICATION_CONFIG.get("providers", {}).get(provider, {})
        display_name_map = provider_config.get("model_display_names", {})

        # Add Model Key Table (use provider_models)
        report.append(f"\n**Model Key ({provider.capitalize()})**\n")
        provider_key_lines = ["| Display Name | Full Model ID |", "| --- | --- |"]
        for model_id in provider_models:
            display_name = display_name_map.get(model_id, model_id)
            provider_key_lines.append(f"| {display_name} | `{model_id}` |")
        report.extend(provider_key_lines)
        report.append("\n")

        # Create results table header (use provider_models)
        display_names = [display_name_map.get(m, m) for m in provider_models]
        header = "| Test | " + " | ".join(display_names) + " |"
        separator = "| --- | " + " | ".join(["---"] * len(provider_models)) + " |"
        report.append(header)
        report.append(separator)

        # Get results for this provider from results_dict
        provider_results_data = results_dict.get(provider, {})

        # Add rows for each test (use sorted_tests)
        for test in sorted_tests:
            # Determine display name based on case count
            base_name = base_test_name_map.get(test, test)  # Get base name
            case_count = base_test_case_counts.get(base_name, 1)  # Get count
            display_test_name = base_name if case_count == 1 else test  # Choose display name
            row = f"| {display_test_name} |"  # Use display name

            for model_id in provider_models:
                if model_id in provider_results_data and test in provider_results_data[model_id]:
                    result = pass_icon if provider_results_data[model_id][test] else fail_icon
                else:
                    result = na_icon
                row += f" {result} |"
            report.append(row)

    # Write to file
    with open(output_file, "w") as f:
        f.write("\n".join(report))
        f.write("\n")

    print(f"Report generated: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate test report")
    parser.add_argument("--run-tests", action="store_true", help="Run tests before generating report")
    parser.add_argument(
        "--providers",
        type=str,
        nargs="+",
        help="Specify providers to include/test (comma-separated or space-separated, default: uses DEFAULT_PROVIDERS)",
    )
    parser.add_argument("--output", type=str, help="Output file location (default: tests/verifications/REPORT.md)")
    parser.add_argument("--k", type=str, help="Keyword expression to filter tests (passed to pytest -k)")
    args = parser.parse_args()

    all_results = {}
    final_providers_order = {}  # Dictionary to store results, preserving processing order
    aggregated_tests = set()
    provider_timestamps = {}

    # 1. Determine the desired list and order of providers
    if args.providers:
        desired_providers = []
        for provider_arg in args.providers:
            desired_providers.extend([p.strip() for p in provider_arg.split(",")])
    else:
        desired_providers = DEFAULT_PROVIDERS  # Use default order/list

    # 2. Run tests if requested (using the desired provider list)
    if args.run_tests:
        run_multiple_tests(desired_providers, args.k)

    for provider in desired_providers:
        # Construct the expected result file path directly
        result_file = RESULTS_DIR / f"{provider}.json"

        if result_file.exists():  # Check if the specific file exists
            print(f"Loading results for {provider} from {result_file}")
            try:
                parsed_data = parse_results(result_file)
                parsed_results, providers_in_file, tests_in_file, run_timestamp = parsed_data
                all_results.update(parsed_results)
                aggregated_tests.update(tests_in_file)

                # Add models for this provider, ensuring it's added in the correct report order
                if provider in providers_in_file:
                    if provider not in final_providers_order:
                        final_providers_order[provider] = set()
                    final_providers_order[provider].update(providers_in_file[provider])
                    if run_timestamp != "Unknown":
                        provider_timestamps[provider] = run_timestamp
                else:
                    print(
                        f"Warning: Provider '{provider}' found in desired list but not within its result file data ({result_file})."
                    )

            except Exception as e:
                print(f"Error parsing results for provider {provider} from {result_file}: {e}")
        else:
            # Only print warning if we expected results (i.e., provider was in the desired list)
            print(f"Result file for desired provider '{provider}' not found at {result_file}. Skipping.")

    # 5. Generate the report using the filtered & ordered results
    print(f"Final Provider Order for Report: {list(final_providers_order.keys())}")
    generate_report(all_results, final_providers_order, aggregated_tests, provider_timestamps, args.output)


if __name__ == "__main__":
    main()
