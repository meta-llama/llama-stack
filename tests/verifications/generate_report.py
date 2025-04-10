# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Test Report Generator

Requirements:
    pip install pytest-json-report

Usage:
    # Generate a report using existing test results
    python tests/verifications/generate_report.py

    # Run tests and generate a report
    python tests/verifications/generate_report.py --run-tests

    # Run tests for specific providers
    python tests/verifications/generate_report.py --run-tests --providers fireworks openai

    # Save the report to a custom location
    python tests/verifications/generate_report.py --output custom_report.md

    # Clean up old test result files
    python tests/verifications/generate_report.py --cleanup
"""

import argparse
import json
import os
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

# Define the root directory for test results
RESULTS_DIR = Path(__file__).parent / "test_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Maximum number of test result files to keep per provider
MAX_RESULTS_PER_PROVIDER = 1

# Custom order of providers
PROVIDER_ORDER = ["together", "fireworks", "groq", "cerebras", "openai"]

# Dictionary to store providers and their models (will be populated dynamically)
PROVIDERS = defaultdict(set)

# Tests will be dynamically extracted from results
ALL_TESTS = set()


def run_tests(provider):
    """Run pytest for a specific provider and save results"""
    print(f"Running tests for provider: {provider}")

    timestamp = int(time.time())
    result_file = RESULTS_DIR / f"{provider}_{timestamp}.json"
    temp_json_file = RESULTS_DIR / f"temp_{provider}_{timestamp}.json"

    # Run pytest with JSON output
    cmd = [
        "python",
        "-m",
        "pytest",
        "tests/verifications/openai/test_chat_completion.py",
        f"--provider={provider}",
        "-v",
        "--json-report",
        f"--json-report-file={temp_json_file}",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(f"Pytest exit code: {result.returncode}")

        # Check if the JSON file was created
        if temp_json_file.exists():
            # Read the JSON file and save it to our results format
            with open(temp_json_file, "r") as f:
                test_results = json.load(f)

            # Save results to our own format with a trailing newline
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


def parse_results(result_file):
    """Parse the test results file and extract pass/fail by model and test"""
    if not os.path.exists(result_file):
        print(f"Results file does not exist: {result_file}")
        return {}

    with open(result_file, "r") as f:
        results = json.load(f)

    # Initialize results dictionary
    parsed_results = defaultdict(lambda: defaultdict(dict))
    provider = os.path.basename(result_file).split("_")[0]

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
        return parsed_results

    # Map for normalizing model names
    model_name_map = {
        "Llama-3.3-8B-Instruct": "Llama-3.3-8B-Instruct",
        "Llama-3.3-70B-Instruct": "Llama-3.3-70B-Instruct",
        "Llama-3.2-11B-Vision-Instruct": "Llama-3.2-11B-Vision-Instruct",
        "Llama-4-Scout-17B-16E": "Llama-4-Scout-17B-16E-Instruct",
        "Llama-4-Scout-17B-16E-Instruct": "Llama-4-Scout-17B-16E-Instruct",
        "Llama-4-Maverick-17B-128E": "Llama-4-Maverick-17B-128E-Instruct",
        "Llama-4-Maverick-17B-128E-Instruct": "Llama-4-Maverick-17B-128E-Instruct",
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
    }

    # Keep track of all models found for this provider
    provider_models = set()

    # Track all unique test cases for each base test
    test_case_counts = defaultdict(int)

    # First pass: count the number of cases for each test
    for test in results["tests"]:
        test_id = test.get("nodeid", "")

        if "call" in test:
            test_name = test_id.split("::")[1].split("[")[0]
            input_output_match = re.search(r"\[input_output(\d+)-", test_id)
            if input_output_match:
                test_case_counts[test_name] += 1

    # Second pass: process the tests with case numbers only for tests with multiple cases
    for test in results["tests"]:
        test_id = test.get("nodeid", "")
        outcome = test.get("outcome", "")

        # Only process tests that have been executed (not setup errors)
        if "call" in test:
            # Regular test that actually ran
            test_name = test_id.split("::")[1].split("[")[0]

            # Extract input_output parameter to differentiate between test cases
            input_output_match = re.search(r"\[input_output(\d+)-", test_id)
            input_output_index = input_output_match.group(1) if input_output_match else ""

            # Create a more detailed test name with case number only if there are multiple cases
            detailed_test_name = test_name
            if input_output_index and test_case_counts[test_name] > 1:
                detailed_test_name = f"{test_name} (case {input_output_index})"

            # Track all unique test names
            ALL_TESTS.add(detailed_test_name)

            # Extract model name from test_id using a more robust pattern
            model_match = re.search(r"\[input_output\d+-([^\]]+)\]", test_id)
            if model_match:
                raw_model = model_match.group(1)
                model = model_name_map.get(raw_model, raw_model)

                # Add to set of known models for this provider
                provider_models.add(model)

                # Also update the global PROVIDERS dictionary
                PROVIDERS[provider].add(model)

                # Store the result
                if outcome == "passed":
                    parsed_results[provider][model][detailed_test_name] = True
                else:
                    parsed_results[provider][model][detailed_test_name] = False

                print(f"Parsed test result: {detailed_test_name} for model {model}: {outcome}")
        elif outcome == "error" and "setup" in test and test.get("setup", {}).get("outcome") == "failed":
            # This is a setup failure, which likely means a configuration issue
            # Extract the base test name and model name
            parts = test_id.split("::")
            if len(parts) > 1:
                test_name = parts[1].split("[")[0]

                # Extract input_output parameter to differentiate between test cases
                input_output_match = re.search(r"\[input_output(\d+)-", test_id)
                input_output_index = input_output_match.group(1) if input_output_match else ""

                # Create a more detailed test name with case number only if there are multiple cases
                detailed_test_name = test_name
                if input_output_index and test_case_counts[test_name] > 1:
                    detailed_test_name = f"{test_name} (case {input_output_index})"

                if detailed_test_name in ALL_TESTS:
                    # Use a more robust pattern for model extraction
                    model_match = re.search(r"\[input_output\d+-([^\]]+)\]", test_id)
                    if model_match:
                        raw_model = model_match.group(1)
                        model = model_name_map.get(raw_model, raw_model)

                        # Add to set of known models for this provider
                        provider_models.add(model)

                        # Also update the global PROVIDERS dictionary
                        PROVIDERS[provider].add(model)

                        # Mark setup failures as false (failed)
                        parsed_results[provider][model][detailed_test_name] = False
                        print(f"Parsed setup failure: {detailed_test_name} for model {model}")

    # Debug: Print parsed results
    if not parsed_results[provider]:
        print(f"Warning: No test results parsed for provider {provider}")
    else:
        for model, tests in parsed_results[provider].items():
            print(f"Model {model}: {len(tests)} test results")

    return parsed_results


def cleanup_old_results():
    """Clean up old test result files, keeping only the newest N per provider"""
    for provider in PROVIDERS.keys():
        # Get all result files for this provider
        provider_files = list(RESULTS_DIR.glob(f"{provider}_*.json"))

        # Sort by timestamp (newest first)
        provider_files.sort(key=lambda x: int(x.stem.split("_")[1]), reverse=True)

        # Remove old files beyond the max to keep
        if len(provider_files) > MAX_RESULTS_PER_PROVIDER:
            for old_file in provider_files[MAX_RESULTS_PER_PROVIDER:]:
                try:
                    old_file.unlink()
                    print(f"Removed old result file: {old_file}")
                except Exception as e:
                    print(f"Error removing file {old_file}: {e}")


def get_latest_results_by_provider():
    """Get the latest test result file for each provider"""
    provider_results = {}

    # Get all result files
    result_files = list(RESULTS_DIR.glob("*.json"))

    # Extract all provider names from filenames
    all_providers = set()
    for file in result_files:
        # File format is provider_timestamp.json
        parts = file.stem.split("_")
        if len(parts) >= 2:
            all_providers.add(parts[0])

    # Group by provider
    for provider in all_providers:
        provider_files = [f for f in result_files if f.name.startswith(f"{provider}_")]

        # Sort by timestamp (newest first)
        provider_files.sort(key=lambda x: int(x.stem.split("_")[1]), reverse=True)

        if provider_files:
            provider_results[provider] = provider_files[0]

    return provider_results


def generate_report(results_dict, output_file=None):
    """Generate the markdown report"""
    if output_file is None:
        # Default to creating the report in the same directory as this script
        output_file = Path(__file__).parent / "REPORT.md"
    else:
        output_file = Path(output_file)

    # Get the timestamp from result files
    provider_timestamps = {}
    provider_results = get_latest_results_by_provider()
    for provider, result_file in provider_results.items():
        # Extract timestamp from filename (format: provider_timestamp.json)
        try:
            timestamp_str = result_file.stem.split("_")[1]
            timestamp = int(timestamp_str)
            formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
            provider_timestamps[provider] = formatted_time
        except (IndexError, ValueError):
            provider_timestamps[provider] = "Unknown"

    # Convert provider model sets to sorted lists
    for provider in PROVIDERS:
        PROVIDERS[provider] = sorted(PROVIDERS[provider])

    # Sort tests alphabetically
    sorted_tests = sorted(ALL_TESTS)

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

    # Count total tests and passes
    total_tests = 0
    passed_tests = 0
    provider_totals = {}

    # Prepare summary data
    for provider in PROVIDERS.keys():
        provider_passed = 0
        provider_total = 0

        if provider in results_dict:
            provider_models = PROVIDERS[provider]
            for model in provider_models:
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

    # Add summary table
    report.append("| Provider | Pass Rate | Tests Passed | Total Tests |")
    report.append("| --- | --- | --- | --- |")

    # Use the custom order for summary table
    for provider in [p for p in PROVIDER_ORDER if p in PROVIDERS]:
        passed, total = provider_totals.get(provider, (0, 0))
        pass_rate = f"{(passed / total * 100):.1f}%" if total > 0 else "N/A"
        report.append(f"| {provider.capitalize()} | {pass_rate} | {passed} | {total} |")

    # Add providers not in the custom order
    for provider in [p for p in PROVIDERS if p not in PROVIDER_ORDER]:
        passed, total = provider_totals.get(provider, (0, 0))
        pass_rate = f"{(passed / total * 100):.1f}%" if total > 0 else "N/A"
        report.append(f"| {provider.capitalize()} | {pass_rate} | {passed} | {total} |")

    report.append("\n")

    # Process each provider in the custom order, then any additional providers
    for provider in sorted(
        PROVIDERS.keys(), key=lambda p: (PROVIDER_ORDER.index(p) if p in PROVIDER_ORDER else float("inf"), p)
    ):
        if not PROVIDERS[provider]:
            # Skip providers with no models
            continue

        report.append(f"\n## {provider.capitalize()}\n")

        # Add timestamp when test was run
        if provider in provider_timestamps:
            report.append(f"*Tests run on: {provider_timestamps[provider]}*\n")

        # Add test command for reproducing results
        test_cmd = f"pytest tests/verifications/openai/test_chat_completion.py --provider={provider} -v"
        report.append(f"```bash\n{test_cmd}\n```\n")

        # Get the relevant models for this provider
        provider_models = PROVIDERS[provider]

        # Create table header with models as columns
        header = "| Test | " + " | ".join(provider_models) + " |"
        separator = "| --- | " + " | ".join(["---"] * len(provider_models)) + " |"

        report.append(header)
        report.append(separator)

        # Get results for this provider
        provider_results = results_dict.get(provider, {})

        # Add rows for each test
        for test in sorted_tests:
            row = f"| {test} |"

            # Add results for each model in this test
            for model in provider_models:
                if model in provider_results and test in provider_results[model]:
                    result = pass_icon if provider_results[model][test] else fail_icon
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
        help="Specify providers to test (comma-separated or space-separated, default: all)",
    )
    parser.add_argument("--output", type=str, help="Output file location (default: tests/verifications/REPORT.md)")
    args = parser.parse_args()

    all_results = {}

    if args.run_tests:
        # Get list of available providers from command line or use detected providers
        if args.providers:
            # Handle both comma-separated and space-separated lists
            test_providers = []
            for provider_arg in args.providers:
                # Split by comma if commas are present
                if "," in provider_arg:
                    test_providers.extend(provider_arg.split(","))
                else:
                    test_providers.append(provider_arg)
        else:
            # Default providers to test
            test_providers = PROVIDER_ORDER

        for provider in test_providers:
            provider = provider.strip()  # Remove any whitespace
            result_file = run_tests(provider)
            if result_file:
                provider_results = parse_results(result_file)
                all_results.update(provider_results)
    else:
        # Use existing results
        provider_result_files = get_latest_results_by_provider()

        for result_file in provider_result_files.values():
            provider_results = parse_results(result_file)
            all_results.update(provider_results)

    # Generate the report
    generate_report(all_results, args.output)

    cleanup_old_results()


if __name__ == "__main__":
    main()
