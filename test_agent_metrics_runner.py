#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Test runner for agent metrics tests.

This script provides a convenient way to run different categories of agent metrics tests.

Usage:
    python test_agent_metrics_runner.py [unit|integration|performance|all]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_tests(test_type: str, verbose: bool = False) -> int:
    """Run agent metrics tests of the specified type"""

    # Base pytest command
    cmd = ["python", "-m", "pytest"]

    if verbose:
        cmd.append("-v")

    # Add test paths based on type
    if test_type == "unit":
        cmd.extend(
            [
                "tests/unit/providers/agents/test_agent_metrics.py",
                "tests/unit/providers/telemetry/test_agent_metrics_histogram.py",
            ]
        )
        cmd.extend(["-m", "agent_metrics_unit"])

    elif test_type == "integration":
        cmd.extend(
            [
                "tests/integration/agents/test_agent_metrics_integration.py",
            ]
        )
        cmd.extend(["-m", "agent_metrics_integration"])

    elif test_type == "performance":
        cmd.extend(
            [
                "tests/performance/test_agent_metrics_performance.py",
            ]
        )
        cmd.extend(["-m", "agent_metrics_performance"])

    elif test_type == "all":
        cmd.extend(
            [
                "tests/unit/providers/agents/test_agent_metrics.py",
                "tests/unit/providers/telemetry/test_agent_metrics_histogram.py",
                "tests/integration/agents/test_agent_metrics_integration.py",
                "tests/performance/test_agent_metrics_performance.py",
            ]
        )
        cmd.extend(["-m", "agent_metrics"])

    else:
        print(f"Unknown test type: {test_type}")
        return 1

    # Add additional pytest options
    cmd.extend(
        [
            "--tb=short",  # Shorter traceback format
            "--disable-warnings",  # Reduce noise
        ]
    )

    print(f"Running {test_type} tests...")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except FileNotFoundError:
        print("Error: pytest not found. Please install pytest.")
        return 1
    except KeyboardInterrupt:
        print("\nTests interrupted by user.")
        return 1


def validate_test_setup() -> bool:
    """Validate that test files exist and are valid"""

    test_files = [
        "tests/unit/providers/agents/test_agent_metrics.py",
        "tests/unit/providers/telemetry/test_agent_metrics_histogram.py",
        "tests/integration/agents/test_agent_metrics_integration.py",
        "tests/performance/test_agent_metrics_performance.py",
    ]

    missing_files = []
    for test_file in test_files:
        if not Path(test_file).exists():
            missing_files.append(test_file)

    if missing_files:
        print("Error: Missing test files:")
        for file in missing_files:
            print(f"  - {file}")
        return False

    print("✓ All test files found")

    # Try to compile test files
    import py_compile

    for test_file in test_files:
        try:
            py_compile.compile(test_file, doraise=True)
        except py_compile.PyCompileError as e:
            print(f"Error: Syntax error in {test_file}: {e}")
            return False

    print("✓ All test files compile successfully")
    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run agent metrics tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test types:
  unit         Run unit tests only (fast)
  integration  Run integration tests (slower)
  performance  Run performance tests (slowest)
  all          Run all agent metrics tests

Examples:
  python test_agent_metrics_runner.py unit -v
  python test_agent_metrics_runner.py all
  python test_agent_metrics_runner.py performance
        """,
    )

    parser.add_argument(
        "test_type",
        choices=["unit", "integration", "performance", "all"],
        nargs="?",
        default="unit",
        help="Type of tests to run (default: unit)",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    parser.add_argument("--validate", action="store_true", help="Validate test setup without running tests")

    args = parser.parse_args()

    # Validate test setup
    if not validate_test_setup():
        return 1

    if args.validate:
        print("✓ Test setup validation passed")
        return 0

    # Run tests
    return run_tests(args.test_type, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
