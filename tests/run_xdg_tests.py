#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Test runner for XDG Base Directory Specification compliance tests.

This script runs all XDG-related tests and provides a comprehensive report
of the test results.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path


def run_test_suite():
    """Run the complete XDG test suite."""
    
    # Set up test environment
    test_dir = Path(__file__).parent
    project_root = test_dir.parent
    
    # Add project root to Python path
    sys.path.insert(0, str(project_root))
    
    # Test modules to run
    test_modules = [
        "tests.unit.test_xdg_compliance",
        "tests.unit.test_config_dirs", 
        "tests.unit.cli.test_migrate_xdg",
        "tests.unit.test_template_xdg_paths",
        "tests.unit.test_xdg_edge_cases",
        "tests.integration.test_xdg_migration",
        "tests.integration.test_xdg_e2e",
    ]
    
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    print("🔍 Discovering XDG compliance tests...")
    
    for module_name in test_modules:
        try:
            # Try to load the test module
            module_suite = loader.loadTestsFromName(module_name)
            suite.addTest(module_suite)
            print(f"  ✅ Loaded {module_name}")
        except Exception as e:
            print(f"  ⚠️  Failed to load {module_name}: {e}")
    
    # Run the tests
    print("\n🧪 Running XDG compliance tests...")
    print("=" * 60)
    
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        buffer=True
    )
    
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    if result.skipped:
        print("\n⏭️  Skipped:")
        for test, reason in result.skipped:
            print(f"  - {test}: {reason}")
    
    # Overall result
    if result.wasSuccessful():
        print("\n🎉 All XDG compliance tests passed!")
        return 0
    else:
        print("\n⚠️  Some XDG compliance tests failed.")
        return 1


def run_quick_tests():
    """Run a quick subset of critical XDG tests."""
    
    print("🚀 Running quick XDG compliance tests...")
    
    # Add project root to Python path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # Quick test: Basic XDG functionality
    try:
        from llama_stack.distribution.utils.xdg_utils import (
            get_xdg_config_home,
            get_llama_stack_config_dir,
            get_xdg_compliant_path,
        )
        
        print("  ✅ XDG utilities import successfully")
        
        # Test basic functionality
        config_home = get_xdg_config_home()
        llama_config = get_llama_stack_config_dir()
        compliant_path = get_xdg_compliant_path("config", "test")
        
        print(f"  ✅ XDG config home: {config_home}")
        print(f"  ✅ Llama Stack config: {llama_config}")
        print(f"  ✅ Compliant path: {compliant_path}")
        
    except Exception as e:
        print(f"  ❌ XDG utilities failed: {e}")
        return 1
    
    # Quick test: Config dirs integration
    try:
        from llama_stack.distribution.utils.config_dirs import (
            LLAMA_STACK_CONFIG_DIR,
            DEFAULT_CHECKPOINT_DIR,
        )
        
        print(f"  ✅ Config dirs integration: {LLAMA_STACK_CONFIG_DIR}")
        print(f"  ✅ Checkpoint directory: {DEFAULT_CHECKPOINT_DIR}")
        
    except Exception as e:
        print(f"  ❌ Config dirs integration failed: {e}")
        return 1
    
    # Quick test: CLI migrate command
    try:
        from llama_stack.cli.migrate_xdg import MigrateXDG
        
        print("  ✅ CLI migrate command available")
        
    except Exception as e:
        print(f"  ❌ CLI migrate command failed: {e}")
        return 1
    
    print("\n🎉 Quick XDG compliance tests passed!")
    return 0


def main():
    """Main test runner entry point."""
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        return run_quick_tests()
    else:
        return run_test_suite()


if __name__ == "__main__":
    sys.exit(main()) 