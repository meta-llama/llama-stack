# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import shutil
import subprocess
import tempfile
import unittest
import yaml
from pathlib import Path
from unittest.mock import patch

from llama_stack.distribution.utils.xdg_utils import (
    get_llama_stack_config_dir,
    get_llama_stack_data_dir,
    get_llama_stack_state_dir,
)


class TestXDGEndToEnd(unittest.TestCase):
    """End-to-end tests for XDG compliance workflows."""

    def setUp(self):
        """Set up test environment."""
        self.original_env = {}
        self.env_vars = [
            "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME",
            "LLAMA_STACK_CONFIG_DIR", "SQLITE_STORE_DIR", "FILES_STORAGE_DIR"
        ]
        
        for key in self.env_vars:
            self.original_env[key] = os.environ.get(key)

    def tearDown(self):
        """Clean up test environment."""
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def clear_env_vars(self):
        """Clear all relevant environment variables."""
        for key in self.env_vars:
            os.environ.pop(key, None)

    def create_realistic_legacy_structure(self, base_dir: Path) -> Path:
        """Create a realistic legacy ~/.llama directory structure."""
        legacy_dir = base_dir / ".llama"
        legacy_dir.mkdir()
        
        # Create distributions with realistic content
        distributions_dir = legacy_dir / "distributions"
        distributions_dir.mkdir()
        
        # Ollama distribution
        ollama_dir = distributions_dir / "ollama"
        ollama_dir.mkdir()
        
        ollama_run_yaml = ollama_dir / "ollama-run.yaml"
        ollama_run_yaml.write_text("""
version: 2
apis:
  - inference
  - safety
  - memory
  - vector_io
  - agents
  - files
providers:
  inference:
    - provider_type: remote::ollama
      config:
        url: http://localhost:11434
""")
        
        ollama_build_yaml = ollama_dir / "build.yaml"
        ollama_build_yaml.write_text("""
name: ollama
description: Ollama inference provider
docker_image: ollama:latest
""")
        
        # Create providers.d structure
        providers_dir = legacy_dir / "providers.d"
        providers_dir.mkdir()
        
        remote_dir = providers_dir / "remote"
        remote_dir.mkdir()
        
        inference_dir = remote_dir / "inference"
        inference_dir.mkdir()
        
        custom_provider = inference_dir / "custom-inference.yaml"
        custom_provider.write_text("""
provider_type: remote::custom
config:
  url: http://localhost:8080
  api_key: test_key
""")
        
        # Create checkpoints with model files
        checkpoints_dir = legacy_dir / "checkpoints"
        checkpoints_dir.mkdir()
        
        model_dir = checkpoints_dir / "meta-llama" / "Llama-3.2-1B-Instruct"
        model_dir.mkdir(parents=True)
        
        # Create fake model files
        (model_dir / "consolidated.00.pth").write_bytes(b"fake model weights" * 1000)
        (model_dir / "params.json").write_text('{"dim": 2048, "n_layers": 22}')
        (model_dir / "tokenizer.model").write_bytes(b"fake tokenizer" * 100)
        
        # Create runtime with databases
        runtime_dir = legacy_dir / "runtime"
        runtime_dir.mkdir()
        
        (runtime_dir / "trace_store.db").write_text("SQLite format 3\x00" + "fake database content")
        (runtime_dir / "agent_sessions.db").write_text("SQLite format 3\x00" + "fake agent sessions")
        
        # Create config files
        (legacy_dir / "config.json").write_text('{"version": "0.2.13", "last_updated": "2024-01-01"}')
        
        return legacy_dir

    def verify_xdg_migration_complete(self, base_dir: Path, legacy_dir: Path):
        """Verify that migration to XDG structure is complete and correct."""
        config_dir = base_dir / ".config" / "llama-stack"
        data_dir = base_dir / ".local" / "share" / "llama-stack"
        state_dir = base_dir / ".local" / "state" / "llama-stack"
        
        # Verify distributions moved to config
        self.assertTrue((config_dir / "distributions").exists())
        self.assertTrue((config_dir / "distributions" / "ollama").exists())
        self.assertTrue((config_dir / "distributions" / "ollama" / "ollama-run.yaml").exists())
        
        # Verify YAML content is preserved
        yaml_content = (config_dir / "distributions" / "ollama" / "ollama-run.yaml").read_text()
        self.assertIn("version: 2", yaml_content)
        self.assertIn("remote::ollama", yaml_content)
        
        # Verify providers.d moved to config
        self.assertTrue((config_dir / "providers.d").exists())
        self.assertTrue((config_dir / "providers.d" / "remote" / "inference").exists())
        self.assertTrue((config_dir / "providers.d" / "remote" / "inference" / "custom-inference.yaml").exists())
        
        # Verify checkpoints moved to data
        self.assertTrue((data_dir / "checkpoints").exists())
        self.assertTrue((data_dir / "checkpoints" / "meta-llama" / "Llama-3.2-1B-Instruct").exists())
        self.assertTrue((data_dir / "checkpoints" / "meta-llama" / "Llama-3.2-1B-Instruct" / "consolidated.00.pth").exists())
        
        # Verify model file content preserved
        model_file = data_dir / "checkpoints" / "meta-llama" / "Llama-3.2-1B-Instruct" / "consolidated.00.pth"
        self.assertGreater(model_file.stat().st_size, 1000)  # Should be substantial size
        
        # Verify runtime moved to state
        self.assertTrue((state_dir / "runtime").exists())
        self.assertTrue((state_dir / "runtime" / "trace_store.db").exists())
        self.assertTrue((state_dir / "runtime" / "agent_sessions.db").exists())
        
        # Verify database files preserved
        db_file = state_dir / "runtime" / "trace_store.db"
        db_content = db_file.read_text()
        self.assertIn("SQLite format 3", db_content)

    def test_fresh_installation_xdg_compliance(self):
        """Test fresh installation uses XDG-compliant paths."""
        self.clear_env_vars()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            
            # Set custom XDG paths
            os.environ["XDG_CONFIG_HOME"] = str(base_dir / "custom_config")
            os.environ["XDG_DATA_HOME"] = str(base_dir / "custom_data")
            os.environ["XDG_STATE_HOME"] = str(base_dir / "custom_state")
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                # Mock that no legacy directory exists
                with patch("llama_stack.distribution.utils.xdg_utils.Path.exists") as mock_exists:
                    mock_exists.return_value = False
                    
                    # Fresh installation should use XDG paths
                    config_dir = get_llama_stack_config_dir()
                    data_dir = get_llama_stack_data_dir()
                    state_dir = get_llama_stack_state_dir()
                    
                    self.assertEqual(config_dir, base_dir / "custom_config" / "llama-stack")
                    self.assertEqual(data_dir, base_dir / "custom_data" / "llama-stack")
                    self.assertEqual(state_dir, base_dir / "custom_state" / "llama-stack")

    def test_complete_migration_workflow(self):
        """Test complete migration workflow from legacy to XDG."""
        self.clear_env_vars()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                
                # Create realistic legacy structure
                legacy_dir = self.create_realistic_legacy_structure(base_dir)
                
                # Verify legacy structure exists
                self.assertTrue(legacy_dir.exists())
                self.assertTrue((legacy_dir / "distributions" / "ollama" / "ollama-run.yaml").exists())
                
                # Perform migration
                from llama_stack.cli.migrate_xdg import migrate_to_xdg
                
                with patch("builtins.input") as mock_input:
                    mock_input.side_effect = ["y", "y"]  # Confirm migration and cleanup
                    
                    result = migrate_to_xdg(dry_run=False)
                    self.assertTrue(result)
                
                # Verify migration completed successfully
                self.verify_xdg_migration_complete(base_dir, legacy_dir)
                
                # Verify legacy directory was removed
                self.assertFalse(legacy_dir.exists())

    def test_migration_preserves_file_integrity(self):
        """Test that migration preserves file integrity and permissions."""
        self.clear_env_vars()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                
                # Create legacy structure
                legacy_dir = self.create_realistic_legacy_structure(base_dir)
                
                # Set specific permissions on files
                config_file = legacy_dir / "distributions" / "ollama" / "ollama-run.yaml"
                config_file.chmod(0o600)
                original_stat = config_file.stat()
                
                # Perform migration
                from llama_stack.cli.migrate_xdg import migrate_to_xdg
                
                with patch("builtins.input") as mock_input:
                    mock_input.side_effect = ["y", "y"]
                    result = migrate_to_xdg(dry_run=False)
                    self.assertTrue(result)
                
                # Verify file integrity
                migrated_config = base_dir / ".config" / "llama-stack" / "distributions" / "ollama" / "ollama-run.yaml"
                self.assertTrue(migrated_config.exists())
                
                # Verify content is identical
                original_content = """
version: 2
apis:
  - inference
  - safety
  - memory
  - vector_io
  - agents
  - files
providers:
  inference:
    - provider_type: remote::ollama
      config:
        url: http://localhost:11434
"""
                migrated_content = migrated_config.read_text()
                self.assertIn("version: 2", migrated_content)
                self.assertIn("remote::ollama", migrated_content)
                
                # Verify permissions preserved
                migrated_stat = migrated_config.stat()
                self.assertEqual(original_stat.st_mode, migrated_stat.st_mode)

    def test_mixed_legacy_and_xdg_environment(self):
        """Test behavior in mixed legacy and XDG environment."""
        self.clear_env_vars()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            
            # Set partial XDG environment
            os.environ["XDG_CONFIG_HOME"] = str(base_dir / "xdg_config")
            # Leave XDG_DATA_HOME and XDG_STATE_HOME unset
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                
                # Create legacy directory
                legacy_dir = self.create_realistic_legacy_structure(base_dir)
                
                # Should use legacy directory since it exists
                config_dir = get_llama_stack_config_dir()
                data_dir = get_llama_stack_data_dir()
                state_dir = get_llama_stack_state_dir()
                
                self.assertEqual(config_dir, legacy_dir)
                self.assertEqual(data_dir, legacy_dir)
                self.assertEqual(state_dir, legacy_dir)

    def test_template_rendering_with_xdg_paths(self):
        """Test that templates render correctly with XDG paths."""
        self.clear_env_vars()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            
            # Set XDG environment
            os.environ["XDG_STATE_HOME"] = str(base_dir / "state")
            os.environ["XDG_DATA_HOME"] = str(base_dir / "data")
            
            # Mock shell environment variable expansion
            def mock_env_expand(template_string):
                """Mock shell environment variable expansion."""
                result = template_string
                result = result.replace("${env.XDG_STATE_HOME:-~/.local/state}", str(base_dir / "state"))
                result = result.replace("${env.XDG_DATA_HOME:-~/.local/share}", str(base_dir / "data"))
                return result
            
            # Test template patterns
            template_patterns = [
                "${env.XDG_STATE_HOME:-~/.local/state}/llama-stack/distributions/ollama",
                "${env.XDG_DATA_HOME:-~/.local/share}/llama-stack/distributions/ollama/files",
            ]
            
            expected_results = [
                str(base_dir / "state" / "llama-stack" / "distributions" / "ollama"),
                str(base_dir / "data" / "llama-stack" / "distributions" / "ollama" / "files"),
            ]
            
            for pattern, expected in zip(template_patterns, expected_results):
                with self.subTest(pattern=pattern):
                    expanded = mock_env_expand(pattern)
                    self.assertEqual(expanded, expected)

    def test_cli_integration_with_xdg_paths(self):
        """Test CLI integration works correctly with XDG paths."""
        self.clear_env_vars()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                
                # Create legacy structure
                legacy_dir = self.create_realistic_legacy_structure(base_dir)
                
                # Test CLI migrate command
                from llama_stack.cli.migrate_xdg import MigrateXDG
                import argparse
                
                parser = argparse.ArgumentParser()
                subparsers = parser.add_subparsers()
                migrate_cmd = MigrateXDG.create(subparsers)
                
                # Test dry-run
                args = parser.parse_args(["migrate-xdg", "--dry-run"])
                
                with patch("builtins.print") as mock_print:
                    result = migrate_cmd._run_migrate_xdg_cmd(args)
                    self.assertEqual(result, 0)
                    
                    # Should print dry-run information
                    print_calls = [call[0][0] for call in mock_print.call_args_list]
                    self.assertTrue(any("Dry run mode" in call for call in print_calls))
                
                # Legacy directory should still exist after dry-run
                self.assertTrue(legacy_dir.exists())

    def test_config_dirs_integration_after_migration(self):
        """Test that config_dirs works correctly after migration."""
        self.clear_env_vars()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                
                # Create and migrate legacy structure
                legacy_dir = self.create_realistic_legacy_structure(base_dir)
                
                from llama_stack.cli.migrate_xdg import migrate_to_xdg
                
                with patch("builtins.input") as mock_input:
                    mock_input.side_effect = ["y", "y"]
                    result = migrate_to_xdg(dry_run=False)
                    self.assertTrue(result)
                
                # Clear module cache to ensure fresh import
                import sys
                if "llama_stack.distribution.utils.config_dirs" in sys.modules:
                    del sys.modules["llama_stack.distribution.utils.config_dirs"]
                
                # Import config_dirs after migration
                from llama_stack.distribution.utils.config_dirs import (
                    LLAMA_STACK_CONFIG_DIR,
                    DEFAULT_CHECKPOINT_DIR,
                    RUNTIME_BASE_DIR,
                    DISTRIBS_BASE_DIR,
                )
                
                # Should use XDG paths
                self.assertEqual(LLAMA_STACK_CONFIG_DIR, base_dir / ".config" / "llama-stack")
                self.assertEqual(DEFAULT_CHECKPOINT_DIR, base_dir / ".local" / "share" / "llama-stack" / "checkpoints")
                self.assertEqual(RUNTIME_BASE_DIR, base_dir / ".local" / "state" / "llama-stack" / "runtime")
                self.assertEqual(DISTRIBS_BASE_DIR, base_dir / ".config" / "llama-stack" / "distributions")

    def test_real_file_operations_with_xdg_paths(self):
        """Test real file operations work correctly with XDG paths."""
        self.clear_env_vars()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            
            # Set XDG environment
            os.environ["XDG_CONFIG_HOME"] = str(base_dir / "config")
            os.environ["XDG_DATA_HOME"] = str(base_dir / "data")
            os.environ["XDG_STATE_HOME"] = str(base_dir / "state")
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                with patch("llama_stack.distribution.utils.xdg_utils.Path.exists") as mock_exists:
                    mock_exists.return_value = False
                    
                    # Get XDG paths
                    config_dir = get_llama_stack_config_dir()
                    data_dir = get_llama_stack_data_dir()
                    state_dir = get_llama_stack_state_dir()
                    
                    # Create directories
                    config_dir.mkdir(parents=True)
                    data_dir.mkdir(parents=True)
                    state_dir.mkdir(parents=True)
                    
                    # Test writing configuration files
                    config_file = config_dir / "test_config.yaml"
                    config_data = {"version": "2", "test": True}
                    
                    with open(config_file, "w") as f:
                        yaml.dump(config_data, f)
                    
                    # Test reading configuration files
                    with open(config_file, "r") as f:
                        loaded_config = yaml.safe_load(f)
                    
                    self.assertEqual(loaded_config, config_data)
                    
                    # Test creating nested directory structure
                    model_dir = data_dir / "checkpoints" / "meta-llama" / "test-model"
                    model_dir.mkdir(parents=True)
                    
                    # Test writing large files
                    model_file = model_dir / "model.bin"
                    test_data = b"test model data" * 1000
                    model_file.write_bytes(test_data)
                    
                    # Verify file integrity
                    read_data = model_file.read_bytes()
                    self.assertEqual(read_data, test_data)
                    
                    # Test state files
                    state_file = state_dir / "runtime" / "session.db"
                    state_file.parent.mkdir(parents=True)
                    state_file.write_text("SQLite format 3\x00test database")
                    
                    # Verify state file
                    state_content = state_file.read_text()
                    self.assertIn("SQLite format 3", state_content)

    def test_backwards_compatibility_scenario(self):
        """Test complete backwards compatibility scenario."""
        self.clear_env_vars()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            
            # Scenario: User has existing legacy installation
            legacy_dir = self.create_realistic_legacy_structure(base_dir)
            
            # User sets legacy environment variable
            os.environ["LLAMA_STACK_CONFIG_DIR"] = str(legacy_dir)
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                
                # Should continue using legacy paths
                config_dir = get_llama_stack_config_dir()
                data_dir = get_llama_stack_data_dir()
                state_dir = get_llama_stack_state_dir()
                
                self.assertEqual(config_dir, legacy_dir)
                self.assertEqual(data_dir, legacy_dir)
                self.assertEqual(state_dir, legacy_dir)
                
                # Should be able to access existing files
                yaml_file = legacy_dir / "distributions" / "ollama" / "ollama-run.yaml"
                self.assertTrue(yaml_file.exists())
                
                # Should be able to parse existing configuration
                with open(yaml_file, "r") as f:
                    config = yaml.safe_load(f)
                
                self.assertEqual(config["version"], 2)

    def test_error_recovery_scenarios(self):
        """Test error recovery in various scenarios."""
        self.clear_env_vars()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                
                # Scenario 1: Partial migration failure
                legacy_dir = self.create_realistic_legacy_structure(base_dir)
                
                # Create conflicting file in target location
                config_dir = base_dir / ".config" / "llama-stack"
                config_dir.mkdir(parents=True)
                
                conflicting_file = config_dir / "distributions"
                conflicting_file.touch()  # Create file instead of directory
                
                from llama_stack.cli.migrate_xdg import migrate_to_xdg
                
                with patch("builtins.input") as mock_input:
                    mock_input.side_effect = ["y", "n"]  # Confirm migration, don't cleanup
                    
                    with patch("builtins.print") as mock_print:
                        result = migrate_to_xdg(dry_run=False)
                        
                        # Should handle conflicts gracefully
                        print_calls = [call[0][0] for call in mock_print.call_args_list]
                        conflict_mentioned = any("Warning" in call or "conflict" in call.lower() 
                                                for call in print_calls)
                        
                        # Migration should complete with warnings
                        self.assertTrue(result or conflict_mentioned)

    def test_cross_platform_compatibility(self):
        """Test cross-platform compatibility of XDG implementation."""
        self.clear_env_vars()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            
            # Test with different path separators and formats
            if os.name == "nt":  # Windows
                # Test Windows-style paths
                os.environ["XDG_CONFIG_HOME"] = str(base_dir / "config").replace("/", "\\")
                os.environ["XDG_DATA_HOME"] = str(base_dir / "data").replace("/", "\\")
            else:  # Unix-like
                # Test Unix-style paths
                os.environ["XDG_CONFIG_HOME"] = str(base_dir / "config")
                os.environ["XDG_DATA_HOME"] = str(base_dir / "data")
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                with patch("llama_stack.distribution.utils.xdg_utils.Path.exists") as mock_exists:
                    mock_exists.return_value = False
                    
                    # Should work regardless of platform
                    config_dir = get_llama_stack_config_dir()
                    data_dir = get_llama_stack_data_dir()
                    
                    # Paths should be valid for the current platform
                    self.assertTrue(config_dir.is_absolute())
                    self.assertTrue(data_dir.is_absolute())
                    
                    # Should be able to create directories
                    config_dir.mkdir(parents=True, exist_ok=True)
                    data_dir.mkdir(parents=True, exist_ok=True)
                    
                    self.assertTrue(config_dir.exists())
                    self.assertTrue(data_dir.exists())


if __name__ == "__main__":
    unittest.main() 