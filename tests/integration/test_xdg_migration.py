# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import shutil
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from llama_stack.cli.migrate_xdg import migrate_to_xdg
from llama_stack.distribution.utils.xdg_utils import (
    get_llama_stack_config_dir,
    get_llama_stack_data_dir,
    get_llama_stack_state_dir,
)


class TestXDGMigrationIntegration(unittest.TestCase):
    """Integration tests for XDG migration functionality."""

    def setUp(self):
        """Set up test environment."""
        # Store original environment variables
        self.original_env = {}
        for key in ["XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "LLAMA_STACK_CONFIG_DIR"]:
            self.original_env[key] = os.environ.get(key)
        
        # Clear environment variables
        for key in self.original_env:
            os.environ.pop(key, None)

    def tearDown(self):
        """Clean up test environment."""
        # Restore original environment variables
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def create_legacy_structure(self, base_dir: Path) -> Path:
        """Create a realistic legacy ~/.llama directory structure."""
        legacy_dir = base_dir / ".llama"
        legacy_dir.mkdir()
        
        # Create distributions
        distributions_dir = legacy_dir / "distributions"
        distributions_dir.mkdir()
        
        # Create sample distribution
        ollama_dir = distributions_dir / "ollama"
        ollama_dir.mkdir()
        (ollama_dir / "ollama-run.yaml").write_text("version: 2\napis: []\n")
        (ollama_dir / "build.yaml").write_text("name: ollama\n")
        
        # Create providers.d
        providers_dir = legacy_dir / "providers.d"
        providers_dir.mkdir()
        (providers_dir / "remote").mkdir()
        (providers_dir / "remote" / "inference").mkdir()
        (providers_dir / "remote" / "inference" / "custom.yaml").write_text("provider_type: remote::custom\n")
        
        # Create checkpoints
        checkpoints_dir = legacy_dir / "checkpoints"
        checkpoints_dir.mkdir()
        model_dir = checkpoints_dir / "meta-llama" / "Llama-3.2-1B-Instruct"
        model_dir.mkdir(parents=True)
        (model_dir / "consolidated.00.pth").write_text("fake model weights")
        (model_dir / "params.json").write_text('{"dim": 2048}')
        
        # Create runtime
        runtime_dir = legacy_dir / "runtime"
        runtime_dir.mkdir()
        (runtime_dir / "trace_store.db").write_text("fake sqlite database")
        
        # Create some fake files in various subdirectories
        (legacy_dir / "config.json").write_text('{"version": "0.2.13"}')
        
        return legacy_dir

    def verify_xdg_structure(self, base_dir: Path, legacy_dir: Path):
        """Verify that the XDG structure was created correctly."""
        config_dir = base_dir / ".config" / "llama-stack"
        data_dir = base_dir / ".local" / "share" / "llama-stack"
        state_dir = base_dir / ".local" / "state" / "llama-stack"
        
        # Verify distributions moved to config
        self.assertTrue((config_dir / "distributions").exists())
        self.assertTrue((config_dir / "distributions" / "ollama" / "ollama-run.yaml").exists())
        
        # Verify providers.d moved to config
        self.assertTrue((config_dir / "providers.d").exists())
        self.assertTrue((config_dir / "providers.d" / "remote" / "inference" / "custom.yaml").exists())
        
        # Verify checkpoints moved to data
        self.assertTrue((data_dir / "checkpoints").exists())
        self.assertTrue((data_dir / "checkpoints" / "meta-llama" / "Llama-3.2-1B-Instruct" / "consolidated.00.pth").exists())
        
        # Verify runtime moved to state
        self.assertTrue((state_dir / "runtime").exists())
        self.assertTrue((state_dir / "runtime" / "trace_store.db").exists())

    def test_full_migration_workflow(self):
        """Test complete migration workflow from legacy to XDG."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            
            # Set up fake home directory
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                
                # Create legacy structure
                legacy_dir = self.create_legacy_structure(base_dir)
                
                # Verify legacy structure exists
                self.assertTrue(legacy_dir.exists())
                self.assertTrue((legacy_dir / "distributions").exists())
                self.assertTrue((legacy_dir / "checkpoints").exists())
                
                # Perform migration with user confirmation
                with patch("builtins.input") as mock_input:
                    mock_input.side_effect = ["y", "y"]  # Confirm migration and cleanup
                    
                    result = migrate_to_xdg(dry_run=False)
                    self.assertTrue(result)
                
                # Verify XDG structure was created
                self.verify_xdg_structure(base_dir, legacy_dir)
                
                # Verify legacy directory was removed
                self.assertFalse(legacy_dir.exists())

    def test_migration_dry_run(self):
        """Test dry run migration (no actual file movement)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                
                # Create legacy structure
                legacy_dir = self.create_legacy_structure(base_dir)
                
                # Perform dry run migration
                with patch("builtins.print") as mock_print:
                    result = migrate_to_xdg(dry_run=True)
                    self.assertTrue(result)
                    
                    # Check that dry run message was printed
                    print_calls = [call[0][0] for call in mock_print.call_args_list]
                    self.assertTrue(any("Dry run mode" in call for call in print_calls))
                
                # Verify nothing was actually moved
                self.assertTrue(legacy_dir.exists())
                self.assertTrue((legacy_dir / "distributions").exists())
                self.assertFalse((base_dir / ".config" / "llama-stack").exists())

    def test_migration_user_cancellation(self):
        """Test migration when user cancels."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                
                # Create legacy structure
                legacy_dir = self.create_legacy_structure(base_dir)
                
                # User cancels migration
                with patch("builtins.input") as mock_input:
                    mock_input.return_value = "n"
                    
                    result = migrate_to_xdg(dry_run=False)
                    self.assertFalse(result)
                
                # Verify nothing was moved
                self.assertTrue(legacy_dir.exists())
                self.assertTrue((legacy_dir / "distributions").exists())
                self.assertFalse((base_dir / ".config" / "llama-stack").exists())

    def test_migration_with_existing_xdg_directories(self):
        """Test migration when XDG directories already exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                
                # Create legacy structure
                legacy_dir = self.create_legacy_structure(base_dir)
                
                # Create existing XDG structure with conflicting files
                config_dir = base_dir / ".config" / "llama-stack"
                config_dir.mkdir(parents=True)
                (config_dir / "distributions").mkdir()
                (config_dir / "distributions" / "existing.yaml").write_text("existing config")
                
                # Perform migration
                with patch("builtins.input") as mock_input:
                    mock_input.side_effect = ["y", "n"]  # Confirm migration, don't cleanup
                    with patch("builtins.print") as mock_print:
                        result = migrate_to_xdg(dry_run=False)
                        self.assertTrue(result)
                        
                        # Check that warning was printed
                        print_calls = [call[0][0] for call in mock_print.call_args_list]
                        self.assertTrue(any("Warning: Target already exists" in call for call in print_calls))
                
                # Verify existing file wasn't overwritten
                self.assertTrue((config_dir / "distributions" / "existing.yaml").exists())
                
                # Legacy distributions should still exist due to conflict
                self.assertTrue((legacy_dir / "distributions").exists())

    def test_migration_partial_success(self):
        """Test migration when some items succeed and others fail."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                
                # Create legacy structure
                legacy_dir = self.create_legacy_structure(base_dir)
                
                # Create readonly target directory to simulate permission error
                config_dir = base_dir / ".config" / "llama-stack"
                config_dir.mkdir(parents=True)
                distributions_target = config_dir / "distributions"
                distributions_target.mkdir()
                distributions_target.chmod(0o444)  # Read-only
                
                try:
                    # Perform migration
                    with patch("builtins.input") as mock_input:
                        mock_input.side_effect = ["y", "n"]  # Confirm migration, don't cleanup
                        
                        result = migrate_to_xdg(dry_run=False)
                        # Should return True even with partial success
                        
                        # Some items should have been migrated successfully
                        self.assertTrue((base_dir / ".local" / "share" / "llama-stack" / "checkpoints").exists())
                        
                finally:
                    # Restore permissions for cleanup
                    distributions_target.chmod(0o755)

    def test_migration_empty_legacy_directory(self):
        """Test migration when legacy directory exists but is empty."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                
                # Create empty legacy directory
                legacy_dir = base_dir / ".llama"
                legacy_dir.mkdir()
                
                result = migrate_to_xdg(dry_run=False)
                self.assertTrue(result)

    def test_migration_preserves_file_permissions(self):
        """Test that migration preserves file permissions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                
                # Create legacy structure with specific permissions
                legacy_dir = self.create_legacy_structure(base_dir)
                
                # Set specific permissions on a file
                config_file = legacy_dir / "distributions" / "ollama" / "ollama-run.yaml"
                config_file.chmod(0o600)
                original_stat = config_file.stat()
                
                # Perform migration
                with patch("builtins.input") as mock_input:
                    mock_input.side_effect = ["y", "y"]
                    result = migrate_to_xdg(dry_run=False)
                    self.assertTrue(result)
                
                # Verify permissions were preserved
                migrated_file = base_dir / ".config" / "llama-stack" / "distributions" / "ollama" / "ollama-run.yaml"
                self.assertTrue(migrated_file.exists())
                migrated_stat = migrated_file.stat()
                self.assertEqual(original_stat.st_mode, migrated_stat.st_mode)

    def test_migration_preserves_directory_structure(self):
        """Test that migration preserves complex directory structures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                
                # Create complex legacy structure
                legacy_dir = base_dir / ".llama"
                legacy_dir.mkdir()
                
                # Create nested structure
                complex_path = legacy_dir / "checkpoints" / "org" / "model" / "variant" / "files"
                complex_path.mkdir(parents=True)
                (complex_path / "model.bin").write_text("model data")
                (complex_path / "config.json").write_text("config data")
                
                # Perform migration
                with patch("builtins.input") as mock_input:
                    mock_input.side_effect = ["y", "y"]
                    result = migrate_to_xdg(dry_run=False)
                    self.assertTrue(result)
                
                # Verify structure was preserved
                migrated_path = base_dir / ".local" / "share" / "llama-stack" / "checkpoints" / "org" / "model" / "variant" / "files"
                self.assertTrue(migrated_path.exists())
                self.assertTrue((migrated_path / "model.bin").exists())
                self.assertTrue((migrated_path / "config.json").exists())

    def test_migration_with_symlinks(self):
        """Test migration with symbolic links in legacy directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                
                # Create legacy structure
                legacy_dir = self.create_legacy_structure(base_dir)
                
                # Create a symlink
                actual_file = legacy_dir / "actual_config.yaml"
                actual_file.write_text("actual config content")
                
                symlink_file = legacy_dir / "distributions" / "symlinked_config.yaml"
                symlink_file.symlink_to(actual_file)
                
                # Perform migration
                with patch("builtins.input") as mock_input:
                    mock_input.side_effect = ["y", "y"]
                    result = migrate_to_xdg(dry_run=False)
                    self.assertTrue(result)
                
                # Verify symlink was preserved
                migrated_symlink = base_dir / ".config" / "llama-stack" / "distributions" / "symlinked_config.yaml"
                self.assertTrue(migrated_symlink.exists())
                self.assertTrue(migrated_symlink.is_symlink())

    def test_migration_large_files(self):
        """Test migration with large files (simulated)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                
                # Create legacy structure
                legacy_dir = self.create_legacy_structure(base_dir)
                
                # Create a larger file (1MB)
                large_file = legacy_dir / "checkpoints" / "large_model.bin"
                large_file.write_bytes(b"0" * (1024 * 1024))
                
                # Perform migration
                with patch("builtins.input") as mock_input:
                    mock_input.side_effect = ["y", "y"]
                    result = migrate_to_xdg(dry_run=False)
                    self.assertTrue(result)
                
                # Verify large file was moved correctly
                migrated_file = base_dir / ".local" / "share" / "llama-stack" / "checkpoints" / "large_model.bin"
                self.assertTrue(migrated_file.exists())
                self.assertEqual(migrated_file.stat().st_size, 1024 * 1024)

    def test_migration_with_unicode_filenames(self):
        """Test migration with unicode filenames."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                
                # Create legacy structure with unicode filenames
                legacy_dir = base_dir / ".llama"
                legacy_dir.mkdir()
                
                unicode_dir = legacy_dir / "distributions" / "配置"
                unicode_dir.mkdir(parents=True)
                unicode_file = unicode_dir / "模型.yaml"
                unicode_file.write_text("unicode content")
                
                # Perform migration
                with patch("builtins.input") as mock_input:
                    mock_input.side_effect = ["y", "y"]
                    result = migrate_to_xdg(dry_run=False)
                    self.assertTrue(result)
                
                # Verify unicode files were migrated
                migrated_file = base_dir / ".config" / "llama-stack" / "distributions" / "配置" / "模型.yaml"
                self.assertTrue(migrated_file.exists())
                self.assertEqual(migrated_file.read_text(), "unicode content")


class TestXDGMigrationCLI(unittest.TestCase):
    """Test the CLI interface for XDG migration."""

    def setUp(self):
        """Set up test environment."""
        self.original_env = {}
        for key in ["XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "LLAMA_STACK_CONFIG_DIR"]:
            self.original_env[key] = os.environ.get(key)
            os.environ.pop(key, None)

    def tearDown(self):
        """Clean up test environment."""
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_cli_migrate_command_exists(self):
        """Test that the migrate-xdg CLI command is properly registered."""
        from llama_stack.cli.llama import LlamaCLIParser
        
        parser = LlamaCLIParser()
        
        # Parse help to see if migrate-xdg is listed
        with patch("sys.argv", ["llama", "--help"]):
            with patch("sys.exit"):
                with patch("builtins.print") as mock_print:
                    try:
                        parser.parse_args()
                    except SystemExit:
                        pass
                    
                    # Check if migrate-xdg appears in help output
                    help_output = "\n".join([call[0][0] for call in mock_print.call_args_list])
                    self.assertIn("migrate-xdg", help_output)

    def test_cli_migrate_dry_run(self):
        """Test CLI migrate command with dry-run flag."""
        from llama_stack.cli.migrate_xdg import MigrateXDG
        import argparse
        
        # Create parser and add migrate command
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        migrate_cmd = MigrateXDG.create(subparsers)
        
        # Test dry-run flag
        args = parser.parse_args(["migrate-xdg", "--dry-run"])
        self.assertTrue(args.dry_run)
        
        # Test without dry-run flag
        args = parser.parse_args(["migrate-xdg"])
        self.assertFalse(args.dry_run)

    def test_cli_migrate_execution(self):
        """Test CLI migrate command execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                
                # Create legacy directory
                legacy_dir = base_dir / ".llama"
                legacy_dir.mkdir()
                (legacy_dir / "test_file").touch()
                
                from llama_stack.cli.migrate_xdg import MigrateXDG
                import argparse
                
                # Create parser and command
                parser = argparse.ArgumentParser()
                subparsers = parser.add_subparsers()
                migrate_cmd = MigrateXDG.create(subparsers)
                
                # Parse arguments
                args = parser.parse_args(["migrate-xdg", "--dry-run"])
                
                # Execute command
                with patch("builtins.print") as mock_print:
                    migrate_cmd._run_migrate_xdg_cmd(args)
                    
                    # Verify output was printed
                    print_calls = [call[0][0] for call in mock_print.call_args_list]
                    self.assertTrue(any("Found legacy directory" in call for call in print_calls))


if __name__ == "__main__":
    unittest.main() 