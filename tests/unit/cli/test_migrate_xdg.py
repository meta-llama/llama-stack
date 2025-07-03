# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import os
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock

from llama_stack.cli.migrate_xdg import MigrateXDG, migrate_to_xdg


class TestMigrateXDGCLI(unittest.TestCase):
    """Test the MigrateXDG CLI command."""

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

    def create_parser_with_migrate_cmd(self):
        """Create parser with migrate-xdg command."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        migrate_cmd = MigrateXDG.create(subparsers)
        return parser, migrate_cmd

    def test_migrate_xdg_command_creation(self):
        """Test that MigrateXDG command can be created."""
        parser, migrate_cmd = self.create_parser_with_migrate_cmd()
        
        self.assertIsInstance(migrate_cmd, MigrateXDG)
        self.assertEqual(migrate_cmd.parser.prog, "llama migrate-xdg")
        self.assertEqual(migrate_cmd.parser.description, "Migrate from legacy ~/.llama to XDG-compliant directories")

    def test_migrate_xdg_argument_parsing(self):
        """Test argument parsing for migrate-xdg command."""
        parser, _ = self.create_parser_with_migrate_cmd()
        
        # Test with dry-run flag
        args = parser.parse_args(["migrate-xdg", "--dry-run"])
        self.assertEqual(args.command, "migrate-xdg")
        self.assertTrue(args.dry_run)
        
        # Test without dry-run flag
        args = parser.parse_args(["migrate-xdg"])
        self.assertEqual(args.command, "migrate-xdg")
        self.assertFalse(args.dry_run)

    def test_migrate_xdg_help_text(self):
        """Test help text for migrate-xdg command."""
        parser, _ = self.create_parser_with_migrate_cmd()
        
        # Capture help output
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            with patch("sys.exit"):
                try:
                    parser.parse_args(["migrate-xdg", "--help"])
                except SystemExit:
                    pass
                
                help_text = mock_stdout.getvalue()
                self.assertIn("migrate-xdg", help_text)
                self.assertIn("XDG-compliant directories", help_text)
                self.assertIn("--dry-run", help_text)

    def test_migrate_xdg_command_execution_no_legacy(self):
        """Test command execution when no legacy directory exists."""
        parser, migrate_cmd = self.create_parser_with_migrate_cmd()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = Path(temp_dir)
                
                args = parser.parse_args(["migrate-xdg"])
                
                with patch("builtins.print") as mock_print:
                    result = migrate_cmd._run_migrate_xdg_cmd(args)
                    
                    # Should succeed when no migration needed
                    self.assertEqual(result, 0)
                    
                    # Should print appropriate message
                    print_calls = [call[0][0] for call in mock_print.call_args_list]
                    self.assertTrue(any("No legacy directory found" in call for call in print_calls))

    def test_migrate_xdg_command_execution_with_legacy(self):
        """Test command execution when legacy directory exists."""
        parser, migrate_cmd = self.create_parser_with_migrate_cmd()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            legacy_dir = base_dir / ".llama"
            legacy_dir.mkdir()
            (legacy_dir / "test_file").touch()
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                
                args = parser.parse_args(["migrate-xdg"])
                
                with patch("builtins.print") as mock_print:
                    result = migrate_cmd._run_migrate_xdg_cmd(args)
                    
                    # Should succeed
                    self.assertEqual(result, 0)
                    
                    # Should print migration information
                    print_calls = [call[0][0] for call in mock_print.call_args_list]
                    self.assertTrue(any("Found legacy directory" in call for call in print_calls))

    def test_migrate_xdg_command_execution_dry_run(self):
        """Test command execution with dry-run flag."""
        parser, migrate_cmd = self.create_parser_with_migrate_cmd()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            legacy_dir = base_dir / ".llama"
            legacy_dir.mkdir()
            (legacy_dir / "test_file").touch()
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                
                args = parser.parse_args(["migrate-xdg", "--dry-run"])
                
                with patch("builtins.print") as mock_print:
                    result = migrate_cmd._run_migrate_xdg_cmd(args)
                    
                    # Should succeed
                    self.assertEqual(result, 0)
                    
                    # Should print dry-run information
                    print_calls = [call[0][0] for call in mock_print.call_args_list]
                    self.assertTrue(any("Dry run mode" in call for call in print_calls))

    def test_migrate_xdg_command_execution_error_handling(self):
        """Test command execution with error handling."""
        parser, migrate_cmd = self.create_parser_with_migrate_cmd()
        
        args = parser.parse_args(["migrate-xdg"])
        
        # Mock migrate_to_xdg to raise an exception
        with patch("llama_stack.cli.migrate_xdg.migrate_to_xdg") as mock_migrate:
            mock_migrate.side_effect = Exception("Test error")
            
            with patch("builtins.print") as mock_print:
                result = migrate_cmd._run_migrate_xdg_cmd(args)
                
                # Should return error code
                self.assertEqual(result, 1)
                
                # Should print error message
                print_calls = [call[0][0] for call in mock_print.call_args_list]
                self.assertTrue(any("Error during migration" in call for call in print_calls))

    def test_migrate_xdg_command_integration(self):
        """Test full integration of migrate-xdg command."""
        from llama_stack.cli.llama import LlamaCLIParser
        
        # Create main parser
        main_parser = LlamaCLIParser()
        
        # Test that migrate-xdg is in the subcommands
        with patch("sys.argv", ["llama", "migrate-xdg", "--help"]):
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                with patch("sys.exit"):
                    try:
                        main_parser.parse_args()
                    except SystemExit:
                        pass
                    
                    help_text = mock_stdout.getvalue()
                    self.assertIn("migrate-xdg", help_text)


class TestMigrateXDGFunction(unittest.TestCase):
    """Test the migrate_to_xdg function directly."""

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

    def create_legacy_structure(self, base_dir: Path) -> Path:
        """Create a test legacy directory structure."""
        legacy_dir = base_dir / ".llama"
        legacy_dir.mkdir()
        
        # Create distributions
        (legacy_dir / "distributions").mkdir()
        (legacy_dir / "distributions" / "ollama").mkdir()
        (legacy_dir / "distributions" / "ollama" / "run.yaml").write_text("version: 2\n")
        
        # Create checkpoints
        (legacy_dir / "checkpoints").mkdir()
        (legacy_dir / "checkpoints" / "model.bin").write_text("fake model")
        
        # Create providers.d
        (legacy_dir / "providers.d").mkdir()
        (legacy_dir / "providers.d" / "provider.yaml").write_text("provider: test\n")
        
        # Create runtime
        (legacy_dir / "runtime").mkdir()
        (legacy_dir / "runtime" / "trace.db").write_text("fake database")
        
        return legacy_dir

    def test_migrate_to_xdg_no_legacy_directory(self):
        """Test migrate_to_xdg when no legacy directory exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = Path(temp_dir)
                
                result = migrate_to_xdg(dry_run=False)
                self.assertTrue(result)

    def test_migrate_to_xdg_dry_run(self):
        """Test migrate_to_xdg with dry_run=True."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            legacy_dir = self.create_legacy_structure(base_dir)
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                
                with patch("builtins.print") as mock_print:
                    result = migrate_to_xdg(dry_run=True)
                    self.assertTrue(result)
                    
                    # Should print dry run information
                    print_calls = [call[0][0] for call in mock_print.call_args_list]
                    self.assertTrue(any("Dry run mode" in call for call in print_calls))
                    
                    # Legacy directory should still exist
                    self.assertTrue(legacy_dir.exists())

    def test_migrate_to_xdg_user_confirms(self):
        """Test migrate_to_xdg when user confirms migration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            legacy_dir = self.create_legacy_structure(base_dir)
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                
                with patch("builtins.input") as mock_input:
                    mock_input.side_effect = ["y", "y"]  # Confirm migration and cleanup
                    
                    result = migrate_to_xdg(dry_run=False)
                    self.assertTrue(result)
                    
                    # Legacy directory should be removed
                    self.assertFalse(legacy_dir.exists())
                    
                    # XDG directories should be created
                    self.assertTrue((base_dir / ".config" / "llama-stack").exists())
                    self.assertTrue((base_dir / ".local" / "share" / "llama-stack").exists())

    def test_migrate_to_xdg_user_cancels(self):
        """Test migrate_to_xdg when user cancels migration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            legacy_dir = self.create_legacy_structure(base_dir)
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                
                with patch("builtins.input") as mock_input:
                    mock_input.return_value = "n"  # Cancel migration
                    
                    result = migrate_to_xdg(dry_run=False)
                    self.assertFalse(result)
                    
                    # Legacy directory should still exist
                    self.assertTrue(legacy_dir.exists())

    def test_migrate_to_xdg_partial_migration(self):
        """Test migrate_to_xdg with partial migration (some files fail)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            legacy_dir = self.create_legacy_structure(base_dir)
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                
                # Create target directory with conflicting file
                config_dir = base_dir / ".config" / "llama-stack"
                config_dir.mkdir(parents=True)
                (config_dir / "distributions").mkdir()
                (config_dir / "distributions" / "existing.yaml").write_text("existing")
                
                with patch("builtins.input") as mock_input:
                    mock_input.side_effect = ["y", "n"]  # Confirm migration, don't cleanup
                    
                    with patch("builtins.print") as mock_print:
                        result = migrate_to_xdg(dry_run=False)
                        self.assertTrue(result)
                        
                        # Should print warning about conflicts
                        print_calls = [call[0][0] for call in mock_print.call_args_list]
                        self.assertTrue(any("Warning: Target already exists" in call for call in print_calls))

    def test_migrate_to_xdg_permission_error(self):
        """Test migrate_to_xdg with permission errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            legacy_dir = self.create_legacy_structure(base_dir)
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                
                # Create readonly target directory
                config_dir = base_dir / ".config" / "llama-stack"
                config_dir.mkdir(parents=True)
                config_dir.chmod(0o444)  # Read-only
                
                try:
                    with patch("builtins.input") as mock_input:
                        mock_input.side_effect = ["y", "n"]  # Confirm migration, don't cleanup
                        
                        with patch("builtins.print") as mock_print:
                            result = migrate_to_xdg(dry_run=False)
                            
                            # Should handle permission errors gracefully
                            print_calls = [call[0][0] for call in mock_print.call_args_list]
                            # Should contain some error or warning message
                            self.assertTrue(len(print_calls) > 0)
                            
                finally:
                    # Restore permissions for cleanup
                    config_dir.chmod(0o755)

    def test_migrate_to_xdg_empty_legacy_directory(self):
        """Test migrate_to_xdg with empty legacy directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            legacy_dir = base_dir / ".llama"
            legacy_dir.mkdir()  # Empty directory
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                
                result = migrate_to_xdg(dry_run=False)
                self.assertTrue(result)

    def test_migrate_to_xdg_preserves_file_content(self):
        """Test that migrate_to_xdg preserves file content correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            legacy_dir = self.create_legacy_structure(base_dir)
            
            # Add specific content to test
            test_content = "test configuration content"
            (legacy_dir / "distributions" / "ollama" / "run.yaml").write_text(test_content)
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                
                with patch("builtins.input") as mock_input:
                    mock_input.side_effect = ["y", "y"]  # Confirm migration and cleanup
                    
                    result = migrate_to_xdg(dry_run=False)
                    self.assertTrue(result)
                    
                    # Check content was preserved
                    migrated_file = base_dir / ".config" / "llama-stack" / "distributions" / "ollama" / "run.yaml"
                    self.assertTrue(migrated_file.exists())
                    self.assertEqual(migrated_file.read_text(), test_content)

    def test_migrate_to_xdg_with_symlinks(self):
        """Test migrate_to_xdg with symbolic links."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            legacy_dir = self.create_legacy_structure(base_dir)
            
            # Create symlink
            actual_file = legacy_dir / "actual_config.yaml"
            actual_file.write_text("actual config")
            
            symlink_file = legacy_dir / "distributions" / "symlinked.yaml"
            symlink_file.symlink_to(actual_file)
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                
                with patch("builtins.input") as mock_input:
                    mock_input.side_effect = ["y", "y"]  # Confirm migration and cleanup
                    
                    result = migrate_to_xdg(dry_run=False)
                    self.assertTrue(result)
                    
                    # Check symlink was preserved
                    migrated_symlink = base_dir / ".config" / "llama-stack" / "distributions" / "symlinked.yaml"
                    self.assertTrue(migrated_symlink.exists())
                    self.assertTrue(migrated_symlink.is_symlink())

    def test_migrate_to_xdg_nested_directory_structure(self):
        """Test migrate_to_xdg with nested directory structures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            legacy_dir = self.create_legacy_structure(base_dir)
            
            # Create nested structure
            nested_dir = legacy_dir / "checkpoints" / "org" / "model" / "variant"
            nested_dir.mkdir(parents=True)
            (nested_dir / "model.bin").write_text("nested model")
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                
                with patch("builtins.input") as mock_input:
                    mock_input.side_effect = ["y", "y"]  # Confirm migration and cleanup
                    
                    result = migrate_to_xdg(dry_run=False)
                    self.assertTrue(result)
                    
                    # Check nested structure was preserved
                    migrated_nested = base_dir / ".local" / "share" / "llama-stack" / "checkpoints" / "org" / "model" / "variant"
                    self.assertTrue(migrated_nested.exists())
                    self.assertTrue((migrated_nested / "model.bin").exists())

    def test_migrate_to_xdg_user_input_variations(self):
        """Test migrate_to_xdg with various user input variations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            legacy_dir = self.create_legacy_structure(base_dir)
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = base_dir
                
                # Test various forms of "yes"
                for yes_input in ["y", "Y", "yes", "Yes", "YES"]:
                    # Recreate legacy directory for each test
                    if legacy_dir.exists():
                        import shutil
                        shutil.rmtree(legacy_dir)
                    self.create_legacy_structure(base_dir)
                    
                    with patch("builtins.input") as mock_input:
                        mock_input.side_effect = [yes_input, "n"]  # Confirm migration, don't cleanup
                        
                        result = migrate_to_xdg(dry_run=False)
                        self.assertTrue(result, f"Failed with input: {yes_input}")


if __name__ == "__main__":
    unittest.main() 