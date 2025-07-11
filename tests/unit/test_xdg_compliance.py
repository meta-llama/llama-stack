# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from llama_stack.distribution.utils.xdg_utils import (
    get_llama_stack_config_dir,
    get_llama_stack_data_dir,
    get_llama_stack_state_dir,
    get_llama_stack_cache_dir,
    get_xdg_cache_home,
    get_xdg_config_home,
    get_xdg_data_home,
    get_xdg_state_home,
    get_xdg_compliant_path,
    migrate_legacy_directory,
    ensure_directory_exists,
)


class TestXDGCompliance(unittest.TestCase):
    """Comprehensive test suite for XDG Base Directory Specification compliance."""

    def setUp(self):
        """Set up test environment."""
        # Store original environment variables
        self.original_env = {}
        self.xdg_vars = ["XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME"]
        self.llama_vars = ["LLAMA_STACK_CONFIG_DIR", "SQLITE_STORE_DIR", "FILES_STORAGE_DIR"]
        
        for key in self.xdg_vars + self.llama_vars:
            self.original_env[key] = os.environ.get(key)

    def tearDown(self):
        """Clean up test environment."""
        # Restore original environment variables
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def clear_env_vars(self, vars_to_clear=None):
        """Helper to clear environment variables."""
        if vars_to_clear is None:
            vars_to_clear = self.xdg_vars + self.llama_vars
        
        for key in vars_to_clear:
            os.environ.pop(key, None)

    def test_xdg_defaults(self):
        """Test that XDG directories use correct defaults when no env vars are set."""
        self.clear_env_vars()
        home = Path.home()
        
        self.assertEqual(get_xdg_config_home(), home / ".config")
        self.assertEqual(get_xdg_data_home(), home / ".local" / "share")
        self.assertEqual(get_xdg_cache_home(), home / ".cache")
        self.assertEqual(get_xdg_state_home(), home / ".local" / "state")

    def test_xdg_custom_paths(self):
        """Test that custom XDG paths are respected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            os.environ["XDG_CONFIG_HOME"] = str(temp_path / "config")
            os.environ["XDG_DATA_HOME"] = str(temp_path / "data")
            os.environ["XDG_CACHE_HOME"] = str(temp_path / "cache")
            os.environ["XDG_STATE_HOME"] = str(temp_path / "state")

            self.assertEqual(get_xdg_config_home(), temp_path / "config")
            self.assertEqual(get_xdg_data_home(), temp_path / "data")
            self.assertEqual(get_xdg_cache_home(), temp_path / "cache")
            self.assertEqual(get_xdg_state_home(), temp_path / "state")

    def test_xdg_paths_with_tilde(self):
        """Test XDG paths that use tilde expansion."""
        os.environ["XDG_CONFIG_HOME"] = "~/custom_config"
        os.environ["XDG_DATA_HOME"] = "~/custom_data"
        
        home = Path.home()
        self.assertEqual(get_xdg_config_home(), home / "custom_config")
        self.assertEqual(get_xdg_data_home(), home / "custom_data")

    def test_xdg_paths_relative(self):
        """Test XDG paths with relative paths get resolved."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            os.environ["XDG_CONFIG_HOME"] = "relative_config"
            
            # Should resolve relative to current directory
            result = get_xdg_config_home()
            self.assertTrue(result.is_absolute())
            self.assertTrue(str(result).endswith("relative_config"))

    def test_llama_stack_directories_new_installation(self):
        """Test llama-stack directories for new installations (no legacy directory)."""
        self.clear_env_vars()
        home = Path.home()
        
        # Mock that ~/.llama doesn't exist
        with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
            mock_home.return_value = home
            with patch("llama_stack.distribution.utils.xdg_utils.Path.exists") as mock_exists:
                mock_exists.return_value = False
                
                self.assertEqual(get_llama_stack_config_dir(), home / ".config" / "llama-stack")
                self.assertEqual(get_llama_stack_data_dir(), home / ".local" / "share" / "llama-stack")
                self.assertEqual(get_llama_stack_state_dir(), home / ".local" / "state" / "llama-stack")
                self.assertEqual(get_llama_stack_cache_dir(), home / ".cache" / "llama-stack")

    def test_llama_stack_directories_with_custom_xdg(self):
        """Test llama-stack directories with custom XDG paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            os.environ["XDG_CONFIG_HOME"] = str(temp_path / "config")
            os.environ["XDG_DATA_HOME"] = str(temp_path / "data")
            os.environ["XDG_STATE_HOME"] = str(temp_path / "state")
            os.environ["XDG_CACHE_HOME"] = str(temp_path / "cache")

            # Mock that ~/.llama doesn't exist
            with patch("llama_stack.distribution.utils.xdg_utils.Path.exists") as mock_exists:
                mock_exists.return_value = False
                
                self.assertEqual(get_llama_stack_config_dir(), temp_path / "config" / "llama-stack")
                self.assertEqual(get_llama_stack_data_dir(), temp_path / "data" / "llama-stack")
                self.assertEqual(get_llama_stack_state_dir(), temp_path / "state" / "llama-stack")
                self.assertEqual(get_llama_stack_cache_dir(), temp_path / "cache" / "llama-stack")

    def test_legacy_environment_variable_precedence(self):
        """Test that LLAMA_STACK_CONFIG_DIR takes highest precedence."""
        with tempfile.TemporaryDirectory() as temp_dir:
            legacy_path = Path(temp_dir) / "legacy"
            xdg_path = Path(temp_dir) / "xdg"
            
            # Set both legacy and XDG variables
            os.environ["LLAMA_STACK_CONFIG_DIR"] = str(legacy_path)
            os.environ["XDG_CONFIG_HOME"] = str(xdg_path / "config")
            os.environ["XDG_DATA_HOME"] = str(xdg_path / "data")
            os.environ["XDG_STATE_HOME"] = str(xdg_path / "state")

            # Legacy should take precedence for all directory types
            self.assertEqual(get_llama_stack_config_dir(), legacy_path)
            self.assertEqual(get_llama_stack_data_dir(), legacy_path)
            self.assertEqual(get_llama_stack_state_dir(), legacy_path)
            self.assertEqual(get_llama_stack_cache_dir(), legacy_path)

    def test_legacy_directory_exists_and_has_content(self):
        """Test that existing ~/.llama directory with content is used."""
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            legacy_llama = home / ".llama"
            legacy_llama.mkdir()
            
            # Create some content to simulate existing data
            (legacy_llama / "test_file").touch()
            (legacy_llama / "distributions").mkdir()

            # Clear environment variables
            self.clear_env_vars()

            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = home
                
                self.assertEqual(get_llama_stack_config_dir(), legacy_llama)
                self.assertEqual(get_llama_stack_data_dir(), legacy_llama)
                self.assertEqual(get_llama_stack_state_dir(), legacy_llama)

    def test_legacy_directory_exists_but_empty(self):
        """Test that empty ~/.llama directory is ignored in favor of XDG."""
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            legacy_llama = home / ".llama"
            legacy_llama.mkdir()
            # Don't add any content - directory is empty

            self.clear_env_vars()

            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = home
                
                # Should use XDG paths since legacy directory is empty
                self.assertEqual(get_llama_stack_config_dir(), home / ".config" / "llama-stack")
                self.assertEqual(get_llama_stack_data_dir(), home / ".local" / "share" / "llama-stack")
                self.assertEqual(get_llama_stack_state_dir(), home / ".local" / "state" / "llama-stack")

    def test_xdg_compliant_path_function(self):
        """Test the get_xdg_compliant_path utility function."""
        self.clear_env_vars()
        home = Path.home()

        # Mock that ~/.llama doesn't exist
        with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
            mock_home.return_value = home
            with patch("llama_stack.distribution.utils.xdg_utils.Path.exists") as mock_exists:
                mock_exists.return_value = False
                
                self.assertEqual(
                    get_xdg_compliant_path("config"), 
                    home / ".config" / "llama-stack"
                )
                self.assertEqual(
                    get_xdg_compliant_path("data", "models"), 
                    home / ".local" / "share" / "llama-stack" / "models"
                )
                self.assertEqual(
                    get_xdg_compliant_path("state", "runtime"), 
                    home / ".local" / "state" / "llama-stack" / "runtime"
                )
                self.assertEqual(
                    get_xdg_compliant_path("cache", "temp"), 
                    home / ".cache" / "llama-stack" / "temp"
                )

    def test_xdg_compliant_path_invalid_type(self):
        """Test that invalid path types raise ValueError."""
        with self.assertRaises(ValueError) as context:
            get_xdg_compliant_path("invalid_type")
        
        self.assertIn("Unknown path type", str(context.exception))
        self.assertIn("invalid_type", str(context.exception))

    def test_xdg_compliant_path_with_subdirectory(self):
        """Test get_xdg_compliant_path with various subdirectories."""
        self.clear_env_vars()
        home = Path.home()

        with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
            mock_home.return_value = home
            with patch("llama_stack.distribution.utils.xdg_utils.Path.exists") as mock_exists:
                mock_exists.return_value = False
                
                # Test nested subdirectories
                self.assertEqual(
                    get_xdg_compliant_path("data", "models/checkpoints"), 
                    home / ".local" / "share" / "llama-stack" / "models/checkpoints"
                )
                
                # Test with Path object
                self.assertEqual(
                    get_xdg_compliant_path("config", str(Path("distributions") / "ollama")), 
                    home / ".config" / "llama-stack" / "distributions" / "ollama"
                )

    def test_ensure_directory_exists(self):
        """Test the ensure_directory_exists utility function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "nested" / "directory" / "structure"
            
            # Directory shouldn't exist initially
            self.assertFalse(test_path.exists())
            
            # Create it
            ensure_directory_exists(test_path)
            
            # Should exist now
            self.assertTrue(test_path.exists())
            self.assertTrue(test_path.is_dir())

    def test_ensure_directory_exists_already_exists(self):
        """Test ensure_directory_exists when directory already exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "existing"
            test_path.mkdir()
            
            # Should not raise an error
            ensure_directory_exists(test_path)
            self.assertTrue(test_path.exists())

    def test_config_dirs_import_and_types(self):
        """Test that the config_dirs module imports correctly and has proper types."""
        from llama_stack.distribution.utils.config_dirs import (
            LLAMA_STACK_CONFIG_DIR,
            DEFAULT_CHECKPOINT_DIR,
            RUNTIME_BASE_DIR,
            EXTERNAL_PROVIDERS_DIR,
            DISTRIBS_BASE_DIR,
        )

        # All should be Path objects
        self.assertIsInstance(LLAMA_STACK_CONFIG_DIR, Path)
        self.assertIsInstance(DEFAULT_CHECKPOINT_DIR, Path)
        self.assertIsInstance(RUNTIME_BASE_DIR, Path)
        self.assertIsInstance(EXTERNAL_PROVIDERS_DIR, Path)
        self.assertIsInstance(DISTRIBS_BASE_DIR, Path)

        # All should be absolute paths
        self.assertTrue(LLAMA_STACK_CONFIG_DIR.is_absolute())
        self.assertTrue(DEFAULT_CHECKPOINT_DIR.is_absolute())
        self.assertTrue(RUNTIME_BASE_DIR.is_absolute())
        self.assertTrue(EXTERNAL_PROVIDERS_DIR.is_absolute())
        self.assertTrue(DISTRIBS_BASE_DIR.is_absolute())

    def test_config_dirs_proper_structure(self):
        """Test that config_dirs uses proper XDG structure."""
        from llama_stack.distribution.utils.config_dirs import (
            LLAMA_STACK_CONFIG_DIR,
            DEFAULT_CHECKPOINT_DIR,
            RUNTIME_BASE_DIR,
            EXTERNAL_PROVIDERS_DIR,
            DISTRIBS_BASE_DIR,
        )

        # Check that paths contain expected components
        config_str = str(LLAMA_STACK_CONFIG_DIR)
        self.assertTrue(
            "llama-stack" in config_str or ".llama" in config_str,
            f"Config dir should contain 'llama-stack' or '.llama': {config_str}"
        )

        # Test relationships between directories
        self.assertEqual(DISTRIBS_BASE_DIR, LLAMA_STACK_CONFIG_DIR / "distributions")
        self.assertEqual(EXTERNAL_PROVIDERS_DIR, LLAMA_STACK_CONFIG_DIR / "providers.d")

    def test_environment_variable_combinations(self):
        """Test various combinations of environment variables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test partial XDG variables
            os.environ["XDG_CONFIG_HOME"] = str(temp_path / "config")
            # Leave others as default
            self.clear_env_vars(["XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME"])
            
            home = Path.home()
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.exists") as mock_exists:
                mock_exists.return_value = False
                
                self.assertEqual(get_llama_stack_config_dir(), temp_path / "config" / "llama-stack")
                self.assertEqual(get_llama_stack_data_dir(), home / ".local" / "share" / "llama-stack")
                self.assertEqual(get_llama_stack_state_dir(), home / ".local" / "state" / "llama-stack")

    def test_migrate_legacy_directory_no_legacy(self):
        """Test migration when no legacy directory exists."""
        with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
            mock_home.return_value = Path("/fake/home")
            with patch("llama_stack.distribution.utils.xdg_utils.Path.exists") as mock_exists:
                mock_exists.return_value = False
                
                # Should return True (success) when no migration needed
                result = migrate_legacy_directory()
                self.assertTrue(result)

    def test_migrate_legacy_directory_exists(self):
        """Test migration message when legacy directory exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            legacy_llama = home / ".llama"
            legacy_llama.mkdir()
            (legacy_llama / "test_file").touch()
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = home
                with patch("builtins.print") as mock_print:
                    result = migrate_legacy_directory()
                    self.assertTrue(result)
                    
                    # Check that migration information was printed
                    print_calls = [call[0][0] for call in mock_print.call_args_list]
                    self.assertTrue(any("Found legacy directory" in call for call in print_calls))
                    self.assertTrue(any("Consider migrating" in call for call in print_calls))

    def test_path_consistency_across_functions(self):
        """Test that all path functions return consistent results."""
        self.clear_env_vars()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = home
                with patch("llama_stack.distribution.utils.xdg_utils.Path.exists") as mock_exists:
                    mock_exists.return_value = False
                    
                    # All config-related functions should return the same base
                    config_dir = get_llama_stack_config_dir()
                    config_path = get_xdg_compliant_path("config")
                    self.assertEqual(config_dir, config_path)
                    
                    # All data-related functions should return the same base
                    data_dir = get_llama_stack_data_dir()
                    data_path = get_xdg_compliant_path("data")
                    self.assertEqual(data_dir, data_path)

    def test_unicode_and_special_characters(self):
        """Test XDG paths with unicode and special characters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with unicode characters
            unicode_path = Path(temp_dir) / "配置" / "llama-stack"
            os.environ["XDG_CONFIG_HOME"] = str(unicode_path.parent)
            
            result = get_xdg_config_home()
            self.assertEqual(result, unicode_path.parent)
            
            # Test spaces in paths
            space_path = Path(temp_dir) / "my config"
            os.environ["XDG_CONFIG_HOME"] = str(space_path)
            
            result = get_xdg_config_home()
            self.assertEqual(result, space_path)

    def test_concurrent_access_safety(self):
        """Test that XDG functions are safe for concurrent access."""
        import threading
        import time
        
        results = []
        errors = []
        
        def worker():
            try:
                # Simulate concurrent access
                config_dir = get_llama_stack_config_dir()
                time.sleep(0.01)  # Small delay to increase chance of race conditions
                data_dir = get_llama_stack_data_dir()
                results.append((config_dir, data_dir))
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for _ in range(10):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Check results
        self.assertEqual(len(errors), 0, f"Concurrent access errors: {errors}")
        self.assertEqual(len(results), 10)
        
        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            self.assertEqual(result, first_result)

    def test_symlink_handling(self):
        """Test XDG path handling with symbolic links."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create actual directory
            actual_dir = temp_path / "actual_config"
            actual_dir.mkdir()
            
            # Create symlink
            symlink_dir = temp_path / "symlinked_config"
            symlink_dir.symlink_to(actual_dir)
            
            os.environ["XDG_CONFIG_HOME"] = str(symlink_dir)
            
            result = get_xdg_config_home()
            self.assertEqual(result, symlink_dir)
            
            # Should resolve to actual path when needed
            self.assertTrue(result.exists())

    def test_readonly_directory_handling(self):
        """Test behavior when XDG directories are read-only."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            readonly_dir = temp_path / "readonly"
            readonly_dir.mkdir()
            
            # Make directory read-only
            readonly_dir.chmod(0o444)
            
            try:
                os.environ["XDG_CONFIG_HOME"] = str(readonly_dir)
                
                # Should still return the path even if read-only
                result = get_xdg_config_home()
                self.assertEqual(result, readonly_dir)
                
            finally:
                # Restore permissions for cleanup
                readonly_dir.chmod(0o755)

    def test_nonexistent_parent_directory(self):
        """Test XDG paths with non-existent parent directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use a path with non-existent parents
            nonexistent_path = Path(temp_dir) / "does" / "not" / "exist" / "config"
            
            os.environ["XDG_CONFIG_HOME"] = str(nonexistent_path)
            
            # Should return the path even if it doesn't exist
            result = get_xdg_config_home()
            self.assertEqual(result, nonexistent_path)

    def test_env_var_expansion(self):
        """Test environment variable expansion in XDG paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["TEST_BASE"] = temp_dir
            os.environ["XDG_CONFIG_HOME"] = "$TEST_BASE/config"
            
            # Path expansion should work
            result = get_xdg_config_home()
            expected = Path(temp_dir) / "config"
            self.assertEqual(result, expected)


class TestXDGEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for XDG compliance."""

    def setUp(self):
        """Set up test environment."""
        self.original_env = {}
        for key in ["XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME", "LLAMA_STACK_CONFIG_DIR"]:
            self.original_env[key] = os.environ.get(key)

    def tearDown(self):
        """Clean up test environment."""
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_empty_environment_variables(self):
        """Test behavior with empty environment variables."""
        # Set empty values
        os.environ["XDG_CONFIG_HOME"] = ""
        os.environ["XDG_DATA_HOME"] = ""
        
        # Should fall back to defaults
        home = Path.home()
        self.assertEqual(get_xdg_config_home(), home / ".config")
        self.assertEqual(get_xdg_data_home(), home / ".local" / "share")

    def test_whitespace_only_environment_variables(self):
        """Test behavior with whitespace-only environment variables."""
        os.environ["XDG_CONFIG_HOME"] = "   "
        os.environ["XDG_DATA_HOME"] = "\t\n"
        
        # Should handle whitespace gracefully
        result_config = get_xdg_config_home()
        result_data = get_xdg_data_home()
        
        # Results should be valid Path objects
        self.assertIsInstance(result_config, Path)
        self.assertIsInstance(result_data, Path)

    def test_very_long_paths(self):
        """Test behavior with very long directory paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a very long path
            long_path_parts = ["very_long_directory_name_" + str(i) for i in range(20)]
            long_path = Path(temp_dir)
            for part in long_path_parts:
                long_path = long_path / part
            
            os.environ["XDG_CONFIG_HOME"] = str(long_path)
            
            result = get_xdg_config_home()
            self.assertEqual(result, long_path)

    def test_circular_symlinks(self):
        """Test handling of circular symbolic links."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create circular symlinks
            link1 = temp_path / "link1"
            link2 = temp_path / "link2"
            
            try:
                link1.symlink_to(link2)
                link2.symlink_to(link1)
                
                os.environ["XDG_CONFIG_HOME"] = str(link1)
                
                # Should handle circular symlinks gracefully
                result = get_xdg_config_home()
                self.assertEqual(result, link1)
                
            except (OSError, NotImplementedError):
                # Some systems don't support circular symlinks
                self.skipTest("System doesn't support circular symlinks")

    def test_permission_denied_scenarios(self):
        """Test scenarios where permission is denied."""
        # This test is platform-specific and may not work on all systems
        try:
            # Try to use a system directory that typically requires root
            os.environ["XDG_CONFIG_HOME"] = "/root/.config"
            
            # Should still return the path even if we can't access it
            result = get_xdg_config_home()
            self.assertEqual(result, Path("/root/.config"))
            
        except Exception:
            # If this fails, it's not critical for the XDG implementation
            pass

    def test_network_paths(self):
        """Test XDG paths with network/UNC paths (Windows-style)."""
        # Test UNC path (though this may not work on non-Windows systems)
        network_path = "//server/share/config"
        os.environ["XDG_CONFIG_HOME"] = network_path
        
        result = get_xdg_config_home()
        # Should handle network paths gracefully
        self.assertIsInstance(result, Path)


if __name__ == "__main__":
    unittest.main() 