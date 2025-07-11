# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Import after we set up environment to avoid module-level imports affecting tests
class TestConfigDirs(unittest.TestCase):
    """Test the config_dirs module with XDG compliance and backwards compatibility."""

    def setUp(self):
        """Set up test environment."""
        # Store original environment variables
        self.original_env = {}
        self.env_vars = [
            "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME",
            "LLAMA_STACK_CONFIG_DIR", "SQLITE_STORE_DIR", "FILES_STORAGE_DIR"
        ]
        
        for key in self.env_vars:
            self.original_env[key] = os.environ.get(key)

    def tearDown(self):
        """Clean up test environment."""
        # Restore original environment variables
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        
        # Clear module cache to ensure fresh imports
        import sys
        modules_to_clear = [
            "llama_stack.distribution.utils.config_dirs",
            "llama_stack.distribution.utils.xdg_utils"
        ]
        for module in modules_to_clear:
            if module in sys.modules:
                del sys.modules[module]

    def clear_env_vars(self):
        """Clear all relevant environment variables."""
        for key in self.env_vars:
            os.environ.pop(key, None)

    def test_config_dirs_xdg_defaults(self):
        """Test config_dirs with XDG default paths."""
        self.clear_env_vars()
        
        # Mock that no legacy directory exists
        with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
            mock_home.return_value = Path("/home/testuser")
            with patch("llama_stack.distribution.utils.xdg_utils.Path.exists") as mock_exists:
                mock_exists.return_value = False
                
                # Import after setting up mocks
                from llama_stack.distribution.utils.config_dirs import (
                    LLAMA_STACK_CONFIG_DIR,
                    DEFAULT_CHECKPOINT_DIR,
                    RUNTIME_BASE_DIR,
                    EXTERNAL_PROVIDERS_DIR,
                    DISTRIBS_BASE_DIR,
                )
                
                # Verify XDG-compliant paths
                self.assertEqual(LLAMA_STACK_CONFIG_DIR, Path("/home/testuser/.config/llama-stack"))
                self.assertEqual(DEFAULT_CHECKPOINT_DIR, Path("/home/testuser/.local/share/llama-stack/checkpoints"))
                self.assertEqual(RUNTIME_BASE_DIR, Path("/home/testuser/.local/state/llama-stack/runtime"))
                self.assertEqual(EXTERNAL_PROVIDERS_DIR, Path("/home/testuser/.config/llama-stack/providers.d"))
                self.assertEqual(DISTRIBS_BASE_DIR, Path("/home/testuser/.config/llama-stack/distributions"))

    def test_config_dirs_custom_xdg_paths(self):
        """Test config_dirs with custom XDG paths."""
        self.clear_env_vars()
        
        # Set custom XDG paths
        os.environ["XDG_CONFIG_HOME"] = "/custom/config"
        os.environ["XDG_DATA_HOME"] = "/custom/data"
        os.environ["XDG_STATE_HOME"] = "/custom/state"
        
        # Mock that no legacy directory exists
        with patch("llama_stack.distribution.utils.xdg_utils.Path.exists") as mock_exists:
            mock_exists.return_value = False
            
            from llama_stack.distribution.utils.config_dirs import (
                LLAMA_STACK_CONFIG_DIR,
                DEFAULT_CHECKPOINT_DIR,
                RUNTIME_BASE_DIR,
                EXTERNAL_PROVIDERS_DIR,
                DISTRIBS_BASE_DIR,
            )
            
            # Verify custom XDG paths are used
            self.assertEqual(LLAMA_STACK_CONFIG_DIR, Path("/custom/config/llama-stack"))
            self.assertEqual(DEFAULT_CHECKPOINT_DIR, Path("/custom/data/llama-stack/checkpoints"))
            self.assertEqual(RUNTIME_BASE_DIR, Path("/custom/state/llama-stack/runtime"))
            self.assertEqual(EXTERNAL_PROVIDERS_DIR, Path("/custom/config/llama-stack/providers.d"))
            self.assertEqual(DISTRIBS_BASE_DIR, Path("/custom/config/llama-stack/distributions"))

    def test_config_dirs_legacy_environment_variable(self):
        """Test config_dirs with legacy LLAMA_STACK_CONFIG_DIR."""
        self.clear_env_vars()
        
        # Set legacy environment variable
        os.environ["LLAMA_STACK_CONFIG_DIR"] = "/legacy/llama"
        
        from llama_stack.distribution.utils.config_dirs import (
            LLAMA_STACK_CONFIG_DIR,
            DEFAULT_CHECKPOINT_DIR,
            RUNTIME_BASE_DIR,
            EXTERNAL_PROVIDERS_DIR,
            DISTRIBS_BASE_DIR,
        )
        
        # All paths should use the legacy base
        legacy_base = Path("/legacy/llama")
        self.assertEqual(LLAMA_STACK_CONFIG_DIR, legacy_base)
        self.assertEqual(DEFAULT_CHECKPOINT_DIR, legacy_base / "checkpoints")
        self.assertEqual(RUNTIME_BASE_DIR, legacy_base / "runtime")
        self.assertEqual(EXTERNAL_PROVIDERS_DIR, legacy_base / "providers.d")
        self.assertEqual(DISTRIBS_BASE_DIR, legacy_base / "distributions")

    def test_config_dirs_legacy_directory_exists(self):
        """Test config_dirs when legacy ~/.llama directory exists."""
        self.clear_env_vars()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            home_dir = Path(temp_dir)
            legacy_dir = home_dir / ".llama"
            legacy_dir.mkdir()
            (legacy_dir / "test_file").touch()  # Add content
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = home_dir
                
                from llama_stack.distribution.utils.config_dirs import (
                    LLAMA_STACK_CONFIG_DIR,
                    DEFAULT_CHECKPOINT_DIR,
                    RUNTIME_BASE_DIR,
                    EXTERNAL_PROVIDERS_DIR,
                    DISTRIBS_BASE_DIR,
                )
                
                # Should use legacy directory
                self.assertEqual(LLAMA_STACK_CONFIG_DIR, legacy_dir)
                self.assertEqual(DEFAULT_CHECKPOINT_DIR, legacy_dir / "checkpoints")
                self.assertEqual(RUNTIME_BASE_DIR, legacy_dir / "runtime")
                self.assertEqual(EXTERNAL_PROVIDERS_DIR, legacy_dir / "providers.d")
                self.assertEqual(DISTRIBS_BASE_DIR, legacy_dir / "distributions")

    def test_config_dirs_precedence_order(self):
        """Test precedence order: LLAMA_STACK_CONFIG_DIR > legacy directory > XDG."""
        self.clear_env_vars()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            home_dir = Path(temp_dir)
            legacy_dir = home_dir / ".llama"
            legacy_dir.mkdir()
            (legacy_dir / "test_file").touch()
            
            # Set both legacy env var and XDG vars
            os.environ["LLAMA_STACK_CONFIG_DIR"] = "/priority/path"
            os.environ["XDG_CONFIG_HOME"] = "/custom/config"
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = home_dir
                
                from llama_stack.distribution.utils.config_dirs import LLAMA_STACK_CONFIG_DIR
                
                # Legacy env var should take precedence
                self.assertEqual(LLAMA_STACK_CONFIG_DIR, Path("/priority/path"))

    def test_config_dirs_all_path_types(self):
        """Test that all path objects are of correct type and absolute."""
        self.clear_env_vars()
        
        from llama_stack.distribution.utils.config_dirs import (
            LLAMA_STACK_CONFIG_DIR,
            DEFAULT_CHECKPOINT_DIR,
            RUNTIME_BASE_DIR,
            EXTERNAL_PROVIDERS_DIR,
            DISTRIBS_BASE_DIR,
        )
        
        # All should be Path objects
        paths = [
            LLAMA_STACK_CONFIG_DIR,
            DEFAULT_CHECKPOINT_DIR,
            RUNTIME_BASE_DIR,
            EXTERNAL_PROVIDERS_DIR,
            DISTRIBS_BASE_DIR,
        ]
        
        for path in paths:
            self.assertIsInstance(path, Path, f"Path {path} should be Path object")
            self.assertTrue(path.is_absolute(), f"Path {path} should be absolute")

    def test_config_dirs_directory_relationships(self):
        """Test relationships between different directory paths."""
        self.clear_env_vars()
        
        from llama_stack.distribution.utils.config_dirs import (
            LLAMA_STACK_CONFIG_DIR,
            EXTERNAL_PROVIDERS_DIR,
            DISTRIBS_BASE_DIR,
        )
        
        # Test parent-child relationships
        self.assertEqual(EXTERNAL_PROVIDERS_DIR.parent, LLAMA_STACK_CONFIG_DIR)
        self.assertEqual(DISTRIBS_BASE_DIR.parent, LLAMA_STACK_CONFIG_DIR)
        
        # Test expected subdirectory names
        self.assertEqual(EXTERNAL_PROVIDERS_DIR.name, "providers.d")
        self.assertEqual(DISTRIBS_BASE_DIR.name, "distributions")

    def test_config_dirs_environment_isolation(self):
        """Test that config_dirs is properly isolated between tests."""
        self.clear_env_vars()
        
        # First import with one set of environment variables
        os.environ["LLAMA_STACK_CONFIG_DIR"] = "/first/path"
        
        # Clear module cache
        import sys
        if "llama_stack.distribution.utils.config_dirs" in sys.modules:
            del sys.modules["llama_stack.distribution.utils.config_dirs"]
        
        from llama_stack.distribution.utils.config_dirs import LLAMA_STACK_CONFIG_DIR as first_config
        
        # Change environment and re-import
        os.environ["LLAMA_STACK_CONFIG_DIR"] = "/second/path"
        
        # Clear module cache again
        if "llama_stack.distribution.utils.config_dirs" in sys.modules:
            del sys.modules["llama_stack.distribution.utils.config_dirs"]
        
        from llama_stack.distribution.utils.config_dirs import LLAMA_STACK_CONFIG_DIR as second_config
        
        # Should get different paths
        self.assertEqual(first_config, Path("/first/path"))
        self.assertEqual(second_config, Path("/second/path"))

    def test_config_dirs_with_tilde_expansion(self):
        """Test config_dirs with tilde in paths."""
        self.clear_env_vars()
        
        os.environ["LLAMA_STACK_CONFIG_DIR"] = "~/custom_llama"
        
        from llama_stack.distribution.utils.config_dirs import LLAMA_STACK_CONFIG_DIR
        
        # Should expand tilde
        expected = Path.home() / "custom_llama"
        self.assertEqual(LLAMA_STACK_CONFIG_DIR, expected)

    def test_config_dirs_empty_environment_variables(self):
        """Test config_dirs with empty environment variables."""
        self.clear_env_vars()
        
        # Set empty values
        os.environ["XDG_CONFIG_HOME"] = ""
        os.environ["XDG_DATA_HOME"] = ""
        
        # Mock no legacy directory
        with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
            mock_home.return_value = Path("/home/testuser")
            with patch("llama_stack.distribution.utils.xdg_utils.Path.exists") as mock_exists:
                mock_exists.return_value = False
                
                from llama_stack.distribution.utils.config_dirs import (
                    LLAMA_STACK_CONFIG_DIR,
                    DEFAULT_CHECKPOINT_DIR,
                )
                
                # Should fall back to defaults
                self.assertEqual(LLAMA_STACK_CONFIG_DIR, Path("/home/testuser/.config/llama-stack"))
                self.assertEqual(DEFAULT_CHECKPOINT_DIR, Path("/home/testuser/.local/share/llama-stack/checkpoints"))

    def test_config_dirs_relative_paths(self):
        """Test config_dirs with relative paths in environment variables."""
        self.clear_env_vars()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            
            # Use relative path
            os.environ["LLAMA_STACK_CONFIG_DIR"] = "relative/config"
            
            from llama_stack.distribution.utils.config_dirs import LLAMA_STACK_CONFIG_DIR
            
            # Should be resolved to absolute path
            self.assertTrue(LLAMA_STACK_CONFIG_DIR.is_absolute())
            self.assertTrue(str(LLAMA_STACK_CONFIG_DIR).endswith("relative/config"))

    def test_config_dirs_with_spaces_in_paths(self):
        """Test config_dirs with spaces in directory paths."""
        self.clear_env_vars()
        
        path_with_spaces = "/path with spaces/llama config"
        os.environ["LLAMA_STACK_CONFIG_DIR"] = path_with_spaces
        
        from llama_stack.distribution.utils.config_dirs import LLAMA_STACK_CONFIG_DIR
        
        self.assertEqual(LLAMA_STACK_CONFIG_DIR, Path(path_with_spaces))

    def test_config_dirs_unicode_paths(self):
        """Test config_dirs with unicode characters in paths."""
        self.clear_env_vars()
        
        unicode_path = "/配置/llama-stack"
        os.environ["LLAMA_STACK_CONFIG_DIR"] = unicode_path
        
        from llama_stack.distribution.utils.config_dirs import LLAMA_STACK_CONFIG_DIR
        
        self.assertEqual(LLAMA_STACK_CONFIG_DIR, Path(unicode_path))

    def test_config_dirs_compatibility_import(self):
        """Test that config_dirs can be imported without errors in various scenarios."""
        self.clear_env_vars()
        
        # Test import with no environment variables
        try:
            from llama_stack.distribution.utils.config_dirs import (
                LLAMA_STACK_CONFIG_DIR,
                DEFAULT_CHECKPOINT_DIR,
                RUNTIME_BASE_DIR,
                EXTERNAL_PROVIDERS_DIR,
                DISTRIBS_BASE_DIR,
            )
            # If we get here without exception, the import succeeded
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Import failed: {e}")

    def test_config_dirs_multiple_imports(self):
        """Test that multiple imports of config_dirs return consistent results."""
        self.clear_env_vars()
        
        os.environ["LLAMA_STACK_CONFIG_DIR"] = "/consistent/path"
        
        # First import
        from llama_stack.distribution.utils.config_dirs import LLAMA_STACK_CONFIG_DIR as first_import
        
        # Second import (should get cached result)
        from llama_stack.distribution.utils.config_dirs import LLAMA_STACK_CONFIG_DIR as second_import
        
        self.assertEqual(first_import, second_import)
        self.assertIs(first_import, second_import)  # Should be the same object


class TestConfigDirsIntegration(unittest.TestCase):
    """Integration tests for config_dirs with other modules."""

    def setUp(self):
        """Set up test environment."""
        self.original_env = {}
        for key in ["XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "LLAMA_STACK_CONFIG_DIR"]:
            self.original_env[key] = os.environ.get(key)

    def tearDown(self):
        """Clean up test environment."""
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        
        # Clear module cache
        import sys
        modules_to_clear = [
            "llama_stack.distribution.utils.config_dirs",
            "llama_stack.distribution.utils.xdg_utils"
        ]
        for module in modules_to_clear:
            if module in sys.modules:
                del sys.modules[module]

    def test_config_dirs_with_model_utils(self):
        """Test that config_dirs works correctly with model_utils."""
        for key in self.original_env:
            os.environ.pop(key, None)
        
        from llama_stack.distribution.utils.config_dirs import DEFAULT_CHECKPOINT_DIR
        from llama_stack.distribution.utils.model_utils import model_local_dir
        
        # Test that model_local_dir uses the correct base directory
        model_descriptor = "meta-llama/Llama-3.2-1B-Instruct"
        expected_path = str(DEFAULT_CHECKPOINT_DIR / model_descriptor.replace(":", "-"))
        actual_path = model_local_dir(model_descriptor)
        
        self.assertEqual(actual_path, expected_path)

    def test_config_dirs_consistency_across_modules(self):
        """Test that all modules use consistent directory paths."""
        for key in self.original_env:
            os.environ.pop(key, None)
        
        from llama_stack.distribution.utils.config_dirs import (
            LLAMA_STACK_CONFIG_DIR,
            DEFAULT_CHECKPOINT_DIR,
            RUNTIME_BASE_DIR,
        )
        from llama_stack.distribution.utils.xdg_utils import (
            get_llama_stack_config_dir,
            get_llama_stack_data_dir,
            get_llama_stack_state_dir,
        )
        
        # Paths should be consistent between modules
        self.assertEqual(LLAMA_STACK_CONFIG_DIR, get_llama_stack_config_dir())
        self.assertEqual(DEFAULT_CHECKPOINT_DIR.parent, get_llama_stack_data_dir())
        self.assertEqual(RUNTIME_BASE_DIR.parent, get_llama_stack_state_dir())


if __name__ == "__main__":
    unittest.main() 