# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import platform
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from llama_stack.distribution.utils.xdg_utils import (
    get_llama_stack_config_dir,
    get_llama_stack_data_dir,
    get_llama_stack_state_dir,
    get_xdg_config_home,
    get_xdg_data_home,
    get_xdg_state_home,
    get_xdg_compliant_path,
    ensure_directory_exists,
    migrate_legacy_directory,
)


class TestXDGEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for XDG compliance."""

    def setUp(self):
        """Set up test environment."""
        self.original_env = {}
        for key in ["XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME", "LLAMA_STACK_CONFIG_DIR"]:
            self.original_env[key] = os.environ.get(key)

    def tearDown(self):
        """Clean up test environment."""
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def clear_env_vars(self):
        """Clear all XDG environment variables."""
        for key in self.original_env:
            os.environ.pop(key, None)

    def test_very_long_paths(self):
        """Test XDG functions with very long directory paths."""
        self.clear_env_vars()
        
        # Create a very long path (close to filesystem limits)
        long_components = ["very_long_directory_name_" + str(i) for i in range(50)]
        long_path = "/tmp/" + "/".join(long_components)
        
        # Test with very long XDG paths
        os.environ["XDG_CONFIG_HOME"] = long_path
        
        result = get_xdg_config_home()
        self.assertEqual(result, Path(long_path))
        
        # Should handle long paths in llama-stack functions
        with patch("llama_stack.distribution.utils.xdg_utils.Path.exists") as mock_exists:
            mock_exists.return_value = False
            
            config_dir = get_llama_stack_config_dir()
            self.assertEqual(config_dir, Path(long_path) / "llama-stack")

    def test_paths_with_special_characters(self):
        """Test XDG functions with special characters in paths."""
        self.clear_env_vars()
        
        # Test various special characters
        special_chars = [
            "path with spaces",
            "path-with-hyphens",
            "path_with_underscores",
            "path.with.dots",
            "path@with@symbols",
            "path+with+plus",
            "path&with&ampersand",
            "path(with)parentheses",
        ]
        
        for special_path in special_chars:
            with self.subTest(path=special_path):
                test_path = f"/tmp/{special_path}"
                os.environ["XDG_CONFIG_HOME"] = test_path
                
                result = get_xdg_config_home()
                self.assertEqual(result, Path(test_path))

    def test_unicode_paths(self):
        """Test XDG functions with unicode characters in paths."""
        self.clear_env_vars()
        
        unicode_paths = [
            "/配置/llama-stack",  # Chinese
            "/конфигурация/llama-stack",  # Russian
            "/構成/llama-stack",  # Japanese
            "/구성/llama-stack",  # Korean
            "/تكوين/llama-stack",  # Arabic
            "/configuración/llama-stack",  # Spanish with accents
            "/配置📁/llama-stack",  # With emoji
        ]
        
        for unicode_path in unicode_paths:
            with self.subTest(path=unicode_path):
                os.environ["XDG_CONFIG_HOME"] = unicode_path
                
                result = get_xdg_config_home()
                self.assertEqual(result, Path(unicode_path))

    def test_network_paths(self):
        """Test XDG functions with network/UNC paths."""
        self.clear_env_vars()
        
        if platform.system() == "Windows":
            # Test Windows UNC paths
            unc_paths = [
                "\\\\server\\share\\config",
                "\\\\server.domain.com\\share\\config",
                "\\\\192.168.1.100\\config",
            ]
            
            for unc_path in unc_paths:
                with self.subTest(path=unc_path):
                    os.environ["XDG_CONFIG_HOME"] = unc_path
                    
                    result = get_xdg_config_home()
                    self.assertEqual(result, Path(unc_path))
        else:
            # Test network mount paths on Unix-like systems
            network_paths = [
                "/mnt/nfs/config",
                "/net/server/config",
                "/media/network/config",
            ]
            
            for network_path in network_paths:
                with self.subTest(path=network_path):
                    os.environ["XDG_CONFIG_HOME"] = network_path
                    
                    result = get_xdg_config_home()
                    self.assertEqual(result, Path(network_path))

    def test_nonexistent_paths(self):
        """Test XDG functions with non-existent paths."""
        self.clear_env_vars()
        
        nonexistent_path = "/this/path/does/not/exist/config"
        os.environ["XDG_CONFIG_HOME"] = nonexistent_path
        
        # Should return the path even if it doesn't exist
        result = get_xdg_config_home()
        self.assertEqual(result, Path(nonexistent_path))
        
        # Should work with llama-stack functions too
        with patch("llama_stack.distribution.utils.xdg_utils.Path.exists") as mock_exists:
            mock_exists.return_value = False
            
            config_dir = get_llama_stack_config_dir()
            self.assertEqual(config_dir, Path(nonexistent_path) / "llama-stack")

    def test_circular_symlinks(self):
        """Test XDG functions with circular symbolic links."""
        self.clear_env_vars()
        
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

    def test_broken_symlinks(self):
        """Test XDG functions with broken symbolic links."""
        self.clear_env_vars()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create broken symlink
            target = temp_path / "nonexistent_target"
            link = temp_path / "broken_link"
            
            try:
                link.symlink_to(target)
                
                os.environ["XDG_CONFIG_HOME"] = str(link)
                
                # Should handle broken symlinks gracefully
                result = get_xdg_config_home()
                self.assertEqual(result, link)
                
            except (OSError, NotImplementedError):
                # Some systems might not support this
                self.skipTest("System doesn't support broken symlinks")

    def test_readonly_directories(self):
        """Test XDG functions with read-only directories."""
        self.clear_env_vars()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            readonly_dir = temp_path / "readonly"
            readonly_dir.mkdir()
            
            # Make directory read-only
            readonly_dir.chmod(0o444)
            
            try:
                os.environ["XDG_CONFIG_HOME"] = str(readonly_dir)
                
                # Should still return the path
                result = get_xdg_config_home()
                self.assertEqual(result, readonly_dir)
                
            finally:
                # Restore permissions for cleanup
                readonly_dir.chmod(0o755)

    def test_permission_denied_access(self):
        """Test XDG functions when permission is denied."""
        self.clear_env_vars()
        
        # This test is platform-specific
        if platform.system() != "Windows":
            # Try to use a system directory that typically requires root
            restricted_paths = [
                "/root/.config",
                "/etc/config",
                "/var/root/config",
            ]
            
            for restricted_path in restricted_paths:
                with self.subTest(path=restricted_path):
                    os.environ["XDG_CONFIG_HOME"] = restricted_path
                    
                    # Should still return the path even if we can't access it
                    result = get_xdg_config_home()
                    self.assertEqual(result, Path(restricted_path))

    def test_environment_variable_injection(self):
        """Test XDG functions with environment variable injection attempts."""
        self.clear_env_vars()
        
        # Test potential injection attempts
        injection_attempts = [
            "/tmp/config; rm -rf /",
            "/tmp/config && echo 'injected'",
            "/tmp/config | cat /etc/passwd",
            "/tmp/config`whoami`",
            "/tmp/config$(whoami)",
            "/tmp/config\necho 'newline'",
        ]
        
        for injection_attempt in injection_attempts:
            with self.subTest(attempt=injection_attempt):
                os.environ["XDG_CONFIG_HOME"] = injection_attempt
                
                # Should treat as literal path, not execute
                result = get_xdg_config_home()
                self.assertEqual(result, Path(injection_attempt))

    def test_extremely_nested_paths(self):
        """Test XDG functions with extremely nested directory structures."""
        self.clear_env_vars()
        
        # Create deeply nested path
        nested_parts = ["level" + str(i) for i in range(100)]
        nested_path = "/tmp/" + "/".join(nested_parts)
        
        os.environ["XDG_CONFIG_HOME"] = nested_path
        
        result = get_xdg_config_home()
        self.assertEqual(result, Path(nested_path))

    def test_empty_and_whitespace_paths(self):
        """Test XDG functions with empty and whitespace-only paths."""
        self.clear_env_vars()
        
        empty_values = [
            "",
            " ",
            "\t",
            "\n",
            "\r\n",
            "   \t  \n  ",
        ]
        
        for empty_value in empty_values:
            with self.subTest(value=repr(empty_value)):
                os.environ["XDG_CONFIG_HOME"] = empty_value
                
                # Should fall back to default
                result = get_xdg_config_home()
                self.assertEqual(result, Path.home() / ".config")

    def test_path_with_null_bytes(self):
        """Test XDG functions with null bytes in paths."""
        self.clear_env_vars()
        
        # Test path with null byte
        null_path = "/tmp/config\x00/test"
        os.environ["XDG_CONFIG_HOME"] = null_path
        
        # Should handle null bytes (Path will likely raise an error, which is expected)
        try:
            result = get_xdg_config_home()
            # If it doesn't raise an error, check the result
            self.assertIsInstance(result, Path)
        except (ValueError, OSError):
            # This is expected behavior for null bytes
            pass

    def test_concurrent_access_safety(self):
        """Test that XDG functions are thread-safe."""
        import threading
        import time
        
        self.clear_env_vars()
        
        results = []
        errors = []
        
        def worker(thread_id):
            try:
                # Each thread sets a different XDG path
                os.environ["XDG_CONFIG_HOME"] = f"/tmp/thread_{thread_id}"
                
                # Small delay to increase chance of race conditions
                time.sleep(0.01)
                
                config_dir = get_llama_stack_config_dir()
                results.append((thread_id, config_dir))
                
            except Exception as e:
                errors.append((thread_id, e))
        
        # Start multiple threads
        threads = []
        for i in range(20):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Check for errors
        if errors:
            self.fail(f"Thread errors: {errors}")
        
        # Check that we got results from all threads
        self.assertEqual(len(results), 20)

    def test_filesystem_limits(self):
        """Test XDG functions approaching filesystem limits."""
        self.clear_env_vars()
        
        # Test with very long filename (close to 255 char limit)
        long_filename = "a" * 240
        long_path = f"/tmp/{long_filename}"
        
        os.environ["XDG_CONFIG_HOME"] = long_path
        
        result = get_xdg_config_home()
        self.assertEqual(result, Path(long_path))

    def test_case_sensitivity(self):
        """Test XDG functions with case sensitivity edge cases."""
        self.clear_env_vars()
        
        # Test case variations
        case_variations = [
            "/tmp/Config",
            "/tmp/CONFIG",
            "/tmp/config",
            "/tmp/Config/MixedCase",
        ]
        
        for case_path in case_variations:
            with self.subTest(path=case_path):
                os.environ["XDG_CONFIG_HOME"] = case_path
                
                result = get_xdg_config_home()
                self.assertEqual(result, Path(case_path))

    def test_ensure_directory_exists_edge_cases(self):
        """Test ensure_directory_exists with edge cases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test with file that exists but is not a directory
            file_path = temp_path / "file_not_dir"
            file_path.touch()
            
            with self.assertRaises(FileExistsError):
                ensure_directory_exists(file_path)
            
            # Test with permission denied
            if platform.system() != "Windows":
                readonly_parent = temp_path / "readonly_parent"
                readonly_parent.mkdir()
                readonly_parent.chmod(0o444)
                
                try:
                    nested_path = readonly_parent / "nested"
                    
                    with self.assertRaises(PermissionError):
                        ensure_directory_exists(nested_path)
                        
                finally:
                    # Restore permissions for cleanup
                    readonly_parent.chmod(0o755)

    def test_migrate_legacy_directory_edge_cases(self):
        """Test migrate_legacy_directory with edge cases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            home_dir = Path(temp_dir)
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = home_dir
                
                # Test with legacy directory but no write permissions
                legacy_dir = home_dir / ".llama"
                legacy_dir.mkdir()
                (legacy_dir / "test_file").touch()
                
                # Make home directory read-only
                home_dir.chmod(0o444)
                
                try:
                    # Should handle permission errors gracefully
                    with patch("builtins.print") as mock_print:
                        result = migrate_legacy_directory()
                        
                        # Should print some information
                        self.assertTrue(mock_print.called)
                        
                finally:
                    # Restore permissions for cleanup
                    home_dir.chmod(0o755)
                    legacy_dir.chmod(0o755)

    def test_path_traversal_attempts(self):
        """Test XDG functions with path traversal attempts."""
        self.clear_env_vars()
        
        traversal_attempts = [
            "/tmp/config/../../../etc/passwd",
            "/tmp/config/../../root/.ssh",
            "/tmp/config/../../../../../etc/shadow",
            "/tmp/config/./../../root",
        ]
        
        for traversal_attempt in traversal_attempts:
            with self.subTest(attempt=traversal_attempt):
                os.environ["XDG_CONFIG_HOME"] = traversal_attempt
                
                # Should handle path traversal attempts by treating as literal paths
                result = get_xdg_config_home()
                self.assertEqual(result, Path(traversal_attempt))

    def test_environment_variable_precedence_edge_cases(self):
        """Test environment variable precedence with edge cases."""
        self.clear_env_vars()
        
        # Test with both old and new env vars set
        os.environ["LLAMA_STACK_CONFIG_DIR"] = "/legacy/path"
        os.environ["XDG_CONFIG_HOME"] = "/xdg/path"
        
        # Create fake legacy directory
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_home = Path(temp_dir)
            fake_legacy = fake_home / ".llama"
            fake_legacy.mkdir()
            (fake_legacy / "test_file").touch()
            
            with patch("llama_stack.distribution.utils.xdg_utils.Path.home") as mock_home:
                mock_home.return_value = fake_home
                
                # LLAMA_STACK_CONFIG_DIR should take precedence
                config_dir = get_llama_stack_config_dir()
                self.assertEqual(config_dir, Path("/legacy/path"))

    def test_malformed_environment_variables(self):
        """Test XDG functions with malformed environment variables."""
        self.clear_env_vars()
        
        malformed_values = [
            "not_an_absolute_path",
            "~/tilde_not_expanded",
            "$HOME/variable_not_expanded",
            "relative/path/config",
            "./relative/path",
            "../parent/path",
        ]
        
        for malformed_value in malformed_values:
            with self.subTest(value=malformed_value):
                os.environ["XDG_CONFIG_HOME"] = malformed_value
                
                # Should handle malformed values gracefully
                result = get_xdg_config_home()
                self.assertIsInstance(result, Path)


if __name__ == "__main__":
    unittest.main() 
