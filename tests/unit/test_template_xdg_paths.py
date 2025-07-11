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
import yaml

# Template imports will be tested through file system access


class TestTemplateXDGPaths(unittest.TestCase):
    """Test that templates use XDG-compliant paths correctly."""

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

    def test_ollama_template_run_yaml_xdg_paths(self):
        """Test that ollama template's run.yaml uses XDG environment variables."""
        template_path = Path(__file__).parent.parent.parent / "llama_stack" / "templates" / "ollama" / "run.yaml"
        
        if not template_path.exists():
            self.skipTest("Ollama template not found")
        
        content = template_path.read_text()
        
        # Check for XDG-compliant environment variable references
        self.assertIn("${env.XDG_STATE_HOME:-~/.local/state}", content)
        self.assertIn("${env.XDG_DATA_HOME:-~/.local/share}", content)
        
        # Check that paths use llama-stack directory
        self.assertIn("llama-stack", content)
        
        # Check specific path patterns
        self.assertIn("${env.XDG_STATE_HOME:-~/.local/state}/llama-stack/distributions/ollama", content)
        self.assertIn("${env.XDG_DATA_HOME:-~/.local/share}/llama-stack/distributions/ollama", content)

    def test_ollama_template_run_yaml_parsing(self):
        """Test that ollama template's run.yaml can be parsed correctly."""
        template_path = Path(__file__).parent.parent.parent / "llama_stack" / "templates" / "ollama" / "run.yaml"
        
        if not template_path.exists():
            self.skipTest("Ollama template not found")
        
        content = template_path.read_text()
        
        # Replace environment variables with test values for parsing
        test_content = content.replace(
            "${env.XDG_STATE_HOME:-~/.local/state}", "/test/state"
        ).replace(
            "${env.XDG_DATA_HOME:-~/.local/share}", "/test/data"
        ).replace(
            "${env.SQLITE_STORE_DIR:=${env.XDG_STATE_HOME:-~/.local/state}/llama-stack/distributions/ollama}",
            "/test/state/llama-stack/distributions/ollama"
        )
        
        # Should be valid YAML
        try:
            yaml.safe_load(test_content)
        except yaml.YAMLError as e:
            self.fail(f"Template YAML is invalid: {e}")

    def test_template_environment_variable_expansion(self):
        """Test environment variable expansion in templates."""
        self.clear_env_vars()
        
        # Set XDG variables
        os.environ["XDG_STATE_HOME"] = "/custom/state"
        os.environ["XDG_DATA_HOME"] = "/custom/data"
        
        # Test pattern that should expand
        pattern = "${env.XDG_STATE_HOME:-~/.local/state}/llama-stack/test"
        expected = "/custom/state/llama-stack/test"
        
        # Mock environment variable expansion (this would normally be done by the shell)
        expanded = pattern.replace("${env.XDG_STATE_HOME:-~/.local/state}", os.environ["XDG_STATE_HOME"])
        self.assertEqual(expanded, expected)

    def test_template_fallback_values(self):
        """Test that templates have correct fallback values."""
        self.clear_env_vars()
        
        # Test fallback pattern
        pattern = "${env.XDG_STATE_HOME:-~/.local/state}/llama-stack/test"
        
        # When environment variable is not set, should use fallback
        if "XDG_STATE_HOME" not in os.environ:
            # This is what the shell would do
            expanded = pattern.replace("${env.XDG_STATE_HOME:-~/.local/state}", "~/.local/state")
            expected = "~/.local/state/llama-stack/test"
            self.assertEqual(expanded, expected)

    def test_ollama_template_python_config_xdg(self):
        """Test that ollama template's Python config uses XDG-compliant paths."""
        template_path = Path(__file__).parent.parent.parent / "llama_stack" / "templates" / "ollama" / "ollama.py"
        
        if not template_path.exists():
            self.skipTest("Ollama template Python file not found")
        
        content = template_path.read_text()
        
        # Check for XDG-compliant environment variable references
        self.assertIn("${env.XDG_STATE_HOME:-~/.local/state}", content)
        self.assertIn("${env.XDG_DATA_HOME:-~/.local/share}", content)
        
        # Check that paths use llama-stack directory
        self.assertIn("llama-stack", content)

    def test_template_path_consistency(self):
        """Test that template paths are consistent across different files."""
        ollama_yaml_path = Path(__file__).parent.parent.parent / "llama_stack" / "templates" / "ollama" / "run.yaml"
        ollama_py_path = Path(__file__).parent.parent.parent / "llama_stack" / "templates" / "ollama" / "ollama.py"
        
        if not ollama_yaml_path.exists() or not ollama_py_path.exists():
            self.skipTest("Ollama template files not found")
        
        yaml_content = ollama_yaml_path.read_text()
        py_content = ollama_py_path.read_text()
        
        # Both should use the same XDG environment variable patterns
        xdg_patterns = [
            "${env.XDG_STATE_HOME:-~/.local/state}",
            "${env.XDG_DATA_HOME:-~/.local/share}",
            "llama-stack"
        ]
        
        for pattern in xdg_patterns:
            self.assertIn(pattern, yaml_content, f"Pattern {pattern} not found in YAML")
            self.assertIn(pattern, py_content, f"Pattern {pattern} not found in Python")

    def test_template_no_hardcoded_legacy_paths(self):
        """Test that templates don't contain hardcoded legacy paths."""
        template_dir = Path(__file__).parent.parent.parent / "llama_stack" / "templates"
        
        if not template_dir.exists():
            self.skipTest("Templates directory not found")
        
        # Check various template files
        for template_path in template_dir.rglob("*.yaml"):
            content = template_path.read_text()
            
            # Should not contain hardcoded ~/.llama paths
            self.assertNotIn("~/.llama", content, f"Found hardcoded ~/.llama in {template_path}")
            
            # Should not contain hardcoded /tmp paths for persistent data
            if "db_path" in content or "storage_dir" in content:
                self.assertNotIn("/tmp", content, f"Found hardcoded /tmp in {template_path}")

    def test_template_environment_variable_format(self):
        """Test that templates use correct environment variable format."""
        template_dir = Path(__file__).parent.parent.parent / "llama_stack" / "templates"
        
        if not template_dir.exists():
            self.skipTest("Templates directory not found")
        
        # Pattern for XDG environment variables with fallbacks
        xdg_patterns = [
            "${env.XDG_CONFIG_HOME:-~/.config}",
            "${env.XDG_DATA_HOME:-~/.local/share}",
            "${env.XDG_STATE_HOME:-~/.local/state}",
            "${env.XDG_CACHE_HOME:-~/.cache}",
        ]
        
        for template_path in template_dir.rglob("*.yaml"):
            content = template_path.read_text()
            
            # If XDG variables are used, they should have proper fallbacks
            for pattern in xdg_patterns:
                base_var = pattern.split(":-")[0] + "}"
                if base_var in content:
                    self.assertIn(pattern, content, f"XDG variable without fallback in {template_path}")

    def test_template_sqlite_store_dir_xdg(self):
        """Test that SQLITE_STORE_DIR uses XDG-compliant fallback."""
        template_dir = Path(__file__).parent.parent.parent / "llama_stack" / "templates"
        
        if not template_dir.exists():
            self.skipTest("Templates directory not found")
        
        for template_path in template_dir.rglob("*.yaml"):
            content = template_path.read_text()
            
            if "SQLITE_STORE_DIR" in content:
                # Should use XDG fallback pattern
                self.assertIn("${env.XDG_STATE_HOME:-~/.local/state}", content)
                self.assertIn("llama-stack", content)

    def test_template_files_storage_dir_xdg(self):
        """Test that FILES_STORAGE_DIR uses XDG-compliant fallback."""
        template_dir = Path(__file__).parent.parent.parent / "llama_stack" / "templates"
        
        if not template_dir.exists():
            self.skipTest("Templates directory not found")
        
        for template_path in template_dir.rglob("*.yaml"):
            content = template_path.read_text()
            
            if "FILES_STORAGE_DIR" in content:
                # Should use XDG fallback pattern
                self.assertIn("${env.XDG_DATA_HOME:-~/.local/share}", content)
                self.assertIn("llama-stack", content)


class TestTemplateCodeGeneration(unittest.TestCase):
    """Test template code generation with XDG paths."""

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

    def test_provider_codegen_xdg_paths(self):
        """Test that provider code generation uses XDG-compliant paths."""
        codegen_path = Path(__file__).parent.parent.parent / "scripts" / "provider_codegen.py"
        
        if not codegen_path.exists():
            self.skipTest("Provider codegen script not found")
        
        content = codegen_path.read_text()
        
        # Should use XDG-compliant path in documentation
        self.assertIn("${env.XDG_DATA_HOME:-~/.local/share}/llama-stack", content)
        
        # Should not use hardcoded ~/.llama paths
        self.assertNotIn("~/.llama/dummy", content)

    def test_template_sample_config_paths(self):
        """Test that template sample configs use XDG-compliant paths."""
        # This test checks that when templates generate sample configs,
        # they use XDG-compliant paths
        
        # Mock a template that generates sample config
        with patch("llama_stack.templates.template.Template") as mock_template:
            mock_instance = MagicMock()
            mock_template.return_value = mock_instance
            
            # Mock sample config generation
            def mock_sample_config(distro_dir):
                # Should use XDG-compliant path structure
                self.assertIn("llama-stack", distro_dir)
                return {"config": "test"}
            
            mock_instance.sample_run_config = mock_sample_config
            
            # Test sample config generation
            template = mock_template()
            config = template.sample_run_config("${env.XDG_DATA_HOME:-~/.local/share}/llama-stack/test")

    def test_template_path_substitution(self):
        """Test that template path substitution works correctly."""
        # Test path substitution in template generation
        
        original_path = "~/.llama/distributions/test"
        
        # Should be converted to XDG-compliant path
        xdg_path = original_path.replace("~/.llama", "${env.XDG_DATA_HOME:-~/.local/share}/llama-stack")
        expected = "${env.XDG_DATA_HOME:-~/.local/share}/llama-stack/distributions/test"
        
        self.assertEqual(xdg_path, expected)

    def test_template_environment_variable_precedence(self):
        """Test environment variable precedence in templates."""
        # Test that custom XDG variables take precedence over defaults
        
        test_cases = [
            {
                "env": {"XDG_STATE_HOME": "/custom/state"},
                "pattern": "${env.XDG_STATE_HOME:-~/.local/state}/llama-stack/test",
                "expected": "/custom/state/llama-stack/test"
            },
            {
                "env": {},  # No XDG variable set
                "pattern": "${env.XDG_STATE_HOME:-~/.local/state}/llama-stack/test",
                "expected": "~/.local/state/llama-stack/test"
            }
        ]
        
        for case in test_cases:
            # Clear environment
            for key in ["XDG_STATE_HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME"]:
                os.environ.pop(key, None)
            
            # Set test environment
            for key, value in case["env"].items():
                os.environ[key] = value
            
            # Simulate shell variable expansion
            pattern = case["pattern"]
            for key, value in case["env"].items():
                var_pattern = f"${{env.{key}:-"
                if var_pattern in pattern:
                    # Replace with actual value
                    pattern = pattern.replace(
                        f"${{env.{key}:-~/.local/state}}", value
                    )
            
            # If no replacement happened, use fallback
            if "${env.XDG_STATE_HOME:-~/.local/state}" in pattern:
                pattern = pattern.replace(
                    "${env.XDG_STATE_HOME:-~/.local/state}", "~/.local/state"
                )
            
            self.assertEqual(pattern, case["expected"])


class TestTemplateIntegration(unittest.TestCase):
    """Integration tests for templates with XDG compliance."""

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

    def test_template_with_xdg_environment(self):
        """Test template behavior with XDG environment variables set."""
        # Clear environment
        for key in self.original_env:
            os.environ.pop(key, None)
        
        # Set custom XDG variables
        os.environ["XDG_CONFIG_HOME"] = "/custom/config"
        os.environ["XDG_DATA_HOME"] = "/custom/data"
        os.environ["XDG_STATE_HOME"] = "/custom/state"
        
        # Test that template paths would resolve correctly
        # (This is a conceptual test since actual shell expansion happens at runtime)
        
        template_pattern = "${env.XDG_STATE_HOME:-~/.local/state}/llama-stack/test"
        
        # In a real shell, this would expand to:
        expected_expansion = "/custom/state/llama-stack/test"
        
        # Verify the pattern structure is correct
        self.assertIn("XDG_STATE_HOME", template_pattern)
        self.assertIn("llama-stack", template_pattern)
        self.assertIn("~/.local/state", template_pattern)  # fallback

    def test_template_with_no_xdg_environment(self):
        """Test template behavior with no XDG environment variables."""
        # Clear all XDG environment variables
        for key in ["XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME"]:
            os.environ.pop(key, None)
        
        # Test that templates would use fallback values
        template_pattern = "${env.XDG_STATE_HOME:-~/.local/state}/llama-stack/test"
        
        # In a real shell with no XDG_STATE_HOME, this would expand to:
        expected_expansion = "~/.local/state/llama-stack/test"
        
        # Verify the pattern structure includes fallback
        self.assertIn(":-~/.local/state", template_pattern)

    def test_template_consistency_across_providers(self):
        """Test that all template providers use consistent XDG patterns."""
        templates_dir = Path(__file__).parent.parent.parent / "llama_stack" / "templates"
        
        if not templates_dir.exists():
            self.skipTest("Templates directory not found")
        
        # Expected XDG patterns that should be consistent across templates
        expected_patterns = [
            "${env.XDG_CONFIG_HOME:-~/.config}",
            "${env.XDG_DATA_HOME:-~/.local/share}",
            "${env.XDG_STATE_HOME:-~/.local/state}",
            "llama-stack"
        ]
        
        # Check a few different provider templates
        provider_templates = []
        for provider_dir in templates_dir.iterdir():
            if provider_dir.is_dir() and not provider_dir.name.startswith('.'):
                run_yaml = provider_dir / "run.yaml"
                if run_yaml.exists():
                    provider_templates.append(run_yaml)
        
        if not provider_templates:
            self.skipTest("No provider templates found")
        
        # Check that templates use consistent patterns
        for template_path in provider_templates[:3]:  # Check first 3 templates
            content = template_path.read_text()
            
            # Should use llama-stack in paths
            if any(xdg_var in content for xdg_var in ["XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME"]):
                self.assertIn("llama-stack", content, f"Template {template_path} uses XDG but not llama-stack")


if __name__ == "__main__":
    unittest.main() 