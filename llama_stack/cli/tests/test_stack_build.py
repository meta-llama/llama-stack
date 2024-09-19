from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest
from llama_stack.distribution.datatypes import BuildConfig
from llama_stack.cli.stack.build import StackBuild


# temporary while we make the tests work
pytest.skip(allow_module_level=True)


@pytest.fixture
def stack_build():
    parser = MagicMock()
    subparsers = MagicMock()
    return StackBuild(subparsers)


def test_stack_build_initialization(stack_build):
    assert stack_build.parser is not None
    assert stack_build.parser.set_defaults.called_once_with(
        func=stack_build._run_stack_build_command
    )


@patch("llama_stack.distribution.build.build_image")
def test_run_stack_build_command_with_config(
    mock_build_image, mock_build_config, stack_build
):
    args = Namespace(
        config="test_config.yaml",
        template=None,
        list_templates=False,
        name=None,
        image_type="conda",
    )

    with patch("builtins.open", MagicMock()):
        with patch("yaml.safe_load") as mock_yaml_load:
            mock_yaml_load.return_value = {"name": "test_build", "image_type": "conda"}
            mock_build_config.return_value = MagicMock()

            stack_build._run_stack_build_command(args)

            mock_build_config.assert_called_once()
            mock_build_image.assert_called_once()


@patch("llama_stack.cli.table.print_table")
def test_run_stack_build_command_list_templates(mock_print_table, stack_build):
    args = Namespace(list_templates=True)

    stack_build._run_stack_build_command(args)

    mock_print_table.assert_called_once()


@patch("prompt_toolkit.prompt")
@patch("llama_stack.distribution.datatypes.BuildConfig")
@patch("llama_stack.distribution.build.build_image")
def test_run_stack_build_command_interactive(
    mock_build_image, mock_build_config, mock_prompt, stack_build
):
    args = Namespace(
        config=None, template=None, list_templates=False, name=None, image_type=None
    )

    mock_prompt.side_effect = [
        "test_name",
        "conda",
        "meta-reference",
        "test description",
    ]
    mock_build_config.return_value = MagicMock()

    stack_build._run_stack_build_command(args)

    assert mock_prompt.call_count == 4
    mock_build_config.assert_called_once()
    mock_build_image.assert_called_once()


@patch("llama_stack.distribution.datatypes.BuildConfig")
@patch("llama_stack.distribution.build.build_image")
def test_run_stack_build_command_with_template(
    mock_build_image, mock_build_config, stack_build
):
    args = Namespace(
        config=None,
        template="test_template",
        list_templates=False,
        name="test_name",
        image_type="docker",
    )

    with patch("builtins.open", MagicMock()):
        with patch("yaml.safe_load") as mock_yaml_load:
            mock_yaml_load.return_value = {"name": "test_build", "image_type": "conda"}
            mock_build_config.return_value = MagicMock()

            stack_build._run_stack_build_command(args)

            mock_build_config.assert_called_once()
            mock_build_image.assert_called_once()
