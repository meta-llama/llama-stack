# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import patch

import pytest

from llama_stack.cli.llama import (
    LlamaCLIParser,
    main,
)


# Note: Running the unit tests through 'uv' confuses pytest. pytest will use uv's "sys.argv" values, not the ones we intend to test.
class TestMain:
    @patch("sys.argv", ("llama", "stack", "list-providers"))
    def test_run(self):
        main()


# Note: Running the unit tests through 'uv' confuses pytest. pytest will use uv's "sys.argv" values, not the ones we intend to test.
class TestLlamaCLIParser:
    @patch("sys.argv", ("llama", "fake-choice"))
    def test_invalid_choice(self):
        with pytest.raises(SystemExit):
            parser = LlamaCLIParser()
            parser.parse_args()

    @patch("sys.argv", ("llama", "stack", "list-providers"))
    def test_run(self):
        parser = LlamaCLIParser()
        args = parser.parse_args()
        parser.run(args)


# Note: Running the unit tests through 'uv' confuses pytest. pytest will use uv's "sys.argv" values, not the ones we intend to test.
class TestDownloadSubcommand:
    @pytest.mark.parametrize(
        "test_args, expected",
        [
            (  # Download model from meta, using default number of parallel downloads
                (
                    "llama",
                    "download",
                    "--source",
                    "meta",
                    "--model-id",
                    "Llama3.2-1B",
                    "--meta-url",
                    "url-obtained-from-llama.meta.com",
                ),
                {
                    "source": "meta",
                    "model_id": "Llama3.2-1B",
                    "meta_url": "url-obtained-from-llama.meta.com",
                    "max_parallel": 3,
                },
            ),
            (  # Download model from meta, setting --max-parallel=10
                (
                    "llama",
                    "download",
                    "--source",
                    "meta",
                    "--model-id",
                    "Llama3.2-1B",
                    "--meta-url",
                    "url-obtained-from-llama.meta.com",
                    "--max-parallel",
                    "10",
                ),
                {
                    "source": "meta",
                    "model_id": "Llama3.2-1B",
                    "meta_url": "url-obtained-from-llama.meta.com",
                    "max_parallel": 10,
                },
            ),
            (  # Download two models from meta
                (
                    "llama",
                    "download",
                    "--source",
                    "meta",
                    "--model-id",
                    "Llama3.2-1B,Llama3.2-3B",
                    "--meta-url",
                    "url-obtained-from-llama.meta.com",
                ),
                {
                    "source": "meta",
                    "model_id": "Llama3.2-1B,Llama3.2-3B",
                    "meta_url": "url-obtained-from-llama.meta.com",
                    "max_parallel": 3,
                },
            ),
        ],
    )
    def test_download_from_meta_url(self, test_args, expected):
        with patch("sys.argv", test_args):
            parser = LlamaCLIParser()
            args = parser.parse_args()
        assert args.source == expected["source"]
        assert args.model_id == expected["model_id"]
        assert args.meta_url == expected["meta_url"]
        assert args.max_parallel == expected["max_parallel"]

    @pytest.mark.parametrize(
        "test_args, expected",
        [
            (  # Download model from Hugging Face, using default number of parallel downloads
                (
                    "llama",
                    "download",
                    "--source",
                    "huggingface",
                    "--model-id",
                    "Llama3.2-1B",
                    "--hf-token",
                    "fake-hf-token",
                    "--ignore-patterns",
                    "*.something",
                ),
                {
                    "source": "huggingface",
                    "model_id": "Llama3.2-1B",
                    "hf_token": "fake-hf-token",
                    "ignore_patterns": "*.something",
                    "max_parallel": 3,
                },
            ),
            (  # Download model from Hugging Face, setting --max-parallel=10
                (
                    "llama",
                    "download",
                    "--source",
                    "huggingface",
                    "--model-id",
                    "Llama3.2-1B",
                    "--hf-token",
                    "fake-hf-token",
                    "--ignore-patterns",
                    "*.something",
                    "--max-parallel",
                    "10",
                ),
                {
                    "source": "huggingface",
                    "model_id": "Llama3.2-1B",
                    "hf_token": "fake-hf-token",
                    "ignore_patterns": "*.something",
                    "max_parallel": 10,
                },
            ),
        ],
    )
    def test_download_from_hugging_face(self, test_args, expected):
        with patch("sys.argv", test_args):
            parser = LlamaCLIParser()
            args = parser.parse_args()
        assert args.source == expected["source"]
        assert args.model_id == expected["model_id"]
        assert args.hf_token == expected["hf_token"]
        assert args.max_parallel == expected["max_parallel"]
        assert args.ignore_patterns == expected["ignore_patterns"]


class TestVerifyDownloadSubcommand:
    def test_verify_downloaded_model(self):
        test_args = ("llama", "verify-download", "--model-id", "Llama3.2-1B")
        expected_model_id = "Llama3.2-1B"
        with patch("sys.argv", test_args):
            parser = LlamaCLIParser()
            args = parser.parse_args()
        assert args.model_id == expected_model_id


# Note: Running the unit tests through 'uv' confuses pytest. pytest will use uv's "sys.argv" values, not the ones we intend to test.
class TestVerifyStackSubcommand:
    @pytest.mark.parametrize(
        "test_args, expected",
        [
            (  # Verify stack with --image_type=container
                (
                    "llama",
                    "stack",
                    "build",
                    "--config",
                    "./some/path/to/config",
                    "--template",
                    "some-template",
                    "--image-type",
                    "container",
                ),
                {
                    "config": "./some/path/to/config",
                    "template": "some-template",
                    "image_type": "container",
                    "image_name": None,
                },
            ),
            (  # Verify stack with --image_type=conda
                (
                    "llama",
                    "stack",
                    "build",
                    "--config",
                    "./some/path/to/config",
                    "--template",
                    "some-template",
                    "--image-type",
                    "conda",
                    "--image-name",
                    "name-of-conda-env",
                ),
                {
                    "config": "./some/path/to/config",
                    "template": "some-template",
                    "image_type": "conda",
                    "image_name": "name-of-conda-env",
                },
            ),
            (  # Verify stack with --image_type=venv
                (
                    "llama",
                    "stack",
                    "build",
                    "--config",
                    "./some/path/to/config",
                    "--template",
                    "some-template",
                    "--image-type",
                    "venv",
                    "--image-name",
                    "name-of-venv",
                ),
                {
                    "config": "./some/path/to/config",
                    "template": "some-template",
                    "image_type": "venv",
                    "image_name": "name-of-venv",
                },
            ),
        ],
    )
    def test_stack_build_subcommand(self, test_args, expected):
        with patch("sys.argv", test_args):
            parser = LlamaCLIParser()
            args = parser.parse_args()
        assert args.config == expected["config"]
        assert args.template == expected["template"]
        assert args.image_type == expected["image_type"]
        assert args.image_name == expected["image_name"]

    @patch("sys.argv", ("llama", "stack", "list-apis"))
    def test_list_apis_subcommand(self):
        parser = LlamaCLIParser()
        parser.parse_args()

    @patch("sys.argv", ("llama", "stack", "list-providers"))
    def test_list_providers_subcommand(self):
        parser = LlamaCLIParser()
        parser.parse_args()

    @pytest.mark.parametrize(
        "test_args, expected",
        [
            (  # conda: Using default values
                (
                    "llama",
                    "stack",
                    "run",
                    "--config",
                    "./some/path/to/config",
                    "--image-type",
                    "conda",
                    "--image-name",
                    "some-image",
                    "--env",
                    "SOME_KEY=VALUE",
                    "--tls-certfile",
                    "tlscert",
                    "--tls-keyfile",
                    "tlskeyfile",
                ),
                {
                    "config": "./some/path/to/config",
                    "port": 8321,
                    "env": ["SOME_KEY=VALUE"],
                    "image_name": "some-image",
                    "disable_ipv6": False,
                    "image_type": "conda",
                    "tls_certfile": "tlscert",
                    "tls_keyfile": "tlskeyfile",
                },
            ),
            (  # conda: Using custom inputs to override defaults (like: port, env, etc.)
                (
                    "llama",
                    "stack",
                    "run",
                    "--config",
                    "./some/path/to/config",
                    "--image-type",
                    "conda",
                    "--port",
                    "9999",
                    "--env",
                    "SOME_KEY=VALUE",
                    "--disable-ipv6",
                    "--image-name",
                    "some-image",
                    "--tls-certfile",
                    "tlscert",
                    "--tls-keyfile",
                    "tlskeyfile",
                ),
                {
                    "config": "./some/path/to/config",
                    "port": 9999,
                    "env": ["SOME_KEY=VALUE"],
                    "image_name": "some-image",
                    "disable_ipv6": True,
                    "image_type": "conda",
                    "tls_certfile": "tlscert",
                    "tls_keyfile": "tlskeyfile",
                },
            ),
            (  # container: Using default values
                (
                    "llama",
                    "stack",
                    "run",
                    "--config",
                    "./some/path/to/config",
                    "--image-type",
                    "container",
                    "--image-name",
                    "some-image",
                    "--env",
                    "SOME_KEY=VALUE",
                    "--tls-certfile",
                    "tlscert",
                    "--tls-keyfile",
                    "tlskeyfile",
                ),
                {
                    "config": "./some/path/to/config",
                    "port": 8321,
                    "env": ["SOME_KEY=VALUE"],
                    "image_name": "some-image",
                    "disable_ipv6": False,
                    "image_type": "container",
                    "tls_certfile": "tlscert",
                    "tls_keyfile": "tlskeyfile",
                },
            ),
            (  # container: Using custom inputs to override defaults (like: port, env, etc.)
                (
                    "llama",
                    "stack",
                    "run",
                    "--config",
                    "./some/path/to/config",
                    "--image-type",
                    "container",
                    "--port",
                    "9999",
                    "--env",
                    "SOME_KEY=VALUE",
                    "--disable-ipv6",
                    "--image-name",
                    "some-image",
                    "--tls-certfile",
                    "tlscert",
                    "--tls-keyfile",
                    "tlskeyfile",
                ),
                {
                    "config": "./some/path/to/config",
                    "port": 9999,
                    "env": ["SOME_KEY=VALUE"],
                    "image_name": "some-image",
                    "disable_ipv6": True,
                    "image_type": "container",
                    "tls_certfile": "tlscert",
                    "tls_keyfile": "tlskeyfile",
                },
            ),
            (  # venv: Using default values
                (
                    "llama",
                    "stack",
                    "run",
                    "--config",
                    "./some/path/to/config",
                    "--image-type",
                    "venv",
                    "--image-name",
                    "some-image",
                    "--env",
                    "SOME_KEY=VALUE",
                    "--tls-certfile",
                    "tlscert",
                    "--tls-keyfile",
                    "tlskeyfile",
                ),
                {
                    "config": "./some/path/to/config",
                    "port": 8321,
                    "env": ["SOME_KEY=VALUE"],
                    "image_name": "some-image",
                    "disable_ipv6": False,
                    "image_type": "venv",
                    "tls_certfile": "tlscert",
                    "tls_keyfile": "tlskeyfile",
                },
            ),
            (  # venv: Using custom inputs to override defaults (like: port, env, etc.)
                (
                    "llama",
                    "stack",
                    "run",
                    "--config",
                    "./some/path/to/config",
                    "--image-type",
                    "venv",
                    "--port",
                    "9999",
                    "--env",
                    "SOME_KEY=VALUE",
                    "--disable-ipv6",
                    "--image-name",
                    "some-image",
                    "--tls-certfile",
                    "tlscert",
                    "--tls-keyfile",
                    "tlskeyfile",
                ),
                {
                    "config": "./some/path/to/config",
                    "port": 9999,
                    "env": ["SOME_KEY=VALUE"],
                    "image_name": "some-image",
                    "disable_ipv6": True,
                    "image_type": "venv",
                    "tls_certfile": "tlscert",
                    "tls_keyfile": "tlskeyfile",
                },
            ),
        ],
    )
    def test_run_subcommand(self, test_args, expected):
        with patch("sys.argv", test_args):
            parser = LlamaCLIParser()
            args = parser.parse_args()
            assert args.config == expected["config"]
            assert args.port == expected["port"]
            assert args.env == expected["env"]
            assert args.image_type == expected["image_type"]
            assert args.image_name == expected["image_name"]
            assert args.disable_ipv6 == expected["disable_ipv6"]
            assert args.tls_certfile == expected["tls_certfile"]
            assert args.tls_keyfile == expected["tls_keyfile"]


# Note: Running the unit tests through 'uv' confuses pytest. pytest will use uv's "sys.argv" values, not the ones we intend to test.
class TestModelSubcommand:
    def test_model_download_subcommand(self):
        test_args = ("llama", "model", "download")
        with patch("sys.argv", test_args):
            parser = LlamaCLIParser()
            parser.parse_args()

    @pytest.mark.parametrize(
        "test_args, expected",
        [
            (  # Show all models
                (
                    "llama",
                    "model",
                    "list",
                    "--show-all",
                ),
                {
                    "show_all": True,
                    "downloaded": False,
                    "search": None,
                },
            ),
            (  # List downloaded models
                (
                    "llama",
                    "model",
                    "list",
                    "--downloaded",
                ),
                {
                    "show_all": False,
                    "downloaded": True,
                    "search": None,
                },
            ),
            (  # Search downloaded models
                (
                    "llama",
                    "model",
                    "list",
                    "--search",
                    "some-search-str",
                ),
                {
                    "show_all": False,
                    "downloaded": False,
                    "search": "some-search-str",
                },
            ),
        ],
    )
    def test_model_list_subcommand(self, test_args, expected):
        with patch("sys.argv", test_args):
            parser = LlamaCLIParser()
            args = parser.parse_args()
        assert args.show_all == expected["show_all"]
        assert args.downloaded == expected["downloaded"]
        assert args.search == expected["search"]

    @pytest.mark.parametrize(
        "test_args, expected",
        [
            (  # List all available models
                (
                    "llama",
                    "model",
                    "prompt-format",
                    "--list",
                ),
                {
                    "list": True,
                    "model_name": "llama3_1",
                },
            ),
            (  # Describe model prompt format for a model
                (
                    "llama",
                    "model",
                    "prompt-format",
                    "--model-name",
                    "llamaX_Y",
                ),
                {
                    "list": False,
                    "model_name": "llamaX_Y",
                },
            ),
            (  # Describe model prompt format for a model
                (
                    "llama",
                    "model",
                    "prompt-format",
                    "-m",
                    "llamaX_Y",
                ),
                {
                    "list": False,
                    "model_name": "llamaX_Y",
                },
            ),
        ],
    )
    def test_model_prompt_format_subcommand(self, test_args, expected):
        with patch("sys.argv", test_args):
            parser = LlamaCLIParser()
            args = parser.parse_args()
        assert args.list == expected["list"]
        assert args.model_name == expected["model_name"]

    @pytest.mark.parametrize(
        "test_args, expected",
        [
            (  # Describe existing model
                (
                    "llama",
                    "model",
                    "describe",
                    "--model-id",
                    "llamaX_Y",
                ),
                {
                    "model_id": "llamaX_Y",
                },
            ),
            (  # Call subcommand while omitting "--model-id". Should raise an exception because it's a required arg.
                (
                    "llama",
                    "model",
                    "describe",
                ),
                SystemExit,
            ),
        ],
    )
    def test_model_describe_subcommand(self, test_args, expected):
        with patch("sys.argv", test_args):
            if type(expected) is type and issubclass(expected, BaseException):
                with pytest.raises(expected):
                    parser = LlamaCLIParser()
                    args = parser.parse_args()
            else:
                parser = LlamaCLIParser()
                args = parser.parse_args()
                assert args.model_id == expected["model_id"]

    @pytest.mark.parametrize(
        "test_args, expected",
        [
            (  # Describe existing model
                (
                    "llama",
                    "model",
                    "verify-download",
                    "--model-id",
                    "llamaX_Y",
                ),
                None,
            ),
            (  # Call subcommand while omitting "--model-id". Should raise an exception because it's a required arg.
                (
                    "llama",
                    "model",
                    "verify-download",
                ),
                SystemExit,
            ),
        ],
    )
    def test_model_verify_download_subcommand(self, test_args, expected):
        with patch("sys.argv", test_args):
            if type(expected) is type and issubclass(expected, BaseException):
                with pytest.raises(expected):
                    parser = LlamaCLIParser()
                    parser.parse_args()
            else:
                parser = LlamaCLIParser()
                parser.parse_args()

    @pytest.mark.parametrize(
        "test_args, expected",
        [
            (  # Remove an existing model
                (
                    "llama",
                    "model",
                    "remove",
                    "--model",
                    "llamaX_Y",
                ),
                {
                    "model": "llamaX_Y",
                    "force": False,
                },
            ),
            (  # Forcefully remove an existing model
                (
                    "llama",
                    "model",
                    "remove",
                    "--model",
                    "llamaX_Y",
                    "--force",
                ),
                {
                    "model": "llamaX_Y",
                    "force": True,
                },
            ),
            (  # Call subcommand while omitting "--model". Should raise an exception because it's a required arg.
                (
                    "llama",
                    "model",
                    "remove",
                ),
                SystemExit,
            ),
        ],
    )
    def test_model_remove_subcommand(self, test_args, expected):
        with patch("sys.argv", test_args):
            if type(expected) is type and issubclass(expected, BaseException):
                with pytest.raises(expected):
                    parser = LlamaCLIParser()
                    parser.parse_args()
            else:
                parser = LlamaCLIParser()
                args = parser.parse_args()
                assert args.model == expected["model"]
                assert args.force == expected["force"]
