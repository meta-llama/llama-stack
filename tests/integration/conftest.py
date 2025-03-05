# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import os
import textwrap

from dotenv import load_dotenv

from .report import Report


def pytest_configure(config):
    config.option.tbstyle = "short"
    config.option.disable_warnings = True

    load_dotenv()

    # Load any environment variables passed via --env
    env_vars = config.getoption("--env") or []
    for env_var in env_vars:
        key, value = env_var.split("=", 1)
        os.environ[key] = value

    # Note:
    # if report_path is not provided (aka no option --report in the pytest command),
    # it will be set to False
    # if --report will give None ( in this case we infer report_path)
    # if --report /a/b is provided, it will be set to the path provided
    # We want to handle all these cases and hence explicitly check for False
    report_path = config.getoption("--report")
    if report_path is not False:
        config.pluginmanager.register(Report(report_path))


def pytest_addoption(parser):
    parser.addoption(
        "--stack-config",
        help=textwrap.dedent(
            """
            a 'pointer' to the stack. this can be either be:
            (a) a template name like `fireworks`, or
            (b) a path to a run.yaml file, or
            (c) an adhoc config spec, e.g. `inference=fireworks,safety=llama-guard,agents=meta-reference`
            """
        ),
    )
    parser.addoption("--env", action="append", help="Set environment variables, e.g. --env KEY=value")
    parser.addoption(
        "--text-model",
        help="Specify the text model to use for testing. Fixture name: text_model_id",
    )
    parser.addoption(
        "--vision-model",
        help="Specify the vision model to use for testing. Fixture name: vision_model_id",
    )
    parser.addoption(
        "--embedding-model",
        help="Specify the embedding model to use for testing. Fixture name: embedding_model_id",
    )
    parser.addoption(
        "--safety-shield",
        default="meta-llama/Llama-Guard-3-1B",
        help="Specify the safety shield model to use for testing",
    )
    parser.addoption(
        "--judge-model",
        help="Specify the judge model to use for testing. Fixture name: judge_model_id",
    )
    parser.addoption(
        "--embedding-dimension",
        type=int,
        default=384,
        help="Output dimensionality of the embedding model to use for testing",
    )
    parser.addoption(
        "--record-responses",
        action="store_true",
        help="Record new API responses instead of using cached ones.",
    )
    parser.addoption(
        "--report",
        help="Path where the test report should be written, e.g. --report=/path/to/report.md",
    )


MODEL_SHORT_IDS = {
    "meta-llama/Llama-3.2-3B-Instruct": "3B",
    "meta-llama/Llama-3.1-8B-Instruct": "8B",
    "meta-llama/Llama-3.1-70B-Instruct": "70B",
    "meta-llama/Llama-3.1-405B-Instruct": "405B",
    "meta-llama/Llama-3.2-11B-Vision-Instruct": "11B",
    "meta-llama/Llama-3.2-90B-Vision-Instruct": "90B",
    "meta-llama/Llama-3.3-70B-Instruct": "70B",
    "meta-llama/Llama-Guard-3-1B": "Guard1B",
    "meta-llama/Llama-Guard-3-8B": "Guard8B",
    "all-MiniLM-L6-v2": "MiniLM",
}


def get_short_id(value):
    return MODEL_SHORT_IDS.get(value, value)


def pytest_generate_tests(metafunc):
    params = []
    values = []
    id_parts = []

    if "text_model_id" in metafunc.fixturenames:
        params.append("text_model_id")
        val = metafunc.config.getoption("--text-model")
        values.append(val)
        if val:
            id_parts.append(f"txt={get_short_id(val)}")

    if "vision_model_id" in metafunc.fixturenames:
        params.append("vision_model_id")
        val = metafunc.config.getoption("--vision-model")
        values.append(val)
        if val:
            id_parts.append(f"vis={get_short_id(val)}")

    if "embedding_model_id" in metafunc.fixturenames:
        params.append("embedding_model_id")
        val = metafunc.config.getoption("--embedding-model")
        values.append(val)
        if val:
            id_parts.append(f"emb={get_short_id(val)}")

    if "shield_id" in metafunc.fixturenames:
        params.append("shield_id")
        val = metafunc.config.getoption("--safety-shield")
        values.append(val)
        if val:
            id_parts.append(f"shield={get_short_id(val)}")

    if "judge_model_id" in metafunc.fixturenames:
        params.append("judge_model_id")
        val = metafunc.config.getoption("--judge-model")
        values.append(val)
        if val:
            id_parts.append(f"judge={get_short_id(val)}")

    if "embedding_dimension" in metafunc.fixturenames:
        params.append("embedding_dimension")
        val = metafunc.config.getoption("--embedding-dimension")
        values.append(val)
        if val != 384:
            id_parts.append(f"dim={val}")

    if params:
        # Create a single test ID string
        test_id = ":".join(id_parts)
        metafunc.parametrize(params, [values], scope="session", ids=[test_id])


pytest_plugins = ["tests.integration.fixtures.common"]
