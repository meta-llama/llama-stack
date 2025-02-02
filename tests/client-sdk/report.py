# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import importlib
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import pytest
from llama_models.datatypes import CoreModelId
from llama_models.sku_list import (
    all_registered_models,
    llama3_1_instruct_models,
    llama3_2_instruct_models,
    llama3_3_instruct_models,
    llama3_instruct_models,
    safety_models,
)

from llama_stack.providers.datatypes import Api
from llama_stack.providers.tests.env import get_env_or_fail

from metadata import API_MAPS

from pytest import CollectReport
from termcolor import cprint


def featured_models():
    models = [
        *llama3_instruct_models(),
        *llama3_1_instruct_models(),
        *llama3_2_instruct_models(),
        *llama3_3_instruct_models(),
        *safety_models(),
    ]
    return {model.huggingface_repo: model for model in models if not model.variant}


SUPPORTED_MODELS = {
    "ollama": set(
        [
            CoreModelId.llama3_1_8b_instruct.value,
            CoreModelId.llama3_1_8b_instruct.value,
            CoreModelId.llama3_1_70b_instruct.value,
            CoreModelId.llama3_1_70b_instruct.value,
            CoreModelId.llama3_1_405b_instruct.value,
            CoreModelId.llama3_1_405b_instruct.value,
            CoreModelId.llama3_2_1b_instruct.value,
            CoreModelId.llama3_2_1b_instruct.value,
            CoreModelId.llama3_2_3b_instruct.value,
            CoreModelId.llama3_2_3b_instruct.value,
            CoreModelId.llama3_2_11b_vision_instruct.value,
            CoreModelId.llama3_2_11b_vision_instruct.value,
            CoreModelId.llama3_2_90b_vision_instruct.value,
            CoreModelId.llama3_2_90b_vision_instruct.value,
            CoreModelId.llama3_3_70b_instruct.value,
            CoreModelId.llama_guard_3_8b.value,
            CoreModelId.llama_guard_3_1b.value,
        ]
    ),
    "tgi": set(
        [
            model.core_model_id.value
            for model in all_registered_models()
            if model.huggingface_repo
        ]
    ),
    "vllm": set(
        [
            model.core_model_id.value
            for model in all_registered_models()
            if model.huggingface_repo
        ]
    ),
}


class Report:
    def __init__(self, report_path: Optional[str] = None):
        if os.environ.get("LLAMA_STACK_CONFIG"):
            config_path_or_template_name = get_env_or_fail("LLAMA_STACK_CONFIG")
            if config_path_or_template_name.endswith(".yaml"):
                config_path = Path(config_path_or_template_name)
            else:
                config_path = Path(
                    importlib.resources.files("llama_stack")
                    / f"templates/{config_path_or_template_name}/run.yaml"
                )
            if not config_path.exists():
                raise ValueError(f"Config file {config_path} does not exist")
            self.output_path = Path(config_path.parent / "report.md")
            self.distro_name = None
        elif os.environ.get("LLAMA_STACK_BASE_URL"):
            url = get_env_or_fail("LLAMA_STACK_BASE_URL")
            self.distro_name = urlparse(url).netloc
            if report_path is None:
                raise ValueError(
                    "Report path must be provided when LLAMA_STACK_BASE_URL is set"
                )
            self.output_path = Path(report_path)
        else:
            raise ValueError("LLAMA_STACK_CONFIG or LLAMA_STACK_BASE_URL must be set")

        self.report_data = defaultdict(dict)
        # test function -> test nodeid
        self.test_data = dict()
        self.test_name_to_nodeid = defaultdict(list)
        self.vision_model_id = None
        self.text_model_id = None
        self.client = None

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_logreport(self, report):
        # This hook is called in several phases, including setup, call and teardown
        # The test is considered failed / error if any of the outcomes is not "Passed"
        outcome = self._process_outcome(report)
        if report.nodeid not in self.test_data:
            self.test_data[report.nodeid] = outcome
        elif self.test_data[report.nodeid] != outcome and outcome != "Passed":
            self.test_data[report.nodeid] = outcome

    def pytest_sessionfinish(self, session):
        report = []
        report.append(f"# Report for {self.distro_name} distribution")
        report.append("\n## Supported Models")

        header = f"| Model Descriptor | {self.distro_name} |"
        dividor = "|:---|:---|"

        report.append(header)
        report.append(dividor)

        rows = []
        if self.distro_name in SUPPORTED_MODELS:
            for model in all_registered_models():
                if (
                    "Instruct" not in model.core_model_id.value
                    and "Guard" not in model.core_model_id.value
                ) or (model.variant):
                    continue
                row = f"| {model.core_model_id.value} |"
                if model.core_model_id.value in SUPPORTED_MODELS[self.distro_name]:
                    row += " ✅ |"
                else:
                    row += " ❌ |"
                rows.append(row)
        else:
            supported_models = {m.identifier for m in self.client.models.list()}
            for hf_name, model in featured_models().items():
                row = f"| {model.core_model_id.value} |"
                if hf_name in supported_models:
                    row += " ✅ |"
                else:
                    row += " ❌ |"
                rows.append(row)
        report.extend(rows)

        report.append("\n## Inference")
        test_table = [
            "| Model | API | Capability | Test | Status |",
            "|:----- |:-----|:-----|:-----|:-----|",
        ]
        for api, capa_map in API_MAPS[Api.inference].items():
            for capa, tests in capa_map.items():
                for test_name in tests:
                    model_id = (
                        self.text_model_id
                        if "text" in test_name
                        else self.vision_model_id
                    )
                    test_nodeids = self.test_name_to_nodeid[test_name]
                    assert len(test_nodeids) > 0

                    # There might be more than one parametrizations for the same test function. We take
                    # the result of the first one for now. Ideally we should mark the test as failed if
                    # any of the parametrizations failed.
                    test_table.append(
                        f"| {model_id} | /{api} | {capa} | {test_name} | {self._print_result_icon(self.test_data[test_nodeids[0]])} |"
                    )

        report.extend(test_table)

        name_map = {Api.vector_io: "Vector IO", Api.agents: "Agents"}
        providers = self.client.providers.list()
        for api_group in [Api.vector_io, Api.agents]:
            api_capitalized = name_map[api_group]
            report.append(f"\n## {api_capitalized}")
            test_table = [
                "| Provider | API | Capability | Test | Status |",
                "|:-----|:-----|:-----|:-----|:-----|",
            ]
            provider = [p for p in providers if p.api == str(api_group.name)]
            provider_str = provider[0].provider_type if provider else ""
            for api, capa_map in API_MAPS[api_group].items():
                for capa, tests in capa_map.items():
                    for test_name in tests:
                        test_nodeids = self.test_name_to_nodeid[test_name]
                        assert len(test_nodeids) > 0
                        test_table.append(
                            f"| {provider_str} | /{api} | {capa} | {test_name} | {self._print_result_icon(self.test_data[test_nodeids[0]])} |"
                        )
            report.extend(test_table)

        output_file = self.output_path
        text = "\n".join(report) + "\n"
        output_file.write_text(text)
        cprint(f"\nReport generated: {output_file.absolute()}", "green")

    def pytest_runtest_makereport(self, item, call):
        func_name = getattr(item, "originalname", item.name)
        self.test_name_to_nodeid[func_name].append(item.nodeid)

        # Get values from fixtures for report output
        if "text_model_id" in item.funcargs:
            text_model = item.funcargs["text_model_id"].split("/")[1]
            self.text_model_id = self.text_model_id or text_model
        elif "vision_model_id" in item.funcargs:
            vision_model = item.funcargs["vision_model_id"].split("/")[1]
            self.vision_model_id = self.vision_model_id or vision_model

        if self.client is None and "llama_stack_client" in item.funcargs:
            self.client = item.funcargs["llama_stack_client"]
            self.distro_name = (
                self.distro_name or self.client.async_client.config.image_name
            )

    def _print_result_icon(self, result):
        if result == "Passed":
            return "✅"
        elif result == "Failed" or result == "Error":
            return "❌"
        else:
            #  result == "Skipped":
            return "⏭️"

    def _process_outcome(self, report: CollectReport):
        if self._is_error(report):
            return "Error"
        if hasattr(report, "wasxfail"):
            if report.outcome in ["passed", "failed"]:
                return "XPassed"
            if report.outcome == "skipped":
                return "XFailed"
        return report.outcome.capitalize()

    def _is_error(self, report: CollectReport):
        return (
            report.when in ["setup", "teardown", "collect"]
            and report.outcome == "failed"
        )
