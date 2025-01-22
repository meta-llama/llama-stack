# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import os
from collections import defaultdict
from pathlib import Path

import pytest
from llama_models.datatypes import CoreModelId
from llama_models.sku_list import all_registered_models

from llama_stack.distribution.library_client import LlamaStackAsLibraryClient
from metadata import API_MAPS

from pytest import CollectReport


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
    "fireworks": set(
        [
            CoreModelId.llama3_1_8b_instruct.value,
            CoreModelId.llama3_1_70b_instruct.value,
            CoreModelId.llama3_1_405b_instruct.value,
            CoreModelId.llama3_2_1b_instruct.value,
            CoreModelId.llama3_2_3b_instruct.value,
            CoreModelId.llama3_2_11b_vision_instruct.value,
            CoreModelId.llama3_2_90b_vision_instruct.value,
            CoreModelId.llama3_3_70b_instruct.value,
            CoreModelId.llama_guard_3_8b.value,
            CoreModelId.llama_guard_3_11b_vision.value,
        ]
    ),
    "together": set(
        [
            CoreModelId.llama3_1_8b_instruct.value,
            CoreModelId.llama3_1_70b_instruct.value,
            CoreModelId.llama3_1_405b_instruct.value,
            CoreModelId.llama3_2_3b_instruct.value,
            CoreModelId.llama3_2_11b_vision_instruct.value,
            CoreModelId.llama3_2_90b_vision_instruct.value,
            CoreModelId.llama3_3_70b_instruct.value,
            CoreModelId.llama_guard_3_8b.value,
            CoreModelId.llama_guard_3_11b_vision.value,
        ]
    ),
}


class Report:

    def __init__(self):
        config_file = os.environ.get("LLAMA_STACK_CONFIG")
        if not config_file:
            raise ValueError(
                "Currently we only support generating report for LlamaStackClientLibrary distributions"
            )
        config_path = Path(config_file)
        self.output_path = Path(config_path.parent / "report.md")
        self.client = LlamaStackAsLibraryClient(
            config_file,
            provider_data=None,
            skip_logger_removal=True,
        )
        self.image_name = self.client.async_client.config.image_name
        self.report_data = defaultdict(dict)
        # test function -> test nodeid
        self.test_data = dict()
        self.test_name_to_nodeid = defaultdict(list)

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
        report.append(f"# Report for {self.image_name} distribution")
        report.append("\n## Supported Models: ")

        header = f"| Model Descriptor | {self.image_name} |"
        dividor = "|:---|:---|"

        report.append(header)
        report.append(dividor)

        rows = []
        for model in all_registered_models():
            if (
                "Instruct" not in model.core_model_id.value
                and "Guard" not in model.core_model_id.value
            ) or (model.variant):
                continue
            row = f"| {model.core_model_id.value} |"
            if model.core_model_id.value in SUPPORTED_MODELS[self.image_name]:
                row += " ✅ |"
            else:
                row += " ❌ |"
            rows.append(row)
        report.extend(rows)

        report.append("\n## Inference: ")
        test_table = [
            "| Model | API | Capability | Test | Status |",
            "|:----- |:-----|:-----|:-----|:-----|",
        ]
        for api, capa_map in API_MAPS["inference"].items():
            for capa, tests in capa_map.items():
                vision_tests = filter(lambda test_name: "image" in test_name, tests)
                text_tests = filter(lambda test_name: "text" in test_name, tests)

                for test_name in text_tests:
                    test_nodeids = self.test_name_to_nodeid[test_name]
                    assert len(test_nodeids) > 0
                    # There might be more than one parametrizations for the same test function. We take
                    # the result of the first one for now. Ideally we should mark the test as failed if
                    # any of the parametrizations failed.
                    test_table.append(
                        f"| Text | /{api} | {capa} | {test_name} | {self._print_result_icon(self.test_data[test_nodeids[0]])} |"
                    )

                for test_name in vision_tests:
                    test_nodeids = self.test_name_to_nodeid[test_name]
                    assert len(test_nodeids) > 0
                    test_table.append(
                        f"| Vision | /{api} | {capa} | {test_name} | {self.test_data[test_nodeids[0]]} |"
                    )

        report.extend(test_table)

        for api_group in ["memory", "agents"]:
            api_capitalized = api_group.capitalize()
            report.append(f"\n## {api_capitalized}: ")
            test_table = [
                "| API | Capability | Test | Status |",
                "|:-----|:-----|:-----|:-----|",
            ]
            for api, capa_map in API_MAPS[api_group].items():
                for capa, tests in capa_map.items():
                    for test_name in tests:
                        test_nodeids = self.test_name_to_nodeid[test_name]
                        assert len(test_nodeids) > 0
                        test_table.append(
                            f"| {api} | {capa} | {test_name} | {self._print_result_icon(self.test_data[test_nodeids[0]])} |"
                        )
            report.extend(test_table)
        output_file = self.output_path
        output_file.write_text("\n".join(report))
        print(f"\nReport generated: {output_file.absolute()}")

    def pytest_runtest_makereport(self, item, call):
        func_name = getattr(item, "originalname", item.name)
        self.test_name_to_nodeid[func_name].append(item.nodeid)

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
