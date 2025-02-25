# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from collections import defaultdict
from pathlib import Path

import pytest
from pytest import ExitCode
from pytest_html.basereport import _process_outcome

from llama_stack.models.llama.datatypes import CoreModelId
from llama_stack.models.llama.sku_list import all_registered_models

INFERENCE_APIS = ["chat_completion"]
FUNCTIONALITIES = ["streaming", "structured_output", "tool_calling"]
SUPPORTED_MODELS = {
    "ollama": {
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
    },
    "fireworks": {
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
    },
    "together": {
        CoreModelId.llama3_1_8b_instruct.value,
        CoreModelId.llama3_1_70b_instruct.value,
        CoreModelId.llama3_1_405b_instruct.value,
        CoreModelId.llama3_2_3b_instruct.value,
        CoreModelId.llama3_2_11b_vision_instruct.value,
        CoreModelId.llama3_2_90b_vision_instruct.value,
        CoreModelId.llama3_3_70b_instruct.value,
        CoreModelId.llama_guard_3_8b.value,
        CoreModelId.llama_guard_3_11b_vision.value,
    },
}


class Report:
    def __init__(self, output_path):
        valid_file_format = (
            output_path.split(".")[1] in ["md", "markdown"] if len(output_path.split(".")) == 2 else False
        )
        if not valid_file_format:
            raise ValueError(f"Invalid output file {output_path}. Markdown file is required")
        self.output_path = output_path
        self.test_data = defaultdict(dict)
        self.inference_tests = defaultdict(dict)

    @pytest.hookimpl
    def pytest_runtest_logreport(self, report):
        # This hook is called in several phases, including setup, call and teardown
        # The test is considered failed / error if any of the outcomes is not "Passed"
        outcome = _process_outcome(report)
        data = {
            "outcome": report.outcome,
            "longrepr": report.longrepr,
            "name": report.nodeid,
        }
        if report.nodeid not in self.test_data:
            self.test_data[report.nodeid] = data
        elif self.test_data[report.nodeid] != outcome and outcome != "Passed":
            self.test_data[report.nodeid] = data

    @pytest.hookimpl
    def pytest_sessionfinish(self, session, exitstatus):
        if exitstatus <= ExitCode.INTERRUPTED:
            return
        report = []
        report.append("# Llama Stack Integration Test Results Report")
        report.append("\n## Summary")
        report.append("\n## Supported Models: ")

        header = "| Model Descriptor |"
        dividor = "|:---|"
        for k in SUPPORTED_MODELS.keys():
            header += f"{k} |"
            dividor += ":---:|"

        report.append(header)
        report.append(dividor)

        rows = []
        for model in all_registered_models():
            if "Instruct" not in model.core_model_id.value and "Guard" not in model.core_model_id.value:
                continue
            row = f"| {model.core_model_id.value} |"
            for k in SUPPORTED_MODELS.keys():
                if model.core_model_id.value in SUPPORTED_MODELS[k]:
                    row += " ✅ |"
                else:
                    row += " ❌ |"
            rows.append(row)
        report.extend(rows)

        report.append("\n### Tests:")

        for provider in SUPPORTED_MODELS.keys():
            if provider not in self.inference_tests:
                continue
            report.append(f"\n #### {provider}")
            test_table = [
                "| Area | Model | API | Functionality Test | Status |",
                "|:-----|:-----|:-----|:-----|:-----|",
            ]
            for api in INFERENCE_APIS:
                tests = self.inference_tests[provider][api]
                for test_nodeid in tests:
                    row = "|{area} | {model} | {api} | {test} | {result} ".format(
                        area="Text" if "text" in test_nodeid else "Vision",
                        model=("Llama-3.1-8B-Instruct" if "text" in test_nodeid else "Llama3.2-11B-Vision-Instruct"),
                        api=f"/{api}",
                        test=self.get_simple_function_name(test_nodeid),
                        result=("✅" if self.test_data[test_nodeid]["outcome"] == "passed" else "❌"),
                    )
                    test_table += [row]
            report.extend(test_table)
            report.append("\n")

        output_file = Path(self.output_path)
        output_file.write_text("\n".join(report))
        print(f"\n Report generated: {output_file.absolute()}")

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, session, config, items):
        for item in items:
            inference = item.callspec.params.get("inference_stack")
            if "inference" in item.nodeid:
                func_name = getattr(item, "originalname", item.name)
                for api in INFERENCE_APIS:
                    if api in func_name:
                        api_tests = self.inference_tests[inference].get(api, set())
                        api_tests.add(item.nodeid)
                        self.inference_tests[inference][api] = api_tests

    def get_simple_function_name(self, nodeid):
        """Extract function name from nodeid.

        Examples:
        - 'tests/test_math.py::test_addition' -> 'test_addition'
        - 'tests/test_math.py::TestClass::test_method' -> test_method'
        """
        parts = nodeid.split("::")
        func_name = nodeid  # Fallback to full nodeid if pattern doesn't match
        if len(parts) == 2:  # Simple function
            func_name = parts[1]
        elif len(parts) == 3:  # Class method
            func_name = parts[2]
        return func_name.split("[")[0]
