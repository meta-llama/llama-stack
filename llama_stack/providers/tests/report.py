# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from collections import defaultdict
from pathlib import Path

import pytest
from llama_models.datatypes import CoreModelId

from pytest_html.basereport import _process_outcome


INFERNECE_APIS = ["chat_completion"]
FUNCTIONALITIES = ["streaming", "structured_output", "tool_calling"]
SUPPORTED_MODELS = {
    "ollama": [
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
    ],
    "fireworks": [
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
    ],
    "together": [
        CoreModelId.llama3_1_8b_instruct.value,
        CoreModelId.llama3_1_70b_instruct.value,
        CoreModelId.llama3_1_405b_instruct.value,
        CoreModelId.llama3_2_3b_instruct.value,
        CoreModelId.llama3_2_11b_vision_instruct.value,
        CoreModelId.llama3_2_90b_vision_instruct.value,
        CoreModelId.llama3_3_70b_instruct.value,
        CoreModelId.llama_guard_3_8b.value,
        CoreModelId.llama_guard_3_11b_vision.value,
    ],
}


class Report:

    def __init__(self, _config):
        self.report_data = defaultdict(dict)
        self.test_data = dict()
        self.inference_tests = defaultdict(dict)

    @pytest.hookimpl(tryfirst=True)
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
    def pytest_sessionfinish(self, session):
        report = []
        report.append("# Llama Stack Integration Test Results Report")
        report.append("\n## Summary")
        report.append("\n### Inference Providers:")

        for provider, models in SUPPORTED_MODELS.items():
            report.append(f"\n#### {provider}")
            report.append("\n - **Supported models:**")
            report.extend([f"   - {model}" for model in models])
            if provider not in self.inference_tests:
                continue
            report.append("\n - **APIs:**")
            for api in INFERNECE_APIS:
                test_nodeids = self.inference_tests[provider][api]
                report.append(f"\n   - /{api}:")
                report.extend(self._generate_test_result_short(test_nodeids))

            report.append("\n - **Functionality:**")
            for functionality in FUNCTIONALITIES:
                test_nodeids = self.inference_tests[provider][functionality]
                report.append(f"\n   - {functionality}. Test coverage:")
                report.extend(self._generate_test_result_short(test_nodeids))

        output_file = Path("pytest_report.md")
        output_file.write_text("\n".join(report))
        print(f"\n Report generated: {output_file.absolute()}")

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, session, config, items):
        for item in items:
            inference = item.callspec.params.get("inference_stack")
            if "inference" in item.nodeid:
                api, functionality = self._process_function_name(item.nodeid)
                api_tests = self.inference_tests[inference].get(api, set())
                functionality_tests = self.inference_tests[inference].get(
                    functionality, set()
                )
                api_tests.add(item.nodeid)
                functionality_tests.add(item.nodeid)
                self.inference_tests[inference][api] = api_tests
                self.inference_tests[inference][functionality] = functionality_tests

    def _process_function_name(self, function_name):
        api, functionality = None, None
        for val in INFERNECE_APIS:
            if val in function_name:
                api = val
        for val in FUNCTIONALITIES:
            if val in function_name:
                functionality = val
        return api, functionality

    def _print_result_icon(self, result):
        if result == "Passed":
            return "&#x2705;"
        else:
            #  result == "Failed" or result == "Error":
            return "&#x274C;"

    def get_function_name(self, nodeid):
        """Extract function name from nodeid.

        Examples:
        - 'tests/test_math.py::test_addition' -> 'test_addition'
        - 'tests/test_math.py::TestClass::test_method' -> 'TestClass.test_method'
        """
        parts = nodeid.split("::")
        if len(parts) == 2:  # Simple function
            return parts[1]
        elif len(parts) == 3:  # Class method
            return f"{parts[1]}.{parts[2]}"
        return nodeid  # Fallback to full nodeid if pattern doesn't match

    def _generate_test_result_short(self, test_nodeids):
        report = []
        for nodeid in test_nodeids:
            name = self.get_function_name(self.test_data[nodeid]["name"])
            result = self.test_data[nodeid]["outcome"]
            report.append(f"     - {name}. Result: {result}")
        return report
