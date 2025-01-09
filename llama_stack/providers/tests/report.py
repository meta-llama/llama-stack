# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from collections import defaultdict

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

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_logreport(self, report):
        # This hook is called in several phases, including setup, call and teardown
        # The test is considered failed / error if any of the outcomes is not "Passed"
        outcome = _process_outcome(report)
        if report.nodeid not in self.test_data:
            self.test_data[report.nodeid] = outcome
        elif self.test_data[report.nodeid] != outcome and outcome != "Passed":
            self.test_data[report.nodeid] = outcome

    def pytest_html_results_summary(self, prefix, summary, postfix):
        prefix.append("<h3> Inference Providers: </h3>")
        for provider in self.report_data.keys():
            prefix.extend(
                [
                    f"<h4> { provider } </h4>",
                    "<ul>",
                    "<li><b> Supported models: </b></li>",
                ]
            )
            supported_models = (
                ["<ul>"]
                + [f"<li> {model} </li>" for model in SUPPORTED_MODELS[provider]]
                + ["</ul>"]
            )

            prefix.extend(supported_models)

            api_section = ["<li><h4> APIs: </h4></li>", "<ul>"]

            for api in INFERNECE_APIS:
                tests = self.report_data[provider].get(api, set())
                api_section.append(f"<li> {api} </li>")
                api_section.append("<ul>")
                for test in tests:
                    result = self.test_data[test]
                    api_section.append(
                        f"<li> test: {test} {self._print_result_icon(result) }  </li>"
                    )
                api_section.append("</ul>")
            api_section.append("</ul>")

            prefix.extend(api_section)

            prefix.append("<li><h4> Model capabilities: </h4> </li>")
            prefix.append("<ul>")
            for functionality in FUNCTIONALITIES:
                tests = self.report_data[provider].get(functionality, set())
                prefix.append(f"<li> <b>{functionality}</b>  </li>")
                prefix.append("<ul>")
                for test in tests:
                    result = self.test_data[test]
                    prefix.append(
                        f"<li> tests: {test} { self._print_result_icon(result) } </li>"
                    )
                prefix.append("</ul>")
            prefix.append("</ul>")
            prefix.append("</ul>")

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_makereport(self, item, call):
        if call.when != "setup":
            return
        # generate the mapping from provider, api/functionality to test nodeid
        provider = item.callspec.params.get("inference_stack")
        if provider is not None:
            api, functionality = self._process_function_name(item.name.split("[")[0])

            api_test_funcs = self.report_data[provider].get(api, set())
            functionality_test_funcs = self.report_data[provider].get(
                functionality, set()
            )
            api_test_funcs.add(item.nodeid)
            functionality_test_funcs.add(item.nodeid)
            self.report_data[provider][api] = api_test_funcs
            self.report_data[provider][functionality] = functionality_test_funcs

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
