# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


def pytest_configure(config):
    config.skip_llama_stack_client_instantiation = True
