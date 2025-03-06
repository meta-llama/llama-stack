# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

POST_TRAINING_PROVIDER_TYPES = ["remote::nvidia"]


@pytest.fixture(scope="session")
def post_training_provider_available(llama_stack_client):
    providers = llama_stack_client.providers.list()
    post_training_providers = [p for p in providers if p.provider_type in POST_TRAINING_PROVIDER_TYPES]
    return len(post_training_providers) > 0


def test_post_training_provider_registration(llama_stack_client, post_training_provider_available):
    """Check if post_training is in the api list.
    This is a sanity check to ensure the provider is registered."""
    if not post_training_provider_available:
        pytest.skip("post training provider not available")

    providers = llama_stack_client.providers.list()
    post_training_providers = [p for p in providers if p.provider_type in POST_TRAINING_PROVIDER_TYPES]

    assert len(post_training_providers) > 0

    assert any("post_training" in provider.api for provider in post_training_providers)


def test_list_training_jobs(llama_stack_client, post_training_provider_available):
    """Check if the list_jobs method returns a list of jobs."""
    if not post_training_provider_available:
        pytest.skip("post training provider not available")

    jobs = llama_stack_client.post_training.job.list()

    assert jobs is not None
    assert isinstance(jobs, list)
