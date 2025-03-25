# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

POST_TRAINING_PROVIDER_TYPES = ["remote::nvidia"]


@pytest.mark.integration
@pytest.fixture(scope="session")
def post_training_provider_available(llama_stack_client):
    providers = llama_stack_client.providers.list()
    post_training_providers = [p for p in providers if p.provider_type in POST_TRAINING_PROVIDER_TYPES]
    return len(post_training_providers) > 0


@pytest.mark.integration
def test_post_training_provider_registration(llama_stack_client, post_training_provider_available):
    """Check if post_training is in the api list.
    This is a sanity check to ensure the provider is registered."""
    if not post_training_provider_available:
        pytest.skip("post training provider not available")

    providers = llama_stack_client.providers.list()
    post_training_providers = [p for p in providers if p.provider_type in POST_TRAINING_PROVIDER_TYPES]
    assert len(post_training_providers) > 0


@pytest.mark.integration
def test_get_training_jobs(llama_stack_client, post_training_provider_available):
    """Test listing all training jobs."""
    if not post_training_provider_available:
        pytest.skip("post training provider not available")

    jobs = llama_stack_client.post_training.get_training_jobs()
    assert isinstance(jobs, dict)
    assert "data" in jobs
    assert isinstance(jobs["data"], list)


@pytest.mark.integration
def test_get_training_job_status(llama_stack_client, post_training_provider_available):
    """Test getting status of a specific training job."""
    if not post_training_provider_available:
        pytest.skip("post training provider not available")

    jobs = llama_stack_client.post_training.get_training_jobs()
    if not jobs["data"]:
        pytest.skip("No training jobs available to check status")

    job_uuid = jobs["data"][0]["job_uuid"]
    job_status = llama_stack_client.post_training.get_training_job_status(job_uuid=job_uuid)

    assert job_status is not None
    assert "job_uuid" in job_status
    assert "status" in job_status
    assert job_status["job_uuid"] == job_uuid
