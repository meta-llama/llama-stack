# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel

from llama_stack.apis.jobs import (
    JobInfo,
    Jobs,
    ListJobsResponse,
)
from llama_stack.distribution.datatypes import StackRunConfig


class DistributionJobsConfig(BaseModel):
    run_config: StackRunConfig


async def get_provider_impl(config, deps):
    impl = DistributionJobsImpl(config, deps)
    await impl.initialize()
    return impl


class DistributionJobsImpl(Jobs):
    def __init__(self, config, deps):
        self.config = config
        self.deps = deps

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def list_jobs(self) -> ListJobsResponse:
        raise NotImplementedError

    async def delete_job(self, job_id: str) -> None:
        raise NotImplementedError

    async def cancel_job(self, job_id: str) -> None:
        raise NotImplementedError

    async def get_job(self, job_id: str) -> JobInfo:
        raise NotImplementedError
