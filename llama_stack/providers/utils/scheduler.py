# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import abc
import asyncio
import functools
import multiprocessing
import threading
from collections.abc import Callable, Coroutine, Iterable
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, TypeAlias

from pydantic import BaseModel

from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="scheduler")


# TODO: revisit the list of possible statuses when defining a more coherent
# Jobs API for all API flows; e.g. do we need new vs scheduled?
class JobStatus(Enum):
    new = "new"
    scheduled = "scheduled"
    running = "running"
    failed = "failed"
    completed = "completed"


JobID: TypeAlias = str
JobType: TypeAlias = str


class JobArtifact(BaseModel):
    type: JobType
    name: str
    # TODO: uri should be a reference to /files API; revisit when /files is implemented
    uri: str | None = None
    metadata: dict[str, Any]


JobHandler = Callable[
    [Callable[[str], None], Callable[[JobStatus], None], Callable[[JobArtifact], None]], Coroutine[Any, Any, None]
]


LogMessage: TypeAlias = tuple[datetime, str]


_COMPLETED_STATUSES = {JobStatus.completed, JobStatus.failed}


class Job:
    def __init__(self, job_type: JobType, job_id: JobID, handler: JobHandler | None):
        super().__init__()
        self.id = job_id
        self._type = job_type
        self._handler = handler
        self._artifacts: list[JobArtifact] = []
        self._logs: list[LogMessage] = []
        self._state_transitions: list[tuple[datetime, JobStatus]] = [(datetime.now(timezone.utc), JobStatus.new)]
        self._child_processes: list[multiprocessing.Process] = []
        self._world_size: int = 1  # Number of processes for distributed training
        self.run_args: dict[str, Any] = {}  # Dictionary to store run arguments

    @property
    def world_size(self) -> int:
        return self._world_size

    @world_size.setter
    def world_size(self, size: int) -> None:
        self._world_size = size

    def add_child_process(self, process: multiprocessing.Process) -> None:
        self._child_processes.append(process)

    def cancel(self) -> None:
        """Cancel the job and all its child processes."""
        for process in self._child_processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        self.status = JobStatus.failed

    def cleanup(self) -> None:
        """Clean up any remaining child processes."""
        for process in self._child_processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    @property
    def handler(self) -> JobHandler | None:
        return self._handler

    @property
    def status(self) -> JobStatus:
        return self._state_transitions[-1][1]

    @status.setter
    def status(self, status: JobStatus):
        if status in _COMPLETED_STATUSES and self.status in _COMPLETED_STATUSES:
            raise ValueError(f"Job is already in a completed state ({self.status})")
        if self.status == status:
            return
        self._state_transitions.append((datetime.now(timezone.utc), status))

    @property
    def artifacts(self) -> list[JobArtifact]:
        return self._artifacts

    def register_artifact(self, artifact: JobArtifact) -> None:
        self._artifacts.append(artifact)

    def _find_state_transition_date(self, status: Iterable[JobStatus]) -> datetime | None:
        for date, s in reversed(self._state_transitions):
            if s in status:
                return date
        return None

    @property
    def scheduled_at(self) -> datetime | None:
        return self._find_state_transition_date([JobStatus.scheduled])

    @property
    def started_at(self) -> datetime | None:
        return self._find_state_transition_date([JobStatus.running])

    @property
    def completed_at(self) -> datetime | None:
        return self._find_state_transition_date(_COMPLETED_STATUSES)

    @property
    def logs(self) -> list[LogMessage]:
        return self._logs[:]

    def append_log(self, message: LogMessage) -> None:
        self._logs.append(message)


class _SchedulerBackend(abc.ABC):
    @abc.abstractmethod
    def on_log_message_cb(self, job: Job, message: LogMessage) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def on_status_change_cb(self, job: Job, status: JobStatus) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def on_artifact_collected_cb(self, job: Job, artifact: JobArtifact) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def shutdown(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def schedule(
        self,
        job: Job,
        on_log_message_cb: Callable[[str], None],
        on_status_change_cb: Callable[[JobStatus], None],
        on_artifact_collected_cb: Callable[[JobArtifact], None],
    ) -> None:
        raise NotImplementedError


class _NaiveSchedulerBackend(_SchedulerBackend):
    def __init__(self, timeout: int = 5):
        self._timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

        # When stopping the loop, give tasks a chance to finish
        for task in asyncio.all_tasks(self._loop):
            self._loop.run_until_complete(task)
        self._loop.close()

    async def shutdown(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()

    def schedule(
        self,
        job: Job,
        on_log_message_cb: Callable[[str], None],
        on_status_change_cb: Callable[[JobStatus], None],
        on_artifact_collected_cb: Callable[[JobArtifact], None],
    ) -> None:
        async def do():
            try:
                job.status = JobStatus.running
                await job.handler(on_log_message_cb, on_status_change_cb, on_artifact_collected_cb)
                job.status = JobStatus.completed
            except Exception as e:
                on_log_message_cb(str(e))
                job.status = JobStatus.failed
                logger.exception(f"Job {job.id} failed.")

        asyncio.run_coroutine_threadsafe(do(), self._loop)

    def on_log_message_cb(self, job: Job, message: LogMessage) -> None:
        pass

    def on_status_change_cb(self, job: Job, status: JobStatus) -> None:
        pass

    def on_artifact_collected_cb(self, job: Job, artifact: JobArtifact) -> None:
        pass


class DistributedJobScheduler(_SchedulerBackend):
    """A scheduler backend that supports distributed training jobs.

    This scheduler uses torchrun to handle distributed training process spawning and coordination.
    torchrun automatically handles:
    - Process spawning
    - Environment variable setup
    - Process group initialization
    - Error handling and process cleanup
    """

    def __init__(self, timeout: int = 5):
        self._timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._active_jobs: dict[JobID, asyncio.subprocess.Process] = {}

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

        # When stopping the loop, give tasks a chance to finish
        for task in asyncio.all_tasks(self._loop):
            self._loop.run_until_complete(task)
        self._loop.close()

    async def shutdown(self) -> None:
        # Clean up any remaining processes
        for process in self._active_jobs.values():
            if process.returncode is None:  # Process is still running
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()

        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()

    def schedule(
        self,
        job: Job,
        on_log_message_cb: Callable[[str], None],
        on_status_change_cb: Callable[[JobStatus], None],
        on_artifact_collected_cb: Callable[[JobArtifact], None],
    ) -> None:
        async def do():
            try:
                job.status = JobStatus.running

                # If this is a distributed training job, use torchrun
                if job.world_size > 1:
                    # Find the path to finetune_handler.py
                    from llama_stack.providers.inline.post_training.huggingface import finetune_handler

                    handler_path = Path(finetune_handler.__file__)

                    # Prepare arguments for the handler script
                    args = [
                        "torchrun",
                        f"--nproc_per_node={job.world_size}",
                        "--master_addr=localhost",
                        "--master_port=29500",
                        str(handler_path),
                    ]

                    # Add arguments from the job.run_args dictionary as proper command-line flags
                    for arg_name, arg_value in job.run_args.items():
                        # Skip world_size as we've already handled it
                        if arg_name == "world_size":
                            continue

                        if arg_value is not None:
                            # Handle boolean flags
                            if isinstance(arg_value, bool):
                                if arg_value:
                                    args.append(f"--{arg_name}")
                            else:
                                # For non-boolean values, we add the argument as a separate flag and value
                                args.append(f"--{arg_name}")
                                args.append(str(arg_value))

                    # Launch torchrun using asyncio
                    on_log_message_cb(f"Launching distributed training with {job.world_size} processes")
                    on_log_message_cb(f"Command: {' '.join(args)}")

                    # Make sure we capture stdout and stderr
                    process = await asyncio.create_subprocess_exec(
                        *args,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                    )

                    # Store process for this job
                    self._active_jobs[job.id] = process

                    # Start monitoring in a separate task so we don't block
                    asyncio.create_task(
                        self._monitor_process(job, process, None, on_log_message_cb, on_status_change_cb)
                    )
                else:
                    # For single-device training, call the handler directly if provided
                    if job.handler:
                        await job.handler(on_log_message_cb, on_status_change_cb, on_artifact_collected_cb)
                        job.status = JobStatus.completed
                    else:
                        on_log_message_cb("No handler function provided for single-device training")
                        job.status = JobStatus.failed
            except Exception as e:
                on_log_message_cb(str(e))
                job.status = JobStatus.failed
                logger.exception(f"Job {job.id} failed.")

        asyncio.run_coroutine_threadsafe(do(), self._loop)

    async def _monitor_process(
        self,
        job: Job,
        process: asyncio.subprocess.Process,
        script_path: Path | None,
        on_log_message_cb: Callable[[str], None],
        on_status_change_cb: Callable[[JobStatus], None],
    ) -> None:
        """Monitor a process until completion."""
        try:
            # Stream output from the process if stdout is available
            if process.stdout is not None:
                while True:
                    line = await process.stdout.readline()
                    if not line and process.returncode is not None:
                        break
                    if line:
                        on_log_message_cb(line.decode().strip())
            else:
                # If stdout is not available, just wait for the process to complete
                on_log_message_cb("Process stdout not available, waiting for completion")
                await process.wait()

            # Wait for process to complete if not already done
            if process.returncode is None:
                await process.wait()

            # Check if process failed
            if process.returncode != 0:
                on_log_message_cb(f"Training failed with return code {process.returncode}")
                job.status = JobStatus.failed
            else:
                on_status_change_cb(JobStatus.completed)
                job.status = JobStatus.completed
        except Exception as e:
            on_log_message_cb(f"Error monitoring process: {str(e)}")
            job.status = JobStatus.failed
            logger.exception(f"Error monitoring process for job {job.id}")
        finally:
            # Clean up temporary files
            if script_path and script_path.exists():
                script_path.unlink()

            # Remove from active jobs
            if job.id in self._active_jobs:
                del self._active_jobs[job.id]

    def on_log_message_cb(self, job: Job, message: LogMessage) -> None:
        pass

    def on_status_change_cb(self, job: Job, status: JobStatus) -> None:
        pass

    def on_artifact_collected_cb(self, job: Job, artifact: JobArtifact) -> None:
        pass


_BACKENDS = {
    "naive": _NaiveSchedulerBackend,
    "distributed": DistributedJobScheduler,
}


def _get_backend_impl(backend: str) -> _SchedulerBackend:
    try:
        return _BACKENDS[backend]()
    except KeyError as e:
        raise ValueError(f"Unknown backend {backend}") from e


class Scheduler:
    def __init__(self, backend: str = "naive"):
        # TODO: if server crashes, job states are lost; we need to persist jobs on disc
        self._jobs: dict[JobID, Job] = {}
        self._backend = _get_backend_impl(backend)

    def _on_log_message_cb(self, job: Job, message: str) -> None:
        msg = (datetime.now(timezone.utc), message)
        # At least for the time being, until there's a better way to expose
        # logs to users, log messages on console
        logger.info(f"Job {job.id}: {message}")
        job.append_log(msg)
        self._backend.on_log_message_cb(job, msg)

    def _on_status_change_cb(self, job: Job, status: JobStatus) -> None:
        job.status = status
        self._backend.on_status_change_cb(job, status)

    def _on_artifact_collected_cb(self, job: Job, artifact: JobArtifact) -> None:
        job.register_artifact(artifact)
        self._backend.on_artifact_collected_cb(job, artifact)

    def schedule(self, type_: JobType, job_id: JobID, handler: JobHandler | None, run_params: dict[str, Any]) -> JobID:
        job = Job(type_, job_id, handler)
        if job.id in self._jobs:
            raise ValueError(f"Job {job.id} already exists")

        # Set world size if provided
        if "world_size" in run_params:
            job.world_size = run_params["world_size"]

        # Store all run parameters in the job's run_args dictionary
        job.run_args = run_params

        self._jobs[job.id] = job
        job.status = JobStatus.scheduled
        self._backend.schedule(
            job,
            functools.partial(self._on_log_message_cb, job),
            functools.partial(self._on_status_change_cb, job),
            functools.partial(self._on_artifact_collected_cb, job),
        )

        return job.id

    def cancel(self, job_id: JobID) -> None:
        self.get_job(job_id).cancel()

    def get_job(self, job_id: JobID) -> Job:
        try:
            return self._jobs[job_id]
        except KeyError as e:
            raise ValueError(f"Job {job_id} not found") from e

    def get_jobs(self, type_: JobType | None = None) -> list[Job]:
        jobs = list(self._jobs.values())
        if type_:
            jobs = [job for job in jobs if job._type == type_]
        return jobs

    async def shutdown(self):
        # TODO: also cancel jobs once implemented
        await self._backend.shutdown()
