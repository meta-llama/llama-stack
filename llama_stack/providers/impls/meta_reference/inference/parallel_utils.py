# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import multiprocessing
import os
import pickle
import tempfile
import time
import uuid

from typing import Callable, Generator

import torch

import zmq

from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_group,
    get_model_parallel_rank,
    get_model_parallel_src_rank,
)

from torch.distributed.launcher.api import elastic_launch, LaunchConfig


_END_SENTINEL = "__end_sentinel__"
_CANCEL_SENTINEL = "__cancel_sentinel__"


def mp_rank_0() -> bool:
    return get_model_parallel_rank() == 0


def retrieve_requests(reply_socket_url: str):
    if mp_rank_0():
        context = zmq.Context()
        reply_socket = context.socket(zmq.ROUTER)
        reply_socket.connect(reply_socket_url)

        while True:
            client_id, obj = maybe_get_work(reply_socket)
            if obj is None:
                time.sleep(0.01)
                continue

            reply_socket.send_multipart([client_id, pickle.dumps("YES READY")])
            break

    def send_obj(obj):
        reply_socket.send_multipart([client_id, pickle.dumps(obj)])

    while True:
        tasks = [None]
        if mp_rank_0():
            client_id, task = maybe_get_work(reply_socket)
            # there is still an unknown unclean GeneratorExit happening resulting in a
            # cancel sentinel getting queued _after_ we have finished sending everything :/
            # kind of a hack this is :/
            if task != _CANCEL_SENTINEL:
                tasks = [task]

        torch.distributed.broadcast_object_list(
            tasks,
            src=get_model_parallel_src_rank(),
            group=get_model_parallel_group(),
        )

        task = tasks[0]
        if task is None:
            time.sleep(0.1)
        else:
            try:
                out = yield task
                if out is None:
                    break

                for obj in out:
                    updates = [None]
                    if mp_rank_0():
                        _, update = maybe_get_work(reply_socket)
                        if update == _CANCEL_SENTINEL:
                            updates = [update]
                        else:
                            # only send the update if it's not cancelled otherwise the object sits in the socket
                            # and gets pulled in the next request lol
                            send_obj(obj)

                    torch.distributed.broadcast_object_list(
                        updates,
                        src=get_model_parallel_src_rank(),
                        group=get_model_parallel_group(),
                    )
                    if updates[0] == _CANCEL_SENTINEL:
                        print("quitting generation loop because request was cancelled")
                        break

                if mp_rank_0():
                    send_obj(_END_SENTINEL)
            except Exception as e:
                print(f"[debug] got exception {e}")
                import traceback

                traceback.print_exc()
                if mp_rank_0():
                    send_obj(e)

    if mp_rank_0():
        send_obj("DONE")


def maybe_get_work(sock: zmq.Socket):
    message = None
    client_id = None
    try:
        client_id, obj = sock.recv_multipart(zmq.NOBLOCK)
        message = pickle.loads(obj)
    except zmq.ZMQError as e:
        if e.errno != zmq.EAGAIN:
            raise e

    return client_id, message


def worker_process_entrypoint(
    reply_socket_url: str,
    init_model_cb: Callable,
) -> None:
    model = init_model_cb()
    torch.distributed.barrier()
    time.sleep(1)

    # run the requests co-routine which retrieves requests from the socket
    # and sends responses (we provide) back to the caller
    req_gen = retrieve_requests(reply_socket_url)
    result = None
    while True:
        try:
            task = req_gen.send(result)
            if isinstance(task, str) and task == _END_SENTINEL:
                break

            result = model(task)
        except StopIteration:
            break

    print("[debug] worker process done")


def launch_dist_group(
    reply_socket_url: str,
    model_parallel_size: int,
    init_model_cb: Callable,
    **kwargs,
) -> None:
    id = uuid.uuid4().hex
    dist_url = f"file:///tmp/llama3_{id}_{time.time()}"

    with tempfile.TemporaryDirectory() as tmpdir:
        # TODO: track workers and if they terminate, tell parent process about it so cleanup can happen
        launch_config = LaunchConfig(
            max_nodes=1,
            min_nodes=1,
            nproc_per_node=model_parallel_size,
            start_method="fork",
            rdzv_backend="c10d",
            rdzv_endpoint=os.path.join(tmpdir, "rdzv"),
            rdzv_configs={"store_type": "file", "timeout": 90},
            max_restarts=0,
            monitor_interval=1,
            run_id=str(uuid.uuid4()),
        )
        elastic_launch(launch_config, entrypoint=worker_process_entrypoint)(
            reply_socket_url,
            init_model_cb,
        )


def start_model_parallel_process(
    model_parallel_size: int,
    init_model_cb: Callable,
    **kwargs,
):
    context = zmq.Context()
    request_socket = context.socket(zmq.DEALER)

    # Binding the request socket to a random port
    request_socket.bind("tcp://127.0.0.1:0")

    main_process_url = request_socket.getsockopt_string(zmq.LAST_ENDPOINT)

    ctx = multiprocessing.get_context("fork")
    process = ctx.Process(
        target=launch_dist_group,
        args=(
            main_process_url,
            model_parallel_size,
            init_model_cb,
        ),
        kwargs=kwargs,
    )
    process.start()

    # wait until the model is loaded; rank 0 will send a message to indicate it's ready

    request_socket.send_pyobj("READY?")
    response = request_socket.recv_pyobj()
    print(f"Finished model load {response}")

    return request_socket, process


class ModelParallelProcessGroup:
    def __init__(
        self,
        model_parallel_size: int,
        init_model_cb: Callable,
        **kwargs,
    ):
        self.model_parallel_size = model_parallel_size
        self.init_model_cb = init_model_cb
        self.started = False
        self.running = False

    def start(self):
        assert not self.started, "process group already started"
        self.request_socket, self.process = start_model_parallel_process(
            self.model_parallel_size,
            self.init_model_cb,
        )
        self.started = True

    def stop(self):
        assert self.started, "process group not started"
        if self.process.is_alive():
            self.request_socket.send_pyobj(_END_SENTINEL, zmq.NOBLOCK)
            self.process.join()
        self.started = False

    def run_inference(self, request) -> Generator:
        assert not self.running, "inference already running"

        self.running = True
        self.request_socket.send_pyobj(request)
        try:
            while True:
                obj = self.request_socket.recv_pyobj()
                if obj == _END_SENTINEL:
                    break

                if isinstance(obj, Exception):
                    print(f"[debug] got exception {obj}")
                    raise obj

                yield obj
        except GeneratorExit as e:
            self.request_socket.send_pyobj(_CANCEL_SENTINEL)
            while True:
                obj = self.request_socket.recv_pyobj()
                if obj == _END_SENTINEL:
                    break
        finally:
            self.running = False
