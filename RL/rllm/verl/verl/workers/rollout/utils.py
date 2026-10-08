# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import ipaddress
import logging
import os
import socket

import uvicorn
from fastapi import FastAPI

logger = logging.getLogger(__file__)


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def get_free_port(address: str) -> tuple[int, socket.socket]:
    family = socket.AF_INET
    if is_valid_ipv6_address(address):
        family = socket.AF_INET6

    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((address, 0))

    port = sock.getsockname()[1]
    return port, sock


async def run_unvicorn(app: FastAPI, server_args, server_address, max_retries=5) -> tuple[int, asyncio.Task]:
    server_port, server_task = None, None

    for i in range(max_retries):
        try:
            server_port, sock = get_free_port(server_address)
            # Keep the kernel-assigned socket reserved and hand that exact
            # socket to uvicorn.  Closing it here and asking uvicorn to bind the
            # numeric port later creates a TOCTOU race: when many TP=1 rollout
            # actors start concurrently, another process can claim the port in
            # that gap and uvicorn exits with EADDRINUSE.  A failed actor then
            # leaves the whole rollout waiting forever at the server barrier.
            app.server_args = server_args
            config = uvicorn.Config(app, host=server_address, port=server_port, log_level="warning")
            server = uvicorn.Server(config)
            # Run the server in the background and WAIT until it is actually
            # listening. The previous code set `should_exit = True` before
            # `await server.serve()`, which made uvicorn exit immediately so the
            # HTTP endpoint never accepted connections -> clients got
            # ClientConnectorError on release_memory_occupation / generate.
            server_task = asyncio.create_task(server.serve(sockets=[sock]))
            # uvicorn sets `server.started = True` once the socket is bound and
            # the app lifespan has started. Poll for it (with a timeout).
            for _ in range(600):  # up to ~60s
                if server_task.done():
                    # serve() returned/raised early -> surface the error and retry
                    exc = server_task.exception()
                    raise OSError(f"uvicorn server task exited early: {exc}")
                if getattr(server, "started", False):
                    break
                await asyncio.sleep(0.1)
            else:
                raise OSError("uvicorn server did not become ready within 60s")
            break
        except (OSError, SystemExit) as e:
            logger.error(f"Failed to start HTTP server on port {server_port} at try {i}, error: {e}")
            if server_task is not None and not server_task.done():
                server.should_exit = True
                try:
                    await server_task
                except Exception:
                    pass
                server_task = None
            # Safe for both pre-start failures and sockets already closed by
            # uvicorn's shutdown path.
            if "sock" in locals():
                sock.close()
    else:
        logger.error(f"Failed to start HTTP server after {max_retries} retries, exiting...")
        os._exit(-1)

    logger.info(f"HTTP server started on port {server_port}")
    return server_port, server_task
