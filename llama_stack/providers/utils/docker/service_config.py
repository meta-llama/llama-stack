# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class LiteralString(str):
    pass  # Marker class for strings we want to format with >


class DockerComposeServiceConfig(BaseModel):
    """Configuration for a single service in docker-compose."""

    image: str
    volumes: Optional[List[str]] = None
    network_mode: str = "bridge"
    ports: Optional[List[str]] = None
    devices: Optional[List[str]] = None
    environment: Optional[Dict[str, str]] = None
    command: Optional[str] = None
    depends_on: Optional[List[str]] = None
    deploy: Optional[Dict[str, Any]] = None
    runtime: Optional[str] = None
    entrypoint: Optional[str] = None
