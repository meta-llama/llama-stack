# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
import io
from urllib.parse import unquote

import pandas

from llama_models.llama3.api.datatypes import URL

from llama_stack.providers.utils.memory.vector_store import parse_data_url


def get_dataframe_from_url(url: URL):
    df = None
    if url.uri.endswith(".csv"):
        df = pandas.read_csv(url.uri)
    elif url.uri.endswith(".xlsx"):
        df = pandas.read_excel(url.uri)
    elif url.uri.startswith("data:"):
        parts = parse_data_url(url.uri)
        data = parts["data"]
        if parts["is_base64"]:
            data = base64.b64decode(data)
        else:
            data = unquote(data)
            encoding = parts["encoding"] or "utf-8"
            data = data.encode(encoding)

        mime_type = parts["mimetype"]
        mime_category = mime_type.split("/")[0]
        data_bytes = io.BytesIO(data)

        if mime_category == "text":
            df = pandas.read_csv(data_bytes)
        else:
            df = pandas.read_excel(data_bytes)
    else:
        raise ValueError(f"Unsupported file type: {url}")

    return df
