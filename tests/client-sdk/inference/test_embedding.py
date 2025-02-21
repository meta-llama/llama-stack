# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


#
# Test plan:
#
#  Types of input:
#   - array of a string
#   - array of a image (ImageContentItem, either URL or base64 string)
#   - array of a text (TextContentItem)
#  Types of output:
#   - list of list of floats
#  Params:
#   - text_truncation
#     - absent w/ long text -> error
#     - none w/ long text -> error
#     - absent w/ short text -> ok
#     - none w/ short text -> ok
#     - end w/ long text -> ok
#     - end w/ short text -> ok
#     - start w/ long text -> ok
#     - start w/ short text -> ok
#   - output_dimension
#     - response dimension matches
#   - task_type, only for asymmetric models
#     - query embedding != passage embedding
#  Negative:
#   - long string
#   - long text
#
# Todo:
#  - negative tests
#    - empty
#      - empty list
#      - empty string
#      - empty text
#      - empty image
#    - long
#      - large image
#      - appropriate combinations
#    - batch size
#      - many inputs
#    - invalid
#      - invalid URL
#      - invalid base64
#
# Notes:
#  - use llama_stack_client fixture
#  - use pytest.mark.parametrize when possible
#  - no accuracy tests: only check the type of output, not the content
#

import pytest
from llama_stack_client import BadRequestError
from llama_stack_client.types import EmbeddingsResponse
from llama_stack_client.types.shared.interleaved_content import (
    ImageContentItem,
    ImageContentItemImage,
    ImageContentItemImageURL,
    TextContentItem,
)

DUMMY_STRING = "hello"
DUMMY_STRING2 = "world"
DUMMY_LONG_STRING = "NVDA " * 10240
DUMMY_TEXT = TextContentItem(text=DUMMY_STRING, type="text")
DUMMY_TEXT2 = TextContentItem(text=DUMMY_STRING2, type="text")
DUMMY_LONG_TEXT = TextContentItem(text=DUMMY_LONG_STRING, type="text")
# TODO(mf): add a real image URL and base64 string
DUMMY_IMAGE_URL = ImageContentItem(
    image=ImageContentItemImage(url=ImageContentItemImageURL(uri="https://example.com/image.jpg")), type="image"
)
DUMMY_IMAGE_BASE64 = ImageContentItem(image=ImageContentItemImage(data="base64string"), type="image")


@pytest.mark.parametrize(
    "contents",
    [
        [DUMMY_STRING, DUMMY_STRING2],
        [DUMMY_TEXT, DUMMY_TEXT2],
    ],
    ids=[
        "list[string]",
        "list[text]",
    ],
)
def test_embedding_text(llama_stack_client, embedding_model_id, contents):
    response = llama_stack_client.inference.embeddings(model_id=embedding_model_id, contents=contents)
    assert isinstance(response, EmbeddingsResponse)
    assert len(response.embeddings) == sum(len(content) if isinstance(content, list) else 1 for content in contents)
    assert isinstance(response.embeddings[0], list)
    assert isinstance(response.embeddings[0][0], float)


@pytest.mark.parametrize(
    "contents",
    [
        [DUMMY_IMAGE_URL, DUMMY_IMAGE_BASE64],
        [DUMMY_IMAGE_URL, DUMMY_STRING, DUMMY_IMAGE_BASE64, DUMMY_TEXT],
    ],
    ids=[
        "list[url,base64]",
        "list[url,string,base64,text]",
    ],
)
@pytest.mark.skip(reason="Media is not supported")
def test_embedding_image(llama_stack_client, embedding_model_id, contents):
    response = llama_stack_client.inference.embeddings(model_id=embedding_model_id, contents=contents)
    assert isinstance(response, EmbeddingsResponse)
    assert len(response.embeddings) == sum(len(content) if isinstance(content, list) else 1 for content in contents)
    assert isinstance(response.embeddings[0], list)
    assert isinstance(response.embeddings[0][0], float)


@pytest.mark.parametrize(
    "text_truncation",
    [
        "end",
        "start",
    ],
)
@pytest.mark.parametrize(
    "contents",
    [
        [DUMMY_LONG_TEXT],
        [DUMMY_STRING],
    ],
    ids=[
        "long",
        "short",
    ],
)
def test_embedding_truncation(llama_stack_client, embedding_model_id, text_truncation, contents):
    response = llama_stack_client.inference.embeddings(
        model_id=embedding_model_id, contents=contents, text_truncation=text_truncation
    )
    assert isinstance(response, EmbeddingsResponse)
    assert len(response.embeddings) == 1
    assert isinstance(response.embeddings[0], list)
    assert isinstance(response.embeddings[0][0], float)


@pytest.mark.parametrize(
    "text_truncation",
    [
        None,
        "none",
    ],
)
@pytest.mark.parametrize(
    "contents",
    [
        [DUMMY_LONG_TEXT],
        [DUMMY_LONG_STRING],
    ],
    ids=[
        "long-text",
        "long-str",
    ],
)
def test_embedding_truncation_error(llama_stack_client, embedding_model_id, text_truncation, contents):
    with pytest.raises(BadRequestError) as excinfo:
        llama_stack_client.inference.embeddings(
            model_id=embedding_model_id, contents=[DUMMY_LONG_TEXT], text_truncation=text_truncation
        )


@pytest.mark.xfail(reason="Only valid for model supporting dimension reduction")
def test_embedding_output_dimension(llama_stack_client, embedding_model_id):
    base_response = llama_stack_client.inference.embeddings(model_id=embedding_model_id, contents=[DUMMY_STRING])
    test_response = llama_stack_client.inference.embeddings(
        model_id=embedding_model_id, contents=[DUMMY_STRING], output_dimension=32
    )
    assert len(base_response.embeddings[0]) != len(test_response.embeddings[0])
    assert len(test_response.embeddings[0]) == 32


@pytest.mark.xfail(reason="Only valid for model supporting task type")
def test_embedding_task_type(llama_stack_client, embedding_model_id):
    query_embedding = llama_stack_client.inference.embeddings(
        model_id=embedding_model_id, contents=[DUMMY_STRING], task_type="query"
    )
    document_embedding = llama_stack_client.inference.embeddings(
        model_id=embedding_model_id, contents=[DUMMY_STRING], task_type="doument"
    )
    assert query_embedding.embeddings != document_embedding.embeddings
