# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Regression tests for #6829: automatic compaction put the new question before the
summary (so the model's last message was its own summary, producing empty replies on
some models), stored a hidden response the caller never asked for, dropped
instructions/max_output_tokens/extra_body, dropped the summarization call's usage,
checked a stale threshold, could silently corrupt history on an empty summary, and
only ever compacted the latest turn in conversation= mode instead of the whole
conversation.
"""

from unittest.mock import AsyncMock

from ogx.providers.inline.responses.builtin.responses.openai_responses import OpenAIResponsesImpl
from ogx.providers.utils.responses.responses_store import _OpenAIResponseObjectWithInputAndMessages
from ogx_api import OpenAIAssistantMessageParam
from ogx_api.inference import (
    OpenAIChatCompletion,
    OpenAIChatCompletionResponseMessage,
    OpenAIChatCompletionToolCall,
    OpenAIChatCompletionToolCallFunction,
    OpenAIChatCompletionUsage,
    OpenAIChoice,
)
from ogx_api.openai_responses import (
    OpenAIResponseCompaction,
    OpenAIResponseInputFunctionToolCallOutput,
    OpenAIResponseInputMessageContentText,
    OpenAIResponseMessage,
    OpenAIResponseOutputMessageContentOutputText,
    OpenAIResponseOutputMessageFunctionToolCall,
    OpenAIResponseText,
    OpenAIResponseTextFormat,
)
from ogx_api.responses.models import CreateResponseRequest
from tests.unit.providers.responses.builtin.test_openai_responses_helpers import fake_stream


def _compaction_completion(summary: str | None, prompt_tokens: int = 50, completion_tokens: int = 20):
    """A non-streaming OpenAIChatCompletion, matching the shape _build_compaction's
    summarization call (stream=False) expects back."""
    return OpenAIChatCompletion(
        id="chatcmpl-compact",
        choices=[
            OpenAIChoice(
                index=0,
                finish_reason="stop",
                message=OpenAIChatCompletionResponseMessage(role="assistant", content=summary),
            )
        ],
        created=0,
        model="test-model",
        usage=OpenAIChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _dual_inference_mock(summary: str | None = "Summary of prior turns.", **usage_kwargs):
    """Route stream=False calls (the summarization call) to a canned completion, and
    stream=True calls (the main turn) to fake_stream() -- the two calls a compacting
    turn makes are otherwise indistinguishable to a single mocked return_value.
    """

    async def side_effect(params):
        if params.stream:
            return fake_stream()
        return _compaction_completion(summary, **usage_kwargs)

    return AsyncMock(side_effect=side_effect)


def _previous_response(
    *,
    input_text: str = "What is 2+2?",
    output_text: str = "2+2 equals 4.",
    usage: OpenAIChatCompletionUsage | None = None,
) -> _OpenAIResponseObjectWithInputAndMessages:
    from ogx_api import OpenAIAssistantMessageParam, OpenAIUserMessageParam
    from ogx_api.openai_responses import OpenAIResponseUsage

    return _OpenAIResponseObjectWithInputAndMessages(
        id="resp-previous-123",
        object="response",
        created_at=1234567890,
        model="meta-llama/Llama-3.1-8B-Instruct",
        status="completed",
        text=OpenAIResponseText(format=OpenAIResponseTextFormat(type="text")),
        input=[
            OpenAIResponseMessage(
                id="msg-prev-user",
                role="user",
                content=[OpenAIResponseInputMessageContentText(text=input_text)],
            )
        ],
        output=[
            OpenAIResponseMessage(
                id="msg-prev-assistant",
                role="assistant",
                content=[OpenAIResponseOutputMessageContentOutputText(text=output_text)],
            )
        ],
        messages=[
            OpenAIUserMessageParam(content=input_text),
            OpenAIAssistantMessageParam(content=output_text),
        ],
        usage=OpenAIResponseUsage(
            input_tokens=5,
            output_tokens=5,
            total_tokens=10,
            input_tokens_details={"cached_tokens": 0},
            output_tokens_details={"reasoning_tokens": 0},
        )
        if usage is None
        else usage,
        store=True,
    )


class TestMaybeAutoCompactOrdering:
    """Point 1: the new input must always end up after the compaction item, never
    before it -- a compaction item renders as an assistant message
    (utils.py's compaction -> OpenAIAssistantMessageParam conversion), so putting the
    new question before it makes the model's own summary its last message.
    """

    async def _make_impl(self, mock_inference_api):
        impl = object.__new__(OpenAIResponsesImpl)
        impl.inference_api = mock_inference_api
        impl.files_api = AsyncMock()
        from ogx.providers.inline.responses.builtin.config import CompactionConfig

        impl.compaction_config = CompactionConfig()
        return impl

    async def test_new_input_follows_the_compaction_item(self, mock_inference_api):
        impl = await self._make_impl(mock_inference_api)
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        history = [
            OpenAIResponseMessage(role="user", content="Turn one."),
            OpenAIResponseMessage(role="assistant", content="Reply one."),
        ]
        new_input = [OpenAIResponseMessage(role="user", content="What is the code word?")]

        effective_input, compaction_item, chat_usage = await impl._maybe_auto_compact(
            history, new_input, "test-model", [{"type": "compaction", "compact_threshold": 1}]
        )

        assert compaction_item is not None
        assert chat_usage is not None
        assert effective_input[-1] is new_input[0]
        assert isinstance(effective_input[-2], OpenAIResponseCompaction)

    async def test_below_threshold_leaves_history_and_new_input_untouched(self, mock_inference_api):
        impl = await self._make_impl(mock_inference_api)
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        history = [OpenAIResponseMessage(role="user", content="hi")]
        new_input = [OpenAIResponseMessage(role="user", content="again")]

        effective_input, compaction_item, chat_usage = await impl._maybe_auto_compact(
            history, new_input, "test-model", [{"type": "compaction", "compact_threshold": 100000}]
        )

        assert compaction_item is None
        assert chat_usage is None
        assert effective_input == history + new_input
        mock_inference_api.openai_chat_completion.assert_not_called()

    async def test_empty_history_short_circuits_without_calling_inference(self, mock_inference_api):
        impl = await self._make_impl(mock_inference_api)
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        new_input = [OpenAIResponseMessage(role="user", content="hello")]
        effective_input, compaction_item, chat_usage = await impl._maybe_auto_compact(
            [], new_input, "test-model", [{"type": "compaction", "compact_threshold": 1}]
        )

        assert compaction_item is None
        assert effective_input == new_input
        mock_inference_api.openai_chat_completion.assert_not_called()

    async def test_empty_summary_keeps_original_history(self, mock_inference_api):
        """Point 6: a reasoning model hitting max_output_tokens mid-thought can return
        content=None. Must not silently replace history with a hollow/empty summary.
        """
        impl = await self._make_impl(mock_inference_api)
        mock_inference_api.openai_chat_completion = _dual_inference_mock(summary=None)

        history = [OpenAIResponseMessage(role="user", content="hi")]
        new_input = [OpenAIResponseMessage(role="user", content="again")]

        effective_input, compaction_item, chat_usage = await impl._maybe_auto_compact(
            history, new_input, "test-model", [{"type": "compaction", "compact_threshold": 1}]
        )

        assert compaction_item is None
        assert chat_usage is None
        assert effective_input == history + new_input


class TestAutoCompactionEndToEnd:
    """End-to-end tests driving the real create_openai_response path (not mocking
    _maybe_auto_compact), so these exercise the full wiring: call-site splitting,
    orchestrator seeding, and storage.
    """

    async def test_no_hidden_response_is_ever_stored(
        self, openai_responses_impl, mock_inference_api, mock_responses_store
    ):
        """Points 2 & 3: the automatic path must never create the kind of orphaned,
        store-ignoring response compact_openai_response creates for the manual endpoint.
        """
        mock_responses_store.get_response_object.return_value = _previous_response()
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="What is the code word?",
                model="meta-llama/Llama-3.1-8B-Instruct",
                previous_response_id="resp-previous-123",
                context_management=[{"type": "compaction", "compact_threshold": 1}],
                store=False,
            )
        )

        mock_responses_store.store_response_object.assert_not_called()

    async def test_instructions_and_extra_body_reach_the_summarization_call(
        self, openai_responses_impl, mock_inference_api, mock_responses_store
    ):
        """Point 3 (plus the extra_body point): these were all silently dropped by the
        automatic path before this fix.
        """
        mock_responses_store.get_response_object.return_value = _previous_response()
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="What is the code word?",
                model="meta-llama/Llama-3.1-8B-Instruct",
                previous_response_id="resp-previous-123",
                instructions="Be terse.",
                context_management=[{"type": "compaction", "compact_threshold": 1}],
                extra_body={"cache_salt": "abc123"},
            )
        )

        compaction_call = next(
            call for call in mock_inference_api.openai_chat_completion.call_args_list if not call.args[0].stream
        )
        params = compaction_call.args[0]
        assert params.cache_salt == "abc123"
        assert any("Be terse." in str(m.content) for m in params.messages)

    async def test_the_answers_max_output_tokens_never_caps_the_summarization_call(
        self, openai_responses_impl, mock_inference_api, mock_responses_store
    ):
        """Review of #6829: the summarization call was capped by the *answer's own*
        max_output_tokens (openai_responses.py's old _maybe_auto_compact signature took
        it straight from request.max_output_tokens), so a short max_output_tokens could
        let the summary alone exhaust it, leaving nothing for the actual answer. The
        summary must get its own independent cap (CompactionConfig.summary_max_output_tokens).
        """
        mock_responses_store.get_response_object.return_value = _previous_response()
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="What is the code word?",
                model="meta-llama/Llama-3.1-8B-Instruct",
                previous_response_id="resp-previous-123",
                max_output_tokens=50,
                context_management=[{"type": "compaction", "compact_threshold": 1}],
            )
        )

        compaction_call = next(
            call for call in mock_inference_api.openai_chat_completion.call_args_list if not call.args[0].stream
        )
        assert (
            compaction_call.args[0].max_completion_tokens
            == openai_responses_impl.compaction_config.summary_max_output_tokens
        )
        assert compaction_call.args[0].max_completion_tokens != 50

    async def test_summarization_usage_does_not_reduce_the_answers_own_output_budget(
        self, openai_responses_impl, mock_inference_api, mock_responses_store
    ):
        """The summarization call's completion tokens must still show up in the final
        usage (test_compaction_usage_is_folded_into_the_final_response_usage covers
        that), but must not count against max_output_tokens' budget for the main answer
        -- otherwise a response can come back incomplete with only the compaction item
        and no answer, purely because the summary itself used up the whole budget.
        """
        mock_responses_store.get_response_object.return_value = _previous_response()
        # A summary that alone would exhaust a small max_output_tokens if it counted
        # against the answer's budget.
        mock_inference_api.openai_chat_completion = _dual_inference_mock(completion_tokens=40)

        result = await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="What is the code word?",
                model="meta-llama/Llama-3.1-8B-Instruct",
                previous_response_id="resp-previous-123",
                max_output_tokens=50,
                context_management=[{"type": "compaction", "compact_threshold": 1}],
            )
        )

        assert result.status == "completed"
        assert result.incomplete_details is None
        main_call = next(
            call for call in mock_inference_api.openai_chat_completion.call_args_list if call.args[0].stream
        )
        # The main call must still see the full max_output_tokens budget, not
        # max_output_tokens minus the summary's own completion tokens.
        assert main_call.args[0].max_completion_tokens == 50

    async def test_compaction_usage_is_folded_into_the_final_response_usage(
        self, openai_responses_impl, mock_inference_api, mock_responses_store
    ):
        """Point 4: the summarization call's tokens must show up somewhere, not be
        silently dropped while still being billed by the provider.
        """
        mock_responses_store.get_response_object.return_value = _previous_response()
        mock_inference_api.openai_chat_completion = _dual_inference_mock(prompt_tokens=100, completion_tokens=50)

        result = await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="What is the code word?",
                model="meta-llama/Llama-3.1-8B-Instruct",
                previous_response_id="resp-previous-123",
                context_management=[{"type": "compaction", "compact_threshold": 1}],
            )
        )

        assert result.usage is not None
        # 150 from the compaction call, plus whatever the main turn's fake_stream usage is.
        assert result.usage.input_tokens >= 100
        assert result.usage.output_tokens >= 50

    async def test_threshold_checks_current_context_not_stale_previous_usage(
        self, openai_responses_impl, mock_inference_api, mock_responses_store
    ):
        """Point 5: previous_usage.total_tokens reflects the turn *before* the new
        question existed. A request that's only over threshold once the new question is
        counted must still trigger compaction.
        """
        # previous_usage.total_tokens (10) is comfortably under the threshold (50); only
        # counting the actual current text (a long history + new question) exceeds it.
        long_text = "word " * 100
        previous_response = _previous_response(input_text=long_text, output_text=long_text)
        mock_responses_store.get_response_object.return_value = previous_response
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="What is the code word?",
                model="meta-llama/Llama-3.1-8B-Instruct",
                previous_response_id="resp-previous-123",
                context_management=[{"type": "compaction", "compact_threshold": 50}],
            )
        )

        compaction_calls = [
            call for call in mock_inference_api.openai_chat_completion.call_args_list if not call.args[0].stream
        ]
        assert len(compaction_calls) == 1

    async def test_compaction_item_is_the_first_output_item(
        self, openai_responses_impl, mock_inference_api, mock_responses_store
    ):
        """Point 9: OpenAI returns the compaction item in output; it must not stay
        server-only state the client never sees.
        """
        mock_responses_store.get_response_object.return_value = _previous_response()
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        result = await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="What is the code word?",
                model="meta-llama/Llama-3.1-8B-Instruct",
                previous_response_id="resp-previous-123",
                context_management=[{"type": "compaction", "compact_threshold": 1}],
            )
        )

        assert len(result.output) >= 1
        assert isinstance(result.output[0], OpenAIResponseCompaction)

    async def test_fresh_single_call_with_large_input_still_compacts(
        self, openai_responses_impl, mock_inference_api, mock_responses_store
    ):
        """No previous_response_id or conversation at all -- a single oversized first
        call must still split its own trailing turn off and compact everything before
        it (mirrors the integration test test_context_management_auto_compacts_large_input).
        """
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        large_input = []
        for i in range(10):
            large_input.append(OpenAIResponseMessage(role="user", content=f"Tell me about topic {i}."))
            large_input.append(OpenAIResponseMessage(role="assistant", content=f"Here is topic {i}. " * 20))
        large_input.append(OpenAIResponseMessage(role="user", content="Summarize what we discussed."))

        result = await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input=large_input,
                model="meta-llama/Llama-3.1-8B-Instruct",
                context_management=[{"type": "compaction", "compact_threshold": 50}],
            )
        )

        assert isinstance(result.output[0], OpenAIResponseCompaction)
        compaction_calls = [
            call for call in mock_inference_api.openai_chat_completion.call_args_list if not call.args[0].stream
        ]
        assert len(compaction_calls) == 1
        # The final question must still be the last message sent to the model for the
        # main turn, not swallowed into the compacted summary.
        main_call = next(
            call for call in mock_inference_api.openai_chat_completion.call_args_list if call.args[0].stream
        )
        assert main_call.args[0].messages[-1].content == "Summarize what we discussed."


class TestConversationModeCompaction:
    """Point 8: conversation= mode must compact the whole conversation, not just the
    newly submitted turn.
    """

    def _conversation_pages(self, total_items: int, page_size: int = 100):
        """Build the sequence of ConversationItemList pages _fetch_full_conversation_items
        will walk through -- more than one page when total_items exceeds page_size, so the
        pagination loop itself is exercised, not just a single-page call.
        """
        from ogx_api.conversations.models import ConversationItemList

        items = [OpenAIResponseMessage(id=f"item-{i}", role="user", content=f"Message {i}") for i in range(total_items)]
        pages = []
        for start in range(0, total_items, page_size):
            chunk = items[start : start + page_size]
            has_more = start + page_size < total_items
            pages.append(
                ConversationItemList(
                    data=chunk,
                    first_id=chunk[0].id if chunk else None,
                    last_id=chunk[-1].id if chunk else None,
                    has_more=has_more,
                )
            )
        return pages

    async def test_full_paginated_conversation_is_compacted_not_just_the_new_turn(
        self, openai_responses_impl, mock_inference_api, mock_responses_store, mock_conversations_api
    ):
        from ogx_api import OpenAIUserMessageParam

        pages = self._conversation_pages(total_items=25, page_size=10)
        mock_conversations_api.list_items = AsyncMock(side_effect=pages)
        mock_responses_store.get_conversation_messages.return_value = [
            OpenAIUserMessageParam(content=f"Message {i}") for i in range(25)
        ]
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="What did we discuss?",
                model="meta-llama/Llama-3.1-8B-Instruct",
                conversation="conv_272fa1af1b67b662ee7ef05f72ec300d92425257645a531a",
                context_management=[{"type": "compaction", "compact_threshold": 1}],
            )
        )

        # All 3 pages (10+10+5) must have been walked, not just the first.
        assert mock_conversations_api.list_items.call_count == 3
        compaction_call = next(
            call for call in mock_inference_api.openai_chat_completion.call_args_list if not call.args[0].stream
        )
        # The summarization call's messages must reflect the full 25-item history, not
        # just the single new turn.
        assert len(compaction_call.args[0].messages) > 2

    async def test_non_compacting_conversation_turn_still_stores_only_the_new_turn(
        self, openai_responses_impl, mock_inference_api, mock_responses_store, mock_conversations_api
    ):
        """A turn where compaction doesn't fire must keep conversation= mode's existing
        O(1)-per-response storage behavior -- this fix must not make every conversation
        turn snapshot the full history.
        """
        from ogx_api import OpenAIUserMessageParam

        pages = self._conversation_pages(total_items=5, page_size=10)
        mock_conversations_api.list_items = AsyncMock(side_effect=pages)
        mock_responses_store.get_conversation_messages.return_value = [
            OpenAIUserMessageParam(content=f"Message {i}") for i in range(5)
        ]
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="A new question.",
                model="meta-llama/Llama-3.1-8B-Instruct",
                conversation="conv_272fa1af1b67b662ee7ef05f72ec300d92425257645a531a",
                context_management=[{"type": "compaction", "compact_threshold": 100000}],
            )
        )

        store_call_args = mock_responses_store.upsert_response_object.call_args
        stored_input = store_call_args.kwargs["input"]
        assert len(stored_input) == 1
        assert stored_input[0].content[0].text == "A new question."


class TestManualCompactEndpoint:
    """The manual /v1/responses/compact endpoint's own dead-parameter bugs (point 3's
    "unlike the manual endpoint" claim didn't hold up under inspection -- it drops
    extra_body/max_output_tokens too) and its empty-summary handling, which differs
    deliberately from the automatic path: the caller explicitly asked to compact here,
    so a hollow result is surfaced as an error rather than silently kept as-is.
    """

    async def test_extra_body_and_max_output_tokens_reach_the_summarization_call(
        self, openai_responses_impl, mock_inference_api
    ):
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        await openai_responses_impl.compact_openai_response(
            model="test-model",
            input="Some content to compact.",
            extra_body={"max_output_tokens": 77, "cache_salt": "xyz"},
        )

        params = mock_inference_api.openai_chat_completion.call_args.args[0]
        assert params.max_completion_tokens == 77
        assert params.cache_salt == "xyz"

    async def test_empty_summary_raises(self, openai_responses_impl, mock_inference_api):
        mock_inference_api.openai_chat_completion = _dual_inference_mock(summary=None)

        try:
            await openai_responses_impl.compact_openai_response(model="test-model", input="Some content.")
        except ValueError as e:
            assert "empty summary" in str(e)
        else:
            raise AssertionError("expected ValueError for an empty summary")


class TestOpenToolCallSurvivesCompaction:
    """Code review of #6838: if the new input answers a function_call still sitting in
    history (a tool loop's pending call), _build_compaction's "keep only user messages
    verbatim" rule dropped that function_call -- leaving the incoming
    function_call_output with no corresponding function_call anywhere in the final
    input, which convert_response_input_to_chat_messages rejects with a 400-worthy
    ValueError. The open call must survive compaction, placed right after the
    compaction item and before the input answering it.
    """

    async def test_pending_function_call_survives_compaction_with_previous_response_id(
        self, openai_responses_impl, mock_inference_api, mock_responses_store
    ):
        previous_response = _previous_response()
        previous_response.output = [
            *previous_response.output,
            OpenAIResponseOutputMessageFunctionToolCall(
                call_id="call_abc123", name="get_weather", arguments='{"city": "Paris"}'
            ),
        ]
        # _process_input_with_previous_response always builds a "default" messages
        # list from previous_response.messages before compaction is even considered
        # (it's discarded and rebuilt from the compacted input if compaction fires,
        # but must not raise before getting there) -- it needs the matching tool-call
        # message too, mirroring the function_call item added to .output above.
        previous_response.messages = [
            *previous_response.messages,
            OpenAIAssistantMessageParam(
                tool_calls=[
                    OpenAIChatCompletionToolCall(
                        id="call_abc123",
                        function=OpenAIChatCompletionToolCallFunction(
                            name="get_weather", arguments='{"city": "Paris"}'
                        ),
                    )
                ]
            ),
        ]
        mock_responses_store.get_response_object.return_value = previous_response
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        result = await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input=[OpenAIResponseInputFunctionToolCallOutput(call_id="call_abc123", output="72F and sunny")],
                model="meta-llama/Llama-3.1-8B-Instruct",
                previous_response_id="resp-previous-123",
                context_management=[{"type": "compaction", "compact_threshold": 1}],
            )
        )

        assert result.status == "completed"
        main_call = next(
            call for call in mock_inference_api.openai_chat_completion.call_args_list if call.args[0].stream
        )
        messages = main_call.args[0].messages
        tool_call_message = next(m for m in messages if getattr(m, "tool_calls", None))
        assert tool_call_message.tool_calls[0].id == "call_abc123"
        tool_result_message = next(m for m in messages if getattr(m, "tool_call_id", None) == "call_abc123")
        assert tool_result_message.content == "72F and sunny"

    async def test_pending_function_call_survives_compaction_in_a_single_oversized_call(
        self, openai_responses_impl, mock_inference_api
    ):
        """Same bug, but via the other path that can separate a function_call from its
        output: a single oversized first call with no previous_response_id/conversation,
        where the call-site split (everything but the last item becomes compactable
        history) happens to cut right between the call and its result.
        """
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        large_input: list = [OpenAIResponseMessage(role="user", content=f"Tell me about topic {i}.") for i in range(5)]
        large_input.append(
            OpenAIResponseOutputMessageFunctionToolCall(
                call_id="call_xyz789", name="get_weather", arguments='{"city": "Paris"}'
            )
        )
        large_input.append(OpenAIResponseInputFunctionToolCallOutput(call_id="call_xyz789", output="72F and sunny"))

        result = await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input=large_input,
                model="meta-llama/Llama-3.1-8B-Instruct",
                context_management=[{"type": "compaction", "compact_threshold": 1}],
            )
        )

        assert result.status == "completed"
        main_call = next(
            call for call in mock_inference_api.openai_chat_completion.call_args_list if call.args[0].stream
        )
        tool_result_message = next(
            m for m in main_call.args[0].messages if getattr(m, "tool_call_id", None) == "call_xyz789"
        )
        assert tool_result_message.content == "72F and sunny"


class TestConversationModeCompactionStaysBounded:
    """Code review of #6838: conversation= mode's raw conversation item log
    (_fetch_full_conversation_items) never shrinks after a compaction -- compaction
    items are deliberately excluded from it (_sync_response_to_conversation) -- so
    using it as the compaction input/threshold source on every turn re-summarizes an
    ever-growing, uncompacted transcript every single turn after the first
    compaction, until it eventually exceeds the model's own context and the
    summarization call itself fails (it has no truncation). Compacting from the
    stored chat-message log instead (which already reflects any earlier compaction)
    keeps each compaction's input bounded to what's new since the last one.
    """

    def _conversation_pages(self, total_items: int, page_size: int = 100):
        from ogx_api.conversations.models import ConversationItemList

        items = [OpenAIResponseMessage(id=f"item-{i}", role="user", content=f"Message {i}") for i in range(total_items)]
        pages = []
        for start in range(0, total_items, page_size):
            chunk = items[start : start + page_size]
            has_more = start + page_size < total_items
            pages.append(
                ConversationItemList(
                    data=chunk,
                    first_id=chunk[0].id if chunk else None,
                    last_id=chunk[-1].id if chunk else None,
                    has_more=has_more,
                )
            )
        return pages

    async def test_recompaction_uses_the_stored_message_log_not_the_ever_growing_raw_items(
        self, openai_responses_impl, mock_inference_api, mock_responses_store, mock_conversations_api
    ):
        from ogx_api import OpenAIAssistantMessageParam, OpenAIUserMessageParam

        # The raw conversation item log has already grown large, as if several turns
        # happened after an earlier compaction -- since compaction items are never
        # synced into it, it keeps every raw item forever and requires pagination.
        pages = self._conversation_pages(total_items=150, page_size=100)
        mock_conversations_api.list_items = AsyncMock(side_effect=pages)
        # But the stored chat-message log already reflects that earlier compaction: a
        # short, bounded log, not the 150-item raw transcript.
        mock_responses_store.get_conversation_messages.return_value = [
            OpenAIAssistantMessageParam(content="Summary of everything before this point."),
            OpenAIUserMessageParam(content="A recent question."),
            OpenAIAssistantMessageParam(content="A recent answer."),
        ]
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="What did we discuss?",
                model="meta-llama/Llama-3.1-8B-Instruct",
                conversation="conv_272fa1af1b67b662ee7ef05f72ec300d92425257645a531a",
                context_management=[{"type": "compaction", "compact_threshold": 1}],
            )
        )

        compaction_call = next(
            call for call in mock_inference_api.openai_chat_completion.call_args_list if not call.args[0].stream
        )
        # 3 stored messages + the summarization prompt _build_compaction appends --
        # not the 150-item raw transcript.
        assert len(compaction_call.args[0].messages) == 4


class TestCompactionItemNotDuplicatedInNextHistory:
    """Code review of #6838: a compacting turn stored the compaction item in BOTH its
    own .input (as the last element of the compacted history) and its .output
    (surfaced as output[0]) -- so the next previous_response_id turn's history
    (previous.input + previous.output, _process_input_with_previous_response) counted
    it twice: double-counting its tokens, re-summarizing it twice on the next
    compaction, and listing it twice via the input_items endpoint.
    """

    async def test_compacting_turn_does_not_store_the_compaction_item_in_its_own_input(
        self, openai_responses_impl, mock_inference_api, mock_responses_store
    ):
        mock_responses_store.get_response_object.return_value = _previous_response()
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        result = await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="What is the code word?",
                model="meta-llama/Llama-3.1-8B-Instruct",
                previous_response_id="resp-previous-123",
                context_management=[{"type": "compaction", "compact_threshold": 1}],
            )
        )

        assert isinstance(result.output[0], OpenAIResponseCompaction)

        final_store_call = next(
            call
            for call in mock_responses_store.upsert_response_object.call_args_list
            if call.kwargs["response_object"].status == "completed"
        )
        stored_input = final_store_call.kwargs["input"]
        assert not any(isinstance(item, OpenAIResponseCompaction) for item in stored_input)
