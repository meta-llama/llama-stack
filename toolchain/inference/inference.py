from typing import AsyncGenerator

from models.llama3.datatypes import StopReason

from .api.config import CheckpointType, GeneratorArgs, InlineImplConfig
from .api.datatypes import (
    ChatCompletionResponseEvent,
    ChatCompletionResponseEventType,
    ToolCallDelta,
    ToolCallParseStatus,
)
from .api.endpoints import (
    ChatCompletionRequest,
    ChatCompletionResponseStreamChunk,
    CompletionRequest,
    ModelInference,
)
from .model_parallel import LlamaModelParallelGenerator


def generator_args_from_config(config: InlineImplConfig) -> GeneratorArgs:
    if (
        config.checkpoint_config.checkpoint.checkpoint_type
        == CheckpointType.pytorch.value
    ):
        pt_checkpoint = config.checkpoint_config.checkpoint
        return GeneratorArgs(
            ckpt_dir=pt_checkpoint.checkpoint_dir,
            tokenizer_path=pt_checkpoint.tokenizer_path,
            model_parallel_size=pt_checkpoint.model_parallel_size,
            max_seq_len=config.max_seq_len,
            max_batch_size=config.max_batch_size,
        )
    else:
        raise NotImplementedError("HF Checkpoint not supported yet")


class ModelInferenceImpl(ModelInference):

    def __init__(self, config: InlineImplConfig) -> None:
        self.config = config

    async def initialize(self) -> None:
        generator_args = generator_args_from_config(self.config)
        self.generator = LlamaModelParallelGenerator(
            args=generator_args,
        )
        self.generator.start()

    async def shutdown(self) -> None:
        self.generator.stop()

    async def completion(self, request: CompletionRequest) -> AsyncGenerator:
        raise NotImplementedError()

    async def chat_completion(self, request: ChatCompletionRequest) -> AsyncGenerator:
        yield ChatCompletionResponseStreamChunk(
            event=ChatCompletionResponseEvent(
                event_type=ChatCompletionResponseEventType.start,
                delta="",
            )
        )

        tokens = []
        logprobs = []

        stop_reason = None

        buffer = ""
        ipython = False

        for token_result in self.generator.chat_completion(
            messages=request.messages,
            temperature=request.sampling_params.temperature,
            top_p=request.sampling_params.top_p,
            max_gen_len=request.sampling_params.max_tokens,
            logprobs=request.logprobs,
        ):
            buffer += token_result.text
            tokens.append(token_result.token)

            if not ipython and buffer.startswith("<|python_tag|>"):
                ipython = True
                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.progress,
                        delta=ToolCallDelta(
                            content="",
                            parse_status=ToolCallParseStatus.started,
                        ),
                    )
                )
                buffer = buffer[len("<|python_tag|>") :]
                continue

            if not request.stream:
                if request.logprobs:
                    logprobs.append(token_result.logprob)

                continue

            if token_result.text == "<|eot_id|>":
                stop_reason = StopReason.end_of_turn
                text = ""
            elif token_result.text == "<|eom_id|>":
                stop_reason = StopReason.end_of_message
                text = ""
            else:
                text = token_result.text

            if ipython:
                delta = ToolCallDelta(
                    content=text,
                    parse_status=ToolCallParseStatus.in_progress,
                )
            else:
                delta = text
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.progress,
                    delta=delta,
                    stop_reason=stop_reason,
                )
            )

        if stop_reason is None:
            stop_reason = StopReason.out_of_tokens

        # TODO(ashwin): parse tool calls separately here and report errors?
        # if someone breaks the iteration before coming here we are toast
        message = self.generator.formatter.decode_assistant_message(tokens, stop_reason)
        if request.stream:
            parsed_tool_calls = len(message.tool_calls) > 0
            if ipython and not parsed_tool_calls:
                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.progress,
                        delta=ToolCallDelta(
                            content="",
                            parse_status=ToolCallParseStatus.failure,
                        ),
                        stop_reason=stop_reason,
                    )
                )

            for tool_call in message.tool_calls:
                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.progress,
                        delta=ToolCallDelta(
                            content=tool_call,
                            parse_status=ToolCallParseStatus.success,
                        ),
                        stop_reason=stop_reason,
                    )
                )

            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.complete,
                    delta="",
                    stop_reason=stop_reason,
                )
            )

            # TODO(ashwin): what else do we need to send out here when everything finishes?
        else:
            yield ChatCompletionResponse(
                content=message.content,
                tool_calls=message.tool_calls,
                stop_reason=stop_reason,
                logprobs=logprobs if request.logprobs else None,
            )
