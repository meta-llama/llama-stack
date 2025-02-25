import Foundation

import LLaMARunner
import LlamaStackClient

class RunnerHolder: ObservableObject {
  var runner: Runner?
}

public class LocalInference: Inference {
  private var runnerHolder = RunnerHolder()
  private let runnerQueue: DispatchQueue

  public init (queue: DispatchQueue) {
    runnerQueue = queue
  }

  public func loadModel(modelPath: String, tokenizerPath: String, completion: @escaping (Result<Void, Error>) -> Void) {
    runnerHolder.runner = runnerHolder.runner ?? Runner(
      modelPath: modelPath,
      tokenizerPath: tokenizerPath
    )


    runnerQueue.async {
      let runner = self.runnerHolder.runner
      do {
        try runner!.load()
        completion(.success(()))
      } catch let loadError {
        print("error: " + loadError.localizedDescription)
        completion(.failure(loadError))
      }
    }
  }

  public func stop() {
    runnerHolder.runner?.stop()
  }

  public func chatCompletion(request: Components.Schemas.ChatCompletionRequest) -> AsyncStream<Components.Schemas.ChatCompletionResponseStreamChunk> {
    return AsyncStream { continuation in
      let workItem = DispatchWorkItem {
        do {
          var tokens: [String] = []

          let prompt = try encodeDialogPrompt(messages: prepareMessages(request: request))
          var stopReason: Components.Schemas.StopReason? = nil
          var buffer = ""
          var ipython = false
          var echoDropped = false

          try self.runnerHolder.runner?.generate(prompt, sequenceLength: 4096) { token in
            buffer += token

            // HACK: Workaround until LlamaRunner exposes echo param
            if (!echoDropped) {
              if (buffer.hasPrefix(prompt)) {
                buffer = String(buffer.dropFirst(prompt.count))
                echoDropped = true
              }
              return
            }

            tokens.append(token)

            if !ipython && (buffer.starts(with: "<|python_tag|>") || buffer.starts(with: "[") ) {
              ipython = true
              continuation.yield(
                Components.Schemas.ChatCompletionResponseStreamChunk(
                  event: Components.Schemas.ChatCompletionResponseEvent(
                    delta: .tool_call(Components.Schemas.ToolCallDelta(
                      parse_status: Components.Schemas.ToolCallParseStatus.started,
                      tool_call: .case1(""),
                      _type: Components.Schemas.ToolCallDelta._typePayload.tool_call
                      )
                    ),
                    event_type: .progress
                  )
                )
              )

              if (buffer.starts(with: "<|python_tag|>")) {
                buffer = String(buffer.dropFirst("<|python_tag|>".count))
              }
            }

            // TODO: Non-streaming lobprobs

            var text = ""
            if token == "<|eot_id|>" {
              stopReason = Components.Schemas.StopReason.end_of_turn
            } else if token == "<|eom_id|>" {
              stopReason = Components.Schemas.StopReason.end_of_message
            } else {
              text = token
            }

            var delta: Components.Schemas.ContentDelta
            if ipython {
              delta = .tool_call(Components.Schemas.ToolCallDelta(
                parse_status: .in_progress,
                tool_call: .case1(text),
                _type: .tool_call
              ))
            } else {
              delta = .text(Components.Schemas.TextDelta(
                text: text,
                _type: Components.Schemas.TextDelta._typePayload.text)
              )
            }

            if stopReason == nil {
              continuation.yield(
                Components.Schemas.ChatCompletionResponseStreamChunk(
                  event: Components.Schemas.ChatCompletionResponseEvent(
                    delta: delta,
                    event_type: .progress
                  )
                )
              )
            }
          }

          if stopReason == nil {
            stopReason = Components.Schemas.StopReason.out_of_tokens
          }

          let message = decodeAssistantMessage(tokens: tokens.joined(), stopReason: stopReason!)
          // TODO: non-streaming support

          let didParseToolCalls = message.tool_calls.count > 0
          if ipython && !didParseToolCalls {
            continuation.yield(
              Components.Schemas.ChatCompletionResponseStreamChunk(
                event: Components.Schemas.ChatCompletionResponseEvent(
                  delta: .tool_call(Components.Schemas.ToolCallDelta(
                    parse_status: Components.Schemas.ToolCallParseStatus.failed,
                    tool_call: .case1(""),
                    _type: Components.Schemas.ToolCallDelta._typePayload.tool_call
                    )
                  ),
                  event_type: .progress
                )
                // TODO: stopReason
              )
            )
          }

          for toolCall in message.tool_calls {
            continuation.yield(
              Components.Schemas.ChatCompletionResponseStreamChunk(
                event: Components.Schemas.ChatCompletionResponseEvent(
                  delta: .tool_call(Components.Schemas.ToolCallDelta(
                    parse_status: Components.Schemas.ToolCallParseStatus.succeeded,
                    tool_call: Components.Schemas.ToolCallDelta.tool_callPayload.ToolCall(toolCall),
                    _type: Components.Schemas.ToolCallDelta._typePayload.tool_call
                    )
                  ),
                  event_type: .progress
                )
                // TODO: stopReason
              )
            )
          }

          continuation.yield(
            Components.Schemas.ChatCompletionResponseStreamChunk(
              event: Components.Schemas.ChatCompletionResponseEvent(
                delta: .text(Components.Schemas.TextDelta(
                  text: "",
                  _type: Components.Schemas.TextDelta._typePayload.text)
                ),
                event_type: .complete
              )
              // TODO: stopReason
            )
          )
        }
        catch (let error) {
          print("Inference error: " + error.localizedDescription)
        }
      }
      runnerQueue.async(execute: workItem)
    }
  }
}
