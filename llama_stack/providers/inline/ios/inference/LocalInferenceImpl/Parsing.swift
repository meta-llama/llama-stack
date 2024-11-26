import Foundation

import LlamaStackClient

func encodeHeader(role: String) -> String {
  return "<|start_header_id|>\(role)<|end_header_id|>\n\n"
}

func encodeDialogPrompt(messages: [Components.Schemas.ChatCompletionRequest.messagesPayloadPayload]) -> String {
  var prompt = ""

  prompt.append("<|begin_of_text|>")
  for message in messages {
    let msg = encodeMessage(message: message)
    prompt += msg
  }

  prompt.append(encodeHeader(role: "assistant"))

  return prompt
}

func getRole(message: Components.Schemas.ChatCompletionRequest.messagesPayloadPayload) -> String {
  switch (message) {
  case .UserMessage(let m):
    return m.role.rawValue
  case .SystemMessage(let m):
    return m.role.rawValue
  case .ToolResponseMessage(let m):
    return m.role.rawValue
  case .CompletionMessage(let m):
    return m.role.rawValue
  }
}

func encodeMessage(message: Components.Schemas.ChatCompletionRequest.messagesPayloadPayload) -> String {
  var prompt = encodeHeader(role: getRole(message: message))

  switch (message) {
  case .CompletionMessage(let m):
    if (m.tool_calls.count > 0) {
      prompt += "<|python_tag|>"
    }
  default:
    break
  }

  func _processContent(_ content: Any) -> String {
    func _process(_ c: Any) {
      if let str = c as? String {
        prompt += str
      }
    }

    if let str = content as? String {
      _process(str)
    } else if let list = content as? [Any] {
      for c in list {
        _process(c)
      }
    }

    return ""
  }

  switch (message) {
  case .UserMessage(let m):
    prompt += _processContent(m.content)
  case .SystemMessage(let m):
    prompt += _processContent(m.content)
  case .ToolResponseMessage(let m):
    prompt += _processContent(m.content)
  case .CompletionMessage(let m):
    prompt += _processContent(m.content)
  }

  var eom = false

  switch (message) {
  case .UserMessage(let m):
    switch (m.content) {
    case .case1(let c):
      prompt += _processContent(c)
    case .ImageMedia(let c):
      prompt += _processContent(c)
    case .case3(let c):
      prompt += _processContent(c)
    }
  case .CompletionMessage(let m):
    // TODO: Support encoding past tool call history
    // for t in m.tool_calls {
    //  _processContent(t.)
    //}
    eom = m.stop_reason == Components.Schemas.StopReason.end_of_message
  case .SystemMessage(_):
    break
  case .ToolResponseMessage(_):
    break
  }

  if (eom) {
    prompt += "<|eom_id|>"
  } else {
    prompt += "<|eot_id|>"
  }

  return prompt
}

func prepareMessages(request: Components.Schemas.ChatCompletionRequest) throws -> [Components.Schemas.ChatCompletionRequest.messagesPayloadPayload] {
  var existingMessages = request.messages
  var existingSystemMessage: Components.Schemas.ChatCompletionRequest.messagesPayloadPayload?
  // TODO: Existing system message

  var messages: [Components.Schemas.ChatCompletionRequest.messagesPayloadPayload] = []

  let defaultGen = SystemDefaultGenerator()
  let defaultTemplate = defaultGen.gen()

  var sysContent = ""

  // TODO: Built-in tools

  sysContent += try defaultTemplate.render()

  messages.append(.SystemMessage(Components.Schemas.SystemMessage(
    content: .case1(sysContent),
    role: .system))
  )

  if request.tools?.isEmpty == false {
    // TODO: Separate built-ins and custom tools (right now everything treated as custom)
    let toolGen = FunctionTagCustomToolGenerator()
    let toolTemplate = try toolGen.gen(customTools: request.tools!)
    let tools = try toolTemplate.render()
    messages.append(.UserMessage(Components.Schemas.UserMessage(
      content: .case1(tools),
      role: .user)
    ))
  }

  messages.append(contentsOf: existingMessages)

  return messages
}

struct FunctionCall {
    let name: String
    let params: [String: Any]
}

public func maybeExtractCustomToolCalls(input: String) -> [Components.Schemas.ToolCall] {
  guard input.hasPrefix("[") && input.hasSuffix("]") else {
    return []
  }

  do {
    let trimmed = input.trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
    let calls = trimmed.components(separatedBy: "),").map { $0.hasSuffix(")") ? $0 : $0 + ")" }

    var result: [Components.Schemas.ToolCall] = []

    for call in calls {
      guard let nameEndIndex = call.firstIndex(of: "("),
            let paramsStartIndex = call.firstIndex(of: "{"),
            let paramsEndIndex = call.lastIndex(of: "}") else {
        return []
      }

      let name = String(call[..<nameEndIndex]).trimmingCharacters(in: .whitespacesAndNewlines)
      let paramsString = String(call[paramsStartIndex...paramsEndIndex])

      guard let data = paramsString.data(using: .utf8),
            let params = try? JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] else {
        return []
      }

      var props: [String : Components.Schemas.ToolCall.argumentsPayload.additionalPropertiesPayload] = [:]
      for (param_name, param) in params {
        switch (param) {
        case let value as String:
          props[param_name] = .case1(value)
        case let value as Int:
          props[param_name] = .case2(value)
        case let value as Double:
          props[param_name] = .case3(value)
        case let value as Bool:
          props[param_name] = .case4(value)
        default:
          return []
        }
      }

      result.append(
        Components.Schemas.ToolCall(
          arguments: .init(additionalProperties: props),
          call_id: UUID().uuidString,
          tool_name: .case2(name) // custom_tool
        )
      )
    }

    return result.isEmpty ? [] : result
  } catch {
    return []
  }
}

func decodeAssistantMessage(tokens: String, stopReason: Components.Schemas.StopReason) -> Components.Schemas.CompletionMessage {
  var content = tokens

  let roles = ["user", "system", "assistant"]
  for role in roles {
    let headerStr = encodeHeader(role: role)
    if content.hasPrefix(headerStr) {
      content = String(content.dropFirst(encodeHeader(role: role).count))
    }
  }

  if content.hasPrefix("<|python_tag|>") {
    content = String(content.dropFirst("<|python_tag|>".count))
  }


  if content.hasSuffix("<|eot_id|>") {
    content = String(content.dropLast("<|eot_id|>".count))
  } else {
    content = String(content.dropLast("<|eom_id|>".count))
  }

  return Components.Schemas.CompletionMessage(
    content: .case1(content),
    role: .assistant,
    stop_reason: stopReason,
    tool_calls: maybeExtractCustomToolCalls(input: content)
  )
}
