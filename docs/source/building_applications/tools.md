# Tools

Tools are functions that can be invoked by an agent to perform tasks. They are organized into tool groups and registered with specific providers. Each tool group represents a collection of related tools from a single provider.

Tools are treated as any other resource in llama stack like models. you can register them, have providers for them etc.

When instatiating an agent, you can provide it a list of tool groups that it has access to. Agent gets the corresponding tool definitions for the specified tool gropus and pass along to the model

Refer to the [Building AI Applications](../../notebooks/Llama_Stack_Building_AI_Applications.ipynb) notebook for more examples on how to use tools.

## Types of Tool Group providers

There are three types of providers for tool groups that are supported by llama stack.

1. Built-in providers
2. Model Context Protocol (MCP) providers
3. Tools provided by the client

### Built-in providers

Built-in providers come packaged with LlamaStack. These providers provide common functionalities like web search, code interpretation, and computational capabilities.

#### Web Search providers
There are three web search providers that are supported by llama stack.

1. Brave Search
2. Bing Search
3. Tavily Search

Example client SDK call to register a "websearch" toolgroup that is provided by brave-search.

```python
# Register Brave Search tool group
client.toolgroups.register(
    toolgroup_id="builtin::websearch",
    provider_id="brave-search",
    args={"max_results": 5}
)
```

The tool requires an API key which can be provided either in the configuration or through the request header `X-LlamaStack-Provider-Data`.
The api key is required to be passed in the header as `X-LlamaStack-Provider-Data` as `{"brave_search_api_key": <your api key>}` for brave search.



#### Code Interpreter

The Code Interpreter tool allows execution of Python code within a controlled environment. It includes safety measures to prevent potentially dangerous operations.

```python
# Register Code Interpreter tool group
client.toolgroups.register(
    toolgroup_id="builtin::code_interpreter",
    provider_id="code_interpreter"
)
```

Features:
- Secure execution environment using `bwrap` sandboxing
- Matplotlib support for generating plots
- Disabled dangerous system operations
- Configurable execution timeouts

#### WolframAlpha

The WolframAlpha tool provides access to computational knowledge through the WolframAlpha API.

```python
# Register WolframAlpha tool group
client.toolgroups.register(
    toolgroup_id="builtin::wolfram_alpha",
    provider_id="wolfram-alpha"
)
```

Example usage:
```python
result = client.tools.invoke_tool(
    tool_name="wolfram_alpha",
    args={"query": "solve x^2 + 2x + 1 = 0"}
)
```

#### Memory

The Memory tool enables retrieval of context from various types of memory banks (vector, key-value, keyword, and graph).

```python
# Register Memory tool group
client.toolgroups.register(
    toolgroup_id="builtin::memory",
    provider_id="memory",
    args={
        "max_chunks": 5,
        "max_tokens_in_context": 4096
    }
)
```

Features:
- Support for multiple memory bank types
- Configurable query generation
- Context retrieval with token limits


> **Note:** By default, llama stack run.yaml defines toolgroups for web search, code interpreter and memory, that are provided by tavily-search, code-interpreter and memory providers.

## Model Context Protocol (MCP) Tools

MCP tools are special tools that can interact with llama stack over model context protocol. These tools are dynamically discovered from an MCP endpoint and can be used to extend the agent's capabilities.

```python
# Register MCP tools
client.toolgroups.register(
    toolgroup_id="builtin::filesystem",
    provider_id="model-context-protocol",
    mcp_endpoint=URL(uri="http://localhost:8000/sse"),
)
```

MCP tools require:
- A valid MCP endpoint URL
- The endpoint must implement the Model Context Protocol
- Tools are discovered dynamically from the endpoint

## Tool Structure

Each tool has the following components:

- `name`: Unique identifier for the tool
- `description`: Human-readable description of the tool's functionality
- `parameters`: List of parameters the tool accepts
  - `name`: Parameter name
  - `parameter_type`: Data type (string, number, etc.)
  - `description`: Parameter description
  - `required`: Whether the parameter is required (default: true)
  - `default`: Default value if any

Example tool definition:
```python
{
    "name": "web_search",
    "description": "Search the web for information",
    "parameters": [
        {
            "name": "query",
            "parameter_type": "string",
            "description": "The query to search for",
            "required": True
        }
    ]
}
```

## Tool Invocation

Tools can be invoked using the `invoke_tool` method:

```python
result = client.tools.invoke_tool(
    tool_name="web_search",
    kwargs={"query": "What is the capital of France?"}
)
```

The result contains:
- `content`: The tool's output
- `error_message`: Optional error message if the tool failed
- `error_code`: Optional error code if the tool failed

## Listing Available Tools

You can list all available tools or filter by tool group:

```python
# List all tools
all_tools = client.tools.list_tools()

# List tools in a specific group
group_tools = client.tools.list_tools(toolgroup_id="search_tools")
```
