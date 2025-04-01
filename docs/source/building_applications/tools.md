# Tools

Tools are functions that can be invoked by an agent to perform tasks. They are organized into tool groups and registered with specific providers. Each tool group represents a collection of related tools from a single provider. They are organized into groups so that state can be externalized: the collection operates on the same state typically.
An example of this would be a "db_access" tool group that contains tools for interacting with a database. "list_tables", "query_table", "insert_row" could be examples of tools in this group.

Tools are treated as any other resource in llama stack like models. You can register them, have providers for them etc.

When instantiating an agent, you can provide it a list of tool groups that it has access to. Agent gets the corresponding tool definitions for the specified tool groups and passes them along to the model.

Refer to the [Building AI Applications](https://github.com/meta-llama/llama-stack/blob/main/docs/getting_started.ipynb) notebook for more examples on how to use tools.

## Types of Tool Group providers

There are three types of providers for tool groups that are supported by Llama Stack.

1. Built-in providers
2. Model Context Protocol (MCP) providers
3. Client provided tools

### Built-in providers

Built-in providers come packaged with Llama Stack. These providers provide common functionalities like web search, code interpretation, and computational capabilities.

#### Web Search providers
There are three web search providers that are supported by Llama Stack.

1. Brave Search
2. Bing Search
3. Tavily Search

Example client SDK call to register a "websearch" toolgroup that is provided by brave-search.

```python
# Register Brave Search tool group
client.toolgroups.register(
    toolgroup_id="builtin::websearch",
    provider_id="brave-search",
    args={"max_results": 5},
)
```

The tool requires an API key which can be provided either in the configuration or through the request header `X-LlamaStack-Provider-Data`. The format of the header is `{"<provider_name>_api_key": <your api key>}`.



#### Code Interpreter

The Code Interpreter allows execution of Python code within a controlled environment.

```python
# Register Code Interpreter tool group
client.toolgroups.register(
    toolgroup_id="builtin::code_interpreter", provider_id="code_interpreter"
)
```

Features:
- Secure execution environment using `bwrap` sandboxing
- Matplotlib support for generating plots
- Disabled dangerous system operations
- Configurable execution timeouts

> ⚠️ Important: The code interpreter tool can operate in a controlled environment locally or on Podman containers. To ensure proper functionality in containerized environments:
> - The container requires privileged access (e.g., --privileged).
> - Users without sufficient permissions may encounter permission errors. (`bwrap: Can't mount devpts on /newroot/dev/pts: Permission denied`)
> - 🔒 Security Warning: Privileged mode grants elevated access and bypasses security restrictions. Use only in local, isolated, or controlled environments.

#### WolframAlpha

The WolframAlpha tool provides access to computational knowledge through the WolframAlpha API.

```python
# Register WolframAlpha tool group
client.toolgroups.register(
    toolgroup_id="builtin::wolfram_alpha", provider_id="wolfram-alpha"
)
```

Example usage:
```python
result = client.tool_runtime.invoke_tool(
    tool_name="wolfram_alpha", args={"query": "solve x^2 + 2x + 1 = 0"}
)
```

#### RAG

The RAG tool enables retrieval of context from various types of memory banks (vector, key-value, keyword, and graph).

```python
# Register Memory tool group
client.toolgroups.register(
    toolgroup_id="builtin::rag",
    provider_id="faiss",
    args={"max_chunks": 5, "max_tokens_in_context": 4096},
)
```

Features:
- Support for multiple memory bank types
- Configurable query generation
- Context retrieval with token limits


> **Note:** By default, llama stack run.yaml defines toolgroups for web search, code interpreter and rag, that are provided by tavily-search, code-interpreter and rag providers.

## Model Context Protocol (MCP) Tools

MCP tools are special tools that can interact with llama stack over model context protocol. These tools are dynamically discovered from an MCP endpoint and can be used to extend the agent's capabilities.

Refer to [https://github.com/modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers) for available MCP servers.

```shell
# start your MCP server
mkdir /tmp/content
touch /tmp/content/foo
touch /tmp/content/bar
npx -y supergateway --port 8000 --stdio 'npx -y @modelcontextprotocol/server-filesystem /tmp/content'
```

Then register the MCP server as a tool group,
```python
client.toolgroups.register(
    toolgroup_id="mcp::filesystem",
    provider_id="model-context-protocol",
    mcp_endpoint=URL(uri="http://localhost:8000/sse"),
)
```

MCP tools require:
- A valid MCP endpoint URL
- The endpoint must implement the Model Context Protocol
- Tools are discovered dynamically from the endpoint


## Adding Custom Tools

When you want to use tools other than the built-in tools, you just need to implement a python function with a docstring. The content of the docstring will be used to describe the tool and the parameters and passed
along to the generative model.

```python
# Example tool definition
def my_tool(input: int) -> int:
    """
    Runs my awesome tool.

    :param input: some int parameter
    """
    return input * 2
```
> **NOTE:** We employ python docstrings to describe the tool and the parameters. It is important to document the tool and the parameters so that the model can use the tool correctly. It is recommended to experiment with different docstrings to see how they affect the model's behavior.

Once defined, simply pass the tool to the agent config. `Agent` will take care of the rest (calling the model with the tool definition, executing the tool, and returning the result to the model for the next iteration).
```python
# Example agent config with client provided tools
agent = Agent(client, ..., tools=[my_tool])
```

Refer to [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/blob/main/examples/agents/e2e_loop_with_client_tools.py) for an example of how to use client provided tools.


## Tool Invocation

Tools can be invoked using the `invoke_tool` method:

```python
result = client.tool_runtime.invoke_tool(
    tool_name="web_search", kwargs={"query": "What is the capital of France?"}
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

## Simple Example: Using an Agent with the Code-Interpreter Tool

```python
from llama_stack_client import Agent

# Instantiate the AI agent with the given configuration
agent = Agent(
    client,
    name="code-interpreter",
    description="A code interpreter agent for executing Python code snippets",
    instructions="""
    You are a highly reliable, concise, and precise assistant.
    Always show the generated code, never generate your own code, and never anticipate results.
    """,
    model="meta-llama/Llama-3.2-3B-Instruct",
    tools=["builtin::code_interpreter"],
    max_infer_iters=5,
)

# Start a session
session_id = agent.create_session("tool_session")

# Send a query to the AI agent for code execution
response = agent.create_turn(
    messages=[{"role": "user", "content": "Run this code: print(3 ** 4 - 5 * 2)"}],
    session_id=session_id,
)
```
