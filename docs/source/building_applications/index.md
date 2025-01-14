# Building AI Applications

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1F2ksmkoGQPa4pzRjMOE6BXWeOxWFIW6n?usp=sharing)

Llama Stack provides all the building blocks needed to create sophisticated AI applications. This guide will walk you through how to use these components effectively. Check out our Colab notebook on to follow along working examples on how you can build LLM-powered agentic applications using Llama Stack.

## Basic Inference

The foundation of any AI application is the ability to interact with LLM models. Llama Stack provides a simple interface for both completion and chat-based inference:

```python
from llama_stack_client import LlamaStackClient

client = LlamaStackClient(base_url="http://localhost:5001")

# List available models
models = client.models.list()

# Simple chat completion
response = client.inference.chat_completion(
    model_id="Llama3.2-3B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about coding"}
    ]
)
print(response.completion_message.content)
```

## Adding Memory & RAG

Memory enables your applications to reference and recall information from previous interactions or external documents. Llama Stack's memory system is built around the concept of Memory Banks:

1. **Vector Memory Banks**: For semantic search and retrieval
2. **Key-Value Memory Banks**: For structured data storage
3. **Keyword Memory Banks**: For basic text search
4. **Graph Memory Banks**: For relationship-based retrieval

Here's how to set up a vector memory bank for RAG:

```python
# Register a memory bank
bank_id = "my_documents"
response = client.memory_banks.register(
    memory_bank_id=bank_id,
    params={
        "memory_bank_type": "vector",
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_size_in_tokens": 512
    }
)

# Insert documents
documents = [
    {
        "document_id": "doc1",
        "content": "Your document text here",
        "mime_type": "text/plain"
    }
]
client.memory.insert(bank_id, documents)

# Query documents
results = client.memory.query(
    bank_id=bank_id,
    query="What do you know about...",
)
```

## Implementing Safety Guardrails

Safety is a critical component of any AI application. Llama Stack provides a Shield system that can be applied at multiple touchpoints:

```python
# Register a safety shield
shield_id = "content_safety"
client.shields.register(
    shield_id=shield_id,
    provider_shield_id="llama-guard-basic"
)

# Run content through shield
response = client.safety.run_shield(
    shield_id=shield_id,
    messages=[{"role": "user", "content": "User message here"}]
)

if response.violation:
    print(f"Safety violation detected: {response.violation.user_message}")
```

## Building Agents

Agents are the heart of complex AI applications. They combine inference, memory, safety, and tool usage into coherent workflows. At its core, an agent follows a sophisticated execution loop that enables multi-step reasoning, tool usage, and safety checks.

### The Agent Execution Loop

Each agent turn follows these key steps:

1. **Initial Safety Check**: The user's input is first screened through configured safety shields

2. **Context Retrieval**:
   - If RAG is enabled, the agent queries relevant documents from memory banks
   - For new documents, they are first inserted into the memory bank
   - Retrieved context is augmented to the user's prompt

3. **Inference Loop**: The agent enters its main execution loop:
   - The LLM receives the augmented prompt (with context and/or previous tool outputs)
   - The LLM generates a response, potentially with tool calls
   - If tool calls are present:
     - Tool inputs are safety-checked
     - Tools are executed (e.g., web search, code execution)
     - Tool responses are fed back to the LLM for synthesis
   - The loop continues until:
     - The LLM provides a final response without tool calls
     - Maximum iterations are reached
     - Token limit is exceeded

4. **Final Safety Check**: The agent's final response is screened through safety shields

```{mermaid}
sequenceDiagram
    participant U as User
    participant E as Executor
    participant M as Memory Bank
    participant L as LLM
    participant T as Tools
    participant S as Safety Shield

    Note over U,S: Agent Turn Start
    U->>S: 1. Submit Prompt
    activate S
    S->>E: Input Safety Check
    deactivate S

    E->>M: 2.1 Query Context
    M-->>E: 2.2 Retrieved Documents

    loop Inference Loop
        E->>L: 3.1 Augment with Context
        L-->>E: 3.2 Response (with/without tool calls)

        alt Has Tool Calls
            E->>S: Check Tool Input
            S->>T: 4.1 Execute Tool
            T-->>E: 4.2 Tool Response
            E->>L: 5.1 Tool Response
            L-->>E: 5.2 Synthesized Response
        end

        opt Stop Conditions
            Note over E: Break if:
            Note over E: - No tool calls
            Note over E: - Max iterations reached
            Note over E: - Token limit exceeded
        end
    end

    E->>S: Output Safety Check
    S->>U: 6. Final Response
```

Each step in this process can be monitored and controlled through configurations. Here's an example that demonstrates monitoring the agent's execution:

```python
from llama_stack_client.lib.agents.event_logger import EventLogger

agent_config = AgentConfig(
    model="Llama3.2-3B-Instruct",
    instructions="You are a helpful assistant",
    # Enable both RAG and tool usage
    tools=[
        {
            "type": "memory",
            "memory_bank_configs": [{
                "type": "vector",
                "bank_id": "my_docs"
            }],
            "max_tokens_in_context": 4096
        },
        {
            "type": "code_interpreter",
            "enable_inline_code_execution": True
        }
    ],
    # Configure safety
    input_shields=["content_safety"],
    output_shields=["content_safety"],
    # Control the inference loop
    max_infer_iters=5,
    sampling_params={
        "strategy": {
            "type": "top_p",
            "temperature": 0.7,
            "top_p": 0.95
        },
        "max_tokens": 2048
    }
)

agent = Agent(client, agent_config)
session_id = agent.create_session("monitored_session")

# Stream the agent's execution steps
response = agent.create_turn(
    messages=[{"role": "user", "content": "Analyze this code and run it"}],
    attachments=[{
        "content": "https://raw.githubusercontent.com/example/code.py",
        "mime_type": "text/plain"
    }],
    session_id=session_id
)

# Monitor each step of execution
for log in EventLogger().log(response):
    if log.event.step_type == "memory_retrieval":
        print("Retrieved context:", log.event.retrieved_context)
    elif log.event.step_type == "inference":
        print("LLM output:", log.event.model_response)
    elif log.event.step_type == "tool_execution":
        print("Tool call:", log.event.tool_call)
        print("Tool response:", log.event.tool_response)
    elif log.event.step_type == "shield_call":
        if log.event.violation:
            print("Safety violation:", log.event.violation)
```

This example shows how an agent can: Llama Stack provides a high-level agent framework:

```python
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.types.agent_create_params import AgentConfig

# Configure an agent
agent_config = AgentConfig(
    model="Llama3.2-3B-Instruct",
    instructions="You are a helpful assistant",
    tools=[
        {
            "type": "memory",
            "memory_bank_configs": [],
            "query_generator_config": {
                "type": "default",
                "sep": " "
            }
        }
    ],
    input_shields=["content_safety"],
    output_shields=["content_safety"],
    enable_session_persistence=True
)

# Create an agent
agent = Agent(client, agent_config)
session_id = agent.create_session("my_session")

# Run agent turns
response = agent.create_turn(
    messages=[{"role": "user", "content": "Your question here"}],
    session_id=session_id
)
```

### Adding Tools to Agents

Agents can be enhanced with various tools:

1. **Search**: Web search capabilities through providers like Brave
2. **Code Interpreter**: Execute code snippets
3. **RAG**: Memory and document retrieval
4. **Function Calling**: Custom function execution
5. **WolframAlpha**: Mathematical computations
6. **Photogen**: Image generation

Example of configuring an agent with tools:

```python
agent_config = AgentConfig(
    model="Llama3.2-3B-Instruct",
    tools=[
        {
            "type": "brave_search",
            "api_key": "YOUR_API_KEY",
            "engine": "brave"
        },
        {
            "type": "code_interpreter",
            "enable_inline_code_execution": True
        }
    ],
    tool_choice="auto",
    tool_prompt_format="json"
)
```

## Building RAG-Enhanced Agents

One of the most powerful patterns is combining agents with RAG capabilities. Here's a complete example:

```python
from llama_stack_client.types import Attachment

# Create attachments from documents
attachments = [
    Attachment(
        content="https://raw.githubusercontent.com/example/doc.rst",
        mime_type="text/plain"
    )
]

# Configure agent with memory
agent_config = AgentConfig(
    model="Llama3.2-3B-Instruct",
    instructions="You are a helpful assistant",
    tools=[{
        "type": "memory",
        "memory_bank_configs": [],
        "query_generator_config": {"type": "default", "sep": " "},
        "max_tokens_in_context": 4096,
        "max_chunks": 10
    }],
    enable_session_persistence=True
)

agent = Agent(client, agent_config)
session_id = agent.create_session("rag_session")

# Initial document ingestion
response = agent.create_turn(
    messages=[{
        "role": "user",
        "content": "I am providing some documents for reference."
    }],
    attachments=attachments,
    session_id=session_id
)

# Query with RAG
response = agent.create_turn(
    messages=[{
        "role": "user",
        "content": "What are the key topics in the documents?"
    }],
    session_id=session_id
)
```

## Testing & Evaluation

Llama Stack provides built-in tools for evaluating your applications:

1. **Benchmarking**: Test against standard datasets
2. **Application Evaluation**: Score your application's outputs
3. **Custom Metrics**: Define your own evaluation criteria

Here's how to set up basic evaluation:

```python
# Create an evaluation task
response = client.eval_tasks.register(
    eval_task_id="my_eval",
    dataset_id="my_dataset",
    scoring_functions=["accuracy", "relevance"]
)

# Run evaluation
job = client.eval.run_eval(
    task_id="my_eval",
    task_config={
        "type": "app",
        "eval_candidate": {
            "type": "agent",
            "config": agent_config
        }
    }
)

# Get results
result = client.eval.job_result(
    task_id="my_eval",
    job_id=job.job_id
)
```

## Debugging & Monitoring

Llama Stack includes comprehensive telemetry for debugging and monitoring your applications:

1. **Tracing**: Track request flows across components
2. **Metrics**: Measure performance and usage
3. **Logging**: Debug issues and track behavior

The telemetry system supports multiple output formats:

- OpenTelemetry for visualization in tools like Jaeger
- SQLite for local storage and querying
- Console output for development

Example of querying traces:

```python
# Query traces for a session
traces = client.telemetry.query_traces(
    attribute_filters=[{
        "key": "session_id",
        "op": "eq",
        "value": session_id
    }]
)

# Get spans within the root span; indexed by ID
# Use parent_span_id to build a tree out of it
spans_by_id = client.telemetry.get_span_tree(
    span_id=traces[0].root_span_id
)
```

For details on how to use the telemetry system to debug your applications, export traces to a dataset, and run evaluations, see the [Telemetry](telemetry) section.

```{toctree}
:hidden:
:maxdepth: 3

telemetry
```
