## Agent Execution Loop

Agents are the heart of complex AI applications. They combine inference, memory, safety, and tool usage into coherent workflows. At its core, an agent follows a sophisticated execution loop that enables multi-step reasoning, tool usage, and safety checks.

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
    toolgroups=[
        {"name": "builtin::rag", "args": {"vector_db_ids": ["my_docs"]}}.
        "builtin::code_interpreter",
    ],
    # Configure safety
    input_shields=["llama_guard"],
    output_shields=["llama_guard"],
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
