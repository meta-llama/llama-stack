# Quick Start

In this guide, we'll walk through how you can use the Llama Stack (server and client SDK) to test a simple RAG agent.

A Llama Stack agent is a simple integrated system that can perform tasks by combining a Llama model for reasoning with tools (e.g., RAG, web search, code execution, etc.) for taking actions.

In Llama Stack, we provide a server exposing multiple APIs. These APIs are backed by implementations from different providers. For this guide, we will use [Ollama](https://ollama.com/) as the inference provider.


### 1. Start Ollama

```bash
ollama run llama3.2:3b-instruct-fp16 --keepalive 60m
```

By default, Ollama keeps the model loaded in memory for 5 minutes which can be too short. We set the `--keepalive` flag to 60 minutes to ensure the model remains loaded for sometime.

```{admonition} Note
:class: tip

If you do not have ollama, you can install it from [here](https://ollama.com/download).
```


### 2. Use `uv` to install and run Llama Stack

Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Setup venv
```bash
uv venv --python 3.10
source .venv/bin/activate
```
Install llama stack
```bash
uv pip install llama-stack
```

Build llama stack for ollama
```bash
llama stack build --template ollama --image-type venv
```

Run llama stack
```bash
# Use the model from ollama. Run `ollama ps` to see if its still running
INFERENCE_MODEL=llama3.2:3b-instruct-fp16 \
    llama stack run ollama --image-type venv
```

You will see the output like below:
```
...
INFO:     Application startup complete.
INFO:     Uvicorn running on http://['::', '0.0.0.0']:8321 (Press CTRL+C to quit)
```

Now you can use the llama stack client to run inference and build agents!

:::{dropdown} Installing the Llama Stack client CLI and SDK

Open a new terminal and navigate to the same directory you started the server from.

Setup venv (llama-stack already includes the client package)
```bash
source .venv/bin/activate
```
Let's use the `llama-stack-client` CLI to check the connectivity to the server.

```bash
llama-stack-client configure --endpoint http://localhost:$LLAMA_STACK_PORT --api-key none
```
You will see the below:
```
Done! You can now use the Llama Stack Client CLI with endpoint http://localhost:8321
```

List the models
```
llama-stack-client models list
```

```
Available Models

┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ model_type      ┃ identifier                          ┃ provider_resource_id                ┃ metadata                                  ┃ provider_id     ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ embedding       │ all-MiniLM-L6-v2                    │ all-minilm:latest                   │ {'embedding_dimension': 384.0}            │ ollama          │
├─────────────────┼─────────────────────────────────────┼─────────────────────────────────────┼───────────────────────────────────────────┼─────────────────┤
│ llm             │ llama3.2:3b-instruct-fp16           │ llama3.2:3b-instruct-fp16           │                                           │ ollama          │
└─────────────────┴─────────────────────────────────────┴─────────────────────────────────────┴───────────────────────────────────────────┴─────────────────┘

Total models: 2

```

You can test basic Llama inference completion using the CLI too.
```bash
llama-stack-client inference chat-completion --message "tell me a joke"
```
```
ChatCompletionResponse(
    completion_message=CompletionMessage(
        content="Here's one:\n\nWhat do you call a fake noodle?\n\nAn impasta!",
        role='assistant',
        stop_reason='end_of_turn',
        tool_calls=[]
    ),
    logprobs=None,
    metrics=[
        Metric(metric='prompt_tokens', value=14.0, unit=None),
        Metric(metric='completion_tokens', value=27.0, unit=None),
        Metric(metric='total_tokens', value=41.0, unit=None)
    ]
)
```
:::

&nbsp;

### 3. Run inference with Python SDK

Here is a simple example to perform chat completions using the SDK.
```python
## lstest.py
from llama_stack_client import LlamaStackClient

client = LlamaStackClient(base_url=f"http://localhost:8321")

# List available models
models = client.models.list()

# Find the first LLM
llm = next(m for m in models if m.model_type == 'llm')
model_id = llm.identifier

print("Model:", model_id)

response = client.inference.chat_completion(
    model_id=model_id,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about coding"},
    ],
)
print(response.completion_message.content)
```

```bash
python lstest.py
```

```
Model: llama3.2:3b-instruct-fp16
Here is a haiku about coding:

Lines of code unfold
Logic flows through digital night
Beauty in the bits
```

### 4. Your first agent

```python
## lsagent.py

from llama_stack_client import LlamaStackClient
from llama_stack_client import Agent, AgentEventLogger
import uuid

client = LlamaStackClient(base_url=f"http://localhost:8321")

models = client.models.list()
llm = next(m for m in models if m.model_type == 'llm')
model_id = llm.identifier

agent = Agent(client,
    model=model_id,
    instructions="You are a helpful assistant that can answer questions about the Torchtune project."
)

s_id = agent.create_session(session_name=f"s{uuid.uuid4()}")

# Non-streaming example
print("Non-streaming ...")
response = agent.create_turn(
    messages=[ {
        "role": "user",
        "content": "Who are you?"
    }],
    session_id=s_id,
    stream=False
)
print("agent>", response.output_message.content)

# Streamining with print helper
print("Streaming with print helper...")
stream = agent.create_turn(
    messages=[ {
        "role": "user",
        "content": "Who are you?"
    }],
    session_id=s_id,
    stream=True
)
for event in AgentEventLogger().log(stream):
    event.print()


# Streaming example
print("Streaming ...")
stream = agent.create_turn(
    messages=[ {
        "role": "user",
        "content": "Who are you?"
    }],
    session_id=s_id,
    stream=True
)
for event in stream:
    print(event)
```

**Run the agent**

```bash
python lsagent.py
```
Sample output
```
Non-streaming ...
agent> I'm an AI assistant, and I'll be happy to help with any questions or information you have about the Torchtune project.

For those who may not know, Torchtune is a popular open-source music composition tool that allows users to create and share musical compositions using a unique visual interface. It's designed to make music creation more accessible and fun for everyone, regardless of their musical background or experience level.

What would you like to know about Torchtune? Are you looking for information on how to use the software, tutorials, or perhaps something else?
Streaming with print helper...
inference> I am an AI assistant specifically designed to provide information and support related to the Torchtune project. I don't have a personal identity in the classical sense, but I'm here to help answer your questions, provide guidance, and offer assistance with any topics related to Torchtune.

I've been trained on a vast amount of text data, including documentation, tutorials, and community discussions about Torchtune, which enables me to provide accurate and up-to-date information. My goal is to be helpful and informative, so feel free to ask me anything you'd like to know about Torchtune!
Streaming ...
AgentTurnResponseStreamChunk(event=TurnResponseEvent(payload=AgentTurnResponseStepStartPayload(event_type='step_start', step_id='7d40b848-3ba9-419b-86d9-942fd65698e2', step_type='inference', metadata={})))
AgentTurnResponseStreamChunk(event=TurnResponseEvent(payload=AgentTurnResponseStepProgressPayload(delta=TextDelta(text='I', type='text'), event_type='step_progress', step_id='7d40b848-3ba9-419b-86d9-942fd65698e2', step_type='inference')))
AgentTurnResponseStreamChunk(event=TurnResponseEvent(payload=AgentTurnResponseStepProgressPayload(delta=TextDelta(text=' am', type='text'), event_type='step_progress', step_id='7d40b848-3ba9-419b-86d9-942fd65698e2', step_type='inference')))
...
AgentTurnResponseStreamChunk(event=TurnResponseEvent(payload=AgentTurnResponseStepProgressPayload(delta=TextDelta(text='!', type='text'), event_type='step_progress', step_id='7d40b848-3ba9-419b-86d9-942fd65698e2', step_type='inference')))
AgentTurnResponseStreamChunk(event=TurnResponseEvent(payload=AgentTurnResponseStepCompletePayload(event_type='step_complete', step_details=InferenceStep(api_model_response=CompletionMessage(content="I am an artificial intelligence language model designed to assist with a wide range of topics, including the Torchtune project. I'm a computer program created through a process called deep learning, which allows me to understand and generate human-like text.\n\nMy primary function is to provide information, answer questions, and engage in conversation to the best of my abilities based on my training data. I don't have personal experiences, emotions, or consciousness like humans do, but I'm designed to be helpful and informative.\n\nIn the context of Torchtune, I can help with topics such as:\n\n* Providing tutorials and guides\n* Answering questions about the software's features and functionality\n* Offering tips and tricks for using Torchtune effectively\n* Discussing music theory and composition concepts related to Torchtune\n\nFeel free to ask me anything about Torchtune or any other topic, and I'll do my best to help!", role='assistant', stop_reason='end_of_turn', tool_calls=[]), step_id='7d40b848-3ba9-419b-86d9-942fd65698e2', step_type='inference', turn_id='2f0921b0-ece7-4d63-bfde-87f0b08a206a', completed_at=datetime.datetime(2025, 3, 29, 18, 32, 12, 976952, tzinfo=TzInfo(UTC)), started_at=datetime.datetime(2025, 3, 29, 18, 32, 4, 840716, tzinfo=TzInfo(UTC))), step_id='7d40b848-3ba9-419b-86d9-942fd65698e2', step_type='inference')))
AgentTurnResponseStreamChunk(event=TurnResponseEvent(payload=AgentTurnResponseTurnCompletePayload(event_type='turn_complete', turn=Turn(input_messages=[UserMessage(content='Who are you?', role='user', context=None)], output_message=CompletionMessage(content="I am an artificial intelligence language model designed to assist with a wide range of topics, including the Torchtune project. I'm a computer program created through a process called deep learning, which allows me to understand and generate human-like text.\n\nMy primary function is to provide information, answer questions, and engage in conversation to the best of my abilities based on my training data. I don't have personal experiences, emotions, or consciousness like humans do, but I'm designed to be helpful and informative.\n\nIn the context of Torchtune, I can help with topics such as:\n\n* Providing tutorials and guides\n* Answering questions about the software's features and functionality\n* Offering tips and tricks for using Torchtune effectively\n* Discussing music theory and composition concepts related to Torchtune\n\nFeel free to ask me anything about Torchtune or any other topic, and I'll do my best to help!", role='assistant', stop_reason='end_of_turn', tool_calls=[]), session_id='a705b5a1-b9a6-4cf5-a99a-7917cc093755', started_at=datetime.datetime(2025, 3, 29, 18, 32, 4, 840680, tzinfo=TzInfo(UTC)), steps=[InferenceStep(api_model_response=CompletionMessage(content="I am an artificial intelligence language model designed to assist with a wide range of topics, including the Torchtune project. I'm a computer program created through a process called deep learning, which allows me to understand and generate human-like text.\n\nMy primary function is to provide information, answer questions, and engage in conversation to the best of my abilities based on my training data. I don't have personal experiences, emotions, or consciousness like humans do, but I'm designed to be helpful and informative.\n\nIn the context of Torchtune, I can help with topics such as:\n\n* Providing tutorials and guides\n* Answering questions about the software's features and functionality\n* Offering tips and tricks for using Torchtune effectively\n* Discussing music theory and composition concepts related to Torchtune\n\nFeel free to ask me anything about Torchtune or any other topic, and I'll do my best to help!", role='assistant', stop_reason='end_of_turn', tool_calls=[]), step_id='7d40b848-3ba9-419b-86d9-942fd65698e2', step_type='inference', turn_id='2f0921b0-ece7-4d63-bfde-87f0b08a206a', completed_at=datetime.datetime(2025, 3, 29, 18, 32, 12, 976952, tzinfo=TzInfo(UTC)), started_at=datetime.datetime(2025, 3, 29, 18, 32, 4, 840716, tzinfo=TzInfo(UTC)))], turn_id='2f0921b0-ece7-4d63-bfde-87f0b08a206a', completed_at=datetime.datetime(2025, 3, 29, 18, 32, 12, 987353, tzinfo=TzInfo(UTC)), output_attachments=[]))))
```

### 5. RAG agent

```python
## rag_agent.py

from llama_stack_client import LlamaStackClient
from llama_stack_client import Agent, AgentEventLogger
from llama_stack_client.types import Document
import uuid

client = LlamaStackClient(base_url=f"http://localhost:8321")

# Create a vector database instance
embedlm = next(m for m in client.models.list() if m.model_type == 'embedding')
embedding_model = embedlm.identifier
vdb = next(p for p in client.providers.list() if p.api == "vector_io")
vector_db_id = f"v{uuid.uuid4()}"
client.vector_dbs.register(
    provider_id=vdb.provider_id,
    vector_db_id=vector_db_id,
    embedding_model=embedding_model,
)

# Create Documents
urls = [
    "memory_optimizations.rst",
    "chat.rst",
    "llama3.rst",
    "datasets.rst",
    "qat_finetune.rst",
    "lora_finetune.rst",
]
documents = [
    Document(
        document_id=f"num-{i}",
        content=f"https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/{url}",
        mime_type="text/plain",
        metadata={},
    )
    for i, url in enumerate(urls)
]

# Insert documents
client.tool_runtime.rag_tool.insert(
    documents=documents,
    vector_db_id=vector_db_id,
    chunk_size_in_tokens=512,
)

# Get the model being served
llm = next(m for m in client.models.list() if m.model_type == 'llm')
model = llm.identifier

# Create RAG agent
ragagent = Agent(client,
    model=model,
    instructions="You are a helpful assistant that can answer questions about the Torchtune project. Use the RAG tool to answer questions as needed.",
    tools=[{
        "name": "builtin::rag",
        "args": {"vector_db_ids": [vector_db_id]},
    }],
)

s_id = ragagent.create_session(
    session_name=f"s{uuid.uuid4()}"
)

turns = [
    "what is torchtune",
    "tell me about dora"
]

for t in turns:
    print("user>", t)
    stream = ragagent.create_turn(
        messages=[{
            "role": "user",
            "content": t
        }],
        session_id=s_id,
        stream=True
    )
    for chunk in stream:
        event_type = chunk.event.payload.event_type
        if event_type == 'step_progress':
            print(chunk.event.payload.delta.text, end='', flush=True)
```
```
python lsragagent.py
```
Sample output:
```
user> what is torchtune
inference> [knowledge_search(query='TorchTune')]
tool_execution> Tool:knowledge_search Args:{'query': 'TorchTune'}
tool_execution> Tool:knowledge_search Response:[TextContentItem(text='knowledge_search tool found 5 chunks:\nBEGIN of knowledge_search tool results.\n', type='text'), TextContentItem(text='Result 1:\nDocument_id:num-1\nContent:  conversational data, :func:`~torchtune.datasets.chat_dataset` seems to be a good fit. ..., type='text'), TextContentItem(text='END of knowledge_search tool results.\n', type='text')]
inference> Here is a high-level overview of the text:

**LoRA Finetuning with PyTorch Tune**

PyTorch Tune provides a recipe for LoRA (Low-Rank Adaptation) finetuning, which is a technique to adapt pre-trained models to new tasks. The recipe uses the `lora_finetune_distributed` command.
...
Overall, DORA is a powerful reinforcement learning algorithm that can learn complex tasks from human demonstrations. However, it requires careful consideration of the challenges and limitations to achieve optimal results.
```
## Next Steps

- Learn more about Llama Stack [Concepts](../concepts/index.md)
- Learn how to [Build Llama Stacks](../distributions/index.md)
- See [References](../references/index.md) for more details about the llama CLI and Python SDK
- For example applications and more detailed tutorials, visit our [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main/examples) repository.
