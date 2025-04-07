# Quick Start

In this guide, we'll walk through how you can use the Llama Stack (server and client SDK) to test a simple agent.
A Llama Stack agent is a simple integrated system that can perform tasks by combining a Llama model for reasoning with
tools (e.g., RAG, web search, code execution, etc.) for taking actions.
In Llama Stack, we provide a server exposing multiple APIs. These APIs are backed by implementations from different providers.

Llama Stack is a stateful service with REST APIs to support seamless transition of AI applications across different environments. The server can be run in a variety of ways, including as a standalone binary, Docker container, or hosted service. You can build and test using a local server first and deploy to a hosted endpoint for production.

In this guide, we'll walk through how to build a RAG agent locally using Llama Stack with [Ollama](https://ollama.com/)
as the inference [provider](../providers/index.md#inference) for a Llama Model.

## Step 1: Installation and Setup


### i. Install and Start Ollama for Inference

Install Ollama by following the instructions on the [Ollama website](https://ollama.com/download).

To start Ollama run:
```bash
ollama run llama3.2:3b --keepalive 60m
```

By default, Ollama keeps the model loaded in memory for 5 minutes which can be too short. We set the `--keepalive` flag to 60 minutes to ensure the model remains loaded for sometime.

### ii. Install `uv` to Manage your Python packages

Install [uv](https://docs.astral.sh/uv/) to setup your virtual environment

::::{tab-set}

:::{tab-item} macOS and Linux
Use `curl` to download the script and execute it with `sh`:
```console
curl -LsSf https://astral.sh/uv/install.sh | sh
```
:::

:::{tab-item} Windows
Use `irm` to download the script and execute it with `iex`:

```console
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
:::
::::

### iii. Setup your Virtual Environment
```bash
uv venv --python 3.10
source .venv/bin/activate
```
## Step 2: Install Llama Stack
Llama Stack is a server that exposes multiple APIs, you connect with it using the Llama Stack client SDK.

### Install the Llama Stack Server
```bash
uv pip install llama-stack
```
Note the Llama Stack Server includes the client SDK as well.

## Step 3: Build and Run Llama Stack
Llama Stack uses a [configuration file](../distributions/configuration.md) to define the stack.
The config file is a YAML file that specifies the providers and their configurations.

### i. Build and Run the Llama Stack Config for Ollama
```bash
INFERENCE_MODEL=llama3.2:3b llama stack build --template ollama --image-type venv --run
```
You will see output like below:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://['::', '0.0.0.0']:8321 (Press CTRL+C to quit)
```

### ii. Using the Llama Stack Client
Now you can use the llama stack client to run inference and build agents! You can reuse the server setup or use the
[Llama Stack Client](https://github.com/meta-llama/llama-stack-client-python/). Note that the client package is already
included in the `llama-stack` package.

Open a new terminal and navigate to the same directory you started the server from. Then set up or activate your
virtual environment.

```bash
source .venv/bin/activate
```
Now let's use the `llama-stack-client` CLI to check the connectivity to the server.

```bash
llama-stack-client configure --endpoint http://localhost:8321 --api-key none
```
You will see the below:
```
Done! You can now use the Llama Stack Client CLI with endpoint http://localhost:8321
```

#### iii. List available models
List the models
```
llama-stack-client models list
Available Models

┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ model_type      ┃ identifier                          ┃ provider_resource_id                ┃ metadata                                  ┃ provider_id     ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ embedding       │ all-MiniLM-L6-v2                    │ all-minilm:latest                   │ {'embedding_dimension': 384.0}            │ ollama          │
├─────────────────┼─────────────────────────────────────┼─────────────────────────────────────┼───────────────────────────────────────────┼─────────────────┤
│ llm             │ llama3.2:3b                         │ llama3.2:3b                         │                                           │ ollama          │
└─────────────────┴─────────────────────────────────────┴─────────────────────────────────────┴───────────────────────────────────────────┴─────────────────┘

Total models: 2

```

## Step 4: Run Basic Inference
You can test basic Llama inference completion using the CLI.

```bash
llama-stack-client inference chat-completion --message "tell me a joke"
```
Sample output:
```python
ChatCompletionResponse(
    completion_message=CompletionMessage(
        content="Here's one:\n\nWhat do you call a fake noodle?\n\nAn impasta!",
        role="assistant",
        stop_reason="end_of_turn",
        tool_calls=[],
    ),
    logprobs=None,
    metrics=[
        Metric(metric="prompt_tokens", value=14.0, unit=None),
        Metric(metric="completion_tokens", value=27.0, unit=None),
        Metric(metric="total_tokens", value=41.0, unit=None),
    ],
)
```
Alternatively, you can run inference using the Llama Stack client SDK.

### i. Create the Script
Create a file `inference.py` and add the following code:
```python
from llama_stack_client import LlamaStackClient

client = LlamaStackClient(base_url="http://localhost:8321")

# List available models
models = client.models.list()

# Select the first LLM
llm = next(m for m in models if m.model_type == "llm")
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
### ii. Run the Script
Let's run the script using `uv`
```bash
uv run python inference.py
```
Which will output:
```
Model: llama3.2:3b-instruct-fp16
Here is a haiku about coding:

Lines of code unfold
Logic flows through digital night
Beauty in the bits
```

## Step 5: Build a Simple Agent
Now we can move beyond simple inference and build an agent that can perform tasks using the Llama Stack server.
### i. Create the Script
Create a file `agent.py` and add the following code:

```python
from llama_stack_client import LlamaStackClient
from llama_stack_client import Agent, AgentEventLogger
from rich.pretty import pprint
import uuid

client = LlamaStackClient(base_url=f"http://localhost:8321")

models = client.models.list()
llm = next(m for m in models if m.model_type == "llm")
model_id = llm.identifier

agent = Agent(client, model=model_id, instructions="You are a helpful assistant.")

s_id = agent.create_session(session_name=f"s{uuid.uuid4().hex}")

print("Non-streaming ...")
response = agent.create_turn(
    messages=[{"role": "user", "content": "Who are you?"}],
    session_id=s_id,
    stream=False,
)
print("agent>", response.output_message.content)

print("Streaming ...")
stream = agent.create_turn(
    messages=[{"role": "user", "content": "Who are you?"}], session_id=s_id, stream=True
)
for event in stream:
    pprint(event)

print("Streaming with print helper...")
stream = agent.create_turn(
    messages=[{"role": "user", "content": "Who are you?"}], session_id=s_id, stream=True
)
for event in AgentEventLogger().log(stream):
    event.print()
```
### ii. Run the Script
Let's run the script using `uv`
```bash
uv run python agent.py
```
:::{dropdown} `👋 Click here to see the sample output`
```
Non-streaming ...
agent> I'm an artificial intelligence designed to assist and communicate with users like you. I don't have a personal identity, but I'm here to provide information, answer questions, and help with tasks to the best of my abilities.

I can be used for a wide range of purposes, such as:

* Providing definitions and explanations
* Offering suggestions and ideas
* Helping with language translation
* Assisting with writing and proofreading
* Generating text or responses to questions
* Playing simple games or chatting about topics of interest

I'm constantly learning and improving my abilities, so feel free to ask me anything, and I'll do my best to help!

Streaming ...
AgentTurnResponseStreamChunk(
│   event=TurnResponseEvent(
│   │   payload=AgentTurnResponseStepStartPayload(
│   │   │   event_type='step_start',
│   │   │   step_id='69831607-fa75-424a-949b-e2049e3129d1',
│   │   │   step_type='inference',
│   │   │   metadata={}
│   │   )
│   )
)
AgentTurnResponseStreamChunk(
│   event=TurnResponseEvent(
│   │   payload=AgentTurnResponseStepProgressPayload(
│   │   │   delta=TextDelta(text='As', type='text'),
│   │   │   event_type='step_progress',
│   │   │   step_id='69831607-fa75-424a-949b-e2049e3129d1',
│   │   │   step_type='inference'
│   │   )
│   )
)
AgentTurnResponseStreamChunk(
│   event=TurnResponseEvent(
│   │   payload=AgentTurnResponseStepProgressPayload(
│   │   │   delta=TextDelta(text=' a', type='text'),
│   │   │   event_type='step_progress',
│   │   │   step_id='69831607-fa75-424a-949b-e2049e3129d1',
│   │   │   step_type='inference'
│   │   )
│   )
)
...
AgentTurnResponseStreamChunk(
│   event=TurnResponseEvent(
│   │   payload=AgentTurnResponseStepCompletePayload(
│   │   │   event_type='step_complete',
│   │   │   step_details=InferenceStep(
│   │   │   │   api_model_response=CompletionMessage(
│   │   │   │   │   content='As a conversational AI, I don\'t have a personal identity in the classical sense. I exist as a program running on computer servers, designed to process and respond to text-based inputs.\n\nI\'m an instance of a type of artificial intelligence called a "language model," which is trained on vast amounts of text data to generate human-like responses. My primary function is to understand and respond to natural language inputs, like our conversation right now.\n\nThink of me as a virtual assistant, a chatbot, or a conversational interface – I\'m here to provide information, answer questions, and engage in conversation to the best of my abilities. I don\'t have feelings, emotions, or consciousness like humans do, but I\'m designed to simulate human-like interactions to make our conversations feel more natural and helpful.\n\nSo, that\'s me in a nutshell! What can I help you with today?',
│   │   │   │   │   role='assistant',
│   │   │   │   │   stop_reason='end_of_turn',
│   │   │   │   │   tool_calls=[]
│   │   │   │   ),
│   │   │   │   step_id='69831607-fa75-424a-949b-e2049e3129d1',
│   │   │   │   step_type='inference',
│   │   │   │   turn_id='8b360202-f7cb-4786-baa9-166a1b46e2ca',
│   │   │   │   completed_at=datetime.datetime(2025, 4, 3, 1, 15, 21, 716174, tzinfo=TzInfo(UTC)),
│   │   │   │   started_at=datetime.datetime(2025, 4, 3, 1, 15, 14, 28823, tzinfo=TzInfo(UTC))
│   │   │   ),
│   │   │   step_id='69831607-fa75-424a-949b-e2049e3129d1',
│   │   │   step_type='inference'
│   │   )
│   )
)
AgentTurnResponseStreamChunk(
│   event=TurnResponseEvent(
│   │   payload=AgentTurnResponseTurnCompletePayload(
│   │   │   event_type='turn_complete',
│   │   │   turn=Turn(
│   │   │   │   input_messages=[UserMessage(content='Who are you?', role='user', context=None)],
│   │   │   │   output_message=CompletionMessage(
│   │   │   │   │   content='As a conversational AI, I don\'t have a personal identity in the classical sense. I exist as a program running on computer servers, designed to process and respond to text-based inputs.\n\nI\'m an instance of a type of artificial intelligence called a "language model," which is trained on vast amounts of text data to generate human-like responses. My primary function is to understand and respond to natural language inputs, like our conversation right now.\n\nThink of me as a virtual assistant, a chatbot, or a conversational interface – I\'m here to provide information, answer questions, and engage in conversation to the best of my abilities. I don\'t have feelings, emotions, or consciousness like humans do, but I\'m designed to simulate human-like interactions to make our conversations feel more natural and helpful.\n\nSo, that\'s me in a nutshell! What can I help you with today?',
│   │   │   │   │   role='assistant',
│   │   │   │   │   stop_reason='end_of_turn',
│   │   │   │   │   tool_calls=[]
│   │   │   │   ),
│   │   │   │   session_id='abd4afea-4324-43f4-9513-cfe3970d92e8',
│   │   │   │   started_at=datetime.datetime(2025, 4, 3, 1, 15, 14, 28722, tzinfo=TzInfo(UTC)),
│   │   │   │   steps=[
│   │   │   │   │   InferenceStep(
│   │   │   │   │   │   api_model_response=CompletionMessage(
│   │   │   │   │   │   │   content='As a conversational AI, I don\'t have a personal identity in the classical sense. I exist as a program running on computer servers, designed to process and respond to text-based inputs.\n\nI\'m an instance of a type of artificial intelligence called a "language model," which is trained on vast amounts of text data to generate human-like responses. My primary function is to understand and respond to natural language inputs, like our conversation right now.\n\nThink of me as a virtual assistant, a chatbot, or a conversational interface – I\'m here to provide information, answer questions, and engage in conversation to the best of my abilities. I don\'t have feelings, emotions, or consciousness like humans do, but I\'m designed to simulate human-like interactions to make our conversations feel more natural and helpful.\n\nSo, that\'s me in a nutshell! What can I help you with today?',
│   │   │   │   │   │   │   role='assistant',
│   │   │   │   │   │   │   stop_reason='end_of_turn',
│   │   │   │   │   │   │   tool_calls=[]
│   │   │   │   │   │   ),
│   │   │   │   │   │   step_id='69831607-fa75-424a-949b-e2049e3129d1',
│   │   │   │   │   │   step_type='inference',
│   │   │   │   │   │   turn_id='8b360202-f7cb-4786-baa9-166a1b46e2ca',
│   │   │   │   │   │   completed_at=datetime.datetime(2025, 4, 3, 1, 15, 21, 716174, tzinfo=TzInfo(UTC)),
│   │   │   │   │   │   started_at=datetime.datetime(2025, 4, 3, 1, 15, 14, 28823, tzinfo=TzInfo(UTC))
│   │   │   │   │   )
│   │   │   │   ],
│   │   │   │   turn_id='8b360202-f7cb-4786-baa9-166a1b46e2ca',
│   │   │   │   completed_at=datetime.datetime(2025, 4, 3, 1, 15, 21, 727364, tzinfo=TzInfo(UTC)),
│   │   │   │   output_attachments=[]
│   │   │   )
│   │   )
│   )
)


Streaming with print helper...
inference> Déjà vu!

As I mentioned earlier, I'm an artificial intelligence language model. I don't have a personal identity or consciousness like humans do. I exist solely to process and respond to text-based inputs, providing information and assistance on a wide range of topics.

I'm a computer program designed to simulate human-like conversations, using natural language processing (NLP) and machine learning algorithms to understand and generate responses. My purpose is to help users like you with their questions, provide information, and engage in conversation.

Think of me as a virtual companion, a helpful tool designed to make your interactions more efficient and enjoyable. I don't have personal opinions, emotions, or biases, but I'm here to provide accurate and informative responses to the best of my abilities.

So, who am I? I'm just a computer program designed to help you!

```
:::

## Step 6: Build a RAG Agent
For our last demo, we can build a RAG agent that can answer questions about the Torchtune project using the documents
in a vector database.
### i. Create the Script
Create a file `rag_agent.py` and add the following code:

```python
from llama_stack_client import LlamaStackClient
from llama_stack_client import Agent, AgentEventLogger
from llama_stack_client.types import Document
import uuid
from termcolor import cprint

client = LlamaStackClient(base_url="http://localhost:8321")

# Create a vector database instance
embed_lm = next(m for m in client.models.list() if m.model_type == "embedding")
embedding_model = embed_lm.identifier
vector_db_id = f"v{uuid.uuid4().hex}"
client.vector_dbs.register(
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
llm = next(m for m in client.models.list() if m.model_type == "llm")
model = llm.identifier

# Create the RAG agent
rag_agent = Agent(
    client,
    model=model,
    instructions="You are a helpful assistant. Use the RAG tool to answer questions as needed.",
    tools=[
        {
            "name": "builtin::rag/knowledge_search",
            "args": {"vector_db_ids": [vector_db_id]},
        }
    ],
)

session_id = rag_agent.create_session(session_name=f"s{uuid.uuid4().hex}")

turns = ["what is torchtune", "tell me about dora"]

for t in turns:
    print("user>", t)
    stream = rag_agent.create_turn(
        messages=[{"role": "user", "content": t}], session_id=session_id, stream=True
    )
    for event in AgentEventLogger().log(stream):
        event.print()
```
### ii. Run the Script
Let's run the script using `uv`
```bash
uv run python rag_agent.py
```
:::{dropdown} `👋 Click here to see the sample output`
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
:::
## Next Steps
- Go through the [Getting Started Notebook](https://github.com/meta-llama/llama-stack/blob/main/docs/getting_started.ipynb)
- Checkout more [Notebooks on GitHub](https://github.com/meta-llama/llama-stack/tree/main/docs/notebooks)
- Learn more about Llama Stack [Concepts](../concepts/index.md)
- Learn how to [Build Llama Stacks](../distributions/index.md)
- See [References](../references/index.md) for more details about the llama CLI and Python SDK
- For example applications and more detailed tutorials, visit our [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main/examples) repository.
