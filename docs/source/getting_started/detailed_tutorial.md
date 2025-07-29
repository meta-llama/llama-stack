## Detailed Tutorial

In this guide, we'll walk through how you can use the Llama Stack (server and client SDK) to test a simple agent.
A Llama Stack agent is a simple integrated system that can perform tasks by combining a Llama model for reasoning with
tools (e.g., RAG, web search, code execution, etc.) for taking actions.
In Llama Stack, we provide a server exposing multiple APIs. These APIs are backed by implementations from different providers.

Llama Stack is a stateful service with REST APIs to support seamless transition of AI applications across different environments. The server can be run in a variety of ways, including as a standalone binary, Docker container, or hosted service. You can build and test using a local server first and deploy to a hosted endpoint for production.

In this guide, we'll walk through how to build a RAG agent locally using Llama Stack with [Ollama](https://ollama.com/)
as the inference [provider](../providers/index.md#inference) for a Llama Model.

### Step 1: Installation and Setup

Install Ollama by following the instructions on the [Ollama website](https://ollama.com/download), then
download Llama 3.2 3B model, and then start the Ollama service.
```bash
ollama pull llama3.2:3b
ollama run llama3.2:3b --keepalive 60m
```

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

Setup your virtual environment.

```bash
uv sync --python 3.12
source .venv/bin/activate
```
### Step 2:  Run Llama Stack
Llama Stack is a server that exposes multiple APIs, you connect with it using the Llama Stack client SDK.

::::{tab-set}

:::{tab-item} Using `venv`
You can use Python to build and run the Llama Stack server, which is useful for testing and development.

Llama Stack uses a [YAML configuration file](../distributions/configuration.md) to specify the stack setup,
which defines the providers and their settings. The generated configuration serves as a starting point that you can [customize for your specific needs](../distributions/customizing_run_yaml.md).
Now let's build and run the Llama Stack config for Ollama.
We use `starter` as template. By default all providers are disabled, this requires enable ollama by passing environment variables.

```bash
llama stack build --template starter --image-type venv --run
```
:::
:::{tab-item} Using `conda`
You can use Python to build and run the Llama Stack server, which is useful for testing and development.

Llama Stack uses a [YAML configuration file](../distributions/configuration.md) to specify the stack setup,
which defines the providers and their settings.
Now let's build and run the Llama Stack config for Ollama.

```bash
llama stack build --template starter --image-type conda --run
```
:::
:::{tab-item} Using a Container
You can use a container image to run the Llama Stack server. We provide several container images for the server
component that works with different inference providers out of the box. For this guide, we will use
`llamastack/distribution-starter` as the container image. If you'd like to build your own image or customize the
configurations, please check out [this guide](../distributions/building_distro.md).
First lets setup some environment variables and create a local directory to mount into the container’s file system.
```bash
export LLAMA_STACK_PORT=8321
mkdir -p ~/.llama
```
Then start the server using the container tool of your choice.  For example, if you are running Docker you can use the
following command:
```bash
docker run -it \
  --pull always \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ~/.llama:/root/.llama \
  llamastack/distribution-starter \
  --port $LLAMA_STACK_PORT \
  --env OLLAMA_URL=http://host.docker.internal:11434
```
Note to start the container with Podman, you can do the same but replace `docker` at the start of the command with
`podman`. If you are using `podman` older than `4.7.0`, please also replace `host.docker.internal` in the `OLLAMA_URL`
with `host.containers.internal`.

The configuration YAML for the Ollama distribution is available at `distributions/ollama/run.yaml`.

```{tip}

Docker containers run in their own isolated network namespaces on Linux. To allow the container to communicate with services running on the host via `localhost`, you need `--network=host`. This makes the container use the host’s network directly so it can connect to Ollama running on `localhost:11434`.

Linux users having issues running the above command should instead try the following:
```bash
docker run -it \
  --pull always \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ~/.llama:/root/.llama \
  --network=host \
  llamastack/distribution-starter \
  --port $LLAMA_STACK_PORT \
  --env OLLAMA_URL=http://localhost:11434
```
:::
::::
You will see output like below:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://['::', '0.0.0.0']:8321 (Press CTRL+C to quit)
```

Now you can use the Llama Stack client to run inference and build agents!

You can reuse the server setup or use the [Llama Stack Client](https://github.com/meta-llama/llama-stack-client-python/).
Note that the client package is already included in the `llama-stack` package.

### Step 3: Run Client CLI

Open a new terminal and navigate to the same directory you started the server from. Then set up a new or activate your
existing server virtual environment.

::::{tab-set}

:::{tab-item} Reuse Server `venv`
```bash
# The client is included in the llama-stack package so we just activate the server venv
source .venv/bin/activate
```
:::

:::{tab-item} Install with `venv`
```bash
uv venv client --python 3.12
source client/bin/activate
pip install llama-stack-client
```
:::

:::{tab-item} Install with `conda`
```bash
yes | conda create -n stack-client python=3.12
conda activate stack-client
pip install llama-stack-client
```
:::
::::

Now let's use the `llama-stack-client` [CLI](../references/llama_stack_client_cli_reference.md) to check the
connectivity to the server.

```bash
llama-stack-client configure --endpoint http://localhost:8321 --api-key none
```
You will see the below:
```
Done! You can now use the Llama Stack Client CLI with endpoint http://localhost:8321
```

List the models
```bash
llama-stack-client models list
Available Models

┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ model_type      ┃ identifier                          ┃ provider_resource_id                ┃ metadata                                  ┃ provider_id           ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│ embedding       │ ollama/all-minilm:l6-v2             │ all-minilm:l6-v2                    │ {'embedding_dimension': 384.0}            │ ollama                │
├─────────────────┼─────────────────────────────────────┼─────────────────────────────────────┼───────────────────────────────────────────┼───────────────────────┤
│ ...             │ ...                                 │ ...                                 │                                           │ ...                   │
├─────────────────┼─────────────────────────────────────┼─────────────────────────────────────┼───────────────────────────────────────────┼───────────────────────┤
│ llm             │ ollama/Llama-3.2:3b                 │ llama3.2:3b                         │                                           │ ollama                │
└─────────────────┴─────────────────────────────────────┴─────────────────────────────────────┴───────────────────────────────────────────┴───────────────────────┘

```
You can test basic Llama inference completion using the CLI.

```bash
llama-stack-client inference chat-completion --model-id "ollama/llama3.2:3b" --message "tell me a joke"

```
Sample output:
```python
OpenAIChatCompletion(
    id="chatcmpl-08d7b2be-40f3-47ed-8f16-a6f29f2436af",
    choices=[
        OpenAIChatCompletionChoice(
            finish_reason="stop",
            index=0,
            message=OpenAIChatCompletionChoiceMessageOpenAIAssistantMessageParam(
                role="assistant",
                content="Why couldn't the bicycle stand up by itself?\n\nBecause it was two-tired.",
                name=None,
                tool_calls=None,
                refusal=None,
                annotations=None,
                audio=None,
                function_call=None,
            ),
            logprobs=None,
        )
    ],
    created=1751725254,
    model="llama3.2:3b",
    object="chat.completion",
    service_tier=None,
    system_fingerprint="fp_ollama",
    usage={
        "completion_tokens": 18,
        "prompt_tokens": 29,
        "total_tokens": 47,
        "completion_tokens_details": None,
        "prompt_tokens_details": None,
    },
)
```

### Step 4: Run the Demos

Note that these demos show the [Python Client SDK](../references/python_sdk_reference/index.md).
Other SDKs are also available, please refer to the [Client SDK](../index.md#client-sdks) list for the complete options.

::::{tab-set}

:::{tab-item} Basic Inference
Now you can run inference using the Llama Stack client SDK.

#### i. Create the Script

Create a file `inference.py` and add the following code:
```python
from llama_stack_client import LlamaStackClient

client = LlamaStackClient(base_url="http://localhost:8321")

# List available models
models = client.models.list()

# Select the first LLM
llm = next(m for m in models if m.model_type == "llm" and m.provider_id == "ollama")
model_id = llm.identifier

print("Model:", model_id)

response = client.chat.completions.create(
    model=model_id,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about coding"},
    ],
)
print(response)
```

#### ii. Run the Script
Let's run the script using `uv`
```bash
uv run python inference.py
```
Which will output:
```
Model: ollama/llama3.2:3b
OpenAIChatCompletion(id='chatcmpl-30cd0f28-a2ad-4b6d-934b-13707fc60ebf', choices=[OpenAIChatCompletionChoice(finish_reason='stop', index=0, message=OpenAIChatCompletionChoiceMessageOpenAIAssistantMessageParam(role='assistant', content="Lines of code unfold\nAlgorithms dance with ease\nLogic's gentle kiss", name=None, tool_calls=None, refusal=None, annotations=None, audio=None, function_call=None), logprobs=None)], created=1751732480, model='llama3.2:3b', object='chat.completion', service_tier=None, system_fingerprint='fp_ollama', usage={'completion_tokens': 16, 'prompt_tokens': 37, 'total_tokens': 53, 'completion_tokens_details': None, 'prompt_tokens_details': None})
```
:::

:::{tab-item} Build a Simple Agent
Next we can move beyond simple inference and build an agent that can perform tasks using the Llama Stack server.
#### i. Create the Script
Create a file `agent.py` and add the following code:

```python
from llama_stack_client import LlamaStackClient
from llama_stack_client import Agent, AgentEventLogger
from rich.pretty import pprint
import uuid

client = LlamaStackClient(base_url=f"http://localhost:8321")

models = client.models.list()
llm = next(m for m in models if m.model_type == "llm" and m.provider_id == "ollama")
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

```{dropdown} 👋 Click here to see the sample output
    Non-streaming ...
    agent> I'm an artificial intelligence designed to assist and communicate with users like you. I don't have a personal identity, but I can provide information, answer questions, and help with tasks to the best of my abilities.

    I'm a large language model, which means I've been trained on a massive dataset of text from various sources, allowing me to understand and respond to a wide range of topics and questions. My purpose is to provide helpful and accurate information, and I'm constantly learning and improving my responses based on the interactions I have with users like you.

    I can help with:

    * Answering questions on various subjects
    * Providing definitions and explanations
    * Offering suggestions and ideas
    * Assisting with language-related tasks, such as proofreading and editing
    * Generating text and content
    * And more!

    Feel free to ask me anything, and I'll do my best to help!
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
    inference> Déjà vu! You're asking me again!

    As I mentioned earlier, I'm a computer program designed to simulate conversation and answer questions. I don't have a personal identity or consciousness like a human would. I exist solely as a digital entity, running on computer servers and responding to inputs from users like you.

    I'm a type of artificial intelligence (AI) called a large language model, which means I've been trained on a massive dataset of text from various sources. This training allows me to understand and respond to a wide range of questions and topics.

    My purpose is to provide helpful and accurate information, answer questions, and assist users like you with tasks and conversations. I don't have personal preferences, emotions, or opinions like humans do. My goal is to be informative, neutral, and respectful in my responses.

    So, that's me in a nutshell!
```
:::

:::{tab-item} Build a RAG Agent

For our last demo, we can build a RAG agent that can answer questions about the Torchtune project using the documents
in a vector database.
#### i. Create the Script
Create a file `rag_agent.py` and add the following code:

```python
from llama_stack_client import LlamaStackClient
from llama_stack_client import Agent, AgentEventLogger
from llama_stack_client.types import Document
import uuid

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
llm = next(
    m
    for m in client.models.list()
    if m.model_type == "llm" and m.provider_id == "ollama"
)
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
#### ii. Run the Script
Let's run the script using `uv`
```bash
uv run python rag_agent.py
```

```{dropdown} 👋 Click here to see the sample output
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

::::

**You're Ready to Build Your Own Apps!**

Congrats! 🥳 Now you're ready to [build your own Llama Stack applications](../building_applications/index)! 🚀
