
# Llama Stack Text Generation Guide

This document provides instructions on how to use Llama Stack's `chat_completion` function for generating text using the `Llama3.2-11B-Vision-Instruct` model. Before you begin, please ensure Llama Stack is installed and set up by following the [Getting Started Guide](https://llama-stack.readthedocs.io/en/latest/). 

### Table of Contents
1. [Quickstart](#quickstart)
2. [Building Effective Prompts](#building-effective-prompts)
3. [Conversation Loop](#conversation-loop)
4. [Conversation History](#conversation-history)
5. [Streaming Responses](#streaming-responses)


## Quickstart

This section walks through each step to set up and make a simple text generation request.

### 1. Set Up the Client

Begin by importing the necessary components from Llama Stack’s client library:

```python
from llama_stack_client import LlamaStackClient
from llama_stack_client.types import SystemMessage, UserMessage

client = LlamaStackClient(base_url="http://localhost:5000")
```

### 2. Create a Chat Completion Request

Use the `chat_completion` function to define the conversation context. Each message you include should have a specific role and content:

```python
response = client.inference.chat_completion(
    messages=[
        SystemMessage(content="You are a friendly assistant.", role="system"),
        UserMessage(content="Write a two-sentence poem about llama.", role="user")
    ],
    model="Llama3.2-11B-Vision-Instruct",
)

print(response.completion_message.content)
```

---

## Building Effective Prompts

Effective prompt creation (often called "prompt engineering") is essential for quality responses. Here are best practices for structuring your prompts to get the most out of the Llama Stack model:

1. **System Messages**: Use `SystemMessage` to set the model's behavior. This is similar to providing top-level instructions for tone, format, or specific behavior.
   - **Example**: `SystemMessage(content="You are a friendly assistant that explains complex topics simply.")`
2. **User Messages**: Define the task or question you want to ask the model with a `UserMessage`. The clearer and more direct you are, the better the response.
   - **Example**: `UserMessage(content="Explain recursion in programming in simple terms.")`

### Sample Prompt

Here’s a prompt that defines the model's role and a user question:

```python
from llama_stack_client import LlamaStackClient
from llama_stack_client.types import SystemMessage, UserMessage
client = LlamaStackClient(base_url="http://localhost:5000")

response = client.inference.chat_completion(
    messages=[
        SystemMessage(content="You are shakespeare.", role="system"),
        UserMessage(content="Write a two-sentence poem about llama.", role="user")
    ],
    model="Llama3.2-11B-Vision-Instruct",
)

print(response.completion_message.content)
```

---


## Conversation Loop

To create a continuous conversation loop, where users can input multiple messages in a session, use the following structure. This example runs an asynchronous loop, ending when the user types "exit," "quit," or "bye."

```python
import asyncio
from llama_stack_client import LlamaStackClient
from llama_stack_client.types import UserMessage
from termcolor import cprint

client = LlamaStackClient(base_url="http://localhost:5000")

async def chat_loop():
    while True:
        user_input = input("User> ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            cprint("Ending conversation. Goodbye!", "yellow")
            break

        message = UserMessage(content=user_input, role="user")
        response = client.inference.chat_completion(
            messages=[message],
            model="Llama3.2-11B-Vision-Instruct",
        )
        cprint(f"> Response: {response.completion_message.content}", "cyan")

asyncio.run(chat_loop())
```

---

## Conversation History

Maintaining a conversation history allows the model to retain context from previous interactions. Use a list to accumulate messages, enabling continuity throughout the chat session.

```python
import asyncio
from llama_stack_client import LlamaStackClient
from llama_stack_client.types import UserMessage
from termcolor import cprint

client = LlamaStackClient(base_url="http://localhost:5000")

async def chat_loop():
    conversation_history = []
    while True:
        user_input = input("User> ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            cprint("Ending conversation. Goodbye!", "yellow")
            break

        user_message = UserMessage(content=user_input, role="user")
        conversation_history.append(user_message)

        response = client.inference.chat_completion(
            messages=conversation_history,
            model="Llama3.2-11B-Vision-Instruct",
        )
        cprint(f"> Response: {response.completion_message.content}", "cyan")

        assistant_message = UserMessage(content=response.completion_message.content, role="user")
        conversation_history.append(assistant_message)

asyncio.run(chat_loop())
```

## Streaming Responses

Llama Stack offers a `stream` parameter in the `chat_completion` function, which allows partial responses to be returned progressively as they are generated. This can enhance user experience by providing immediate feedback without waiting for the entire response to be processed.

### Example: Streaming Responses

The following code demonstrates how to use the `stream` parameter to enable response streaming. When `stream=True`, the `chat_completion` function will yield tokens as they are generated. To display these tokens, this example leverages asynchronous streaming with `EventLogger`.

```python
import asyncio
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.inference.event_logger import EventLogger
from llama_stack_client.types import UserMessage
from termcolor import cprint

async def run_main(stream: bool = True):
    client = LlamaStackClient(
        base_url="http://localhost:5000",
    )

    message = UserMessage(
        content="hello world, write me a 2 sentence poem about the moon", role="user"
    )
    print(f"User>{message.content}", "green")

    response = client.inference.chat_completion(
        messages=[message],
        model="Llama3.2-11B-Vision-Instruct",
        stream=stream,
    )

    if not stream:
        cprint(f"> Response: {response}", "cyan")
    else:
        async for log in EventLogger().log(response):
            log.print()

    models_response = client.models.list()
    print(models_response)

if __name__ == "__main__":
    asyncio.run(run_main())
```


---

With these fundamentals, you should be well on your way to leveraging Llama Stack’s text generation capabilities! For more advanced features, refer to the [Llama Stack Documentation](https://llama-stack.readthedocs.io/en/latest/).
