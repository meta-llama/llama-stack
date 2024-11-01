
# Few-Shot Inference for LLMs

This guide provides instructions on how to use Llama Stack’s `chat_completion` API with a few-shot learning approach to enhance text generation. Few-shot examples enable the model to recognize patterns by providing labeled prompts, allowing it to complete tasks based on minimal prior examples.

### Overview

Few-shot learning provides the model with multiple examples of input-output pairs. This is particularly useful for guiding the model's behavior in specific tasks, helping it understand the desired completion format and content based on a few sample interactions.

### Implementation

1. **Initialize the Client**

   Begin by setting up the `LlamaStackClient` to connect to the inference endpoint.

   ```python
   from llama_stack_client import LlamaStackClient

   client = LlamaStackClient(base_url="http://localhost:5000")
   ```

2. **Define Few-Shot Examples**

   Construct a series of labeled `UserMessage` and `CompletionMessage` instances to demonstrate the task to the model. Each `UserMessage` represents an input prompt, and each `CompletionMessage` is the desired output. The model uses these examples to infer the appropriate response patterns.

   ```python
   from llama_stack_client.types import CompletionMessage, UserMessage

   few_shot_examples =  messages=[
        UserMessage(content="Have shorter, spear-shaped ears.", role="user"),
        CompletionMessage(
            content="That's Alpaca!",
            role="assistant",
            stop_reason="end_of_message",
            tool_calls=[],
        ),
        UserMessage(
            content="Known for their calm nature and used as pack animals in mountainous regions.",
            role="user",
        ),
        CompletionMessage(
            content="That's Llama!",
            role="assistant",
            stop_reason="end_of_message",
            tool_calls=[],
        ),
        UserMessage(
            content="Has a straight, slender neck and is smaller in size compared to its relative.",
            role="user",
        ),
        CompletionMessage(
            content="That's Alpaca!",
            role="assistant",
            stop_reason="end_of_message",
            tool_calls=[],
        ),
        UserMessage(
            content="Generally taller and more robust, commonly seen as guard animals.",
            role="user",
        ),
    ]
   ```

   ### Note
   - **Few-Shot Examples**: These examples show the model the correct responses for specific prompts.
   - **CompletionMessage**: This defines the model's expected completion for each prompt.

3. **Invoke `chat_completion` with Few-Shot Examples**

   Use the few-shot examples as the message input for `chat_completion`. The model will use the examples to generate contextually appropriate responses, allowing it to infer and complete new queries in a similar format.

   ```python
   response = client.inference.chat_completion(
       messages=few_shot_examples, model="Llama3.2-11B-Vision-Instruct"
   )
   ```

4. **Display the Model’s Response**

   The `completion_message` contains the assistant’s generated content based on the few-shot examples provided. Output this content to see the model's response directly in the console.

   ```python
   from termcolor import cprint

   cprint(f"> Response: {response.completion_message.content}", "cyan")
   ```

Few-shot learning with Llama Stack’s `chat_completion` allows the model to recognize patterns with minimal training data, helping it generate contextually accurate responses based on prior examples. This approach is highly effective for guiding the model in tasks that benefit from clear input-output examples without extensive fine-tuning.


### Complete code
Summing it up, here's the code for few-shot implementation with llama-stack:

```python
from llama_stack_client import LlamaStackClient
from llama_stack_client.types import CompletionMessage, UserMessage
from termcolor import cprint

client = LlamaStackClient(base_url="http://localhost:5000")

response = client.inference.chat_completion(
    messages=[
        UserMessage(content="Have shorter, spear-shaped ears.", role="user"),
        CompletionMessage(
            content="That's Alpaca!",
            role="assistant",
            stop_reason="end_of_message",
            tool_calls=[],
        ),
        UserMessage(
            content="Known for their calm nature and used as pack animals in mountainous regions.",
            role="user",
        ),
        CompletionMessage(
            content="That's Llama!",
            role="assistant",
            stop_reason="end_of_message",
            tool_calls=[],
        ),
        UserMessage(
            content="Has a straight, slender neck and is smaller in size compared to its relative.",
            role="user",
        ),
        CompletionMessage(
            content="That's Alpaca!",
            role="assistant",
            stop_reason="end_of_message",
            tool_calls=[],
        ),
        UserMessage(
            content="Generally taller and more robust, commonly seen as guard animals.",
            role="user",
        ),
    ],
    model="Llama3.2-11B-Vision-Instruct",
)

cprint(f"> Response: {response.completion_message.content}", "cyan")
```

---

With this fundamental, you should be well on your way to leveraging Llama Stack’s text generation capabilities! For more advanced features, refer to the [Llama Stack Documentation](https://llama-stack.readthedocs.io/en/latest/).

