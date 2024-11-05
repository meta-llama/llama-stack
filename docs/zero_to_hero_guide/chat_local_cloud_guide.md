
# Switching between Local and Cloud Model with Llama Stack

This guide provides a streamlined setup to switch between local and cloud clients for text generation with Llama Stack’s `chat_completion` API. This setup enables automatic fallback to a cloud instance if the local client is unavailable.


### Pre-requisite
Before you begin, please ensure Llama Stack is installed and the distribution are set up by following the [Getting Started Guide](https://llama-stack.readthedocs.io/en/latest/). You will need to run two distribution, a local and a cloud distribution, for this demo to work.

<!--- [TODO: show how to create two distributions] --->

### Implementation

1. **Set Up Local and Cloud Clients**

   Initialize both clients, specifying the `base_url` for you intialized each instance. In this case, we have the local distribution running on `http://localhost:5000` and the cloud distribution running on `http://localhost:5001`.

   ```python
   from llama_stack_client import LlamaStackClient

   # Configure local and cloud clients
   local_client = LlamaStackClient(base_url="http://localhost:5000")
   cloud_client = LlamaStackClient(base_url="http://localhost:5001")
   ```

2. **Client Selection with Fallback**

   The `select_client` function checks if the local client is available using a lightweight `/health` check. If the local client is unavailable, it automatically switches to the cloud client.

   ```python
   import httpx
   from termcolor import cprint

   async def select_client() -> LlamaStackClient:
       """Use local client if available; otherwise, switch to cloud client."""
       try:
           async with httpx.AsyncClient() as http_client:
               response = await http_client.get(f"{local_client.base_url}/health")
               if response.status_code == 200:
                   cprint("Using local client.", "yellow")
                   return local_client
       except httpx.RequestError:
           pass
       cprint("Local client unavailable. Switching to cloud client.", "yellow")
       return cloud_client
   ```

3. **Generate a Response**

   After selecting the client, you can generate text using `chat_completion`. This example sends a sample prompt to the model and prints the response.

   ```python
   from llama_stack_client.types import UserMessage

   async def get_llama_response(stream: bool = True):
       client = await select_client()  # Selects the available client
       message = UserMessage(content="hello world, write me a 2 sentence poem about the moon", role="user")
       cprint(f"User> {message.content}", "green")

       response = client.inference.chat_completion(
           messages=[message],
           model="Llama3.2-11B-Vision-Instruct",
           stream=stream,
       )

       if not stream:
           cprint(f"> Response: {response}", "cyan")
       else:
           # Stream tokens progressively
           async for log in EventLogger().log(response):
               log.print()
   ```

4. **Run the Asynchronous Response Generation**

   Use `asyncio.run()` to execute `get_llama_response` in an asynchronous event loop.

   ```python
   import asyncio

   # Initiate the response generation process
   asyncio.run(get_llama_response())
   ```


### Complete code
Summing it up, here's the code for local-cloud model implementation with llama-stack:

```python
import asyncio

import httpx
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.inference.event_logger import EventLogger
from llama_stack_client.types import UserMessage
from termcolor import cprint

local_client = LlamaStackClient(base_url="http://localhost:5000")
cloud_client = LlamaStackClient(base_url="http://localhost:5001")


async def select_client() -> LlamaStackClient:
    try:
        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(f"{local_client.base_url}/health")
            if response.status_code == 200:
                cprint("Using local client.", "yellow")
                return local_client
    except httpx.RequestError:
        pass
    cprint("Local client unavailable. Switching to cloud client.", "yellow")
    return cloud_client


async def get_llama_response(stream: bool = True):
    client = await select_client()
    message = UserMessage(
        content="hello world, write me a 2 sentence poem about the moon", role="user"
    )
    cprint(f"User> {message.content}", "green")

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


asyncio.run(get_llama_response())
```

---

With this fundamental, you should be well on your way to leveraging Llama Stack’s text generation capabilities! For more advanced features, refer to the [Llama Stack Documentation](https://llama-stack.readthedocs.io/en/latest/).
