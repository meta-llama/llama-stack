import asyncio

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.inference.event_logger import EventLogger
from llama_stack_client.types import UserMessage
from termcolor import cprint


async def run_main(stream: bool = True):
    client = LlamaStackClient(
        base_url=f"http://localhost:5000",
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
