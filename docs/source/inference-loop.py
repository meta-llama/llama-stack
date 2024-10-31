import asyncio

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.inference.event_logger import EventLogger
from llama_stack_client.types import UserMessage
from termcolor import cprint

client = LlamaStackClient(
    base_url="http://localhost:5000",
)


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
