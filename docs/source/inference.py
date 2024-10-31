import asyncio

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.inference.event_logger import EventLogger
from llama_stack_client.types import UserMessage
from termcolor import cprint


client = LlamaStackClient(
    base_url=f"http://localhost:5000",
)
message = UserMessage(
    content="hello world, write me a 2 sentence poem about the moon", role="user"
)

cprint(f"User>{message.content}", "green")
response = client.inference.chat_completion(
    messages=[message],
    model="Llama3.2-11B-Vision-Instruct",
)

cprint(f"> Response: {response.completion_message.content}", "cyan")
