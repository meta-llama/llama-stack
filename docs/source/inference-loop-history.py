import asyncio

from llama_stack_client import LlamaStackClient
from llama_stack_client.types import UserMessage
from termcolor import cprint

client = LlamaStackClient(
    base_url="http://localhost:5000",
)


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

        assistant_message = UserMessage(
            content=response.completion_message.content, role="user"
        )
        conversation_history.append(assistant_message)


asyncio.run(chat_loop())
