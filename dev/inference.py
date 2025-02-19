import os
import sys


def create_http_client():
    from llama_stack_client import LlamaStackClient

    return LlamaStackClient(
        base_url=f"http://localhost:{os.environ['LLAMA_STACK_PORT']}"
    )


def create_library_client(template="ollama"):
    from llama_stack import LlamaStackAsLibraryClient

    client = LlamaStackAsLibraryClient(template)
    if not client.initialize():
        print("llama stack not built properly")
        sys.exit(1)
    return client


client = (
    create_library_client()
)  # or create_http_client() depending on the environment you picked

# List available models
models = client.models.list()
print("--- Available models: ---")
for m in models:
    print(f"- {m.identifier}")
print()

response = client.inference.chat_completion(
    model_id=os.environ["INFERENCE_MODEL"],
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about coding"},
    ],
)
print(response.completion_message.content)