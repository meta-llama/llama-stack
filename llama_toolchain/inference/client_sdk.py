import fire
from llama_stack import LlamaStack
from llama_stack.types import UserMessage

def main(host: str, port: int):
    client = LlamaStack(
        base_url=f"http://{host}:{port}",
    )

    response = client.inference.chat_completion(
        request={
            "messages": [
                UserMessage(content="hello world, troll me in two-paragraphs about 42", role="user"),
            ],
            "model": "Meta-Llama3.1-8B-Instruct",
            "stream": False,
        },
    )

    print(response)
    # TODO (xiyan). This does not work with current server, need to fix
    # response = client.inference.chat_completion(
    #     messages=[
    #         UserMessage(content="hello world, troll me in two-paragraphs about 42", role="user"),
    #     ],
    #     model="Meta-Llama3.1-8B-Instruct",
    #     stream=True,
    # )

if __name__ == "__main__":
    fire.Fire(main)
