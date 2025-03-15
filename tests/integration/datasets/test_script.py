from llama_stack_client import LlamaStackClient
from rich.pretty import pprint


def test_register_dataset():
    client = LlamaStackClient(base_url="http://localhost:8321")
    dataset = client.datasets.register(
        purpose="eval/messages-answer",
        source={"type": "uri", "uri": "huggingface://llamastack/simpleqa?split=train"},
    )
    pprint(dataset)


if __name__ == "__main__":
    test_register_dataset()
