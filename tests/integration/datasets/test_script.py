from llama_stack_client import LlamaStackClient
from rich.pretty import pprint


def test_register_dataset():
    client = LlamaStackClient(base_url="http://localhost:8321")
    dataset = client.datasets.register(
        purpose="eval/messages-answer",
        source={
            "type": "uri",
            "uri": "huggingface://datasets/llamastack/simpleqa?split=train",
        },
    )
    dataset_id = dataset.identifier
    pprint(dataset)
    rows = client.datasets.iterrows(dataset_id=dataset_id, limit=10)
    pprint(rows)


if __name__ == "__main__":
    test_register_dataset()
