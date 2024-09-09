import fire
from llama_stack import LlamaStack
from llama_stack.types import UserMessage

def main(host: str, port: int):
    client = LlamaStack(
        base_url=f"http://{host}:{port}",
    )

    agentic_system_create_response = client.agentic_system.create(
        agent_config={
            "instructions": "You are a helpful assistant",
            "model": "Meta-Llama3.1-8B-Instruct",
        },
    )
    print(agentic_system_create_response)

    agentic_system_create_session_response = client.agentic_system.sessions.create(
        agent_id=agentic_system_create_response.agent_id,
        session_name="test_session",
    )
    print(agentic_system_create_session_response)

    # TODO(xiyan): remove request wrapper
    response = client.agentic_system.turns.create(
        request={
            "agent_id": agentic_system_create_response.agent_id,
            "session_id": agentic_system_create_session_response.session_id,
            "messages": [
                UserMessage(content="What is the capital of France?", role="user"),
            ],
            "stream": False,
        }
    )

    print(response)


if __name__ == "__main__":
    fire.Fire(main)
