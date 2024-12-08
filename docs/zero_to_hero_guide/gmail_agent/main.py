import argparse
import gmagent
import asyncio
from gmagent import *
from functions_prompt import * #system_prompt

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import (
    AgentConfig,
)

LLAMA_STACK_API_TOGETHER_URL="https://llama-stack.together.ai"
LLAMA31_8B_INSTRUCT = "Llama3.1-8B-Instruct"

async def create_gmail_agent(client: LlamaStackClient) -> Agent:
    """Create an agent with gmail tool capabilities."""

    listEmailsTool = ListEmailsTool()
    getEmailTool = GetEmailTool()
    sendEmailTool = SendEmailTool()
    getPDFSummaryTool = GetPDFSummaryTool()
    createDraftTool = CreateDraftTool()
    sendDraftTool = SendDraftTool()

    agent_config = AgentConfig(
        model=LLAMA31_8B_INSTRUCT,
        instructions=system_prompt,
        sampling_params={
            "strategy": "greedy",
            "temperature": 0.0,
            "top_p": 0.9,
        },
        tools=[
            listEmailsTool.get_tool_definition(),
            getEmailTool.get_tool_definition(),
            sendEmailTool.get_tool_definition(),
            getPDFSummaryTool.get_tool_definition(),
            createDraftTool.get_tool_definition(),
            sendDraftTool.get_tool_definition(),

        ],
        tool_choice="auto",
        tool_prompt_format="json",
        input_shields=[],
        output_shields=[],
        enable_session_persistence=True
    )

    agent = Agent(
        client=client,
        agent_config=agent_config,
        custom_tools=[listEmailsTool,
                      getEmailTool,
                      sendEmailTool,
                      getPDFSummaryTool,
                      createDraftTool,
                      sendDraftTool]
    )

    return agent





async def main():
    parser = argparse.ArgumentParser(description="Set email address")
    parser.add_argument("--gmail", type=str, required=True, help="Your Gmail address")
    args = parser.parse_args()

    gmagent.set_email_service(args.gmail)

    greeting = llama31("hello", "Your name is Gmagent, an assistant that can perform all Gmail related tasks for your user.")
    agent_response = f"{greeting}\n\nYour ask: "
    #agent = Agent(system_prompt)

    while True:
        ask = input(agent_response)
        if ask == "bye":
            print(llama31("bye"))
            break
        print("\n-------------------------\nCalling Llama...")
        # agent(ask)
        # agent_response = "Your ask: "


        client = LlamaStackClient(base_url=LLAMA_STACK_API_TOGETHER_URL)
        agent = await create_gmail_agent(client)
        session_id = agent.create_session("email-session")

        queries = [
            "do i have any emails with attachments?",
            "what's the content of the email from LangSmith",
        ]

        for query in queries:
            print(f"\nQuery: {query}")
            print("-" * 50)

            response = agent.create_turn(
                messages=[{"role": "user", "content": query}],
                session_id=session_id,
            )

            async for log in EventLogger().log(response):
                log.print()



if __name__ == "__main__":
    asyncio.run(main())



