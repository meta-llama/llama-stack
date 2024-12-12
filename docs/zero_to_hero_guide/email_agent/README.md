# A Llama and Llama Stack Powered Email Agent

This is a Llama Stack port of the [Llama Powered Email Agent](https://github.com/meta-llama/llama-recipes/tree/gmagent/recipes/use_cases/email_agent) app that shows how to build an email agent app powered by Llama 3.1 8B and Llama Stack, using Llama Stack custom tool and agent APIs. 

Currently implemented features of the agent include:
* search for emails and attachments
* get email detail
* reply to a specific email 
* forward an email
* get summary of a PDF attachment
* draft and send an email

We'll mainly cover here how to port a Llama app using native custom tools supported in Llama 3.1 (and later) and an agent implementation from scratch to using Llama Stack APIs. See the link above for a comprehensive overview, definition, and resources of LLM agents.

# Setup and Installation

See the link above for Enable Gmail API and Install Ollama with Llama 3.1 8B.

## Install required packages
First, create a Conda or virtual env, then activate it and install the required Python libraries (slightly different from the original app because here we'll also install the `llama-stack-client` package):
```
git clone https://github.com/meta-llama/llama-stack
cd llama-stack/docs/zero_to_hero_guide/email_agent
pip install -r requirements.txt
```

# Run Email Agent

The steps are also the same as the [original app]((https://github.com/meta-llama/llama-recipes/tree/gmagent/recipes/use_cases/email_agent):

```
python main.py --gmail <your_gmail_address>
```

# Implementation Notes
Notes here mainly cover how custom tools (functions) are defined and how the Llama Stack Agent class is used with the custom tools.

## Available Custom Tool Definition
The `functions_prompt.py` defines the following six custom tools (functions), each as a subclass of Llama Stack's `CustomTool`, along with examples for each function call spec that Llama should return):

* ListEmailsTool
* GetEmailDetailTool
* SendEmailTool
* GetPDFSummaryTool
* CreateDraftTool
* SendDraftTool

Below is an example custom tool call spec in JSON format, for the user asks such as "do i have emails with attachments larger than 5mb", "any attachments larger than 5mb" or "let me know if i have large attachments over 5mb":
```
{"name": "list_emails", "parameters": {"query": "has:attachment larger:5mb"}}
```

Porting the custom function definition in the original app to Llama Stack's CustomTool subclass is straightforward. Below is an example of the original custom function definition:
```
list_emails_function = """
{
    "type": "function",
    "function": {
        "name": "list_emails",
        "description": "Return a list of emails matching an optionally specified query.",
        "parameters": {
            "type": "dic",
            "properties": [
                {
                    "maxResults": {
                        "type": "integer",
                        "description": "The default maximum number of emails to return is 100; the maximum allowed value for this field is 500."
                    }
                },              
                {
                    "query": {
                        "type": "string",
                        "description": "One or more keywords in the email subject and body, or one or more filters. There can be 6 types of filters: 1) Field-specific Filters: from, to, cc, bcc, subject; 2) Date Filters: before, after, older than, newer than); 3) Status Filters: read, unread, starred, importatant; 4) Attachment Filters: has, filename or type; 5) Size Filters: larger, smaller; 6) logical operators (or, and, not)."
                    }
                }
            ],
            "required": []
        }
    }
}
"""
```

And its Llama Stack CustomTool subclass implementation is:
```
class ListEmailsTool(CustomTool):
    """Custom tool for List Emails."""

    def get_name(self) -> str:
        return "list_emails"

    def get_description(self) -> str:
        return "Return a list of emails matching an optionally specified query."

    def get_params_definition(self) -> Dict[str, ToolParamDefinitionParam]:
        return {
            "maxResults": ToolParamDefinitionParam(
                param_type="int",
                description="The default maximum number of emails to return is 100; the maximum allowed value for this field is 500.",
                required=False
            ),
            "query": ToolParamDefinitionParam(
                param_type="str",
                description="One or more keywords in the email subject and body, or one or more filters. There can be 6 types of filters: 1) Field-specific Filters: from, to, cc, bcc, subject; 2) Date Filters: before, after, older than, newer than); 3) Status Filters: read, unread, starred, importatant; 4) Attachment Filters: has, filename or type; 5) Size Filters: larger, smaller; 6) logical operators (or, and, not).",
                required=False
            )
        }
    async def run(self, messages: List[CompletionMessage]) -> List[ToolResponseMessage]:
        assert len(messages) == 1, "Expected single message"

        message = messages[0]

        tool_call = message.tool_calls[0]
        try:
            response = await self.run_impl(**tool_call.arguments)
            response_str = json.dumps(response, ensure_ascii=False)
        except Exception as e:
            response_str = f"Error when running tool: {e}"

        message = ToolResponseMessage(
            call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            content=response_str,
            role="ipython",
        )
        return [message]

    async def run_impl(self, query: str, maxResults: int = 100) -> Dict[str, Any]:
        """Query to get a list of emails matching the query."""
        emails = list_emails(query)
        return {"name": self.get_name(), "result": emails}
```

Each CustomTool subclass has a `run_impl` method that calls actual Gmail API-based tool call implementation (same as the original app), which, in the example above, is `list_emails`.

## The Llama Stack Agent class

The `create_email_agent` in main.py creates a Llama Stack Agent with 6 custom tools using a `LlamaStackClient` instance that connects to Together.ai's Llama Stack server. The agent then creates a session, uses the same session in a loop to create a turn for each user ask. Inside each turn, a tool call spec is generated based on the user ask and, if needed after processing of the tool call spec to match what the actual Gmail API expects (e.g. get_email_detail requires an email id but the tool call spec generated by Llama doesn't have the id), actual tool calling happens. After post-processing of the tool call result, a user-friendly message is generated to respond to the user's original ask. 

## Memory

In `shared.py` we define a simple dictionary `memory`, used to hold short-term results such as a list of found emails based on the user ask, or the draft id of a created email draft. They're needed to answer follow up user asks such as "what attachments does the email with subject xxx have" or "send the draft". 


# TODOs

1. Improve the search, reply, forward, create email draft, and query about types of attachments.
2. Improve the fallback and error handling mechanism when the user asks don't lead to a correct function calling spec or the function calling fails. 
3. Improve the user experience by showing progress when some Gmail search API calls take long (minutes) to complete.
4. Implement the async behavior of the agent - schedule an email to be sent later.
5. Implement the agent planning - decomposing a complicated ask into sub-tasks, using ReAct and other methods.
6. Implement the agent long-term memory - longer context and memory across sessions (consider using Llama Stack/MemGPT/Letta)
7. Implement reflection - on the tool calling spec and results.
8. Introduce multiple-agent collaboration. 
9. Implement the agent observability. 
10. Compare different agent frameworks using the agent as the case study.
11. Add and implement a test plan and productionize the email agent.


# Resources
1. Lilian Weng's blog [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) 
2. Andrew Ng's posts [Agentic Design Patterns](https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/) with basic [implementations from scratch](https://github.com/neural-maze/agentic_patterns).
3. LangChain's survey [State of AI Agents](https://www.langchain.com/stateofaiagents)
4. Deloitte's report [AI agents and multiagent systems](https://www2.deloitte.com/content/dam/Deloitte/us/Documents/consulting/us-ai-institute-generative-ai-agents-multiagent-systems.pdf)
5. Letta's blog [The AI agents stack](https://www.letta.com/blog/ai-agents-stack)
6. Microsoft's multi-agent system [Magentic-One](https://www.microsoft.com/en-us/research/articles/magentic-one-a-generalist-multi-agent-system-for-solving-complex-tasks)
7. Amazon's [Multi-Agent Orchestrator framework](https://awslabs.github.io/multi-agent-orchestrator/)
8. Deeplearning.ai's [agent related courses](https://www.deeplearning.ai/courses/?courses_date_desc%5Bquery%5D=agents) (Meta, AWS, Microsoft, LangChain, LlamaIndex, crewAI, AutoGen, Letta) and some [lessons ported to using Llama](https://github.com/meta-llama/llama-recipes/tree/main/recipes/quickstart/agents/DeepLearningai_Course_Notebooks). 
9. Felicis's [The Agentic Web](https://www.felicis.com/insight/the-agentic-web)
10. A pretty complete [list of AI agents](https://github.com/e2b-dev/awesome-ai-agents), not including [/dev/agents](https://sdsa.ai/), a very new startup building the next-gen OS for AI agents, though.
11. Sequoia's [post](https://www.linkedin.com/posts/konstantinebuhler_the-ai-landscape-is-shifting-from-simple-activity-7270111755710672897-ZHnr/) on 2024 being the year of AI agents and 2025 networks of AI agents.
