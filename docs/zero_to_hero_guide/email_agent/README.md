# A Llama and Llama Stack Powered Email Agent

This is a Llama Stack port of the [Llama Powered Email Agent](https://github.com/meta-llama/llama-recipes/tree/main/recipes/use_cases/email_agent) app that shows how to build an email agent app powered by Llama 3.1 8B and Llama Stack, using Llama Stack custom tool and agent APIs. 

Currently implemented features of the agent include:
* search for emails and attachments
* get email detail
* reply to a specific email 
* forward an email
* get summary of a PDF attachment
* draft and send an email

We'll mainly cover here how to port a Llama app using native custom tools supported in Llama 3.1 (and later) and an agent implementation from scratch to using Llama Stack APIs. See the link above for a comprehensive overview, definition, and resources of LLM agents, and a detailed list of TODOs for the email agent.

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

The steps are also the same as the [original app](https://github.com/meta-llama/llama-recipes/tree/main/recipes/use_cases/email_agent):

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
