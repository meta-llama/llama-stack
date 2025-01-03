from typing import List, Dict, Any
from llama_stack_client.types.tool_param_definition_param import ToolParamDefinitionParam
from llama_stack_client.types import CompletionMessage, ToolResponseMessage
from llama_stack_client.lib.agents.custom_tool import CustomTool
from email_agent import *
import json

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


class GetEmailDetailTool(CustomTool):
    """Custom tool for Get Email Detail."""

    def get_name(self) -> str:
        return "get_email_detail"

    def get_description(self) -> str:
        return "Get detailed info about a specific email"

    def get_params_definition(self) -> Dict[str, ToolParamDefinitionParam]:
        return {
            "detail": ToolParamDefinitionParam(
                param_type="str",
                description="what detail the user wants to know about - two possible values: body or attachment",
                required=True
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

    async def run_impl(self, detail: str, query: str) -> Dict[str, Any]:
        """Query to get the detail of an email."""

        detail = get_email_detail(detail, query)
        return {"name": self.get_name(), "result": detail}


class SendEmailTool(CustomTool):
    """Compose, reply, or forward email."""

    def get_name(self) -> str:
        return "send_email"

    def get_description(self) -> str:
        return "Compose, reply, or forward email"

    def get_params_definition(self) -> Dict[str, ToolParamDefinitionParam]:
        return {
            "action": ToolParamDefinitionParam(
                param_type="string",
                description="Whether to compose, reply, or forward an email",
                required=True
            ),
            "to": ToolParamDefinitionParam(
                param_type="str",
                description="The recipient of the email",
                required=True
            ),
            "subject": ToolParamDefinitionParam(
                param_type="str",
                description="The subject of the email",
                required=True
            ),
            "body": ToolParamDefinitionParam(
                param_type="str",
                description="The content of the email",
                required=True
            ),
            "email_id": ToolParamDefinitionParam(
                param_type="str",
                description="The email id to reply or forward to",
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

    async def run_impl(self, action, to, subject, body="", email_id="") -> Dict[str, Any]:
        """Send an email."""

        result = send_email(action, to, subject, body, email_id)
        return {"name": self.get_name(), "result": result}


class GetPDFSummaryTool(CustomTool):
    """Get a summary of a PDF attachment."""

    def get_name(self) -> str:
        return "get_pdf_summary"

    def get_description(self) -> str:
        return "Get a summary of a PDF attachment"

    def get_params_definition(self) -> Dict[str, ToolParamDefinitionParam]:
        return {
            "file_name": ToolParamDefinitionParam(
                param_type="string",
                description="The name of the PDF file",
                required=True
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

    async def run_impl(self, file_name: str) -> Dict[str, Any]:
        """Get the summary of a PDF file."""

        summary = get_pdf_summary(file_name)
        return {"name": self.get_name(), "result": summary}


class CreateDraftTool(CustomTool):
    """Create a new, reply, or forward email draft."""

    def get_name(self) -> str:
        return "create_draft"

    def get_description(self) -> str:
        return "Create a new, reply, or forward email draft"

    def get_params_definition(self) -> Dict[str, ToolParamDefinitionParam]:
        return {
            "action": ToolParamDefinitionParam(
                param_type="string",
                description="Whether to compose, reply, or forward an email",
                required=True
            ),
            "to": ToolParamDefinitionParam(
                param_type="str",
                description="The recipient of the email",
                required=True
            ),
            "subject": ToolParamDefinitionParam(
                param_type="str",
                description="The subject of the email",
                required=True
            ),
            "body": ToolParamDefinitionParam(
                param_type="str",
                description="The content of the email",
                required=True
            ),
            "email_id": ToolParamDefinitionParam(
                param_type="str",
                description="The email id to reply or forward to, or empty if draft a new email.",
                required=True
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

    async def run_impl(self, action, to, subject, body="", email_id="") -> Dict[str, Any]:
        """Create an email draft."""

        result = create_draft(action, to, subject, body, email_id)
        return {"name": self.get_name(), "result": result}


class SendDraftTool(CustomTool):
    """Send a draft email."""

    def get_name(self) -> str:
        return "send_draft"

    def get_description(self) -> str:
        return "Send a draft email"

    def get_params_definition(self) -> Dict[str, ToolParamDefinitionParam]:
        return {
            "id": ToolParamDefinitionParam(
                param_type="str",
                description="The email draft id.",
                required=True
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

    async def run_impl(self, id: str) -> Dict[str, Any]:
        """Send the last draft email."""

        result = send_draft(memory['draft_id'])
        return {"name": self.get_name(), "result": result}


examples = """
{"name": "list_emails", "parameters": {"query": "has:attachment larger:5mb"}}
{"name": "list_emails", "parameters": {"query": "has:attachment"}}
{"name": "list_emails", "parameters": {"query": "newer_than:1d"}}
{"name": "list_emails", "parameters": {"query": "older_than:1d"}}
{"name": "list_emails", "parameters": {"query": "is:unread"}}
{"name": "list_emails", "parameters":  {"query": "<query> is:unread"}}
{"name": "list_emails", "parameters":  {"query": "<query> is:read"}}
{"name": "get_email_detail", "parameters": {"detail": "body", "which": "first"}}
{"name": "get_email_detail", "parameters": {"detail": "body", "which": "last"}}
{"name": "get_email_detail", "parameters": {"detail": "body", "which": "second"}}
{"name": "get_email_detail", "parameters": {"detail": "body", "which": "subject <subject info>"}}
{"name": "get_email_detail", "parameters": {"detail": "attachment", "which": "from <sender info>"}}
{"name": "get_email_detail", "parameters": {"detail": "attachment", "which": "first"}}
{"name": "get_email_detail", "parameters": {"detail": "attachment", "which": "last"}}
{"name": "get_email_detail", "parameters": {"detail": "attachment", "which": "<email id>"}}
{"name": "send_email", "parameters": {"action": "compose", "to": "jeffxtang@meta.com", "subject": "xxxxx", "body": "xxxxx"}}
{"name": "send_email", "parameters": {"action": "reply", "to": "", "subject": "xxxxx", "body": "xxxxx", "email_id": "xxxxx"}}
{"name": "send_email", "parameters": {"action": "forward", "to": "jeffxtang@meta.com", "subject": "xxxxx", "body": "xxxxx", "email_id": "xxxxx"}}
{"name": "create_draft", "parameters": {"action": "new", "to": "jeffxtang@meta.com", "subject": "xxxxx", "body": "xxxxx", "email_id": ""}}
{"name": "create_draft", "parameters": {"action": "reply", "to": "", "subject": "xxxxx", "body": "xxxxx", "email_id": "xxxxx"}}
{"name": "create_draft", "parameters": {"action": "forward", "to": "jeffxtang@meta.com", "subject": "xxxxx", "body": "xxxxx", "email_id": "xxxxx"}}
{"name": "send_draft", "parameters": {"id": "..."}}
{"name": "get_pdf_summary", "parameters": {"file_name": "..."}}
"""

system_prompt = f"""
Your name is Email Agent, an assistant that can perform all email related tasks for your user.
Respond to the user's ask by making use of the following functions if needed.
If no available functions can be used, just say "I don't know" and don't make up facts.

Example responses:
{examples}

"""
