"use client";

import {
  OpenAIResponse,
  ResponseInput,
  ResponseInputMessageContent,
} from "@/lib/types";
import { LogsTable, LogTableRow } from "@/components/logs/logs-table";
import {
  isMessageInput,
  isMessageItem,
  isFunctionCallItem,
  isWebSearchCallItem,
  MessageItem,
  FunctionCallItem,
  WebSearchCallItem,
} from "./utils/item-types";

interface ResponsesTableProps {
  data: OpenAIResponse[];
  isLoading: boolean;
  error: Error | null;
}

function getInputText(response: OpenAIResponse): string {
  const firstInput = response.input.find(isMessageInput);
  if (firstInput) {
    return extractContentFromItem(firstInput);
  }
  return "";
}

function getOutputText(response: OpenAIResponse): string {
  const firstMessage = response.output.find((item) =>
    isMessageItem(item as any),
  );
  if (firstMessage) {
    const content = extractContentFromItem(firstMessage as MessageItem);
    if (content) {
      return content;
    }
  }

  const functionCall = response.output.find((item) =>
    isFunctionCallItem(item as any),
  );
  if (functionCall) {
    return formatFunctionCall(functionCall as FunctionCallItem);
  }

  const webSearchCall = response.output.find((item) =>
    isWebSearchCallItem(item as any),
  );
  if (webSearchCall) {
    return formatWebSearchCall(webSearchCall as WebSearchCallItem);
  }

  return JSON.stringify(response.output);
}

function extractContentFromItem(item: {
  content?: string | ResponseInputMessageContent[];
}): string {
  if (!item.content) {
    return "";
  }

  if (typeof item.content === "string") {
    return item.content;
  } else if (Array.isArray(item.content)) {
    const textContent = item.content.find(
      (c: ResponseInputMessageContent) =>
        c.type === "input_text" || c.type === "output_text",
    );
    return textContent?.text || "";
  }
  return "";
}

function formatFunctionCall(functionCall: FunctionCallItem): string {
  const args = functionCall.arguments || "{}";
  const name = functionCall.name || "unknown";
  return `${name}(${args})`;
}

function formatWebSearchCall(webSearchCall: WebSearchCallItem): string {
  return `web_search_call(status: ${webSearchCall.status})`;
}

function formatResponseToRow(response: OpenAIResponse): LogTableRow {
  return {
    id: response.id,
    input: getInputText(response),
    output: getOutputText(response),
    model: response.model,
    createdTime: new Date(response.created_at * 1000).toLocaleString(),
    detailPath: `/logs/responses/${response.id}`,
  };
}

export function ResponsesTable({
  data,
  isLoading,
  error,
}: ResponsesTableProps) {
  const formattedData = data.map(formatResponseToRow);

  return (
    <LogsTable
      data={formattedData}
      isLoading={isLoading}
      error={error}
      caption="A list of your recent responses."
      emptyMessage="No responses found."
    />
  );
}
