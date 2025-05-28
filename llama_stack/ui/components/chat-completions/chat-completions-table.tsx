"use client";

import { ChatCompletion } from "@/lib/types";
import { LogsTable, LogTableRow } from "@/components/logs/logs-table";
import {
  extractTextFromContentPart,
  extractDisplayableText,
} from "@/lib/format-message-content";

interface ChatCompletionsTableProps {
  data: ChatCompletion[];
  isLoading: boolean;
  error: Error | null;
}

function formatChatCompletionToRow(completion: ChatCompletion): LogTableRow {
  return {
    id: completion.id,
    input: extractTextFromContentPart(completion.input_messages?.[0]?.content),
    output: extractDisplayableText(completion.choices?.[0]?.message),
    model: completion.model,
    createdTime: new Date(completion.created * 1000).toLocaleString(),
    detailPath: `/logs/chat-completions/${completion.id}`,
  };
}

export function ChatCompletionsTable({
  data,
  isLoading,
  error,
}: ChatCompletionsTableProps) {
  const formattedData = data.map(formatChatCompletionToRow);

  return (
    <LogsTable
      data={formattedData}
      isLoading={isLoading}
      error={error}
      caption="A list of your recent chat completions."
      emptyMessage="No chat completions found."
    />
  );
}
