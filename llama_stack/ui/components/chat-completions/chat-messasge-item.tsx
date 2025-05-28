"use client";

import { ChatMessage } from "@/lib/types";
import React from "react";
import { formatToolCallToString } from "@/lib/format-tool-call";
import { extractTextFromContentPart } from "@/lib/format-message-content";

// Sub-component or helper for the common label + content structure
const MessageBlock: React.FC<{
  label: string;
  labelDetail?: string;
  content: React.ReactNode;
}> = ({ label, labelDetail, content }) => {
  return (
    <div>
      <p className="py-1 font-semibold text-gray-800 mb-1">
        {label}
        {labelDetail && (
          <span className="text-xs text-gray-500 font-normal ml-1">
            {labelDetail}
          </span>
        )}
      </p>
      <div className="py-1">{content}</div>
    </div>
  );
};

interface ToolCallBlockProps {
  children: React.ReactNode;
  className?: string;
}

const ToolCallBlock = ({ children, className }: ToolCallBlockProps) => {
  // Common styling for both function call arguments and tool output blocks
  // Let's use slate-50 background as it's good for code-like content.
  const baseClassName =
    "p-3 bg-slate-50 border border-slate-200 rounded-md text-sm";

  return (
    <div className={`${baseClassName} ${className || ""}`}>
      <pre className="whitespace-pre-wrap text-xs">{children}</pre>
    </div>
  );
};

interface ChatMessageItemProps {
  message: ChatMessage;
}
export function ChatMessageItem({ message }: ChatMessageItemProps) {
  switch (message.role) {
    case "system":
      return (
        <MessageBlock
          label="System"
          content={extractTextFromContentPart(message.content)}
        />
      );
    case "user":
      return (
        <MessageBlock
          label="User"
          content={extractTextFromContentPart(message.content)}
        />
      );

    case "assistant":
      if (message.tool_calls && message.tool_calls.length > 0) {
        return (
          <>
            {message.tool_calls.map((toolCall: any, index: number) => {
              const formattedToolCall = formatToolCallToString(toolCall);
              const toolCallContent = (
                <ToolCallBlock>
                  {formattedToolCall || "Error: Could not display tool call"}
                </ToolCallBlock>
              );
              return (
                <MessageBlock
                  key={index}
                  label="Tool Call"
                  content={toolCallContent}
                />
              );
            })}
          </>
        );
      } else {
        return (
          <MessageBlock
            label="Assistant"
            content={extractTextFromContentPart(message.content)}
          />
        );
      }
    case "tool":
      const toolOutputContent = (
        <ToolCallBlock>
          {extractTextFromContentPart(message.content)}
        </ToolCallBlock>
      );
      return (
        <MessageBlock label="Tool Call Output" content={toolOutputContent} />
      );
  }
  return null;
}
