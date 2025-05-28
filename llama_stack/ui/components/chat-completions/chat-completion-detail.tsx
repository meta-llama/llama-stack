"use client";

import { ChatMessage, ChatCompletion } from "@/lib/types";
import { ChatMessageItem } from "@/components/chat-completions/chat-messasge-item";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

function ChatCompletionDetailLoadingView() {
  return (
    <>
      <Skeleton className="h-8 w-3/4 mb-6" /> {/* Title Skeleton */}
      <div className="flex flex-col md:flex-row gap-6">
        <div className="flex-grow md:w-2/3 space-y-6">
          {[...Array(2)].map((_, i) => (
            <Card key={`main-skeleton-card-${i}`}>
              <CardHeader>
                <CardTitle>
                  <Skeleton className="h-6 w-1/2" />
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-3/4" />
              </CardContent>
            </Card>
          ))}
        </div>
        <div className="md:w-1/3">
          <div className="p-4 border rounded-lg shadow-sm bg-white space-y-3">
            <Skeleton className="h-6 w-1/3 mb-3" />{" "}
            {/* Properties Title Skeleton */}
            {[...Array(5)].map((_, i) => (
              <div key={`prop-skeleton-${i}`} className="space-y-1">
                <Skeleton className="h-4 w-1/4" />
                <Skeleton className="h-4 w-1/2" />
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}

interface ChatCompletionDetailViewProps {
  completion: ChatCompletion | null;
  isLoading: boolean;
  error: Error | null;
  id: string;
}

export function ChatCompletionDetailView({
  completion,
  isLoading,
  error,
  id,
}: ChatCompletionDetailViewProps) {
  if (error) {
    return (
      <>
        {/* We still want a title for consistency on error pages */}
        <h1 className="text-2xl font-bold mb-6">Chat Completion Details</h1>
        <p>
          Error loading details for ID {id}: {error.message}
        </p>
      </>
    );
  }

  if (isLoading) {
    return <ChatCompletionDetailLoadingView />;
  }

  if (!completion) {
    // This state means: not loading, no error, but no completion data
    return (
      <>
        {/* We still want a title for consistency on not-found pages */}
        <h1 className="text-2xl font-bold mb-6">Chat Completion Details</h1>
        <p>No details found for completion ID: {id}.</p>
      </>
    );
  }

  // If no error, not loading, and completion exists, render the details:
  return (
    <>
      <h1 className="text-2xl font-bold mb-6">Chat Completion Details</h1>
      <div className="flex flex-col md:flex-row gap-6">
        <div className="flex-grow md:w-2/3 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Input</CardTitle>
            </CardHeader>
            <CardContent>
              {completion.input_messages?.map((msg, index) => (
                <ChatMessageItem key={`input-msg-${index}`} message={msg} />
              ))}
              {completion.choices?.[0]?.message?.tool_calls &&
                !completion.input_messages?.some(
                  (im) =>
                    im.role === "assistant" &&
                    im.tool_calls &&
                    im.tool_calls.length > 0,
                ) &&
                completion.choices[0].message.tool_calls.map(
                  (toolCall: any, index: number) => {
                    const assistantToolCallMessage: ChatMessage = {
                      role: "assistant",
                      tool_calls: [toolCall],
                      content: "", // Ensure content is defined, even if empty
                    };
                    return (
                      <ChatMessageItem
                        key={`choice-tool-call-${index}`}
                        message={assistantToolCallMessage}
                      />
                    );
                  },
                )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Output</CardTitle>
            </CardHeader>
            <CardContent>
              {completion.choices?.[0]?.message ? (
                <ChatMessageItem
                  message={completion.choices[0].message as ChatMessage}
                />
              ) : (
                <p className="text-gray-500 italic text-sm">
                  No message found in assistant's choice.
                </p>
              )}
            </CardContent>
          </Card>
        </div>

        <div className="md:w-1/3">
          <Card>
            <CardHeader>
              <CardTitle>Properties</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2 text-sm text-gray-600">
                <li>
                  <strong>Created:</strong>{" "}
                  <span className="text-gray-900 font-medium">
                    {new Date(completion.created * 1000).toLocaleString()}
                  </span>
                </li>
                <li>
                  <strong>ID:</strong>{" "}
                  <span className="text-gray-900 font-medium">
                    {completion.id}
                  </span>
                </li>
                <li>
                  <strong>Model:</strong>{" "}
                  <span className="text-gray-900 font-medium">
                    {completion.model}
                  </span>
                </li>
                <li className="pt-1 mt-1 border-t border-gray-200">
                  <strong>Finish Reason:</strong>{" "}
                  <span className="text-gray-900 font-medium">
                    {completion.choices?.[0]?.finish_reason || "N/A"}
                  </span>
                </li>
                {completion.choices?.[0]?.message?.tool_calls &&
                  completion.choices[0].message.tool_calls.length > 0 && (
                    <li className="pt-1 mt-1 border-t border-gray-200">
                      <strong>Functions/Tools Called:</strong>
                      <ul className="list-disc list-inside pl-4 mt-1">
                        {completion.choices[0].message.tool_calls.map(
                          (toolCall: any, index: number) => (
                            <li key={index}>
                              <span className="text-gray-900 font-medium">
                                {toolCall.function?.name || "N/A"}
                              </span>
                            </li>
                          ),
                        )}
                      </ul>
                    </li>
                  )}
              </ul>
            </CardContent>
          </Card>
        </div>
      </div>
    </>
  );
}
