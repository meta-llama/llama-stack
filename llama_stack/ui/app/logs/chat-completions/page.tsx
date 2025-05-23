"use client";

import { useEffect, useState } from "react";
import LlamaStackClient from "llama-stack-client";
import { ChatCompletion } from "@/lib/types";
import { ChatCompletionsTable } from "@/components/chat-completions/chat-completion-table";

export default function ChatCompletionsPage() {
  const [completions, setCompletions] = useState<ChatCompletion[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const client = new LlamaStackClient({
      baseURL: process.env.NEXT_PUBLIC_LLAMA_STACK_BASE_URL,
    });
    const fetchCompletions = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await client.chat.completions.list();
        const data = Array.isArray(response)
          ? response
          : (response as any).data;

        if (Array.isArray(data)) {
          setCompletions(data);
        } else {
          console.error("Unexpected response structure:", response);
          setError(new Error("Unexpected response structure"));
          setCompletions([]);
        }
      } catch (err) {
        console.error("Error fetching chat completions:", err);
        setError(
          err instanceof Error ? err : new Error("Failed to fetch completions"),
        );
        setCompletions([]);
      } finally {
        setIsLoading(false);
      }
    };

    fetchCompletions();
  }, []);

  return (
    <ChatCompletionsTable
      completions={completions}
      isLoading={isLoading}
      error={error}
    />
  );
}
