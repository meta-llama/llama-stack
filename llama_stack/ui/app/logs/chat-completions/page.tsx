"use client";

import { useEffect, useState } from "react";
import { ChatCompletion } from "@/lib/types";
import { ChatCompletionsTable } from "@/components/chat-completions/chat-completions-table";
import { client } from "@/lib/client";

export default function ChatCompletionsPage() {
  const [completions, setCompletions] = useState<ChatCompletion[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const fetchCompletions = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await client.chat.completions.list();
        const data = Array.isArray(response)
          ? response
          : (response as { data: ChatCompletion[] }).data;

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
      data={completions}
      isLoading={isLoading}
      error={error}
    />
  );
}
