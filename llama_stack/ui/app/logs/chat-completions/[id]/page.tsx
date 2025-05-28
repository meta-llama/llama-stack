"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import LlamaStackClient from "llama-stack-client";
import { ChatCompletion } from "@/lib/types";
import { ChatCompletionDetailView } from "@/components/chat-completions/chat-completion-detail";

export default function ChatCompletionDetailPage() {
  const params = useParams();
  const id = params.id as string;

  const [completionDetail, setCompletionDetail] =
    useState<ChatCompletion | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    if (!id) {
      setError(new Error("Completion ID is missing."));
      setIsLoading(false);
      return;
    }

    const client = new LlamaStackClient({
      baseURL: process.env.NEXT_PUBLIC_LLAMA_STACK_BASE_URL,
    });

    const fetchCompletionDetail = async () => {
      setIsLoading(true);
      setError(null);
      setCompletionDetail(null);
      try {
        const response = await client.chat.completions.retrieve(id);
        setCompletionDetail(response as ChatCompletion);
      } catch (err) {
        console.error(
          `Error fetching chat completion detail for ID ${id}:`,
          err,
        );
        setError(
          err instanceof Error
            ? err
            : new Error("Failed to fetch completion detail"),
        );
      } finally {
        setIsLoading(false);
      }
    };

    fetchCompletionDetail();
  }, [id]);

  return (
    <ChatCompletionDetailView
      completion={completionDetail}
      isLoading={isLoading}
      error={error}
      id={id}
    />
  );
}
