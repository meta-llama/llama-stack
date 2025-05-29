"use client";

import { useEffect, useState } from "react";
import type { ResponseListResponse } from "llama-stack-client/resources/responses/responses";
import { OpenAIResponse } from "@/lib/types";
import { ResponsesTable } from "@/components/responses/responses-table";
import { client } from "@/lib/client";

export default function ResponsesPage() {
  const [responses, setResponses] = useState<OpenAIResponse[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);

  // Helper function to convert ResponseListResponse.Data to OpenAIResponse
  const convertResponseListData = (
    responseData: ResponseListResponse.Data,
  ): OpenAIResponse => {
    return {
      id: responseData.id,
      created_at: responseData.created_at,
      model: responseData.model,
      object: responseData.object,
      status: responseData.status,
      output: responseData.output as OpenAIResponse["output"],
      input: responseData.input as OpenAIResponse["input"],
      error: responseData.error,
      parallel_tool_calls: responseData.parallel_tool_calls,
      previous_response_id: responseData.previous_response_id,
      temperature: responseData.temperature,
      top_p: responseData.top_p,
      truncation: responseData.truncation,
      user: responseData.user,
    };
  };

  useEffect(() => {
    const fetchResponses = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await client.responses.list();
        const responseListData = response as ResponseListResponse;

        const convertedResponses: OpenAIResponse[] = responseListData.data.map(
          convertResponseListData,
        );

        setResponses(convertedResponses);
      } catch (err) {
        console.error("Error fetching responses:", err);
        setError(
          err instanceof Error ? err : new Error("Failed to fetch responses"),
        );
        setResponses([]);
      } finally {
        setIsLoading(false);
      }
    };

    fetchResponses();
  }, []);

  return (
    <ResponsesTable data={responses} isLoading={isLoading} error={error} />
  );
}
