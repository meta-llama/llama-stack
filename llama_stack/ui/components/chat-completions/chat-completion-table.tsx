"use client";

import { useRouter } from "next/navigation";
import { ChatCompletion } from "@/lib/types";
import { truncateText } from "@/lib/truncate-text";
import {
  extractTextFromContentPart,
  extractDisplayableText,
} from "@/lib/format-message-content";
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Skeleton } from "@/components/ui/skeleton";

interface ChatCompletionsTableProps {
  completions: ChatCompletion[];
  isLoading: boolean;
  error: Error | null;
}

export function ChatCompletionsTable({
  completions,
  isLoading,
  error,
}: ChatCompletionsTableProps) {
  const router = useRouter();

  const tableHeader = (
    <TableHeader>
      <TableRow>
        <TableHead>Input</TableHead>
        <TableHead>Output</TableHead>
        <TableHead>Model</TableHead>
        <TableHead className="text-right">Created</TableHead>
      </TableRow>
    </TableHeader>
  );

  if (isLoading) {
    return (
      <Table>
        <TableCaption>
          <Skeleton className="h-4 w-[250px] mx-auto" />
        </TableCaption>
        {tableHeader}
        <TableBody>
          {[...Array(3)].map((_, i) => (
            <TableRow key={`skeleton-${i}`}>
              <TableCell>
                <Skeleton className="h-4 w-full" />
              </TableCell>
              <TableCell>
                <Skeleton className="h-4 w-full" />
              </TableCell>
              <TableCell>
                <Skeleton className="h-4 w-3/4" />
              </TableCell>
              <TableCell className="text-right">
                <Skeleton className="h-4 w-1/2 ml-auto" />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    );
  }

  if (error) {
    return (
      <p>Error fetching data: {error.message || "An unknown error occurred"}</p>
    );
  }

  if (completions.length === 0) {
    return <p>No chat completions found.</p>;
  }

  return (
    <Table>
      <TableCaption>A list of your recent chat completions.</TableCaption>
      {tableHeader}
      <TableBody>
        {completions.map((completion) => (
          <TableRow
            key={completion.id}
            onClick={() =>
              router.push(`/logs/chat-completions/${completion.id}`)
            }
            className="cursor-pointer hover:bg-muted/50"
          >
            <TableCell>
              {truncateText(
                extractTextFromContentPart(
                  completion.input_messages?.[0]?.content,
                ),
              )}
            </TableCell>
            <TableCell>
              {(() => {
                const message = completion.choices?.[0]?.message;
                const outputText = extractDisplayableText(message);
                return truncateText(outputText);
              })()}
            </TableCell>
            <TableCell>{completion.model}</TableCell>
            <TableCell className="text-right">
              {new Date(completion.created * 1000).toLocaleString()}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
