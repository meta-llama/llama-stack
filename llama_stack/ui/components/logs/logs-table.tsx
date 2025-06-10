"use client";

import { useRouter } from "next/navigation";
import { truncateText } from "@/lib/truncate-text";
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

// Generic table row data interface
export interface LogTableRow {
  id: string;
  input: string;
  output: string;
  model: string;
  createdTime: string;
  detailPath: string;
}

interface LogsTableProps {
  data: LogTableRow[];
  isLoading: boolean;
  error: Error | null;
  caption: string;
  emptyMessage: string;
}

export function LogsTable({
  data,
  isLoading,
  error,
  caption,
  emptyMessage,
}: LogsTableProps) {
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

  if (data.length === 0) {
    return <p>{emptyMessage}</p>;
  }

  return (
    <Table>
      <TableCaption>{caption}</TableCaption>
      {tableHeader}
      <TableBody>
        {data.map((row) => (
          <TableRow
            key={row.id}
            onClick={() => router.push(row.detailPath)}
            className="cursor-pointer hover:bg-muted/50"
          >
            <TableCell>{truncateText(row.input)}</TableCell>
            <TableCell>{truncateText(row.output)}</TableCell>
            <TableCell>{row.model}</TableCell>
            <TableCell className="text-right">{row.createdTime}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
