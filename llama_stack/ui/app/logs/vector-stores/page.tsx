"use client";

import React, { useState, useEffect } from "react";
import { useAuthClient } from "@/hooks/use-auth-client";
import type {
  ListVectorStoresResponse,
  VectorStore,
} from "llama-stack-client/resources/vector-stores/vector-stores";
import { useRouter } from "next/navigation";
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

export default function VectorStoresPage() {
  const client = useAuthClient();
  const router = useRouter();
  const [stores, setStores] = useState<VectorStore[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchStores() {
      try {
        const response = await client.vectorStores.list();
        const res = response as ListVectorStoresResponse;
        setStores(res.data);
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "Failed to load vector stores.",
        );
      } finally {
        setLoading(false);
      }
    }
    fetchStores();
  }, [client]);

  if (loading) {
    return (
      <div className="space-y-2">
        <Skeleton className="h-8 w-full" />
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-full" />
      </div>
    );
  }

  if (error) {
    return <div className="text-destructive">Error: {error}</div>;
  }

  return (
    <Table>
      <TableCaption>A list of your vector stores.</TableCaption>
      <TableHeader>
        <TableRow>
          <TableHead>ID</TableHead>
          <TableHead>Name</TableHead>
          <TableHead>Created</TableHead>
          <TableHead>Completed</TableHead>
          <TableHead>Cancelled</TableHead>
          <TableHead>Failed</TableHead>
          <TableHead>In Progress</TableHead>
          <TableHead>Total</TableHead>
          <TableHead>Usage Bytes</TableHead>
          <TableHead>Provider ID</TableHead>
          <TableHead>Provider Vector DB ID</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {stores.map((store) => {
          const fileCounts = store.file_counts;
          const metadata = store.metadata || {};
          const providerId = metadata.provider_id ?? "";
          const providerDbId = metadata.provider_vector_db_id ?? "";

          return (
            <TableRow key={store.id} className="hover:bg-muted/50">
              <TableCell>{store.id}</TableCell>
              <TableCell>{store.name}</TableCell>
              <TableCell>{new Date(store.created_at * 1000).toLocaleString()}</TableCell>
              <TableCell>{fileCounts.completed}</TableCell>
              <TableCell>{fileCounts.cancelled}</TableCell>
              <TableCell>{fileCounts.failed}</TableCell>
              <TableCell>{fileCounts.in_progress}</TableCell>
              <TableCell>{fileCounts.total}</TableCell>
              <TableCell>{store.usage_bytes}</TableCell>
              <TableCell>{providerId}</TableCell>
              <TableCell>{providerDbId}</TableCell>
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
}
