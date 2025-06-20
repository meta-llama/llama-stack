"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { PaginationStatus, UsePaginationOptions } from "@/lib/types";

interface PaginationState<T> {
  data: T[];
  status: PaginationStatus;
  hasMore: boolean;
  error: Error | null;
  lastId: string | null;
}

interface PaginationResponse<T> {
  data: T[];
  has_more: boolean;
  last_id: string;
  first_id: string;
  object: "list";
}

export interface PaginationReturn<T> {
  data: T[];
  status: PaginationStatus;
  hasMore: boolean;
  error: Error | null;
  loadMore: () => void;
}

interface UsePaginationParams<T> extends UsePaginationOptions {
  fetchFunction: (params: {
    after?: string;
    limit: number;
    model?: string;
    order?: string;
  }) => Promise<PaginationResponse<T>>;
  errorMessagePrefix: string;
}

export function usePagination<T>({
  limit = 20,
  model,
  order = "desc",
  fetchFunction,
  errorMessagePrefix,
}: UsePaginationParams<T>): PaginationReturn<T> {
  const [state, setState] = useState<PaginationState<T>>({
    data: [],
    status: "loading",
    hasMore: true,
    error: null,
    lastId: null,
  });

  // Use refs to avoid stale closures
  const stateRef = useRef(state);
  stateRef.current = state;

  // Track if initial data has been fetched
  const hasFetchedInitialData = useRef(false);

  /**
   * Fetches data from the API with cursor-based pagination
   */
  const fetchData = useCallback(
    async (after?: string, targetRows?: number) => {
      const isInitialLoad = !after;
      const fetchLimit = targetRows || limit;

      try {
        setState((prev) => ({
          ...prev,
          status: isInitialLoad ? "loading" : "loading-more",
          error: null,
        }));

        const response = await fetchFunction({
          after: after || undefined,
          limit: fetchLimit,
          ...(model && { model }),
          ...(order && { order }),
        });

        setState((prev) => ({
          ...prev,
          data: isInitialLoad
            ? response.data
            : [...prev.data, ...response.data],
          hasMore: response.has_more,
          lastId: response.last_id || null,
          status: "idle",
        }));
      } catch (err) {
        const errorMessage = isInitialLoad
          ? `Failed to load ${errorMessagePrefix}. Please try refreshing the page.`
          : `Failed to load more ${errorMessagePrefix}. Please try again.`;

        const error =
          err instanceof Error
            ? new Error(`${errorMessage} ${err.message}`)
            : new Error(errorMessage);

        setState((prev) => ({
          ...prev,
          error,
          status: "error",
        }));
      }
    },
    [limit, model, order, fetchFunction, errorMessagePrefix],
  );

  /**
   * Loads more data for infinite scroll
   */
  const loadMore = useCallback(() => {
    const currentState = stateRef.current;
    if (currentState.hasMore && currentState.status === "idle") {
      fetchData(currentState.lastId || undefined);
    }
  }, [fetchData]);

  // Auto-load initial data on mount
  useEffect(() => {
    if (!hasFetchedInitialData.current) {
      hasFetchedInitialData.current = true;
      fetchData();
    }
  }, [fetchData]);

  return {
    data: state.data,
    status: state.status,
    hasMore: state.hasMore,
    error: state.error,
    loadMore,
  };
}
