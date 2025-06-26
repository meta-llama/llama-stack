import LlamaStackClient from "llama-stack-client";
import { getAuthToken } from "./auth";

export function getClient() {
  const token = getAuthToken();

  return new LlamaStackClient({
    baseURL:
      typeof window !== "undefined" ? `${window.location.origin}/api` : "/api",
    apiKey: token || undefined,
  });
}
