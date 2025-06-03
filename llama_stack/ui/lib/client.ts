import LlamaStackClient from "llama-stack-client";

export const client = new LlamaStackClient({
  baseURL:
    typeof window !== "undefined" ? `${window.location.origin}/api` : "/api",
});
