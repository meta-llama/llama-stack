import LlamaStackClient from "llama-stack-client";
import OpenAI from "openai";

export const client =
  process.env.NEXT_PUBLIC_USE_OPENAI_CLIENT === "true" // useful for testing
    ? new OpenAI({
        apiKey: process.env.NEXT_PUBLIC_OPENAI_API_KEY,
        dangerouslyAllowBrowser: true,
      })
    : new LlamaStackClient({
        baseURL: process.env.NEXT_PUBLIC_LLAMA_STACK_BASE_URL,
      });
