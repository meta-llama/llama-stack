/**
 * Server-side configuration for the Llama Stack UI
 * This file should only be imported in server components
 */

// Get backend URL from environment variable or default to localhost for development
export const BACKEND_URL =
  process.env.LLAMA_STACK_BACKEND_URL ||
  `http://localhost:${process.env.LLAMA_STACK_PORT || 8321}`;

export const serverConfig = {
  backendUrl: BACKEND_URL,
} as const;

export default serverConfig;
