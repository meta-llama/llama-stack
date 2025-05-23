export interface TextContentBlock {
  type: "text";
  text: string;
}

export interface ImageUrlDetail {
  url: string;
  detail?: "low" | "high" | "auto";
}

export interface ImageUrlContentBlock {
  type: "image_url";
  // Support both simple URL string and detailed object, though our parser currently just looks for type: "image_url"
  image_url: string | ImageUrlDetail;
}

// Union of known content part types. Add more specific types as needed.
export type ChatMessageContentPart =
  | TextContentBlock
  | ImageUrlContentBlock
  | { type: string; [key: string]: any }; // Fallback for other potential types

export interface ChatMessage {
  role: string;
  content: string | ChatMessageContentPart[]; // Updated content type
  name?: string | null;
  tool_calls?: any | null; // This could also be refined to a more specific ToolCall[] type
}

export interface Choice {
  message: ChatMessage;
  finish_reason: string;
  index: number;
  logprobs?: any | null;
}

export interface ChatCompletion {
  id: string;
  choices: Choice[];
  object: string;
  created: number;
  model: string;
  input_messages: ChatMessage[];
}
