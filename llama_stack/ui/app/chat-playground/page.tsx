"use client";

import { useState, useEffect } from "react";
import { flushSync } from "react-dom";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Chat } from "@/components/chat-playground/chat";
import { type Message } from "@/components/chat-playground/chat-message";
import { useAuthClient } from "@/hooks/use-auth-client";
import type { CompletionCreateParams } from "llama-stack-client/resources/chat/completions";
import type { Model } from "llama-stack-client/resources/models";
import type { VectorDBListResponse } from "llama-stack-client/resources/vector-dbs";
import { VectorDbManager } from "@/components/vector-db/vector-db-manager";
import { SessionManager, SessionUtils } from "@/components/chat-playground/session-manager";

interface ChatSession {
  id: string;
  name: string;
  messages: Message[];
  selectedModel: string;
  selectedVectorDb: string;
  createdAt: number;
  updatedAt: number;
}

export default function ChatPlaygroundPage() {
  const [currentSession, setCurrentSession] = useState<ChatSession | null>(null);
  const [input, setInput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [vectorDbs, setVectorDbs] = useState<VectorDBListResponse>([]);
  const [selectedVectorDb, setSelectedVectorDb] = useState<string>("");
  const [vectorDbsLoading, setVectorDbsLoading] = useState(true);
  const [vectorDbsError, setVectorDbsError] = useState<string | null>(null);
  const client = useAuthClient();

  const isModelsLoading = modelsLoading ?? true;

  // Load current session on mount
  useEffect(() => {
    const savedSession = SessionUtils.loadCurrentSession();
    if (savedSession) {
      setCurrentSession(savedSession);
    } else {
      // Create default session if none exists - will be updated with model when models load
      const defaultSession = SessionUtils.createDefaultSession();
      setCurrentSession(defaultSession);
      SessionUtils.saveCurrentSession(defaultSession);
    }
  }, []);

  // Save session when it changes
  useEffect(() => {
    if (currentSession) {
      SessionUtils.saveCurrentSession(currentSession);
    }
  }, [currentSession]);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        setModelsLoading(true);
        setModelsError(null);
        const modelList = await client.models.list();
        const llmModels = modelList.filter(model => model.model_type === "llm");
        setModels(llmModels);
        if (llmModels.length > 0 && currentSession && !currentSession.selectedModel) {
          setCurrentSession(prev => prev ? { ...prev, selectedModel: llmModels[0].identifier } : null);
        }
      } catch (err) {
        console.error("Error fetching models:", err);
        setModelsError("Failed to fetch available models");
      } finally {
        setModelsLoading(false);
      }
    };

    const fetchVectorDbs = async () => {
      try {
        setVectorDbsLoading(true);
        setVectorDbsError(null);
        const vectorDbList = await client.vectorDBs.list();
        setVectorDbs(vectorDbList);
      } catch (err) {
        console.error("Error fetching vector DBs:", err);
        setVectorDbsError("Failed to fetch available vector databases");
      } finally {
        setVectorDbsLoading(false);
      }
    };

    fetchModels();
    fetchVectorDbs();
  }, [client]);

  const extractTextContent = (content: unknown): string => {
    if (typeof content === "string") {
      return content;
    }
    if (Array.isArray(content)) {
      return content
        .filter(
          item =>
            item &&
            typeof item === "object" &&
            "type" in item &&
            item.type === "text"
        )
        .map(item =>
          item && typeof item === "object" && "text" in item
            ? String(item.text)
            : ""
        )
        .join("");
    }
    if (
      content &&
      typeof content === "object" &&
      "type" in content &&
      content.type === "text" &&
      "text" in content
    ) {
      return String(content.text) || "";
    }
    return "";
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
  };

const handleSubmit = async (event?: { preventDefault?: () => void }) => {
  event?.preventDefault?.();
  if (!input.trim() || !currentSession || !currentSession.selectedModel) return;

    // Add user message to chat
    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
      createdAt: new Date(),
    };

  setCurrentSession(prev => prev ? {
    ...prev,
    messages: [...prev.messages, userMessage],
    updatedAt: Date.now()
  } : null);
  setInput("");

    // Use the helper function with the content
    await handleSubmitWithContent(userMessage.content);
  };

  const handleSubmitWithContent = async (content: string) => {
    setIsGenerating(true);
    setError(null);

  try {
    let enhancedContent = content;

    // If a vector DB is selected, query for relevant context
    if (currentSession?.selectedVectorDb && currentSession.selectedVectorDb !== "none") {
      try {
        const vectorResponse = await client.vectorIo.query({
          query: content,
          vector_db_id: currentSession.selectedVectorDb,
        });

        if (vectorResponse.chunks && vectorResponse.chunks.length > 0) {
          const context = vectorResponse.chunks
            .map(chunk => {
              // Extract text content from the chunk
              const chunkContent = typeof chunk.content === 'string'
                ? chunk.content
                : extractTextContent(chunk.content);
              return chunkContent;
            })
            .join('\n\n');

          enhancedContent = `Please answer the following query using the context below.\n\nCONTEXT:\n${context}\n\nQUERY:\n${content}`;
        }
      } catch (vectorErr) {
        console.error("Error querying vector DB:", vectorErr);
        // Continue with original content if vector query fails
      }
    }

    const messageParams: CompletionCreateParams["messages"] = [
      ...(currentSession?.messages || []).map(msg => {
        const msgContent = typeof msg.content === 'string' ? msg.content : extractTextContent(msg.content);
        if (msg.role === "user") {
          return { role: "user" as const, content: msgContent };
        } else if (msg.role === "assistant") {
          return { role: "assistant" as const, content: msgContent };
        } else {
          return { role: "system" as const, content: msgContent };
        }
      }),
      { role: "user" as const, content: enhancedContent }
    ];

    const response = await client.chat.completions.create({
      model: currentSession?.selectedModel || "",
      messages: messageParams,
      stream: true,
    });

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "",
        createdAt: new Date(),
      };

    setCurrentSession(prev => prev ? {
      ...prev,
      messages: [...prev.messages, assistantMessage],
      updatedAt: Date.now()
    } : null);

    let fullContent = "";
    for await (const chunk of response) {
      if (chunk.choices && chunk.choices[0]?.delta?.content) {
        const deltaContent = chunk.choices[0].delta.content;
        fullContent += deltaContent;

        flushSync(() => {
          setCurrentSession(prev => {
            if (!prev) return null;
            const newMessages = [...prev.messages];
            const lastMessage = newMessages[newMessages.length - 1];
            if (lastMessage.role === "assistant") {
              lastMessage.content = fullContent;
            }
            return { ...prev, messages: newMessages, updatedAt: Date.now() };
          });
        });
      }
    }
  } catch (err) {
    console.error("Error sending message:", err);
    setError("Failed to send message. Please try again.");
    setCurrentSession(prev => prev ? {
      ...prev,
      messages: prev.messages.slice(0, -1),
      updatedAt: Date.now()
    } : null);
  } finally {
    setIsGenerating(false);
  }
};
  const suggestions = [
    "Write a Python function that prints 'Hello, World!'",
    "Explain step-by-step how to solve this math problem: If x² + 6x + 9 = 25, what is x?",
    "Design a simple algorithm to find the longest palindrome in a string.",
  ];

  const append = (message: { role: "user"; content: string }) => {
    const newMessage: Message = {
      id: Date.now().toString(),
      role: message.role,
      content: message.content,
      createdAt: new Date(),
    };
    setCurrentSession(prev => prev ? {
      ...prev,
      messages: [...prev.messages, newMessage],
      updatedAt: Date.now()
    } : null);
    handleSubmitWithContent(newMessage.content);
  };

  const clearChat = () => {
    setCurrentSession(prev => prev ? {
      ...prev,
      messages: [],
      updatedAt: Date.now()
    } : null);
    setError(null);
  };

  const handleSessionChange = (session: ChatSession) => {
    setCurrentSession(session);
    setError(null);
  };

  const handleNewSession = () => {
    const defaultModel = currentSession?.selectedModel || (models.length > 0 ? models[0].identifier : "");
    const defaultVectorDb = currentSession?.selectedVectorDb || "";

    const newSession = {
      ...SessionUtils.createDefaultSession(),
      selectedModel: defaultModel,
      selectedVectorDb: defaultVectorDb,
    };
    setCurrentSession(newSession);
    SessionUtils.saveCurrentSession(newSession);
  };

  const refreshVectorDbs = async () => {
    try {
      setVectorDbsLoading(true);
      setVectorDbsError(null);
      const vectorDbList = await client.vectorDBs.list();
      setVectorDbs(vectorDbList);
    } catch (err) {
      console.error("Error refreshing vector DBs:", err);
      setVectorDbsError("Failed to refresh vector databases");
    } finally {
      setVectorDbsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full w-full max-w-4xl mx-auto">
      <div className="mb-4 flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Chat Playground</h1>
        </div>

        <div className="flex justify-between items-center">
          <SessionManager
            currentSession={currentSession}
            onSessionChange={handleSessionChange}
            onNewSession={handleNewSession}
          />
          <Button variant="outline" onClick={clearChat} disabled={isGenerating}>
            Clear Chat
          </Button>
        </div>

        <div className="flex flex-wrap gap-2 items-center">
          <div className="flex items-center gap-1">
            <span className="text-sm font-medium text-gray-600">Model:</span>
            <Select
              value={currentSession?.selectedModel || ""}
              onValueChange={(value) => setCurrentSession(prev => prev ? { ...prev, selectedModel: value, updatedAt: Date.now() } : null)}
              disabled={isModelsLoading || isGenerating}
            >
              <SelectTrigger className="w-[160px]">
                <SelectValue placeholder={isModelsLoading ? "Loading..." : "Select Model"} />
              </SelectTrigger>
              <SelectContent>
                {models.map((model) => (
                  <SelectItem key={model.identifier} value={model.identifier}>
                    {model.identifier}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center gap-1">
            <span className="text-sm font-medium text-gray-600">Vector DB:</span>
            <Select
              value={currentSession?.selectedVectorDb || ""}
              onValueChange={(value) => setCurrentSession(prev => prev ? { ...prev, selectedVectorDb: value, updatedAt: Date.now() } : null)}
              disabled={vectorDbsLoading || isGenerating}
            >
              <SelectTrigger className="w-[160px]">
                <SelectValue placeholder={vectorDbsLoading ? "Loading..." : "None"} />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">None</SelectItem>
                {vectorDbs.map((vectorDb) => (
                  <SelectItem key={vectorDb.identifier} value={vectorDb.identifier}>
                    {vectorDb.identifier}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <VectorDbManager
            client={client}
            onVectorDbCreated={refreshVectorDbs}
          />
        </div>
      </div>

      {modelsError && (
        <div className="mb-4 p-3 bg-destructive/10 border border-destructive/20 rounded-md">
          <p className="text-destructive text-sm">{modelsError}</p>
        </div>
      )}

      {vectorDbsError && (
        <div className="mb-4 p-3 bg-destructive/10 border border-destructive/20 rounded-md">
          <p className="text-destructive text-sm">{vectorDbsError}</p>
        </div>
      )}

      {error && (
        <div className="mb-4 p-3 bg-destructive/10 border border-destructive/20 rounded-md">
          <p className="text-destructive text-sm">{error}</p>
        </div>
      )}

      <Chat
        className="flex-1"
        messages={currentSession?.messages || []}
        handleSubmit={handleSubmit}
        input={input}
        handleInputChange={handleInputChange}
        isGenerating={isGenerating}
        append={append}
        suggestions={suggestions}
        setMessages={(messages) => setCurrentSession(prev => prev ? { ...prev, messages, updatedAt: Date.now() } : null)}
      />
    </div>
  );
}
