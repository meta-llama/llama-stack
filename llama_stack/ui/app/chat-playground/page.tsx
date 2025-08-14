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
import { VectorDbManager } from "@/components/vector-db/vector-db-manager-simple";
import {
  SessionManager,
  SessionUtils,
} from "@/components/chat-playground/session-manager";
import { DocumentUploader } from "@/components/chat-playground/document-uploader";

/**
 * Unified Chat Playground
 * - Keeps session + system message + VectorDB/RAG & document upload from version B
 * - Preserves simple message flow & suggestions/append helpers from version A
 * - Uses a single state source of truth: currentSession
 */

interface ChatSession {
  id: string;
  name: string;
  messages: Message[];
  selectedModel: string;
  selectedVectorDb: string; // "none" disables RAG
  systemMessage: string;
  createdAt: number;
  updatedAt: number;
}

export default function ChatPlaygroundPage() {
  const [currentSession, setCurrentSession] = useState<ChatSession | null>(
    null
  );
  const [input, setInput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [models, setModels] = useState<Model[]>([]);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelsError, setModelsError] = useState<string | null>(null);

  const [vectorDbs, setVectorDbs] = useState<VectorDBListResponse>([]);
  const [vectorDbsLoading, setVectorDbsLoading] = useState(true);
  const [vectorDbsError, setVectorDbsError] = useState<string | null>(null);

  const client = useAuthClient();
  const isModelsLoading = modelsLoading ?? true;

  // --- Session bootstrapping ---
  useEffect(() => {
    const saved = SessionUtils.loadCurrentSession();
    if (saved) {
      setCurrentSession(saved);
    } else {
      const def = SessionUtils.createDefaultSession();
      // ensure defaults align with our fields
      const defaultSession: ChatSession = {
        ...def,
        selectedModel: "",
        selectedVectorDb: "none",
        systemMessage: def.systemMessage || "You are a helpful assistant.",
      };
      setCurrentSession(defaultSession);
      SessionUtils.saveCurrentSession(defaultSession);
    }
  }, []);

  // Persist session on change
  useEffect(() => {
    if (currentSession) SessionUtils.saveCurrentSession(currentSession);
  }, [currentSession]);

  // --- Fetch models & vector DBs ---
  useEffect(() => {
    const fetchModels = async () => {
      try {
        setModelsLoading(true);
        setModelsError(null);
        const list = await client.models.list();
        const llms = list.filter(m => m.model_type === "llm");
        setModels(llms);
        if (llms.length > 0) {
          setCurrentSession(prev =>
            prev && !prev.selectedModel
              ? {
                  ...prev,
                  selectedModel: llms[0].identifier,
                  updatedAt: Date.now(),
                }
              : prev
          );
        }
      } catch (e) {
        console.error("Error fetching models:", e);
        setModelsError("Failed to fetch available models");
      } finally {
        setModelsLoading(false);
      }
    };

    const fetchVectorDbs = async () => {
      try {
        setVectorDbsLoading(true);
        setVectorDbsError(null);
        const list = await client.vectorDBs.list();
        setVectorDbs(list);
        // default to "none" if not set
        setCurrentSession(prev =>
          prev && !prev.selectedVectorDb
            ? { ...prev, selectedVectorDb: "none", updatedAt: Date.now() }
            : prev
        );
      } catch (e) {
        console.error("Error fetching vector DBs:", e);
        setVectorDbsError("Failed to fetch available vector databases");
      } finally {
        setVectorDbsLoading(false);
      }
    };

    fetchModels();
    fetchVectorDbs();
  }, [client]);

  // --- Utilities ---
  const extractTextContent = (content: unknown): string => {
    if (typeof content === "string") return content;
    if (Array.isArray(content)) {
      return content
        .filter(
          item =>
            item &&
            typeof item === "object" &&
            "type" in item &&
            (item as { type: string }).type === "text"
        )
        .map(item =>
          item && typeof item === "object" && "text" in item
            ? String((item as { text: unknown }).text)
            : ""
        )
        .join("");
    }
    if (
      content &&
      typeof content === "object" &&
      "type" in content &&
      (content as { type: string }).type === "text" &&
      "text" in content
    ) {
      return String((content as { text: unknown }).text) || "";
    }
    return "";
  };

  // --- Handlers ---
  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) =>
    setInput(e.target.value);

  const handleSubmit = async (event?: { preventDefault?: () => void }) => {
    event?.preventDefault?.();
    if (!input.trim() || !currentSession || !currentSession.selectedModel)
      return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
      createdAt: new Date(),
    };

    setCurrentSession(prev =>
      prev
        ? {
            ...prev,
            messages: [...prev.messages, userMessage],
            updatedAt: Date.now(),
          }
        : prev
    );
    setInput("");

  // Use the helper function with the content
  await handleSubmitWithContent(userMessage.content);
};

  const handleSubmitWithContent = async (content: string) => {
    setIsGenerating(true);
    setError(null);

    try {
      let enhancedContent = content;

      // --- RAG augmentation (optional) ---
      if (
        currentSession?.selectedVectorDb &&
        currentSession.selectedVectorDb !== "none"
      ) {
        try {
          const vectorResponse = await client.vectorIo.query({
            query: content,
            vector_db_id: currentSession.selectedVectorDb,
          });

          if (vectorResponse.chunks && vectorResponse.chunks.length > 0) {
            const context = vectorResponse.chunks
              .map(chunk =>
                typeof chunk.content === "string"
                  ? chunk.content
                  : extractTextContent(chunk.content)
              )
              .join("\n\n");

            enhancedContent = `Please answer the following query using the context below.\n\nCONTEXT:\n${context}\n\nQUERY:\n${content}`;
          }
        } catch (vectorErr) {
          console.error("Error querying vector DB:", vectorErr);
          // proceed without augmentation
        }
      }

      const messageParams: CompletionCreateParams["messages"] = [
        ...(currentSession?.systemMessage
          ? [{ role: "system" as const, content: currentSession.systemMessage }]
          : []),
        ...(currentSession?.messages || []).map(msg => {
          const msgContent =
            typeof msg.content === "string"
              ? msg.content
              : extractTextContent(msg.content);
          if (msg.role === "user")
            return { role: "user" as const, content: msgContent };
          if (msg.role === "assistant")
            return { role: "assistant" as const, content: msgContent };
          return { role: "system" as const, content: msgContent };
        }),
        { role: "user" as const, content: enhancedContent },
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
              const last = newMessages[newMessages.length - 1];
              if (last.role === "assistant") last.content = fullContent;
              return { ...prev, messages: newMessages, updatedAt: Date.now() };
            });
          });
        }
      }
    } catch (err) {
      console.error("Error sending message:", err);
      setError("Failed to send message. Please try again.");
      setCurrentSession(prev =>
        prev
          ? {
              ...prev,
              messages: prev.messages.slice(0, -1),
              updatedAt: Date.now(),
            }
          : prev
      );
    } finally {
      setIsGenerating(false);
    }
  };

  // --- UX helpers ---
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
    setCurrentSession(prev =>
      prev
        ? {
            ...prev,
            messages: [...prev.messages, newMessage],
            updatedAt: Date.now(),
          }
        : prev
    );
    handleSubmitWithContent(newMessage.content);
  };

  const clearChat = () => {
    setCurrentSession(prev =>
      prev ? { ...prev, messages: [], updatedAt: Date.now() } : prev
    );
    setError(null);
  };

  const handleSessionChange = (session: ChatSession) => {
    setCurrentSession(session);
    setError(null);
  };

  const handleNewSession = () => {
    const defaultModel =
      currentSession?.selectedModel ||
      (models.length > 0 ? models[0].identifier : "");
    const defaultVectorDb = currentSession?.selectedVectorDb || "none";

    const newSession: ChatSession = {
      ...SessionUtils.createDefaultSession(),
      selectedModel: defaultModel,
      selectedVectorDb: defaultVectorDb,
      systemMessage:
        currentSession?.systemMessage || "You are a helpful assistant.",
      messages: [],
      updatedAt: Date.now(),
      createdAt: Date.now(),
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
    <div className="flex flex-col h-full w-full max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <div className="flex justify-between items-center mb-4">
          <h1 className="text-3xl font-bold">Chat Playground</h1>
          <div className="flex justify-between items-center">
            <SessionManager
              currentSession={currentSession}
              onSessionChange={handleSessionChange}
              onNewSession={handleNewSession}
            />
            <Button
              variant="outline"
              onClick={clearChat}
              disabled={isGenerating}
            >
              Clear Chat
            </Button>
          </div>
        </div>
      </div>

      {/* Main Two-Column Layout */}
      <div className="flex flex-1 gap-6 min-h-0 flex-col lg:flex-row">
        {/* Left Column - Configuration Panel */}
        <div className="w-full lg:w-80 lg:flex-shrink-0 space-y-6 p-4 border border-border rounded-lg bg-muted/30">
          <h2 className="text-lg font-semibold border-b pb-2 text-left">
            Settings
          </h2>

          {/* Model Configuration */}
          <div className="space-y-4 text-left">
            <h3 className="text-lg font-semibold border-b pb-2 text-left">
              Model Configuration
            </h3>
            <div className="space-y-3">
              <div>
                <label className="text-sm font-medium block mb-2">Model</label>
                <Select
                  value={currentSession?.selectedModel || ""}
                  onValueChange={value =>
                    setCurrentSession(prev =>
                      prev
                        ? {
                            ...prev,
                            selectedModel: value,
                            updatedAt: Date.now(),
                          }
                        : prev
                    )
                  }
                  disabled={isModelsLoading || isGenerating}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue
                      placeholder={
                        isModelsLoading ? "Loading..." : "Select Model"
                      }
                    />
                  </SelectTrigger>
                  <SelectContent>
                    {models.map(model => (
                      <SelectItem
                        key={model.identifier}
                        value={model.identifier}
                      >
                        {model.identifier}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {modelsError && (
                  <p className="text-destructive text-xs mt-1">{modelsError}</p>
                )}
              </div>

              <div>
                <label className="text-sm font-medium block mb-2">
                  System Message
                </label>
                <textarea
                  value={currentSession?.systemMessage || ""}
                  onChange={e =>
                    setCurrentSession(prev =>
                      prev
                        ? {
                            ...prev,
                            systemMessage: e.target.value,
                            updatedAt: Date.now(),
                          }
                        : prev
                    )
                  }
                  placeholder="You are a helpful assistant."
                  disabled={isGenerating}
                  className="w-full h-24 px-3 py-2 text-sm border border-input rounded-md resize-none focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
                />
              </div>
            </div>
          </div>

          {/* Vector Database Configuration */}
          <div className="space-y-4 text-left">
            <h3 className="text-lg font-semibold border-b pb-2 text-left">
              VectorDB Configuration
            </h3>

            <div className="space-y-3">
              <div>
                <label className="text-sm font-medium block mb-2">
                  Vector Database
                </label>
                <Select
                  value={currentSession?.selectedVectorDb || "none"}
                  onValueChange={value =>
                    setCurrentSession(prev =>
                      prev
                        ? {
                            ...prev,
                            selectedVectorDb: value,
                            updatedAt: Date.now(),
                          }
                        : prev
                    )
                  }
                  disabled={vectorDbsLoading || isGenerating}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue
                      placeholder={
                        vectorDbsLoading ? "Loading..." : "None (No RAG)"
                      }
                    />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="none">None (No RAG)</SelectItem>
                    {vectorDbs.map(db => (
                      <SelectItem key={db.identifier} value={db.identifier}>
                        {db.identifier}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {vectorDbsError && (
                  <p className="text-destructive text-xs mt-1">
                    {vectorDbsError}
                  </p>
                )}
              </div>

              <div>
                <VectorDbManager
                  client={client}
                  onVectorDbCreated={refreshVectorDbs}
                />
              </div>
            </div>
          </div>

          {/* Document Upload Section */}
          <div className="space-y-4 text-left">
            <DocumentUploader
              client={client}
              selectedVectorDb={currentSession?.selectedVectorDb || "none"}
              disabled={isGenerating}
            />
          </div>
        </div>

        {/* Right Column - Chat Interface */}
        <div className="flex-1 flex flex-col min-h-0 p-4 border border-border rounded-lg bg-background">
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
            setMessages={messages =>
              setCurrentSession(prev =>
                prev ? { ...prev, messages, updatedAt: Date.now() } : prev
              )
            }
          />
        </div>
      </div>
    </div>
  );
}
