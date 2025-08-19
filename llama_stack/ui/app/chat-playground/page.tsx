"use client";

import { useState, useEffect, useCallback, useRef } from "react";
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
import {
  SessionManager,
  SessionUtils,
} from "@/components/chat-playground/session-manager";

interface ChatSession {
  id: string;
  name: string;
  messages: Message[];
  selectedModel: string;
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
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const client = useAuthClient();
  const abortControllerRef = useRef<AbortController | null>(null);

  const isModelsLoading = modelsLoading ?? true;

  useEffect(() => {
    const saved = SessionUtils.loadCurrentSession();
    if (saved) {
      setCurrentSession(saved);
    } else {
      const def = SessionUtils.createDefaultSession();
      const defaultSession: ChatSession = {
        ...def,
        selectedModel: "",
        systemMessage: def.systemMessage || "You are a helpful assistant.",
      };
      setCurrentSession(defaultSession);
      SessionUtils.saveCurrentSession(defaultSession);
    }
  }, []);

  const handleModelChange = useCallback((newModel: string) => {
    setSelectedModel(newModel);
    setCurrentSession(prev =>
      prev
        ? {
            ...prev,
            selectedModel: newModel,
            updatedAt: Date.now(),
          }
        : prev
    );
  }, []);

  useEffect(() => {
    if (currentSession) {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
        setIsGenerating(false);
      }

      SessionUtils.saveCurrentSession(currentSession);
      setSelectedModel(currentSession.selectedModel);
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
        if (llmModels.length > 0) {
          handleModelChange(llmModels[0].identifier);
        }
      } catch (err) {
        console.error("Error fetching models:", err);
        setModelsError("Failed to fetch available models");
      } finally {
        setModelsLoading(false);
      }
    };

    fetchModels();
  }, [client, handleModelChange]);

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
    if (!input.trim()) return;

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

    await handleSubmitWithContent(userMessage.content);
  };

  const handleSubmitWithContent = async (content: string) => {
    setIsGenerating(true);
    setError(null);

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    try {
      const messageParams: CompletionCreateParams["messages"] = [
        ...(currentSession?.systemMessage
          ? [{ role: "system" as const, content: currentSession.systemMessage }]
          : []),
        ...(currentSession?.messages || []).map(msg => {
          const msgContent =
            typeof msg.content === "string"
              ? msg.content
              : extractTextContent(msg.content);
          if (msg.role === "user") {
            return { role: "user" as const, content: msgContent };
          } else if (msg.role === "assistant") {
            return { role: "assistant" as const, content: msgContent };
          } else {
            return { role: "system" as const, content: msgContent };
          }
        }),
        { role: "user" as const, content },
      ];

      const response = await client.chat.completions.create(
        {
          model: selectedModel || "",
          messages: messageParams,
          stream: true,
        },
        {
          signal: abortController.signal,
        } as { signal: AbortSignal }
      );

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "",
        createdAt: new Date(),
      };

      setCurrentSession(prev =>
        prev
          ? {
              ...prev,
              messages: [...prev.messages, assistantMessage],
              updatedAt: Date.now(),
            }
          : null
      );
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
      // don't show error if request was aborted
      if (err instanceof Error && err.name === "AbortError") {
        console.log("Request aborted");
        return;
      }

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
      abortControllerRef.current = null;
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

    const newSession: ChatSession = {
      ...SessionUtils.createDefaultSession(),
      selectedModel: defaultModel,
      systemMessage:
        currentSession?.systemMessage || "You are a helpful assistant.",
      messages: [],
      updatedAt: Date.now(),
      createdAt: Date.now(),
    };
    setCurrentSession(newSession);
    SessionUtils.saveCurrentSession(newSession);
  };

  return (
    <div className="flex flex-col h-full w-full max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <div className="flex justify-between items-center mb-4">
          <h1 className="text-3xl font-bold">Chat Playground</h1>
          <div className="flex items-center gap-3">
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
                  value={selectedModel}
                  onValueChange={handleModelChange}
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
