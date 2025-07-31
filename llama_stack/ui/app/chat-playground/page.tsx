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

export default function ChatPlaygroundPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const client = useAuthClient();

  const isModelsLoading = modelsLoading ?? true;


  useEffect(() => {
    const fetchModels = async () => {
      try {
        setModelsLoading(true);
        setModelsError(null);
        const modelList = await client.models.list();
        const llmModels = modelList.filter(model => model.model_type === 'llm');
        setModels(llmModels);
        if (llmModels.length > 0) {
          setSelectedModel(llmModels[0].identifier);
        }
      } catch (err) {
        console.error("Error fetching models:", err);
        setModelsError("Failed to fetch available models");
      } finally {
        setModelsLoading(false);
      }
    };

    fetchModels();
  }, [client]);

  const extractTextContent = (content: unknown): string => {
    if (typeof content === 'string') {
      return content;
    }
    if (Array.isArray(content)) {
      return content
        .filter(item => item && typeof item === 'object' && 'type' in item && item.type === 'text')
        .map(item => (item && typeof item === 'object' && 'text' in item) ? String(item.text) : '')
        .join('');
    }
    if (content && typeof content === 'object' && 'type' in content && content.type === 'text' && 'text' in content) {
      return String(content.text) || '';
    }
    return '';
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
  };

const handleSubmit = async (event?: { preventDefault?: () => void }) => {
  event?.preventDefault?.();
  if (!input.trim()) return;

  // Add user message to chat
  const userMessage: Message = {
    id: Date.now().toString(),
    role: "user",
    content: input.trim(),
    createdAt: new Date(),
  };

  setMessages(prev => [...prev, userMessage]);
  setInput("");

  // Use the helper function with the content
  await handleSubmitWithContent(userMessage.content);
};

const handleSubmitWithContent = async (content: string) => {
  setIsGenerating(true);
  setError(null);

  try {
    const messageParams: CompletionCreateParams["messages"] = [
      ...messages.map(msg => {
        const msgContent = typeof msg.content === 'string' ? msg.content : extractTextContent(msg.content);
        if (msg.role === "user") {
          return { role: "user" as const, content: msgContent };
        } else if (msg.role === "assistant") {
          return { role: "assistant" as const, content: msgContent };
        } else {
          return { role: "system" as const, content: msgContent };
        }
      }),
      { role: "user" as const, content }
    ];

    const response = await client.chat.completions.create({
      model: selectedModel,
      messages: messageParams,
      stream: true,
    });

    const assistantMessage: Message = {
      id: (Date.now() + 1).toString(),
      role: "assistant",
      content: "",
      createdAt: new Date(),
    };

    setMessages(prev => [...prev, assistantMessage]);
    let fullContent = "";
    for await (const chunk of response) {
      if (chunk.choices && chunk.choices[0]?.delta?.content) {
        const deltaContent = chunk.choices[0].delta.content;
        fullContent += deltaContent;

        flushSync(() => {
          setMessages(prev => {
            const newMessages = [...prev];
            const lastMessage = newMessages[newMessages.length - 1];
            if (lastMessage.role === "assistant") {
              lastMessage.content = fullContent;
            }
            return newMessages;
          });
        });
      }
    }
  } catch (err) {
    console.error("Error sending message:", err);
    setError("Failed to send message. Please try again.");
    setMessages(prev => prev.slice(0, -1));
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
    setMessages(prev => [...prev, newMessage])
    handleSubmitWithContent(newMessage.content);
  };

  const clearChat = () => {
    setMessages([]);
    setError(null);
  };

  return (
    <div className="flex flex-col h-full max-w-4xl mx-auto">
      <div className="mb-4 flex justify-between items-center">
        <h1 className="text-2xl font-bold">Chat Playground</h1>
        <div className="flex gap-2">
          <Select value={selectedModel} onValueChange={setSelectedModel} disabled={isModelsLoading || isGenerating}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder={isModelsLoading ? "Loading models..." : "Select Model"} />
            </SelectTrigger>
            <SelectContent>
              {models.map((model) => (
                <SelectItem key={model.identifier} value={model.identifier}>
                  {model.identifier}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button variant="outline" onClick={clearChat} disabled={isGenerating}>
            Clear Chat
          </Button>
        </div>
      </div>

      {modelsError && (
        <div className="mb-4 p-3 bg-destructive/10 border border-destructive/20 rounded-md">
          <p className="text-destructive text-sm">{modelsError}</p>
        </div>
      )}

      {error && (
        <div className="mb-4 p-3 bg-destructive/10 border border-destructive/20 rounded-md">
          <p className="text-destructive text-sm">{error}</p>
        </div>
      )}

      <Chat
        className="flex-1"
        messages={messages}
        handleSubmit={handleSubmit}
        input={input}
        handleInputChange={handleInputChange}
        isGenerating={isGenerating}
        append={append}
        suggestions={suggestions}
        setMessages={setMessages}
      />
    </div>
  );
}
