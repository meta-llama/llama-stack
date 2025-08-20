"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import type { Message } from "@/components/chat-playground/chat-message";

interface ChatSession {
  id: string;
  name: string;
  messages: Message[];
  selectedModel: string;
  selectedVectorDb: string;
  systemMessage: string;
  createdAt: number;
  updatedAt: number;
}

interface SessionManagerProps {
  currentSession: ChatSession | null;
  onSessionChange: (session: ChatSession) => void;
  onNewSession: () => void;
}

const SESSIONS_STORAGE_KEY = "chat-playground-sessions";
const CURRENT_SESSION_KEY = "chat-playground-current-session";

export function SessionManager({
  currentSession,
  onSessionChange,
}: SessionManagerProps) {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [newSessionName, setNewSessionName] = useState("");

  // Load sessions from localStorage
  useEffect(() => {
    const savedSessions = localStorage.getItem(SESSIONS_STORAGE_KEY);
    if (savedSessions) {
      try {
        setSessions(JSON.parse(savedSessions));
      } catch (err) {
        console.error("Error loading sessions:", err);
      }
    }
  }, []);

  // Save sessions to localStorage
  const saveSessions = (updatedSessions: ChatSession[]) => {
    setSessions(updatedSessions);
    localStorage.setItem(SESSIONS_STORAGE_KEY, JSON.stringify(updatedSessions));
  };

  const createNewSession = () => {
    const sessionName =
      newSessionName.trim() || `Session ${sessions.length + 1}`;
    const newSession: ChatSession = {
      id: Date.now().toString(),
      name: sessionName,
      messages: [],
      selectedModel: currentSession?.selectedModel || "",
      selectedVectorDb: currentSession?.selectedVectorDb || "",
      systemMessage:
        currentSession?.systemMessage || "You are a helpful assistant.",
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };

    const updatedSessions = [...sessions, newSession];
    saveSessions(updatedSessions);

    localStorage.setItem(CURRENT_SESSION_KEY, newSession.id);
    onSessionChange(newSession);

    setNewSessionName("");
    setShowCreateForm(false);
  };

  const switchToSession = (sessionId: string) => {
    const session = sessions.find(s => s.id === sessionId);
    if (session) {
      localStorage.setItem(CURRENT_SESSION_KEY, sessionId);
      onSessionChange(session);
    }
  };

  // Update current session in the sessions list
  useEffect(() => {
    if (currentSession) {
      setSessions(prevSessions => {
        const updatedSessions = prevSessions.map(session =>
          session.id === currentSession.id ? currentSession : session
        );

        // Add session if it doesn't exist
        if (!prevSessions.find(s => s.id === currentSession.id)) {
          updatedSessions.push(currentSession);
        }

        // Save to localStorage
        localStorage.setItem(
          SESSIONS_STORAGE_KEY,
          JSON.stringify(updatedSessions)
        );

        return updatedSessions;
      });
    }
  }, [currentSession]);

  return (
    <div className="relative">
      <div className="flex items-center gap-2">
        <Select
          value={currentSession?.id || ""}
          onValueChange={switchToSession}
        >
          <SelectTrigger className="w-[200px]">
            <SelectValue placeholder="Select Session" />
          </SelectTrigger>
          <SelectContent>
            {sessions.map(session => (
              <SelectItem key={session.id} value={session.id}>
                {session.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Button
          onClick={() => setShowCreateForm(true)}
          variant="outline"
          size="sm"
        >
          + New
        </Button>
      </div>

      {showCreateForm && (
        <Card className="absolute top-full left-0 mt-2 p-4 space-y-3 w-80 z-50 bg-background border shadow-lg">
          <h3 className="text-md font-semibold">Create New Session</h3>

          <Input
            value={newSessionName}
            onChange={e => setNewSessionName(e.target.value)}
            placeholder="Session name (optional)"
            onKeyDown={e => {
              if (e.key === "Enter") {
                createNewSession();
              } else if (e.key === "Escape") {
                setShowCreateForm(false);
                setNewSessionName("");
              }
            }}
          />

          <div className="flex gap-2">
            <Button onClick={createNewSession} className="flex-1">
              Create
            </Button>
            <Button
              variant="outline"
              onClick={() => {
                setShowCreateForm(false);
                setNewSessionName("");
              }}
              className="flex-1"
            >
              Cancel
            </Button>
          </div>
        </Card>
      )}

      {currentSession && sessions.length > 1 && (
        <div className="mt-2 text-xs text-gray-500">
          {sessions.length} sessions • Current: {currentSession.name}
          {currentSession.messages.length > 0 &&
            ` • ${currentSession.messages.length} messages`}
        </div>
      )}
    </div>
  );
}

// Export utility functions for session management
export const SessionUtils = {
  loadCurrentSession: (): ChatSession | null => {
    const currentSessionId = localStorage.getItem(CURRENT_SESSION_KEY);
    const savedSessions = localStorage.getItem(SESSIONS_STORAGE_KEY);

    if (currentSessionId && savedSessions) {
      try {
        const sessions: ChatSession[] = JSON.parse(savedSessions);
        return sessions.find(s => s.id === currentSessionId) || null;
      } catch (err) {
        console.error("Error loading current session:", err);
      }
    }
    return null;
  },

  saveCurrentSession: (session: ChatSession) => {
    const savedSessions = localStorage.getItem(SESSIONS_STORAGE_KEY);
    let sessions: ChatSession[] = [];

    if (savedSessions) {
      try {
        sessions = JSON.parse(savedSessions);
      } catch (err) {
        console.error("Error parsing sessions:", err);
      }
    }

    const existingIndex = sessions.findIndex(s => s.id === session.id);
    if (existingIndex >= 0) {
      sessions[existingIndex] = { ...session, updatedAt: Date.now() };
    } else {
      sessions.push({
        ...session,
        createdAt: Date.now(),
        updatedAt: Date.now(),
      });
    }

    localStorage.setItem(SESSIONS_STORAGE_KEY, JSON.stringify(sessions));
    localStorage.setItem(CURRENT_SESSION_KEY, session.id);
  },

  createDefaultSession: (
    inheritModel?: string,
    inheritVectorDb?: string
  ): ChatSession => ({
    id: Date.now().toString(),
    name: "Default Session",
    messages: [],
    selectedModel: inheritModel || "",
    selectedVectorDb: inheritVectorDb || "",
    systemMessage: "You are a helpful assistant.",
    createdAt: Date.now(),
    updatedAt: Date.now(),
  }),
};
