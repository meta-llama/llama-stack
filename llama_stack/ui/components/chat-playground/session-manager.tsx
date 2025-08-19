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
import { Trash2 } from "lucide-react";
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

// ensures this only happens client side
const safeLocalStorage = {
  getItem: (key: string): string | null => {
    if (typeof window === "undefined") return null;
    try {
      return localStorage.getItem(key);
    } catch (err) {
      console.error("Error accessing localStorage:", err);
      return null;
    }
  },
  setItem: (key: string, value: string): void => {
    if (typeof window === "undefined") return;
    try {
      localStorage.setItem(key, value);
    } catch (err) {
      console.error("Error writing to localStorage:", err);
    }
  },
  removeItem: (key: string): void => {
    if (typeof window === "undefined") return;
    try {
      localStorage.removeItem(key);
    } catch (err) {
      console.error("Error removing from localStorage:", err);
    }
  },
};

function safeJsonParse<T>(jsonString: string | null, fallback: T): T {
  if (!jsonString) return fallback;
  try {
    return JSON.parse(jsonString) as T;
  } catch (err) {
    console.error("Error parsing JSON:", err);
    return fallback;
  }
}

const generateSessionId = (): string => {
  return globalThis.crypto.randomUUID();
};

export function SessionManager({
  currentSession,
  onSessionChange,
}: SessionManagerProps) {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [newSessionName, setNewSessionName] = useState("");

  useEffect(() => {
    const savedSessions = safeLocalStorage.getItem(SESSIONS_STORAGE_KEY);
    const sessions = safeJsonParse<ChatSession[]>(savedSessions, []);
    setSessions(sessions);
  }, []);

  const saveSessions = (updatedSessions: ChatSession[]) => {
    setSessions(updatedSessions);
    safeLocalStorage.setItem(
      SESSIONS_STORAGE_KEY,
      JSON.stringify(updatedSessions)
    );
  };

  const createNewSession = () => {
    const sessionName =
      newSessionName.trim() || `Session ${sessions.length + 1}`;
    const newSession: ChatSession = {
      id: generateSessionId(),
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

    safeLocalStorage.setItem(CURRENT_SESSION_KEY, newSession.id);
    onSessionChange(newSession);

    setNewSessionName("");
    setShowCreateForm(false);
  };

  const switchToSession = (sessionId: string) => {
    const session = sessions.find(s => s.id === sessionId);
    if (session) {
      safeLocalStorage.setItem(CURRENT_SESSION_KEY, sessionId);
      onSessionChange(session);
    }
  };

  const deleteSession = (sessionId: string) => {
    if (sessions.length <= 1) {
      return;
    }

    if (
      confirm(
        "Are you sure you want to delete this session? This action cannot be undone."
      )
    ) {
      const updatedSessions = sessions.filter(s => s.id !== sessionId);
      saveSessions(updatedSessions);

      if (currentSession?.id === sessionId) {
        const newCurrentSession = updatedSessions[0] || null;
        if (newCurrentSession) {
          safeLocalStorage.setItem(CURRENT_SESSION_KEY, newCurrentSession.id);
          onSessionChange(newCurrentSession);
        } else {
          safeLocalStorage.removeItem(CURRENT_SESSION_KEY);
          const defaultSession = SessionUtils.createDefaultSession();
          saveSessions([defaultSession]);
          safeLocalStorage.setItem(CURRENT_SESSION_KEY, defaultSession.id);
          onSessionChange(defaultSession);
        }
      }
    }
  };

  useEffect(() => {
    if (currentSession) {
      setSessions(prevSessions => {
        const updatedSessions = prevSessions.map(session =>
          session.id === currentSession.id ? currentSession : session
        );

        if (!prevSessions.find(s => s.id === currentSession.id)) {
          updatedSessions.push(currentSession);
        }

        safeLocalStorage.setItem(
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

        {currentSession && sessions.length > 1 && (
          <Button
            onClick={() => deleteSession(currentSession.id)}
            variant="outline"
            size="sm"
            className="text-destructive hover:text-destructive hover:bg-destructive/10"
            title="Delete current session"
          >
            <Trash2 className="h-3 w-3" />
          </Button>
        )}
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
        <div className="absolute top-full left-0 mt-1 text-xs text-gray-500 whitespace-nowrap">
          {sessions.length} sessions • Current: {currentSession.name}
          {currentSession.messages.length > 0 &&
            ` • ${currentSession.messages.length} messages`}
        </div>
      )}
    </div>
  );
}

export const SessionUtils = {
  loadCurrentSession: (): ChatSession | null => {
    const currentSessionId = safeLocalStorage.getItem(CURRENT_SESSION_KEY);
    const savedSessions = safeLocalStorage.getItem(SESSIONS_STORAGE_KEY);

    if (currentSessionId && savedSessions) {
      const sessions = safeJsonParse<ChatSession[]>(savedSessions, []);
      return sessions.find(s => s.id === currentSessionId) || null;
    }
    return null;
  },

  saveCurrentSession: (session: ChatSession) => {
    const savedSessions = safeLocalStorage.getItem(SESSIONS_STORAGE_KEY);
    const sessions = safeJsonParse<ChatSession[]>(savedSessions, []);

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

    safeLocalStorage.setItem(SESSIONS_STORAGE_KEY, JSON.stringify(sessions));
    safeLocalStorage.setItem(CURRENT_SESSION_KEY, session.id);
  },

  createDefaultSession: (
    inheritModel?: string,
    inheritVectorDb?: string
  ): ChatSession => ({
    id: generateSessionId(),
    name: "Default Session",
    messages: [],
    selectedModel: inheritModel || "",
    selectedVectorDb: inheritVectorDb || "",
    systemMessage: "You are a helpful assistant.",
    createdAt: Date.now(),
    updatedAt: Date.now(),
  }),

  deleteSession: (
    sessionId: string
  ): {
    deletedSession: ChatSession | null;
    remainingSessions: ChatSession[];
  } => {
    const savedSessions = safeLocalStorage.getItem(SESSIONS_STORAGE_KEY);
    const sessions = safeJsonParse<ChatSession[]>(savedSessions, []);

    const sessionToDelete = sessions.find(s => s.id === sessionId);
    const remainingSessions = sessions.filter(s => s.id !== sessionId);

    safeLocalStorage.setItem(
      SESSIONS_STORAGE_KEY,
      JSON.stringify(remainingSessions)
    );

    const currentSessionId = safeLocalStorage.getItem(CURRENT_SESSION_KEY);
    if (currentSessionId === sessionId) {
      safeLocalStorage.removeItem(CURRENT_SESSION_KEY);
    }

    return { deletedSession: sessionToDelete || null, remainingSessions };
  },
};
