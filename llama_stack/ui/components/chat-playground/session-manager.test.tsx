import React from "react";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  act,
} from "@testing-library/react";
import "@testing-library/jest-dom";
import { SessionManager, SessionUtils } from "./session-manager";
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

const mockOnSessionChange = jest.fn();
const mockOnNewSession = jest.fn();

const localStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
};

Object.defineProperty(window, "localStorage", {
  value: localStorageMock,
  writable: true,
});

// Mock crypto.randomUUID for test environment
let uuidCounter = 0;
Object.defineProperty(globalThis, "crypto", {
  value: {
    randomUUID: jest.fn(() => `test-uuid-${++uuidCounter}`),
  },
  writable: true,
});

describe("SessionManager", () => {
  const mockSession: ChatSession = {
    id: "session_123",
    name: "Test Session",
    messages: [
      {
        id: "msg_1",
        role: "user",
        content: "Hello",
        timestamp: Date.now(),
      },
    ],
    selectedModel: "test-model",
    selectedVectorDb: "test-vector-db",
    systemMessage: "You are a helpful assistant.",
    createdAt: 1710000000,
    updatedAt: 1710001000,
  };

  const mockSessions: ChatSession[] = [
    mockSession,
    {
      id: "session_456",
      name: "Another Session",
      messages: [],
      selectedModel: "another-model",
      selectedVectorDb: "another-vector-db",
      systemMessage: "You are another assistant.",
      createdAt: 1710002000,
      updatedAt: 1710003000,
    },
  ];

  beforeEach(() => {
    jest.clearAllMocks();
    localStorageMock.getItem.mockReturnValue(null);
    localStorageMock.setItem.mockImplementation(() => {});
    uuidCounter = 0; // Reset UUID counter for consistent test behavior
  });

  describe("Component Rendering", () => {
    test("renders session selector with placeholder when no session selected", async () => {
      await act(async () => {
        render(
          <SessionManager
            currentSession={null}
            onSessionChange={mockOnSessionChange}
            onNewSession={mockOnNewSession}
          />
        );
      });

      expect(screen.getByText("Select Session")).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /New/ })).toBeInTheDocument();
    });

    test("renders current session name when session is selected", async () => {
      localStorageMock.getItem.mockImplementation(key => {
        if (key === "chat-playground-sessions") {
          return JSON.stringify(mockSessions);
        }
        return null;
      });

      await act(async () => {
        render(
          <SessionManager
            currentSession={mockSession}
            onSessionChange={mockOnSessionChange}
            onNewSession={mockOnNewSession}
          />
        );
      });

      await waitFor(() => {
        expect(screen.getByText("Test Session")).toBeInTheDocument();
      });
    });

    test("shows session info when multiple sessions exist", async () => {
      localStorageMock.getItem.mockImplementation(key => {
        if (key === "chat-playground-sessions") {
          return JSON.stringify(mockSessions);
        }
        return null;
      });

      await act(async () => {
        render(
          <SessionManager
            currentSession={mockSession}
            onSessionChange={mockOnSessionChange}
            onNewSession={mockOnNewSession}
          />
        );
      });

      await waitFor(() => {
        expect(screen.getByText(/2 sessions/)).toBeInTheDocument();
        expect(screen.getByText(/Current: Test Session/)).toBeInTheDocument();
        expect(screen.getByText(/1 messages/)).toBeInTheDocument();
      });
    });
  });

  describe("Session Creation", () => {
    test("shows create form when New button is clicked", async () => {
      await act(async () => {
        render(
          <SessionManager
            currentSession={mockSession}
            onSessionChange={mockOnSessionChange}
            onNewSession={mockOnNewSession}
          />
        );
      });

      const newButton = screen.getByRole("button", { name: /New/ });
      fireEvent.click(newButton);

      expect(screen.getByText("Create New Session")).toBeInTheDocument();
      expect(
        screen.getByPlaceholderText("Session name (optional)")
      ).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: "Create" })
      ).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: "Cancel" })
      ).toBeInTheDocument();
    });

    test("creates session with custom name", async () => {
      await act(async () => {
        render(
          <SessionManager
            currentSession={mockSession}
            onSessionChange={mockOnSessionChange}
            onNewSession={mockOnNewSession}
          />
        );
      });

      const newButton = screen.getByRole("button", { name: /New/ });
      fireEvent.click(newButton);

      const nameInput = screen.getByPlaceholderText("Session name (optional)");
      fireEvent.change(nameInput, { target: { value: "Custom Session" } });

      const createButton = screen.getByRole("button", { name: "Create" });
      fireEvent.click(createButton);

      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        "chat-playground-sessions",
        expect.stringContaining("Custom Session")
      );
      expect(mockOnSessionChange).toHaveBeenCalled();
    });

    test("creates session with default name when no name provided", async () => {
      localStorageMock.getItem.mockImplementation(key => {
        if (key === "chat-playground-sessions") {
          return JSON.stringify(mockSessions);
        }
        return null;
      });

      await act(async () => {
        render(
          <SessionManager
            currentSession={mockSession}
            onSessionChange={mockOnSessionChange}
            onNewSession={mockOnNewSession}
          />
        );
      });

      const newButton = screen.getByRole("button", { name: /New/ });
      fireEvent.click(newButton);

      const createButton = screen.getByRole("button", { name: "Create" });
      fireEvent.click(createButton);

      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        "chat-playground-sessions",
        expect.stringContaining("Session 3")
      );
    });

    test("cancels session creation", async () => {
      await act(async () => {
        render(
          <SessionManager
            currentSession={null}
            onSessionChange={mockOnSessionChange}
            onNewSession={mockOnNewSession}
          />
        );
      });

      const newButton = screen.getByRole("button", { name: /New/ });
      fireEvent.click(newButton);

      const nameInput = screen.getByPlaceholderText("Session name (optional)");
      fireEvent.change(nameInput, { target: { value: "Test Input" } });

      localStorageMock.setItem.mockClear();

      const cancelButton = screen.getByRole("button", { name: "Cancel" });
      fireEvent.click(cancelButton);

      expect(screen.queryByText("Create New Session")).not.toBeInTheDocument();
      expect(localStorageMock.setItem).not.toHaveBeenCalled();
    });

    test("creates session on Enter key press", async () => {
      await act(async () => {
        render(
          <SessionManager
            currentSession={mockSession}
            onSessionChange={mockOnSessionChange}
            onNewSession={mockOnNewSession}
          />
        );
      });

      const newButton = screen.getByRole("button", { name: /New/ });
      fireEvent.click(newButton);

      const nameInput = screen.getByPlaceholderText("Session name (optional)");
      fireEvent.change(nameInput, { target: { value: "Enter Session" } });
      fireEvent.keyDown(nameInput, { key: "Enter" });

      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        "chat-playground-sessions",
        expect.stringContaining("Enter Session")
      );
    });

    test("cancels session creation on Escape key press", async () => {
      await act(async () => {
        render(
          <SessionManager
            currentSession={mockSession}
            onSessionChange={mockOnSessionChange}
            onNewSession={mockOnNewSession}
          />
        );
      });

      const newButton = screen.getByRole("button", { name: /New/ });
      fireEvent.click(newButton);

      const nameInput = screen.getByPlaceholderText("Session name (optional)");
      fireEvent.keyDown(nameInput, { key: "Escape" });

      expect(screen.queryByText("Create New Session")).not.toBeInTheDocument();
    });
  });

  describe("Session Switching", () => {
    test("switches to selected session", async () => {
      localStorageMock.getItem.mockImplementation(key => {
        if (key === "chat-playground-sessions") {
          return JSON.stringify(mockSessions);
        }
        return null;
      });

      await act(async () => {
        render(
          <SessionManager
            currentSession={mockSession}
            onSessionChange={mockOnSessionChange}
            onNewSession={mockOnNewSession}
          />
        );
      });

      await waitFor(() => {
        expect(screen.getByText("Test Session")).toBeInTheDocument();
      });

      const selectTrigger = screen.getByRole("combobox");
      fireEvent.click(selectTrigger);

      await waitFor(() => {
        const anotherSessionOption = screen.getByText("Another Session");
        fireEvent.click(anotherSessionOption);
      });

      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        "chat-playground-current-session",
        "session_456"
      );
      expect(mockOnSessionChange).toHaveBeenCalledWith(
        expect.objectContaining({
          id: "session_456",
          name: "Another Session",
        })
      );
    });
  });

  describe("LocalStorage Integration", () => {
    test("loads sessions from localStorage on mount", async () => {
      localStorageMock.getItem.mockImplementation(key => {
        if (key === "chat-playground-sessions") {
          return JSON.stringify(mockSessions);
        }
        return null;
      });

      await act(async () => {
        render(
          <SessionManager
            currentSession={mockSession}
            onSessionChange={mockOnSessionChange}
            onNewSession={mockOnNewSession}
          />
        );
      });

      await waitFor(() => {
        expect(localStorageMock.getItem).toHaveBeenCalledWith(
          "chat-playground-sessions"
        );
      });
    });

    test("handles corrupted localStorage data gracefully", async () => {
      localStorageMock.getItem.mockReturnValue("invalid json");
      const consoleSpy = jest.spyOn(console, "error").mockImplementation();

      await act(async () => {
        render(
          <SessionManager
            currentSession={mockSession}
            onSessionChange={mockOnSessionChange}
            onNewSession={mockOnNewSession}
          />
        );
      });

      expect(consoleSpy).toHaveBeenCalledWith(
        "Error parsing JSON:",
        expect.any(Error)
      );

      consoleSpy.mockRestore();
    });

    test("updates localStorage when current session changes", async () => {
      const updatedSession = {
        ...mockSession,
        messages: [
          ...mockSession.messages,
          {
            id: "msg_2",
            role: "assistant" as const,
            content: "Hello back!",
            timestamp: Date.now(),
          },
        ],
        updatedAt: Date.now(),
      };

      localStorageMock.getItem.mockImplementation(key => {
        if (key === "chat-playground-sessions") {
          return JSON.stringify([mockSession]);
        }
        return null;
      });

      const { rerender } = render(
        <SessionManager
          currentSession={mockSession}
          onSessionChange={mockOnSessionChange}
          onNewSession={mockOnNewSession}
        />
      );

      await act(async () => {
        rerender(
          <SessionManager
            currentSession={updatedSession}
            onSessionChange={mockOnSessionChange}
            onNewSession={mockOnNewSession}
          />
        );
      });

      await waitFor(() => {
        expect(localStorageMock.setItem).toHaveBeenCalledWith(
          "chat-playground-sessions",
          expect.stringContaining(updatedSession.id)
        );
      });
    });
  });

  describe("Session Deletion", () => {
    test("shows delete button only when multiple sessions exist", async () => {
      localStorageMock.getItem.mockImplementation(key => {
        if (key === "chat-playground-sessions") {
          return JSON.stringify([mockSession]);
        }
        return null;
      });

      await act(async () => {
        render(
          <SessionManager
            currentSession={mockSession}
            onSessionChange={mockOnSessionChange}
            onNewSession={mockOnNewSession}
          />
        );
      });

      expect(
        screen.queryByTitle("Delete current session")
      ).not.toBeInTheDocument();

      localStorageMock.getItem.mockImplementation(key => {
        if (key === "chat-playground-sessions") {
          return JSON.stringify(mockSessions);
        }
        return null;
      });

      const { rerender } = render(
        <SessionManager
          currentSession={mockSession}
          onSessionChange={mockOnSessionChange}
          onNewSession={mockOnNewSession}
        />
      );

      await act(async () => {
        rerender(
          <SessionManager
            currentSession={mockSession}
            onSessionChange={mockOnSessionChange}
            onNewSession={mockOnNewSession}
          />
        );
      });

      await waitFor(() => {
        expect(screen.getByTitle("Delete current session")).toBeInTheDocument();
      });
    });

    test("deletes current session after confirmation", async () => {
      window.confirm = jest.fn().mockReturnValue(true);

      localStorageMock.getItem.mockImplementation(key => {
        if (key === "chat-playground-sessions") {
          return JSON.stringify(mockSessions);
        }
        return null;
      });

      await act(async () => {
        render(
          <SessionManager
            currentSession={mockSession}
            onSessionChange={mockOnSessionChange}
            onNewSession={mockOnNewSession}
          />
        );
      });

      await waitFor(() => {
        expect(screen.getByTitle("Delete current session")).toBeInTheDocument();
      });

      const deleteButton = screen.getByTitle("Delete current session");
      fireEvent.click(deleteButton);

      expect(window.confirm).toHaveBeenCalledWith(
        "Are you sure you want to delete this session? This action cannot be undone."
      );
      expect(mockOnSessionChange).toHaveBeenCalled();
    });

    test("cancels deletion when user rejects confirmation", async () => {
      window.confirm = jest.fn().mockReturnValue(false);

      localStorageMock.getItem.mockImplementation(key => {
        if (key === "chat-playground-sessions") {
          return JSON.stringify(mockSessions);
        }
        return null;
      });

      await act(async () => {
        render(
          <SessionManager
            currentSession={mockSession}
            onSessionChange={mockOnSessionChange}
            onNewSession={mockOnNewSession}
          />
        );
      });

      await waitFor(() => {
        expect(screen.getByTitle("Delete current session")).toBeInTheDocument();
      });

      const deleteButton = screen.getByTitle("Delete current session");
      fireEvent.click(deleteButton);

      expect(window.confirm).toHaveBeenCalled();
      expect(mockOnSessionChange).not.toHaveBeenCalled();
    });

    test("prevents deletion of the last remaining session", async () => {
      const singleSession = [mockSession];

      localStorageMock.getItem.mockImplementation(key => {
        if (key === "chat-playground-sessions") {
          return JSON.stringify(singleSession);
        }
        return null;
      });

      await act(async () => {
        render(
          <SessionManager
            currentSession={mockSession}
            onSessionChange={mockOnSessionChange}
            onNewSession={mockOnNewSession}
          />
        );
      });

      expect(
        screen.queryByTitle("Delete current session")
      ).not.toBeInTheDocument();
    });
  });

  describe("Error Handling", () => {
    test("component renders without crashing when localStorage is unavailable", async () => {
      await act(async () => {
        render(
          <SessionManager
            currentSession={mockSession}
            onSessionChange={mockOnSessionChange}
            onNewSession={mockOnNewSession}
          />
        );
      });

      expect(screen.getByRole("button", { name: /New/ })).toBeInTheDocument();
      expect(screen.getByText("Test Session")).toBeInTheDocument();
    });
  });
});

describe("SessionUtils", () => {
  const mockSession: ChatSession = {
    id: "utils_session_123",
    name: "Utils Test Session",
    messages: [],
    selectedModel: "utils-model",
    selectedVectorDb: "utils-vector-db",
    systemMessage: "You are a utils assistant.",
    createdAt: 1710000000,
    updatedAt: 1710001000,
  };

  const mockSessions = [mockSession];

  beforeEach(() => {
    jest.clearAllMocks();
    localStorageMock.getItem.mockReturnValue(null);
    localStorageMock.setItem.mockImplementation(() => {});
  });

  describe("loadCurrentSession", () => {
    test("returns null when no current session ID stored", () => {
      const result = SessionUtils.loadCurrentSession();
      expect(result).toBeNull();
    });

    test("returns null when no sessions stored", () => {
      localStorageMock.getItem.mockImplementation(key => {
        if (key === "chat-playground-current-session") {
          return "session_123";
        }
        return null;
      });

      const result = SessionUtils.loadCurrentSession();
      expect(result).toBeNull();
    });

    test("returns current session when found", () => {
      localStorageMock.getItem.mockImplementation(key => {
        if (key === "chat-playground-current-session") {
          return "utils_session_123";
        }
        if (key === "chat-playground-sessions") {
          return JSON.stringify(mockSessions);
        }
        return null;
      });

      const result = SessionUtils.loadCurrentSession();
      expect(result).toEqual(mockSession);
    });

    test("returns null when current session ID not found in sessions", () => {
      localStorageMock.getItem.mockImplementation(key => {
        if (key === "chat-playground-current-session") {
          return "nonexistent_session";
        }
        if (key === "chat-playground-sessions") {
          return JSON.stringify(mockSessions);
        }
        return null;
      });

      const result = SessionUtils.loadCurrentSession();
      expect(result).toBeNull();
    });

    test("handles corrupted sessions data gracefully", () => {
      localStorageMock.getItem.mockImplementation(key => {
        if (key === "chat-playground-current-session") {
          return "session_123";
        }
        if (key === "chat-playground-sessions") {
          return "invalid json";
        }
        return null;
      });

      const consoleSpy = jest.spyOn(console, "error").mockImplementation();
      const result = SessionUtils.loadCurrentSession();

      expect(result).toBeNull();
      expect(consoleSpy).toHaveBeenCalledWith(
        "Error parsing JSON:",
        expect.any(Error)
      );

      consoleSpy.mockRestore();
    });
  });

  describe("saveCurrentSession", () => {
    test("saves new session to localStorage", () => {
      localStorageMock.setItem.mockClear();

      SessionUtils.saveCurrentSession(mockSession);

      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        "chat-playground-sessions",
        expect.stringContaining(mockSession.id)
      );
      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        "chat-playground-current-session",
        mockSession.id
      );
    });

    test("updates existing session in localStorage", () => {
      localStorageMock.setItem.mockClear();
      localStorageMock.getItem.mockImplementation(key => {
        if (key === "chat-playground-sessions") {
          return JSON.stringify(mockSessions);
        }
        return null;
      });

      const updatedSession = {
        ...mockSession,
        name: "Updated Session Name",
        messages: [
          {
            id: "msg_1",
            role: "user" as const,
            content: "Test message",
            timestamp: Date.now(),
          },
        ],
      };

      SessionUtils.saveCurrentSession(updatedSession);

      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        "chat-playground-sessions",
        expect.stringContaining("Updated Session Name")
      );
      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        "chat-playground-current-session",
        updatedSession.id
      );
    });

    test("handles corrupted sessions data gracefully", () => {
      localStorageMock.setItem.mockClear();
      localStorageMock.getItem.mockReturnValue("invalid json");
      const consoleSpy = jest.spyOn(console, "error").mockImplementation();

      SessionUtils.saveCurrentSession(mockSession);

      expect(consoleSpy).toHaveBeenCalledWith(
        "Error parsing JSON:",
        expect.any(Error)
      );
      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        "chat-playground-sessions",
        expect.stringContaining(mockSession.id)
      );

      consoleSpy.mockRestore();
    });

    test("updates timestamps correctly", () => {
      localStorageMock.setItem.mockClear();
      const originalNow = Date.now;
      const mockTime = 1710005000;
      Date.now = jest.fn(() => mockTime);

      SessionUtils.saveCurrentSession(mockSession);

      const savedSessionsCall = localStorageMock.setItem.mock.calls.find(
        call => call[0] === "chat-playground-sessions"
      );
      const savedSessions = JSON.parse(savedSessionsCall[1]);

      expect(savedSessions[0].updatedAt).toBe(mockTime);

      Date.now = originalNow;
    });
  });

  describe("createDefaultSession", () => {
    test("creates default session with default values", () => {
      const result = SessionUtils.createDefaultSession();

      expect(result).toEqual(
        expect.objectContaining({
          name: "Default Session",
          messages: [],
          selectedModel: "",
          selectedVectorDb: "",
          systemMessage: "You are a helpful assistant.",
        })
      );
      expect(result.id).toBeTruthy();
      expect(result.createdAt).toBeTruthy();
      expect(result.updatedAt).toBeTruthy();
    });

    test("creates default session with inherited model", () => {
      const result = SessionUtils.createDefaultSession("inherited-model");

      expect(result.selectedModel).toBe("inherited-model");
      expect(result.selectedVectorDb).toBe("");
    });

    test("creates default session with inherited model and vector db", () => {
      const result = SessionUtils.createDefaultSession(
        "inherited-model",
        "inherited-vector-db"
      );

      expect(result.selectedModel).toBe("inherited-model");
      expect(result.selectedVectorDb).toBe("inherited-vector-db");
    });

    test("creates unique session IDs", () => {
      const originalNow = Date.now;
      let mockTime = 1710005000;
      Date.now = jest.fn(() => ++mockTime);

      const session1 = SessionUtils.createDefaultSession();
      const session2 = SessionUtils.createDefaultSession();

      expect(session1.id).not.toBe(session2.id);

      Date.now = originalNow;
    });

    test("sets creation and update timestamps", () => {
      const result = SessionUtils.createDefaultSession();

      expect(result.createdAt).toBeTruthy();
      expect(result.updatedAt).toBeTruthy();
      expect(typeof result.createdAt).toBe("number");
      expect(typeof result.updatedAt).toBe("number");
    });
  });

  describe("deleteSession", () => {
    test("deletes session and returns deleted session info", () => {
      localStorageMock.setItem.mockClear();
      localStorageMock.getItem.mockImplementation(key => {
        if (key === "chat-playground-sessions") {
          return JSON.stringify(mockSessions);
        }
        if (key === "chat-playground-current-session") {
          return "utils_session_123";
        }
        return null;
      });

      const result = SessionUtils.deleteSession("utils_session_123");

      expect(result.deletedSession).toEqual(mockSession);
      expect(result.remainingSessions).toHaveLength(0);
      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        "chat-playground-sessions",
        "[]"
      );
    });

    test("removes current session key when deleting current session", () => {
      localStorageMock.getItem.mockImplementation(key => {
        if (key === "chat-playground-sessions") {
          return JSON.stringify(mockSessions);
        }
        if (key === "chat-playground-current-session") {
          return "utils_session_123";
        }
        return null;
      });

      SessionUtils.deleteSession("utils_session_123");

      expect(localStorageMock.removeItem).toHaveBeenCalledWith(
        "chat-playground-current-session"
      );
    });

    test("does not remove current session key when deleting different session", () => {
      localStorageMock.getItem.mockImplementation(key => {
        if (key === "chat-playground-sessions") {
          return JSON.stringify([
            mockSession,
            { ...mockSession, id: "other_session" },
          ]);
        }
        if (key === "chat-playground-current-session") {
          return "utils_session_123";
        }
        return null;
      });

      SessionUtils.deleteSession("other_session");

      expect(localStorageMock.removeItem).not.toHaveBeenCalledWith(
        "chat-playground-current-session"
      );
    });

    test("returns null for non-existent session", () => {
      localStorageMock.getItem.mockImplementation(key => {
        if (key === "chat-playground-sessions") {
          return JSON.stringify(mockSessions);
        }
        return null;
      });

      const result = SessionUtils.deleteSession("non_existent");

      expect(result.deletedSession).toBeNull();
      expect(result.remainingSessions).toEqual(mockSessions);
    });

    test("handles corrupted sessions data gracefully", () => {
      localStorageMock.getItem.mockReturnValue("invalid json");
      const consoleSpy = jest.spyOn(console, "error").mockImplementation();

      const result = SessionUtils.deleteSession("any_session");

      expect(result.deletedSession).toBeNull();
      expect(result.remainingSessions).toEqual([]);
      expect(consoleSpy).toHaveBeenCalledWith(
        "Error parsing JSON:",
        expect.any(Error)
      );

      consoleSpy.mockRestore();
    });
  });
});
