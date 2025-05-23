import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import { ChatCompletionsTable } from "./chat-completion-table";
import { ChatCompletion } from "@/lib/types"; // Assuming this path is correct

// Mock next/navigation
const mockPush = jest.fn();
jest.mock("next/navigation", () => ({
  useRouter: () => ({
    push: mockPush,
  }),
}));

// Mock helper functions
// These are hoisted, so their mocks are available throughout the file
jest.mock("@/lib/truncate-text");
jest.mock("@/lib/format-tool-call");

// Import the mocked functions to set up default or specific implementations
import { truncateText as originalTruncateText } from "@/lib/truncate-text";
import { formatToolCallToString as originalFormatToolCallToString } from "@/lib/format-tool-call";

// Cast to jest.Mock for typings
const truncateText = originalTruncateText as jest.Mock;
const formatToolCallToString = originalFormatToolCallToString as jest.Mock;

describe("ChatCompletionsTable", () => {
  const defaultProps = {
    completions: [] as ChatCompletion[],
    isLoading: false,
    error: null,
  };

  beforeEach(() => {
    // Reset all mocks before each test
    mockPush.mockClear();
    truncateText.mockClear();
    formatToolCallToString.mockClear();

    // Default pass-through implementation for tests not focusing on truncation/formatting
    truncateText.mockImplementation((text: string | undefined) => text);
    formatToolCallToString.mockImplementation((toolCall: any) =>
      toolCall && typeof toolCall === "object" && toolCall.name
        ? `[DefaultToolCall:${toolCall.name}]`
        : "[InvalidToolCall]",
    );
  });

  test("renders without crashing with default props", () => {
    render(<ChatCompletionsTable {...defaultProps} />);
    // Check for a unique element that should be present in the non-empty, non-loading, non-error state
    // For now, as per Task 1, we will test the empty state message
    expect(screen.getByText("No chat completions found.")).toBeInTheDocument();
  });

  test("click on a row navigates to the correct URL", () => {
    const { rerender } = render(<ChatCompletionsTable {...defaultProps} />);

    // Simulate a scenario where a completion exists and is clicked
    const mockCompletion: ChatCompletion = {
      id: "comp_123",
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model: "llama-test-model",
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: "Test output" },
          finish_reason: "stop",
        },
      ],
      input_messages: [{ role: "user", content: "Test input" }],
    };

    rerender(
      <ChatCompletionsTable {...defaultProps} completions={[mockCompletion]} />,
    );
    const row = screen.getByText("Test input").closest("tr");
    if (row) {
      fireEvent.click(row);
      expect(mockPush).toHaveBeenCalledWith("/logs/chat-completions/comp_123");
    } else {
      throw new Error('Row with "Test input" not found for router mock test.');
    }
  });

  describe("Loading State", () => {
    test("renders skeleton UI when isLoading is true", () => {
      const { container } = render(
        <ChatCompletionsTable {...defaultProps} isLoading={true} />,
      );

      // The Skeleton component uses data-slot="skeleton"
      const skeletonSelector = '[data-slot="skeleton"]';

      // Check for skeleton in the table caption
      const tableCaption = container.querySelector("caption");
      expect(tableCaption).toBeInTheDocument();
      if (tableCaption) {
        const captionSkeleton = tableCaption.querySelector(skeletonSelector);
        expect(captionSkeleton).toBeInTheDocument();
      }

      // Check for skeletons in the table body cells
      const tableBody = container.querySelector("tbody");
      expect(tableBody).toBeInTheDocument();
      if (tableBody) {
        const bodySkeletons = tableBody.querySelectorAll(
          `td ${skeletonSelector}`,
        );
        expect(bodySkeletons.length).toBeGreaterThan(0); // Ensure at least one skeleton cell exists
      }

      // General check: ensure multiple skeleton elements are present in the table overall
      const allSkeletonsInTable = container.querySelectorAll(
        `table ${skeletonSelector}`,
      );
      expect(allSkeletonsInTable.length).toBeGreaterThan(3); // e.g., caption + at least one row of 3 cells, or just a few
    });
  });

  describe("Error State", () => {
    test("renders error message when error prop is provided", () => {
      const errorMessage = "Network Error";
      render(
        <ChatCompletionsTable
          {...defaultProps}
          error={{ name: "Error", message: errorMessage }}
        />,
      );
      expect(
        screen.getByText(`Error fetching data: ${errorMessage}`),
      ).toBeInTheDocument();
    });

    test("renders default error message when error.message is not available", () => {
      render(
        <ChatCompletionsTable
          {...defaultProps}
          error={{ name: "Error", message: "" }}
        />,
      ); // Error with empty message
      expect(
        screen.getByText("Error fetching data: An unknown error occurred"),
      ).toBeInTheDocument();
    });

    test("renders default error message when error prop is an object without message", () => {
      render(<ChatCompletionsTable {...defaultProps} error={{} as Error} />); // Empty error object
      expect(
        screen.getByText("Error fetching data: An unknown error occurred"),
      ).toBeInTheDocument();
    });
  });

  describe("Empty State", () => {
    test('renders "No chat completions found." and no table when completions array is empty', () => {
      render(
        <ChatCompletionsTable
          completions={[]}
          isLoading={false}
          error={null}
        />,
      );
      expect(
        screen.getByText("No chat completions found."),
      ).toBeInTheDocument();

      // Ensure that the table structure is NOT rendered in the empty state
      const table = screen.queryByRole("table");
      expect(table).not.toBeInTheDocument();
    });
  });

  describe("Data Rendering", () => {
    test("renders table caption, headers, and completion data correctly", () => {
      const mockCompletions = [
        {
          id: "comp_1",
          object: "chat.completion",
          created: 1710000000, // Fixed timestamp for test
          model: "llama-test-model",
          choices: [
            {
              index: 0,
              message: { role: "assistant", content: "Test output" },
              finish_reason: "stop",
            },
          ],
          input_messages: [{ role: "user", content: "Test input" }],
        },
        {
          id: "comp_2",
          object: "chat.completion",
          created: 1710001000,
          model: "llama-another-model",
          choices: [
            {
              index: 0,
              message: { role: "assistant", content: "Another output" },
              finish_reason: "stop",
            },
          ],
          input_messages: [{ role: "user", content: "Another input" }],
        },
      ];

      render(
        <ChatCompletionsTable
          completions={mockCompletions}
          isLoading={false}
          error={null}
        />,
      );

      // Table caption
      expect(
        screen.getByText("A list of your recent chat completions."),
      ).toBeInTheDocument();

      // Table headers
      expect(screen.getByText("Input")).toBeInTheDocument();
      expect(screen.getByText("Output")).toBeInTheDocument();
      expect(screen.getByText("Model")).toBeInTheDocument();
      expect(screen.getByText("Created")).toBeInTheDocument();

      // Data rows
      expect(screen.getByText("Test input")).toBeInTheDocument();
      expect(screen.getByText("Test output")).toBeInTheDocument();
      expect(screen.getByText("llama-test-model")).toBeInTheDocument();
      expect(
        screen.getByText(new Date(1710000000 * 1000).toLocaleString()),
      ).toBeInTheDocument();

      expect(screen.getByText("Another input")).toBeInTheDocument();
      expect(screen.getByText("Another output")).toBeInTheDocument();
      expect(screen.getByText("llama-another-model")).toBeInTheDocument();
      expect(
        screen.getByText(new Date(1710001000 * 1000).toLocaleString()),
      ).toBeInTheDocument();
    });
  });

  describe("Text Truncation and Tool Call Formatting", () => {
    test("truncates long input and output text", () => {
      // Specific mock implementation for this test
      truncateText.mockImplementation(
        (text: string | undefined, maxLength?: number) => {
          const defaultTestMaxLength = 10;
          const effectiveMaxLength = maxLength ?? defaultTestMaxLength;
          return typeof text === "string" && text.length > effectiveMaxLength
            ? text.slice(0, effectiveMaxLength) + "..."
            : text;
        },
      );

      const longInput =
        "This is a very long input message that should be truncated.";
      const longOutput =
        "This is a very long output message that should also be truncated.";
      const mockCompletions = [
        {
          id: "comp_trunc",
          object: "chat.completion",
          created: 1710002000,
          model: "llama-trunc-model",
          choices: [
            {
              index: 0,
              message: { role: "assistant", content: longOutput },
              finish_reason: "stop",
            },
          ],
          input_messages: [{ role: "user", content: longInput }],
        },
      ];

      render(
        <ChatCompletionsTable
          completions={mockCompletions}
          isLoading={false}
          error={null}
        />,
      );

      // The truncated text should be present for both input and output
      const truncatedTexts = screen.getAllByText(
        longInput.slice(0, 10) + "...",
      );
      expect(truncatedTexts.length).toBe(2); // one for input, one for output
      // Optionally, verify each one is in the document if getAllByText doesn't throw on not found
      truncatedTexts.forEach((textElement) =>
        expect(textElement).toBeInTheDocument(),
      );
    });

    test("formats tool call output using formatToolCallToString", () => {
      // Specific mock implementation for this test
      formatToolCallToString.mockImplementation(
        (toolCall: any) => `[TOOL:${toolCall.name}]`,
      );
      // Ensure no truncation interferes for this specific test for clarity of tool call format
      truncateText.mockImplementation((text: string | undefined) => text);

      const toolCall = { name: "search", args: { query: "llama" } };
      const mockCompletions = [
        {
          id: "comp_tool",
          object: "chat.completion",
          created: 1710003000,
          model: "llama-tool-model",
          choices: [
            {
              index: 0,
              message: {
                role: "assistant",
                content: "Tool output", // Content that will be prepended
                tool_calls: [toolCall],
              },
              finish_reason: "stop",
            },
          ],
          input_messages: [{ role: "user", content: "Tool input" }],
        },
      ];

      render(
        <ChatCompletionsTable
          completions={mockCompletions}
          isLoading={false}
          error={null}
        />,
      );

      // The component concatenates message.content and the formatted tool call
      expect(screen.getByText("Tool output [TOOL:search]")).toBeInTheDocument();
    });
  });
});
