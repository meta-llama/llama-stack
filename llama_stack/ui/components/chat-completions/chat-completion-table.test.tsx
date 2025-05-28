import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import { ChatCompletionsTable } from "./chat-completions-table";
import { ChatCompletion } from "@/lib/types";

// Mock next/navigation
const mockPush = jest.fn();
jest.mock("next/navigation", () => ({
  useRouter: () => ({
    push: mockPush,
  }),
}));

// Mock helper functions
jest.mock("@/lib/truncate-text");
jest.mock("@/lib/format-message-content");

// Import the mocked functions to set up default or specific implementations
import { truncateText as originalTruncateText } from "@/lib/truncate-text";
import {
  extractTextFromContentPart as originalExtractTextFromContentPart,
  extractDisplayableText as originalExtractDisplayableText,
} from "@/lib/format-message-content";

// Cast to jest.Mock for typings
const truncateText = originalTruncateText as jest.Mock;
const extractTextFromContentPart =
  originalExtractTextFromContentPart as jest.Mock;
const extractDisplayableText = originalExtractDisplayableText as jest.Mock;

describe("ChatCompletionsTable", () => {
  const defaultProps = {
    data: [] as ChatCompletion[],
    isLoading: false,
    error: null,
  };

  beforeEach(() => {
    // Reset all mocks before each test
    mockPush.mockClear();
    truncateText.mockClear();
    extractTextFromContentPart.mockClear();
    extractDisplayableText.mockClear();

    // Default pass-through implementations
    truncateText.mockImplementation((text: string | undefined) => text);
    extractTextFromContentPart.mockImplementation((content: unknown) =>
      typeof content === "string" ? content : "extracted text",
    );
    extractDisplayableText.mockImplementation(
      (message: unknown) =>
        (message as { content?: string })?.content || "extracted output",
    );
  });

  test("renders without crashing with default props", () => {
    render(<ChatCompletionsTable {...defaultProps} />);
    expect(screen.getByText("No chat completions found.")).toBeInTheDocument();
  });

  test("click on a row navigates to the correct URL", () => {
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

    // Set up mocks to return expected values
    extractTextFromContentPart.mockReturnValue("Test input");
    extractDisplayableText.mockReturnValue("Test output");

    render(<ChatCompletionsTable {...defaultProps} data={[mockCompletion]} />);

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

      // Check for skeleton in the table caption
      const tableCaption = container.querySelector("caption");
      expect(tableCaption).toBeInTheDocument();
      if (tableCaption) {
        const captionSkeleton = tableCaption.querySelector(
          '[data-slot="skeleton"]',
        );
        expect(captionSkeleton).toBeInTheDocument();
      }

      // Check for skeletons in the table body cells
      const tableBody = container.querySelector("tbody");
      expect(tableBody).toBeInTheDocument();
      if (tableBody) {
        const bodySkeletons = tableBody.querySelectorAll(
          '[data-slot="skeleton"]',
        );
        expect(bodySkeletons.length).toBeGreaterThan(0);
      }
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
      );
      expect(
        screen.getByText("Error fetching data: An unknown error occurred"),
      ).toBeInTheDocument();
    });

    test("renders default error message when error prop is an object without message", () => {
      render(<ChatCompletionsTable {...defaultProps} error={{} as Error} />);
      expect(
        screen.getByText("Error fetching data: An unknown error occurred"),
      ).toBeInTheDocument();
    });
  });

  describe("Empty State", () => {
    test('renders "No chat completions found." and no table when data array is empty', () => {
      render(<ChatCompletionsTable data={[]} isLoading={false} error={null} />);
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
          created: 1710000000,
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

      // Set up mocks to return expected values
      extractTextFromContentPart.mockImplementation((content: unknown) => {
        if (content === "Test input") return "Test input";
        if (content === "Another input") return "Another input";
        return "extracted text";
      });
      extractDisplayableText.mockImplementation((message: unknown) => {
        const msg = message as { content?: string };
        if (msg?.content === "Test output") return "Test output";
        if (msg?.content === "Another output") return "Another output";
        return "extracted output";
      });

      render(
        <ChatCompletionsTable
          data={mockCompletions}
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

  describe("Text Truncation and Content Extraction", () => {
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

      extractTextFromContentPart.mockReturnValue(longInput);
      extractDisplayableText.mockReturnValue(longOutput);

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
          data={mockCompletions}
          isLoading={false}
          error={null}
        />,
      );

      // The truncated text should be present for both input and output
      const truncatedTexts = screen.getAllByText(
        longInput.slice(0, 10) + "...",
      );
      expect(truncatedTexts.length).toBe(2); // one for input, one for output
      truncatedTexts.forEach((textElement) =>
        expect(textElement).toBeInTheDocument(),
      );
    });

    test("uses content extraction functions correctly", () => {
      const mockCompletion = {
        id: "comp_extract",
        object: "chat.completion",
        created: 1710003000,
        model: "llama-extract-model",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "Extracted output" },
            finish_reason: "stop",
          },
        ],
        input_messages: [{ role: "user", content: "Extracted input" }],
      };

      extractTextFromContentPart.mockReturnValue("Extracted input");
      extractDisplayableText.mockReturnValue("Extracted output");

      render(
        <ChatCompletionsTable
          data={[mockCompletion]}
          isLoading={false}
          error={null}
        />,
      );

      // Verify the extraction functions were called
      expect(extractTextFromContentPart).toHaveBeenCalledWith(
        "Extracted input",
      );
      expect(extractDisplayableText).toHaveBeenCalledWith({
        role: "assistant",
        content: "Extracted output",
      });

      // Verify the extracted content is displayed
      expect(screen.getByText("Extracted input")).toBeInTheDocument();
      expect(screen.getByText("Extracted output")).toBeInTheDocument();
    });
  });
});
