import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import { ResponsesTable } from "./responses-table";
import { OpenAIResponse } from "@/lib/types";

// Mock next/navigation
const mockPush = jest.fn();
jest.mock("next/navigation", () => ({
  useRouter: () => ({
    push: mockPush,
  }),
}));

// Mock helper functions
jest.mock("@/lib/truncate-text");

// Import the mocked functions
import { truncateText as originalTruncateText } from "@/lib/truncate-text";

// Cast to jest.Mock for typings
const truncateText = originalTruncateText as jest.Mock;

describe("ResponsesTable", () => {
  const defaultProps = {
    data: [] as OpenAIResponse[],
    isLoading: false,
    error: null,
  };

  beforeEach(() => {
    // Reset all mocks before each test
    mockPush.mockClear();
    truncateText.mockClear();

    // Default pass-through implementation
    truncateText.mockImplementation((text: string | undefined) => text);
  });

  test("renders without crashing with default props", () => {
    render(<ResponsesTable {...defaultProps} />);
    expect(screen.getByText("No responses found.")).toBeInTheDocument();
  });

  test("click on a row navigates to the correct URL", () => {
    const mockResponse: OpenAIResponse = {
      id: "resp_123",
      object: "response",
      created_at: Math.floor(Date.now() / 1000),
      model: "llama-test-model",
      status: "completed",
      output: [
        {
          type: "message",
          role: "assistant",
          content: "Test output",
        },
      ],
      input: [
        {
          type: "message",
          role: "user",
          content: "Test input",
        },
      ],
    };

    render(<ResponsesTable {...defaultProps} data={[mockResponse]} />);

    const row = screen.getByText("Test input").closest("tr");
    if (row) {
      fireEvent.click(row);
      expect(mockPush).toHaveBeenCalledWith("/logs/responses/resp_123");
    } else {
      throw new Error('Row with "Test input" not found for router mock test.');
    }
  });

  describe("Loading State", () => {
    test("renders skeleton UI when isLoading is true", () => {
      const { container } = render(
        <ResponsesTable {...defaultProps} isLoading={true} />,
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
        <ResponsesTable
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
        <ResponsesTable
          {...defaultProps}
          error={{ name: "Error", message: "" }}
        />,
      );
      expect(
        screen.getByText("Error fetching data: An unknown error occurred"),
      ).toBeInTheDocument();
    });

    test("renders default error message when error prop is an object without message", () => {
      render(<ResponsesTable {...defaultProps} error={{} as Error} />);
      expect(
        screen.getByText("Error fetching data: An unknown error occurred"),
      ).toBeInTheDocument();
    });
  });

  describe("Empty State", () => {
    test('renders "No responses found." and no table when data array is empty', () => {
      render(<ResponsesTable data={[]} isLoading={false} error={null} />);
      expect(screen.getByText("No responses found.")).toBeInTheDocument();

      // Ensure that the table structure is NOT rendered in the empty state
      const table = screen.queryByRole("table");
      expect(table).not.toBeInTheDocument();
    });
  });

  describe("Data Rendering", () => {
    test("renders table caption, headers, and response data correctly", () => {
      const mockResponses = [
        {
          id: "resp_1",
          object: "response" as const,
          created_at: 1710000000,
          model: "llama-test-model",
          status: "completed",
          output: [
            {
              type: "message" as const,
              role: "assistant" as const,
              content: "Test output",
            },
          ],
          input: [
            {
              type: "message",
              role: "user",
              content: "Test input",
            },
          ],
        },
        {
          id: "resp_2",
          object: "response" as const,
          created_at: 1710001000,
          model: "llama-another-model",
          status: "completed",
          output: [
            {
              type: "message" as const,
              role: "assistant" as const,
              content: "Another output",
            },
          ],
          input: [
            {
              type: "message",
              role: "user",
              content: "Another input",
            },
          ],
        },
      ];

      render(
        <ResponsesTable data={mockResponses} isLoading={false} error={null} />,
      );

      // Table caption
      expect(
        screen.getByText("A list of your recent responses."),
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

  describe("Input Text Extraction", () => {
    test("extracts text from string content", () => {
      const mockResponse: OpenAIResponse = {
        id: "resp_string",
        object: "response",
        created_at: 1710000000,
        model: "test-model",
        status: "completed",
        output: [{ type: "message", role: "assistant", content: "output" }],
        input: [
          {
            type: "message",
            role: "user",
            content: "Simple string input",
          },
        ],
      };

      render(
        <ResponsesTable data={[mockResponse]} isLoading={false} error={null} />,
      );
      expect(screen.getByText("Simple string input")).toBeInTheDocument();
    });

    test("extracts text from array content with input_text type", () => {
      const mockResponse: OpenAIResponse = {
        id: "resp_array",
        object: "response",
        created_at: 1710000000,
        model: "test-model",
        status: "completed",
        output: [{ type: "message", role: "assistant", content: "output" }],
        input: [
          {
            type: "message",
            role: "user",
            content: [
              { type: "input_text", text: "Array input text" },
              { type: "input_text", text: "Should not be used" },
            ],
          },
        ],
      };

      render(
        <ResponsesTable data={[mockResponse]} isLoading={false} error={null} />,
      );
      expect(screen.getByText("Array input text")).toBeInTheDocument();
    });

    test("returns empty string when no message input found", () => {
      const mockResponse: OpenAIResponse = {
        id: "resp_no_input",
        object: "response",
        created_at: 1710000000,
        model: "test-model",
        status: "completed",
        output: [{ type: "message", role: "assistant", content: "output" }],
        input: [
          {
            type: "other_type",
            content: "Not a message",
          },
        ],
      };

      const { container } = render(
        <ResponsesTable data={[mockResponse]} isLoading={false} error={null} />,
      );

      // Find the input cell (first cell in the data row) and verify it's empty
      const inputCell = container.querySelector("tbody tr td:first-child");
      expect(inputCell).toBeInTheDocument();
      expect(inputCell).toHaveTextContent("");
    });
  });

  describe("Output Text Extraction", () => {
    test("extracts text from string message content", () => {
      const mockResponse: OpenAIResponse = {
        id: "resp_string_output",
        object: "response",
        created_at: 1710000000,
        model: "test-model",
        status: "completed",
        output: [
          {
            type: "message",
            role: "assistant",
            content: "Simple string output",
          },
        ],
        input: [{ type: "message", content: "input" }],
      };

      render(
        <ResponsesTable data={[mockResponse]} isLoading={false} error={null} />,
      );
      expect(screen.getByText("Simple string output")).toBeInTheDocument();
    });

    test("extracts text from array message content with output_text type", () => {
      const mockResponse: OpenAIResponse = {
        id: "resp_array_output",
        object: "response",
        created_at: 1710000000,
        model: "test-model",
        status: "completed",
        output: [
          {
            type: "message",
            role: "assistant",
            content: [
              { type: "output_text", text: "Array output text" },
              { type: "output_text", text: "Should not be used" },
            ],
          },
        ],
        input: [{ type: "message", content: "input" }],
      };

      render(
        <ResponsesTable data={[mockResponse]} isLoading={false} error={null} />,
      );
      expect(screen.getByText("Array output text")).toBeInTheDocument();
    });

    test("formats function call output", () => {
      const mockResponse: OpenAIResponse = {
        id: "resp_function_call",
        object: "response",
        created_at: 1710000000,
        model: "test-model",
        status: "completed",
        output: [
          {
            type: "function_call",
            id: "call_123",
            status: "completed",
            name: "search_function",
            arguments: '{"query": "test"}',
          },
        ],
        input: [{ type: "message", content: "input" }],
      };

      render(
        <ResponsesTable data={[mockResponse]} isLoading={false} error={null} />,
      );
      expect(
        screen.getByText('search_function({"query": "test"})'),
      ).toBeInTheDocument();
    });

    test("formats function call output without arguments", () => {
      const mockResponse: OpenAIResponse = {
        id: "resp_function_no_args",
        object: "response",
        created_at: 1710000000,
        model: "test-model",
        status: "completed",
        output: [
          {
            type: "function_call",
            id: "call_123",
            status: "completed",
            name: "simple_function",
          },
        ],
        input: [{ type: "message", content: "input" }],
      };

      render(
        <ResponsesTable data={[mockResponse]} isLoading={false} error={null} />,
      );
      expect(screen.getByText("simple_function({})")).toBeInTheDocument();
    });

    test("formats web search call output", () => {
      const mockResponse: OpenAIResponse = {
        id: "resp_web_search",
        object: "response",
        created_at: 1710000000,
        model: "test-model",
        status: "completed",
        output: [
          {
            type: "web_search_call",
            id: "search_123",
            status: "completed",
          },
        ],
        input: [{ type: "message", content: "input" }],
      };

      render(
        <ResponsesTable data={[mockResponse]} isLoading={false} error={null} />,
      );
      expect(
        screen.getByText("web_search_call(status: completed)"),
      ).toBeInTheDocument();
    });

    test("falls back to JSON.stringify for unknown tool call types", () => {
      const mockResponse: OpenAIResponse = {
        id: "resp_unknown_tool",
        object: "response",
        created_at: 1710000000,
        model: "test-model",
        status: "completed",
        output: [
          {
            type: "unknown_call",
            id: "unknown_123",
            status: "completed",
            custom_field: "custom_value",
          } as any,
        ],
        input: [{ type: "message", content: "input" }],
      };

      render(
        <ResponsesTable data={[mockResponse]} isLoading={false} error={null} />,
      );
      // Should contain the JSON stringified version
      expect(screen.getByText(/unknown_call/)).toBeInTheDocument();
    });

    test("falls back to JSON.stringify for entire output when no message or tool call found", () => {
      const mockResponse: OpenAIResponse = {
        id: "resp_fallback",
        object: "response",
        created_at: 1710000000,
        model: "test-model",
        status: "completed",
        output: [
          {
            type: "unknown_type",
            data: "some data",
          } as any,
        ],
        input: [{ type: "message", content: "input" }],
      };

      render(
        <ResponsesTable data={[mockResponse]} isLoading={false} error={null} />,
      );
      // Should contain the JSON stringified version of the output array
      expect(screen.getByText(/unknown_type/)).toBeInTheDocument();
    });
  });

  describe("Text Truncation", () => {
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

      const mockResponse: OpenAIResponse = {
        id: "resp_trunc",
        object: "response",
        created_at: 1710002000,
        model: "llama-trunc-model",
        status: "completed",
        output: [
          {
            type: "message",
            role: "assistant",
            content: longOutput,
          },
        ],
        input: [
          {
            type: "message",
            role: "user",
            content: longInput,
          },
        ],
      };

      render(
        <ResponsesTable data={[mockResponse]} isLoading={false} error={null} />,
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
  });
});
