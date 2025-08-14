"use client";

import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type LlamaStackClient from "llama-stack-client";

interface DocumentUploaderProps {
  client: LlamaStackClient;
  selectedVectorDb: string;
  disabled?: boolean;
}

interface UploadState {
  isUploading: boolean;
  uploadProgress: string;
  uploadError: string | null;
}

export function DocumentUploader({
  client,
  selectedVectorDb,
  disabled,
}: DocumentUploaderProps) {
  const [uploadState, setUploadState] = useState<UploadState>({
    isUploading: false,
    uploadProgress: "",
    uploadError: null,
  });
  const [urlInput, setUrlInput] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] || null;
    setSelectedFile(file);
    if (file) {
      console.log("File selected:", file.name, file.size, "bytes");
    }
  };

  const handleFileUpload = async () => {
    console.log("Upload button clicked");
    console.log("Files:", fileInputRef.current?.files?.length);
    console.log("Selected Vector DB:", selectedVectorDb);

    if (
      !fileInputRef.current?.files?.length ||
      !selectedVectorDb ||
      selectedVectorDb === "none"
    ) {
      console.log("Upload blocked: missing file or vector DB");
      setUploadState({
        isUploading: false,
        uploadProgress: "",
        uploadError:
          "Please select a file and ensure a Vector Database is selected",
      });
      return;
    }

    const file = fileInputRef.current.files[0];
    console.log("Starting upload for:", file.name);
    setUploadState({
      isUploading: true,
      uploadProgress: `Uploading ${file.name}...`,
      uploadError: null,
    });

    try {
      // Determine MIME type
      let mimeType = file.type;
      if (!mimeType) {
        // Fallback based on file extension
        const extension = file.name.split(".").pop()?.toLowerCase();
        switch (extension) {
          case "pdf":
            mimeType = "application/pdf";
            break;
          case "txt":
            mimeType = "text/plain";
            break;
          case "md":
            mimeType = "text/markdown";
            break;
          default:
            mimeType = "application/octet-stream";
        }
      }

      setUploadState({
        isUploading: true,
        uploadProgress: `Reading ${file.name}...`,
        uploadError: null,
      });

      // Use server-side file processing API for better efficiency
      const formData = new FormData();
      formData.append("file", file);
      formData.append("vectorDbId", selectedVectorDb);

      setUploadState({
        isUploading: true,
        uploadProgress: `Processing ${file.name}...`,
        uploadError: null,
      });

      const uploadResponse = await fetch("/api/upload-document", {
        method: "POST",
        body: formData,
      });

      if (!uploadResponse.ok) {
        throw new Error(`Upload failed: ${uploadResponse.statusText}`);
      }

      const { content, mimeType: processedMimeType } =
        await uploadResponse.json();

      // Use RagTool to insert the document
      console.log("Calling RagTool.insert with:", {
        vector_db_id: selectedVectorDb,
        chunk_size_in_tokens: 512,
        content_type: typeof content,
        content_length: typeof content === "string" ? content.length : "N/A",
        mime_type: processedMimeType,
      });

      await client.toolRuntime.ragTool.insert({
        vector_db_id: selectedVectorDb,
        chunk_size_in_tokens: 512,
        documents: [
          {
            document_id: `file-${Date.now()}-${file.name}`,
            content: content,
            mime_type: processedMimeType,
            metadata: {
              source: file.name,
              uploaded_at: new Date().toISOString(),
              file_size: file.size,
            },
          },
        ],
      });

      console.log("RagTool.insert completed successfully");

      const truncatedName =
        file.name.length > 20 ? file.name.substring(0, 20) + "..." : file.name;
      setUploadState({
        isUploading: false,
        uploadProgress: `Successfully uploaded ${truncatedName}`,
        uploadError: null,
      });

      // Clear file input and selected file state
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
      setSelectedFile(null);
    } catch (err) {
      console.error("Error uploading file:", err);
      setUploadState({
        isUploading: false,
        uploadProgress: "",
        uploadError:
          err instanceof Error ? err.message : "Failed to upload file",
      });
    }
  };

  const handleUrlUpload = async () => {
    if (!urlInput.trim() || !selectedVectorDb || selectedVectorDb === "none")
      return;

    setUploadState({
      isUploading: true,
      uploadProgress: `Fetching content from ${urlInput}...`,
      uploadError: null,
    });

    try {
      // Determine MIME type from URL
      let mimeType = "text/html";
      const url = urlInput.toLowerCase();
      if (url.endsWith(".pdf")) {
        mimeType = "application/pdf";
      } else if (url.endsWith(".txt")) {
        mimeType = "text/plain";
      } else if (url.endsWith(".md")) {
        mimeType = "text/markdown";
      }

      setUploadState({
        isUploading: true,
        uploadProgress: `Processing content from ${urlInput}...`,
        uploadError: null,
      });

      // Use RagTool to insert the document from URL
      await client.toolRuntime.ragTool.insert({
        vector_db_id: selectedVectorDb,
        chunk_size_in_tokens: 512,
        documents: [
          {
            document_id: `url-${Date.now()}-${encodeURIComponent(urlInput)}`,
            content: urlInput,
            mime_type: mimeType,
            metadata: {
              source: urlInput,
              uploaded_at: new Date().toISOString(),
            },
          },
        ],
      });

      const truncatedUrl =
        urlInput.length > 30 ? urlInput.substring(0, 30) + "..." : urlInput;
      setUploadState({
        isUploading: false,
        uploadProgress: `Successfully processed ${truncatedUrl}`,
        uploadError: null,
      });

      setUrlInput("");
    } catch (err) {
      console.error("Error uploading URL:", err);
      setUploadState({
        isUploading: false,
        uploadProgress: "",
        uploadError:
          err instanceof Error ? err.message : "Failed to process URL content",
      });
    }
  };

  const clearStatus = () => {
    setUploadState({
      isUploading: false,
      uploadProgress: "",
      uploadError: null,
    });
  };

  if (!selectedVectorDb || selectedVectorDb === "none") {
    return (
      <div className="text-sm text-gray-500 text-center py-4">
        Select a Vector Database to upload documents
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h3 className="text-md font-medium">Upload Documents</h3>

      {uploadState.uploadError && (
        <div className="p-2 bg-destructive/10 border border-destructive/20 rounded-md">
          <p className="text-sm text-foreground">{uploadState.uploadError}</p>
          <Button
            variant="ghost"
            size="sm"
            onClick={clearStatus}
            className="mt-1 h-6 px-2 text-xs"
          >
            Clear
          </Button>
        </div>
      )}

      {uploadState.uploadProgress && (
        <div className="p-2 bg-muted border border-border rounded-md">
          <p className="text-sm text-foreground">
            {uploadState.uploadProgress}
          </p>
          {uploadState.isUploading && (
            <div className="mt-2 w-full bg-secondary rounded-full h-1">
              <div
                className="bg-primary h-1 rounded-full animate-pulse"
                style={{ width: "60%" }}
              ></div>
            </div>
          )}
        </div>
      )}

      {/* File Upload */}
      <div className="space-y-2">
        <label className="text-sm font-medium block">Upload File</label>
        <input
          ref={fileInputRef}
          type="file"
          accept=".txt,.pdf,.md"
          onChange={handleFileSelect}
          className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-medium file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
          disabled={disabled || uploadState.isUploading}
        />

        {selectedFile && (
          <div className="p-2 bg-muted border border-border rounded-md">
            <p className="text-sm text-foreground">
              Selected:{" "}
              <span className="font-medium">
                {selectedFile.name.length > 25
                  ? selectedFile.name.substring(0, 25) + "..."
                  : selectedFile.name}
              </span>
              <span className="text-muted-foreground ml-2">
                ({(selectedFile.size / 1024).toFixed(1)} KB)
              </span>
            </p>
          </div>
        )}

        <Button
          onClick={handleFileUpload}
          disabled={disabled || !selectedFile || uploadState.isUploading}
          className="w-full"
          variant="outline"
          size="sm"
        >
          {uploadState.isUploading
            ? "Uploading..."
            : selectedFile
              ? "Upload File"
              : "Upload File"}
        </Button>
      </div>

      {/* URL Upload */}
      <div className="space-y-2">
        <label className="text-sm font-medium block">Or Enter URL</label>
        <Input
          value={urlInput}
          onChange={e => setUrlInput(e.target.value)}
          placeholder="https://example.com/document.pdf"
          disabled={disabled || uploadState.isUploading}
        />
        <Button
          onClick={handleUrlUpload}
          disabled={disabled || !urlInput.trim() || uploadState.isUploading}
          className="w-full"
          variant="outline"
          size="sm"
        >
          {uploadState.isUploading ? "Processing..." : "Process URL"}
        </Button>
      </div>

      <p className="text-xs text-gray-500">
        Supported formats: PDF, TXT, MD files and web URLs
      </p>
    </div>
  );
}
