"use client";

import { useState, useRef } from "react";
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
import type LlamaStackClient from "llama-stack-client";

interface VectorDbManagerProps {
  client: LlamaStackClient;
  onVectorDbCreated: () => void;
}

interface UploadState {
  isUploading: boolean;
  uploadProgress: string;
  uploadError: string | null;
}

export function VectorDbManager({
  client,
  onVectorDbCreated,
}: VectorDbManagerProps) {
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [formData, setFormData] = useState({
    vectorDbId: "",
    embeddingModel: "all-MiniLM-L6-v2",
    embeddingDimension: "384",
  });
  const [uploadState, setUploadState] = useState<UploadState>({
    isUploading: false,
    uploadProgress: "",
    uploadError: null,
  });
  const [urlInput, setUrlInput] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleCreateVectorDb = async () => {
    if (!formData.vectorDbId.trim()) {
      setCreateError("Vector DB ID is required");
      return;
    }

    setIsCreating(true);
    setCreateError(null);

    try {
      // Get available providers to find a vector_io provider
      const providers = await client.providers.list();
      const vectorIoProvider = providers.find(p => p.api === "vector_io");

      if (!vectorIoProvider) {
        throw new Error("No vector_io provider found");
      }

      await client.vectorDBs.register({
        vector_db_id: formData.vectorDbId.trim(),
        embedding_model: formData.embeddingModel,
        embedding_dimension: parseInt(formData.embeddingDimension),
        provider_id: vectorIoProvider.provider_id,
      });

      // Reset form and close
      setFormData({
        vectorDbId: "",
        embeddingModel: "all-MiniLM-L6-v2",
        embeddingDimension: "384",
      });
      setShowCreateForm(false);

      // Refresh the vector DB list
      onVectorDbCreated();
    } catch (err) {
      console.error("Error creating vector DB:", err);
      setCreateError(
        err instanceof Error ? err.message : "Failed to create vector database"
      );
    } finally {
      setIsCreating(false);
    }
  };

  const handleCancel = () => {
    setShowCreateForm(false);
    setCreateError(null);
    setFormData({
      vectorDbId: "",
      embeddingModel: "all-MiniLM-L6-v2",
      embeddingDimension: "384",
    });
    setUploadState({
      isUploading: false,
      uploadProgress: "",
      uploadError: null,
    });
    setUrlInput("");
  };

  const chunkText = (text: string, chunkSize: number = 512): string[] => {
    const words = text.split(/\s+/);
    const chunks: string[] = [];

    for (let i = 0; i < words.length; i += chunkSize) {
      chunks.push(words.slice(i, i + chunkSize).join(" "));
    }

    return chunks;
  };

  const ingestDocument = async (
    content: string,
    documentId: string,
    vectorDbId: string
  ) => {
    const chunks = chunkText(content);

    const vectorChunks = chunks.map((chunk, index) => ({
      content: chunk,
      metadata: {
        document_id: documentId,
        chunk_index: index,
        source: documentId,
      },
    }));

    await client.vectorIo.insert({
      vector_db_id: vectorDbId,
      chunks: vectorChunks,
    });
  };

  const handleFileUpload = async (vectorDbId: string) => {
    if (!fileInputRef.current?.files?.length) return;

    const file = fileInputRef.current.files[0];
    setUploadState({
      isUploading: true,
      uploadProgress: `Reading ${file.name}...`,
      uploadError: null,
    });

    try {
      const text = await file.text();

      setUploadState({
        isUploading: true,
        uploadProgress: `Ingesting ${file.name} into vector database...`,
        uploadError: null,
      });

      await ingestDocument(text, file.name, vectorDbId);

      setUploadState({
        isUploading: false,
        uploadProgress: `Successfully ingested ${file.name}`,
        uploadError: null,
      });

      // Clear file input
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
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

  const handleUrlUpload = async (vectorDbId: string) => {
    if (!urlInput.trim()) return;

    setUploadState({
      isUploading: true,
      uploadProgress: `Fetching content from ${urlInput}...`,
      uploadError: null,
    });

    try {
      const response = await fetch(`/api/fetch-url`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: urlInput }),
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch URL: ${response.statusText}`);
      }

      const { content } = await response.json();

      setUploadState({
        isUploading: true,
        uploadProgress: `Ingesting content from ${urlInput}...`,
        uploadError: null,
      });

      await ingestDocument(content, urlInput, vectorDbId);

      setUploadState({
        isUploading: false,
        uploadProgress: `Successfully ingested content from ${urlInput}`,
        uploadError: null,
      });

      setUrlInput("");
    } catch (err) {
      console.error("Error uploading URL:", err);
      setUploadState({
        isUploading: false,
        uploadProgress: "",
        uploadError:
          err instanceof Error ? err.message : "Failed to fetch URL content",
      });
    }
  };

  return (
    <div className="relative">
      {!showCreateForm ? (
        <Button
          onClick={() => setShowCreateForm(true)}
          variant="outline"
          size="default"
        >
          + Vector DB
        </Button>
      ) : (
        <Card className="absolute top-full right-0 mt-2 p-4 space-y-4 w-96 z-50 bg-background border shadow-lg">
          <h3 className="text-lg font-semibold">Create Vector Database</h3>

          {createError && (
            <div className="p-3 bg-destructive/10 border border-destructive/20 rounded-md">
              <p className="text-destructive text-sm">{createError}</p>
            </div>
          )}

          {uploadState.uploadError && (
            <div className="p-3 bg-destructive/10 border border-destructive/20 rounded-md">
              <p className="text-destructive text-sm">
                {uploadState.uploadError}
              </p>
            </div>
          )}

          {uploadState.uploadProgress && (
            <div className="p-3 bg-blue-50 border border-blue-200 rounded-md">
              <p className="text-blue-700 text-sm">
                {uploadState.uploadProgress}
              </p>
            </div>
          )}

          <div className="space-y-3">
            <div>
              <label className="text-sm font-medium block mb-1">
                Vector DB ID *
              </label>
              <Input
                value={formData.vectorDbId}
                onChange={e =>
                  setFormData({ ...formData, vectorDbId: e.target.value })
                }
                placeholder="Enter unique vector DB identifier"
                disabled={isCreating || uploadState.isUploading}
              />
            </div>

            <div>
              <label className="text-sm font-medium block mb-1">
                Embedding Model
              </label>
              <Select
                value={formData.embeddingModel}
                onValueChange={value =>
                  setFormData({ ...formData, embeddingModel: value })
                }
                disabled={isCreating || uploadState.isUploading}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all-MiniLM-L6-v2">
                    all-MiniLM-L6-v2
                  </SelectItem>
                  <SelectItem value="text-embedding-ada-002">
                    text-embedding-ada-002
                  </SelectItem>
                  <SelectItem value="text-embedding-3-small">
                    text-embedding-3-small
                  </SelectItem>
                  <SelectItem value="text-embedding-3-large">
                    text-embedding-3-large
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="text-sm font-medium block mb-1">
                Embedding Dimension
              </label>
              <Select
                value={formData.embeddingDimension}
                onValueChange={value =>
                  setFormData({ ...formData, embeddingDimension: value })
                }
                disabled={isCreating || uploadState.isUploading}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="384">384 (all-MiniLM-L6-v2)</SelectItem>
                  <SelectItem value="1536">1536 (ada-002, 3-small)</SelectItem>
                  <SelectItem value="3072">3072 (3-large)</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="flex gap-2">
            <Button
              onClick={handleCreateVectorDb}
              disabled={isCreating || uploadState.isUploading}
              className="flex-1"
            >
              {isCreating ? "Creating..." : "Create"}
            </Button>
            <Button
              variant="outline"
              onClick={handleCancel}
              disabled={isCreating || uploadState.isUploading}
              className="flex-1"
            >
              Cancel
            </Button>
          </div>

          {/* Document Upload Section */}
          <div className="border-t pt-4 space-y-3">
            <h4 className="text-md font-medium">Upload Documents (Optional)</h4>

            <div>
              <label className="text-sm font-medium block mb-1">
                Upload File (TXT, PDF)
              </label>
              <input
                ref={fileInputRef}
                type="file"
                accept=".txt,.pdf"
                className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                disabled={!formData.vectorDbId || uploadState.isUploading}
              />
              <Button
                onClick={() => handleFileUpload(formData.vectorDbId)}
                disabled={
                  !formData.vectorDbId ||
                  !fileInputRef.current?.files?.length ||
                  uploadState.isUploading
                }
                className="mt-2 w-full"
                variant="outline"
                size="sm"
              >
                {uploadState.isUploading ? "Uploading..." : "Upload File"}
              </Button>
            </div>

            <div>
              <label className="text-sm font-medium block mb-1">
                Or Enter URL
              </label>
              <Input
                value={urlInput}
                onChange={e => setUrlInput(e.target.value)}
                placeholder="https://example.com/article"
                disabled={!formData.vectorDbId || uploadState.isUploading}
              />
              <Button
                onClick={() => handleUrlUpload(formData.vectorDbId)}
                disabled={
                  !formData.vectorDbId ||
                  !urlInput.trim() ||
                  uploadState.isUploading
                }
                className="mt-2 w-full"
                variant="outline"
                size="sm"
              >
                {uploadState.isUploading ? "Fetching..." : "Fetch & Upload"}
              </Button>
            </div>

            <p className="text-xs text-gray-500">
              Note: Create the Vector DB first, then upload documents to it.
            </p>
          </div>
        </Card>
      )}
    </div>
  );
}
