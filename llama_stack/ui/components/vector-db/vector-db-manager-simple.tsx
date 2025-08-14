"use client";

import { useState } from "react";
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
  };

  return (
    <div className="relative">
      {!showCreateForm ? (
        <Button
          onClick={() => setShowCreateForm(true)}
          variant="outline"
          size="default"
          className="w-full"
        >
          + Create Vector DB
        </Button>
      ) : (
        <Card className="absolute top-full right-0 mt-2 p-4 space-y-4 w-80 z-50 bg-background border shadow-lg">
          <h3 className="text-lg font-semibold">Create Vector Database</h3>

          {createError && (
            <div className="p-3 bg-destructive/10 border border-destructive/20 rounded-md">
              <p className="text-destructive text-sm">{createError}</p>
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
                disabled={isCreating}
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
                disabled={isCreating}
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
                disabled={isCreating}
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
              disabled={isCreating}
              className="flex-1"
            >
              {isCreating ? "Creating..." : "Create"}
            </Button>
            <Button
              variant="outline"
              onClick={handleCancel}
              disabled={isCreating}
              className="flex-1"
            >
              Cancel
            </Button>
          </div>
        </Card>
      )}
    </div>
  );
}
