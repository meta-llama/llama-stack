"use client";

import React from "react";
import { usePathname, useParams } from "next/navigation";
import {
  PageBreadcrumb,
  BreadcrumbSegment,
} from "@/components/layout/page-breadcrumb";
import { truncateText } from "@/lib/truncate-text";

export default function ChatCompletionsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const params = useParams();

  let segments: BreadcrumbSegment[] = [];

  // Default for /logs/chat-completions
  if (pathname === "/logs/chat-completions") {
    segments = [{ label: "Chat Completions" }];
  }

  // For /logs/chat-completions/[id]
  const idParam = params?.id;
  if (idParam && typeof idParam === "string") {
    segments = [
      { label: "Chat Completions", href: "/logs/chat-completions" },
      { label: `Details (${truncateText(idParam, 20)})` },
    ];
  }

  return (
    <div className="container mx-auto p-4">
      <>
        {segments.length > 0 && (
          <PageBreadcrumb segments={segments} className="mb-4" />
        )}
        {children}
      </>
    </div>
  );
}
