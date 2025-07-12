"use client";

import React from "react";
import LogsLayout from "@/components/layout/logs-layout";

export default function VectorStoresLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <LogsLayout sectionLabel="Vector Stores" basePath="/logs/vector-stores">
      {children}
    </LogsLayout>
  );
}
