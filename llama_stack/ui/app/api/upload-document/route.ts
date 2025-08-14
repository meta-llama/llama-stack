import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get("file") as File;
    const vectorDbId = formData.get("vectorDbId") as string;

    if (!file || !vectorDbId) {
      return NextResponse.json(
        { error: "File and vectorDbId are required" },
        { status: 400 }
      );
    }

    // Read file content based on type
    let content: string;
    const mimeType = file.type || "application/octet-stream";

    if (mimeType === "text/plain" || mimeType === "text/markdown") {
      content = await file.text();
    } else if (mimeType === "application/pdf") {
      // For PDFs, convert to base64 on the server side
      const arrayBuffer = await file.arrayBuffer();
      const bytes = new Uint8Array(arrayBuffer);
      let binary = "";
      for (let i = 0; i < bytes.byteLength; i++) {
        binary += String.fromCharCode(bytes[i]);
      }
      const base64 = btoa(binary);
      content = `data:${mimeType};base64,${base64}`;
    } else {
      // Try to read as text
      content = await file.text();
    }

    // Return the processed content for the client to send to RagTool
    return NextResponse.json({
      content,
      mimeType,
      fileName: file.name,
      fileSize: file.size,
    });
  } catch (error) {
    console.error("Error processing file upload:", error);
    return NextResponse.json(
      { error: "Failed to process file upload" },
      { status: 500 }
    );
  }
}
