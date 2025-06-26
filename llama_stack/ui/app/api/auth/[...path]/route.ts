import { NextRequest, NextResponse } from "next/server";
import { BACKEND_URL } from "@/lib/server-config";

async function proxyAuthRequest(request: NextRequest, method: string) {
  try {
    const url = new URL(request.url);
    const pathSegments = url.pathname.split("/");

    // Remove /api/auth from the path to get the actual auth path
    // /api/auth/github/login -> /auth/github/login
    const authPath = pathSegments.slice(3).join("/"); // Remove 'api' and 'auth' segments
    const targetUrl = `${BACKEND_URL}/auth/${authPath}${url.search}`;

    // Prepare headers (exclude host and other problematic headers)
    const headers = new Headers();
    request.headers.forEach((value, key) => {
      // Skip headers that might cause issues in proxy
      if (
        !["host", "connection", "content-length"].includes(key.toLowerCase())
      ) {
        headers.set(key, value);
      }
    });

    const requestOptions: RequestInit = {
      method,
      headers,
      // Don't follow redirects automatically - we need to handle them
      redirect: "manual",
    };

    if (["POST", "PUT", "PATCH"].includes(method) && request.body) {
      requestOptions.body = await request.text();
    }

    const response = await fetch(targetUrl, requestOptions);

    // Handle redirects
    if (response.status === 302 || response.status === 307) {
      const location = response.headers.get("location");
      if (location) {
        // For external redirects (like GitHub OAuth), return the redirect
        return NextResponse.redirect(location);
      }
    }

    if (response.ok) {
      const responseText = await response.text();

      const proxyResponse = new NextResponse(responseText, {
        status: response.status,
        statusText: response.statusText,
      });

      response.headers.forEach((value, key) => {
        if (!["connection", "transfer-encoding"].includes(key.toLowerCase())) {
          proxyResponse.headers.set(key, value);
        }
      });

      return proxyResponse;
    }

    const errorText = await response.text();
    return new NextResponse(errorText, {
      status: response.status,
      statusText: response.statusText,
    });
  } catch (error) {
    return NextResponse.json(
      {
        error: "Auth proxy request failed",
        message: error instanceof Error ? error.message : "Unknown error",
        backend_url: BACKEND_URL,
        timestamp: new Date().toISOString(),
      },
      { status: 500 },
    );
  }
}

export async function GET(request: NextRequest) {
  return proxyAuthRequest(request, "GET");
}

export async function POST(request: NextRequest) {
  return proxyAuthRequest(request, "POST");
}
