# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# /// script
# dependencies = [
#     "fastapi",
#     "fastapi-cors",
#     "fastapi-staticfiles",
#     "fastapi-responses",
#     "fastapi-middleware",
#     "fastapi-middleware-cors",
#     "fastapi-middleware-staticfiles",
#     "fastapi-middleware-responses",
#     "uvicorn",
# ]
# ///
#
#
# Before running the script, build the Next.js app:
# cd llama_stack/ui
# npm run build
#
# Run the script like:
# uv run uvicorn server:app --reload

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your actual domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, ".next", "static")
server_dir = os.path.join(current_dir, ".next", "server")
app_dir = os.path.join(server_dir, "app")

# Mount the static files directory at the Next.js path
app.mount("/_next/static", StaticFiles(directory=static_dir), name="static")


# Serve HTML files for each route
@app.get("/{path:path}")
async def serve_page(path: str):
    # Handle root path
    if path == "" or path == "/":
        return FileResponse(os.path.join(app_dir, "index.html"))

    # Handle logs routes
    if path.startswith("logs/"):
        log_name = path.replace("logs/", "")
        log_html = os.path.join(app_dir, "logs", f"{log_name}.html")
        if os.path.exists(log_html):
            return FileResponse(log_html)

    # If no specific route is found, return 404 page
    not_found_path = os.path.join(app_dir, "_not-found.html")
    if os.path.exists(not_found_path):
        return FileResponse(not_found_path)

    raise HTTPException(status_code=404, detail="Page not found")


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "message": "Static file server is running"}
