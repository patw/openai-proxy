import os
import httpx
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from urllib.parse import urlparse

# Load environment variables from .env
load_dotenv()

REAL_BASE_URL = os.getenv("REAL_BASE_URL")
REAL_API_KEY = os.getenv("REAL_API_KEY")
REAL_MODEL_NAME = os.getenv("REAL_MODEL_NAME")
PROXY_PORT = int(os.getenv("PROXY_PORT", 8000))
BIND_HOST = os.getenv("BIND_HOST", "0.0.0.0")

app = FastAPI(title="OpenAI Compatible Proxy")

# Pre-parse the base URL once at startup
PARSED_BASE = urlparse(REAL_BASE_URL)
# Extract segments from the base path (e.g., "/inference/v1" -> ["inference", "v1"])
BASE_PATH_SEGMENTS = [s for s in PARSED_BASE.path.split("/") if s]

# Set timeout to None so the proxy doesn't kill long-running LLM requests.
client = httpx.AsyncClient(timeout=None)


def resolve_path(incoming_path: str) -> str:
    """
    Stitch the incoming request path onto the real base URL,
    avoiding duplicate path segments (like a double /v1).
    """
    # Split incoming path into segments (e.g., "/v1/chat/completions" -> ["v1", "chat", "completions"])
    incoming_segments = [s for s in incoming_path.split("/") if s]

    # If the first incoming segment matches the LAST segment of the base URL path,
    # strip it to avoid duplication (e.g., base=/inference/v1, incoming=/v1/chat -> /chat)
    if incoming_segments and BASE_PATH_SEGMENTS and incoming_segments[0] == BASE_PATH_SEGMENTS[-1]:
        incoming_segments = incoming_segments[1:]

    # Build the final path
    remainder = "/".join(incoming_segments)
    base = REAL_BASE_URL.rstrip("/")
    return f"{base}/{remainder}" if remainder else base


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_handler(request: Request, path: str):
    """
    Catch-all route that captures any path and forwards it to the real backend.
    """
    url = resolve_path(path)
    method = request.method
    body = await request.body()

    # 1. Prepare headers: Use the real API key and correct host
    headers = dict(request.headers)
    headers["host"] = PARSED_BASE.netloc
    headers["authorization"] = f"Bearer {REAL_API_KEY}"

    # Remove content-length so httpx recalculates it if we modify the body
    if "content-length" in headers:
        del headers["content-length"]

    # 2. Handle Body Modification: Force the REAL model name
    new_body = body
    if method == "POST" and body:
        try:
            data = json.loads(body)
            if "model" in data:
                print(f"Proxying request: Overriding model '{data['model']}' -> '{REAL_MODEL_NAME}'")
                data["model"] = REAL_MODEL_NAME
            new_body = json.dumps(data).encode("utf-8")
        except Exception as e:
            print(f"Warning: Could not parse/modify JSON body: {e}")

    # 3. Forward the request
    req = client.build_request(
        method,
        url,
        headers=headers,
        content=new_body
    )

    # We use stream=True to support long-running responses and streaming (SSE)
    rp_resp = await client.send(req, stream=True)

    # 4. Return the response back to the client via StreamingResponse
    return StreamingResponse(
        rp_resp.aiter_raw(),
        status_code=rp_resp.status_code,
        headers=dict(rp_resp.headers)
    )


if __name__ == "__main__":
    import uvicorn
    print(f"Starting proxy on {BIND_HOST}:{PROXY_PORT}")
    print(f"Forwarding all requests to: {REAL_BASE_URL} using model: {REAL_MODEL_NAME}")
    uvicorn.run(app, host=BIND_HOST, port=PROXY_PORT)
