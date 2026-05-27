import os
import httpx
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from urllib.parse import urlparse

# Load environment variables from .env
load_dotenv()

# --- Fast model config (default) ---
# Backward-compatible: falls back to the old REAL_* names
FAST_BASE_URL = os.getenv("FAST_BASE_URL") or os.getenv("REAL_BASE_URL")
FAST_API_KEY = os.getenv("FAST_API_KEY") or os.getenv("REAL_API_KEY")
FAST_MODEL_NAME = os.getenv("FAST_MODEL_NAME") or os.getenv("REAL_MODEL_NAME")

# --- Smart model config (optional) ---
# If not set, smart requests will fall back to the fast model
SMART_BASE_URL = os.getenv("SMART_BASE_URL") or FAST_BASE_URL
SMART_API_KEY = os.getenv("SMART_API_KEY") or FAST_API_KEY
SMART_MODEL_NAME = os.getenv("SMART_MODEL_NAME") or FAST_MODEL_NAME

PROXY_PORT = int(os.getenv("PROXY_PORT", 8000))
BIND_HOST = os.getenv("BIND_HOST", "0.0.0.0")

# Timeout for backend requests (seconds). Use a connect timeout so we don't
# wait forever to establish a connection, and a long read timeout so we can
# detect stalled responses while still supporting slow LLM generation.
# Set PROXY_TIMEOUT=0 to disable timeouts entirely (not recommended).
_raw_timeout = int(os.getenv("PROXY_TIMEOUT", 120))
PROXY_TIMEOUT = httpx.Timeout(
    connect=10.0,
    read=_raw_timeout if _raw_timeout > 0 else None,
    write=30.0,
    pool=10.0,
) if _raw_timeout > 0 else None

app = FastAPI(title="OpenAI Compatible Proxy")

# Shared httpx client — timeout set per-request above
client = httpx.AsyncClient(timeout=PROXY_TIMEOUT)

# ---------------------------------------------------------------------------
# URL path helpers
# ---------------------------------------------------------------------------

def _parse_base_url(url: str):
    parsed = urlparse(url)
    segments = [s for s in parsed.path.split("/") if s]
    return parsed, segments


FAST_PARSED, FAST_PATH_SEGMENTS = _parse_base_url(FAST_BASE_URL)
SMART_PARSED, SMART_PATH_SEGMENTS = _parse_base_url(SMART_BASE_URL)


def resolve_path(incoming_path: str, base_url: str, base_path_segments: list) -> str:
    """
    Stitch the incoming request path onto the given base URL,
    avoiding duplicate path segments (like a double /v1).
    """
    incoming_segments = [s for s in incoming_path.split("/") if s]

    # If the first incoming segment matches the LAST segment of the base URL
    # path, strip it to avoid duplication
    # (e.g., base=/inference/v1, incoming=/v1/chat -> /inference/v1/chat)
    if incoming_segments and base_path_segments and incoming_segments[0] == base_path_segments[-1]:
        incoming_segments = incoming_segments[1:]

    remainder = "/".join(incoming_segments)
    base = base_url.rstrip("/")
    return f"{base}/{remainder}" if remainder else base


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def get_config_for_model(model: str | None) -> dict:
    """
    Determine which backend config to use based on the requested model name.

    - "smart"  -> smart model config (falls back to fast if not separately configured)
    - "fast" / anything else -> fast model config
    """
    if model == "smart":
        return {
            "base_url": SMART_BASE_URL,
            "api_key": SMART_API_KEY,
            "model_name": SMART_MODEL_NAME,
            "parsed": SMART_PARSED,
            "path_segments": SMART_PATH_SEGMENTS,
            "label": "smart",
        }
    else:
        return {
            "base_url": FAST_BASE_URL,
            "api_key": FAST_API_KEY,
            "model_name": FAST_MODEL_NAME,
            "parsed": FAST_PARSED,
            "path_segments": FAST_PATH_SEGMENTS,
            "label": "fast",
        }


def get_opposite_config(model: str | None) -> dict:
    """Return the config for the model NOT currently being requested."""
    if model == "smart":
        return get_config_for_model("fast")
    else:
        return get_config_for_model("smart")


def configs_differ(a: dict, b: dict) -> bool:
    """Return True if two backend configs actually point to different places."""
    return a["base_url"] != b["base_url"] or a["model_name"] != b["model_name"]


def _should_fallback(status_code: int) -> bool:
    """Return True for status codes where falling back makes sense."""
    return status_code >= 500 or status_code == 429


# ---------------------------------------------------------------------------
# Core forwarding logic
# ---------------------------------------------------------------------------

async def _forward_to_backend(
    config: dict,
    method: str,
    path: str,
    incoming_headers: dict,
    body: bytes,
) -> httpx.Response:
    """
    Build and send the request to a single backend.  Returns the raw httpx
    response (streaming).  Raises on connection/timeout errors.
    """
    url = resolve_path(path, config["base_url"], config["path_segments"])

    # Build headers with the backend's API key and correct host.
    # Strip the client's Authorization (if any) so we don't send duplicate
    # auth headers, which causes Cloudflare to reject the request (HTTP 400).
    headers = {}
    for k, v in incoming_headers.items():
        kl = k.lower()
        if kl in ("host", "content-length", "authorization"):
            continue
        headers[k] = v
    headers["host"] = config["parsed"].netloc
    if config["api_key"]:
        headers["authorization"] = f"Bearer {config['api_key']}"

    if "content-length" in headers:
        del headers["content-length"]

    # Rewrite the model name in the JSON body
    new_body = body
    requested_model = None
    if method == "POST" and body:
        try:
            data = json.loads(body)
            requested_model = data.get("model")
            data["model"] = config["model_name"]
            new_body = json.dumps(data).encode("utf-8")
        except Exception:
            pass  # body isn't JSON — forward as-is

    print(
        f"[{config['label']}] Proxying: "
        f"model '{requested_model}' -> '{config['model_name']}' "
        f"→ {url}"
    )

    req = client.build_request(method, url, headers=headers, content=new_body)
    return await client.send(req, stream=True)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_handler(request: Request, path: str):
    """
    Catch-all route — forwards to the appropriate backend with automatic
    fallback if the primary backend fails (server error / timeout / rate limit).
    """
    body = await request.body()
    method = request.method

    # Work out what model the client asked for
    requested_model = None
    if method == "POST" and body:
        try:
            data = json.loads(body)
            requested_model = data.get("model")
        except Exception:
            pass

    primary = get_config_for_model(requested_model)
    fallback = get_opposite_config(requested_model)
    can_fallback = configs_differ(primary, fallback)

    # --- Attempt #1: primary backend ---
    last_error_msg = None
    last_error_status = None

    try:
        resp = await _forward_to_backend(primary, method, path, request.headers, body)
    except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.TimeoutException) as e:
        print(f"[{primary['label']}] Connection/timeout error: {e}")
        resp = None
        last_error_msg = str(e)
        last_error_status = 504  # Gateway Timeout
    except Exception as e:
        print(f"[{primary['label']}] Unexpected error: {e}")
        resp = None
        last_error_msg = str(e)
        last_error_status = 502

    # If primary succeeded with a good status, return immediately
    if resp is not None and not _should_fallback(resp.status_code):
        return StreamingResponse(
            resp.aiter_raw(),
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )

    # Primary returned a server error / rate limit, or connection failed
    if resp is not None:
        print(
            f"[{primary['label']}] Backend returned {resp.status_code}"
            + (f", falling back to {fallback['label']}..." if can_fallback else "")
        )
        last_error_status = resp.status_code
        # Read the error body so we can return it if fallback also fails
        try:
            last_error_msg = (await resp.aread()).decode("utf-8", errors="replace")
        except Exception:
            last_error_msg = f"HTTP {resp.status_code}"
    else:
        print(
            f"[{primary['label']}] Connection failed"
            + (f", falling back to {fallback['label']}..." if can_fallback else "")
        )

    if not can_fallback:
        # No fallback available — return the error as-is
        if resp is not None:
            return StreamingResponse(
                resp.aiter_raw() if resp.status_code != 0 else None,
                status_code=last_error_status,
                headers=dict(resp.headers) if resp is not None else {},
            )
        from fastapi.responses import JSONResponse
        return JSONResponse(
            {"error": last_error_msg or "Unknown error"},
            status_code=last_error_status or 502,
        )

    # --- Attempt #2: fallback backend ---
    print(f"[{fallback['label']}] Attempting fallback...")

    try:
        resp2 = await _forward_to_backend(fallback, method, path, request.headers, body)
    except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.TimeoutException) as e:
        print(f"[{fallback['label']}] Fallback also failed: {e}")
        from fastapi.responses import JSONResponse
        return JSONResponse(
            {"error": last_error_msg or "Both backends unavailable"},
            status_code=last_error_status or 502,
        )
    except Exception as e:
        print(f"[{fallback['label']}] Fallback unexpected error: {e}")
        from fastapi.responses import JSONResponse
        return JSONResponse(
            {"error": last_error_msg or str(e)},
            status_code=last_error_status or 502,
        )

    # Fallback succeeded (any status — we tried our best)
    print(f"[{fallback['label']}] Fallback response: {resp2.status_code}")
    return StreamingResponse(
        resp2.aiter_raw(),
        status_code=resp2.status_code,
        headers=dict(resp2.headers),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    has_smart = configs_differ(
        get_config_for_model("smart"),
        get_config_for_model("fast"),
    )

    print(f"Starting proxy on {BIND_HOST}:{PROXY_PORT}")
    print(f"Timeout:      {PROXY_TIMEOUT or 'none (disabled)'}")
    print(f"Fast model:   {FAST_BASE_URL}  ->  {FAST_MODEL_NAME}")
    if has_smart:
        print(f"Smart model:  {SMART_BASE_URL}  ->  {SMART_MODEL_NAME}")
        print(f"  -> model='fast' (or anything else) falls back to 'smart' on error")
        print(f"  -> model='smart'                   falls back to 'fast' on error")
    else:
        print(f"Smart model:  (not separately configured — fallback disabled)")
    uvicorn.run(app, host=BIND_HOST, port=PROXY_PORT)
