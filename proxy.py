"""
Proxy forwarding logic — sends requests to the real backend, captures usage,
and implements automatic fallback between fast ↔ smart models.
"""

import json
import time
import httpx
from urllib.parse import urlparse
from flask import Response, stream_with_context, jsonify

from usage_tracker import extract_usage_from_json, extract_usage_from_sse_line, record_usage
from models_config import get_model, get_model_by_tag
from storage import get_settings


def get_timeout(settings: dict) -> httpx.Timeout:
    """Build an httpx timeout from global settings."""
    seconds = int(settings.get("proxy_timeout_seconds", 120))
    if seconds <= 0:
        seconds = 120
    return httpx.Timeout(connect=10.0, read=seconds, write=seconds, pool=10.0)



# ---------------------------------------------------------------------------
# Reasoning option sanitization
# ---------------------------------------------------------------------------

REASONING_TOP_LEVEL_KEYS = {
    "reasoning_effort",
    "reasoning",
    "reasoning_history",
    "thinking",
    "reasoning_format",
    "include_reasoning",
}

REASONING_MESSAGE_KEYS = {
    "reasoning",
    "reasoning_content",
    "reasoning_details",
}

UNSUPPORTED_REASONING_ERROR_MARKERS = (
    "reasoning_effort",
    "reasoning_history",
    "reasoning_content",
    "reasoning_details",
    "include_reasoning",
    "reasoning_format",
    "unsupported parameter",
    "unknown parameter",
    "unknown field",
    "extra inputs are not permitted",
    "unrecognized request argument",
)


def _sanitize_reasoning_options(body: bytes) -> tuple[bytes, bool]:
    """Remove reasoning-specific request/message fields for a retry.

    Returns (new_body, changed). If the body isn't JSON/object, returns it
    unchanged.
    """
    try:
        data = json.loads(body)
    except Exception:
        return body, False
    if not isinstance(data, dict):
        return body, False

    changed = False
    for key in list(REASONING_TOP_LEVEL_KEYS):
        if key in data:
            data.pop(key, None)
            changed = True

    messages = data.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                for key in list(REASONING_MESSAGE_KEYS):
                    if key in msg:
                        msg.pop(key, None)
                        changed = True

    if not changed:
        return body, False
    return json.dumps(data).encode("utf-8"), True


def _looks_like_unsupported_reasoning_error(resp) -> bool:
    """Heuristic for providers that reject unknown reasoning parameters."""
    if resp is None or getattr(resp, "status_code", None) != 400:
        return False
    try:
        text = resp.text.lower()
    except Exception:
        return False
    return any(marker in text for marker in UNSUPPORTED_REASONING_ERROR_MARKERS)

# ---------------------------------------------------------------------------
# Path resolution (ported from original)
# ---------------------------------------------------------------------------

def _resolve_path(incoming_path: str, base_url: str) -> str:
    """Stitch the incoming path onto the base URL, avoiding duplicate segments."""
    incoming = [s for s in incoming_path.split("/") if s]
    parsed = urlparse(base_url)
    base_segments = [s for s in parsed.path.split("/") if s]

    if incoming and base_segments and incoming[0] == base_segments[-1]:
        incoming = incoming[1:]

    remainder = "/".join(incoming)
    base = base_url.rstrip("/")
    return f"{base}/{remainder}" if remainder else base


# ---------------------------------------------------------------------------
# Model lookup
# ---------------------------------------------------------------------------

def resolve_model(requested_model: str | None) -> dict | None:
    """
    Resolve the requested model name to a full config dict.
    Returns None if the model is unknown.
    """
    if not requested_model:
        return None

    if requested_model in ("fast", "smart", "local"):
        return get_model_by_tag(requested_model)

    return get_model(requested_model)


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------

def calculate_cost(model: dict, usage: dict, duration_seconds: float, settings: dict) -> float:
    """
    Calculate the cost of a request.

    For remote models:  token-based pricing.
    For local models:   wattage × duration × electricity cost.
    """
    if model.get("type") == "local":
        wattage = settings.get("local_model_max_wattage", 300)
        price_kwh = settings.get("electricity_cost_per_kwh", 0.12)
        hours = duration_seconds / 3600.0
        kwh = (wattage / 1000.0) * hours
        return kwh * price_kwh

    # Remote model — token pricing
    it = usage.get("input_tokens", 0)
    ot = usage.get("output_tokens", 0)
    ct = usage.get("cached_tokens", 0)

    iprice = model.get("input_price_per_million", 0)
    oprice = model.get("output_price_per_million", 0)
    cprice = model.get("cached_price_per_million", 0)

    # NOTE: input_tokens from OpenAI includes cached tokens.
    # We must exclude them here since they're priced separately
    # at the cached rate — otherwise they'd be double-counted.
    non_cached_input = max(it - ct, 0)

    cost = (non_cached_input / 1_000_000) * iprice
    cost += (ot / 1_000_000) * oprice
    cost += (ct / 1_000_000) * cprice
    return cost


# ---------------------------------------------------------------------------
# Request building (shared by streaming and non-streaming paths)
# ---------------------------------------------------------------------------

def _rewrite_body_model(body: bytes, model: dict, method: str):
    """Rewrite the JSON `model` field to the backend's expected name.

    Returns (new_body, requested_model_name). Leaves the body untouched if
    it isn't a POST with a JSON object.
    """
    new_body = body
    requested_model = "?"
    if method == "POST" and body:
        try:
            data = json.loads(body)
            requested_model = data.get("model", "?")
            data["model"] = model["api_model_name"]
            new_body = json.dumps(data).encode("utf-8")
        except Exception:
            pass
    return new_body, requested_model


def _build_forward_headers(headers: dict, model: dict, method: str) -> dict:
    """Copy client headers, dropping hop-by-hop ones and overriding auth/host.

    Supports two authentication styles via the optional ``api_key_header`` field:
      - Default (unset or empty): ``Authorization: Bearer <key>`` (OpenAI-style)
      - ``"api-key"``: ``api-key: <key>`` (Azure OpenAI-style)
      - Any other value: that header name, raw key value (custom providers)
    """
    fwd_headers = {}
    for k, v in headers.items():
        kl = k.lower()
        if kl in ("host", "content-length", "transfer-encoding", "connection",
                  "authorization"):
            continue
        fwd_headers[k] = v

    # Also drop the custom auth header if the client is sending one — we'll
    # replace it with the model's own key.
    api_key_header = model.get("api_key_header", "").strip()
    if api_key_header:
        for k in list(fwd_headers.keys()):
            if k.lower() == api_key_header.lower():
                del fwd_headers[k]

    # Ensure Content-Type is set for POST requests
    if method == "POST" and "content-type" not in {k.lower() for k in fwd_headers}:
        fwd_headers["Content-Type"] = "application/json"

    fwd_headers["host"] = urlparse(model["base_url"]).netloc
    if model.get("api_key"):
        if api_key_header:
            fwd_headers[api_key_header] = model["api_key"]
        else:
            fwd_headers["authorization"] = f"Bearer {model['api_key']}"
    return fwd_headers


# ---------------------------------------------------------------------------
# Forwarding — non-streaming
# ---------------------------------------------------------------------------

def _forward_non_streaming(model: dict, method: str, path: str,
                           headers: dict, body: bytes, settings: dict):
    """
    Forward a non-streaming request to the backend.

    Returns (response, usage_dict_or_None).
    """
    url = _resolve_path(path, model["base_url"])
    new_body, requested_model = _rewrite_body_model(body, model, method)
    fwd_headers = _build_forward_headers(headers, model, method)

    print(f"[{model['name']}] {method} {url}  (requested: '{requested_model}' "
          f"→ using: '{model['api_model_name']}')")

    with httpx.Client(timeout=get_timeout(settings)) as client:
        resp = client.request(method, url, headers=fwd_headers, content=new_body)

    # Log response for non-2xx so we can diagnose backend errors
    if resp.status_code >= 400:
        try:
            body_preview = resp.text[:500]
            print(f"[{model['name']}] ← {resp.status_code}: {body_preview}")
        except Exception:
            print(f"[{model['name']}] ← {resp.status_code} (body unreadable)")

    # Try to extract usage from response body
    usage = None
    if resp.status_code < 400:
        try:
            resp_data = resp.json()
            usage = extract_usage_from_json(resp_data)
        except Exception:
            pass

    return resp, usage


# ---------------------------------------------------------------------------
# Forwarding — streaming (SSE)
# ---------------------------------------------------------------------------

def _open_upstream_stream(model: dict, method: str, path: str,
                          headers: dict, body: bytes, settings: dict):
    """
    Open a streaming request to the backend and return (client, stream_ctx, resp)
    with the response status/headers available but the body not yet consumed.

    This lets the caller inspect resp.status_code *before* committing a status
    to the client — which is what makes streaming error-passthrough and
    fallback possible. The caller owns closing stream_ctx and client (via
    _close_stream) or handing them to the streaming generator.

    Raises the usual httpx errors if the connection can't be established.
    """
    url = _resolve_path(path, model["base_url"])
    new_body, requested_model = _rewrite_body_model(body, model, method)
    fwd_headers = _build_forward_headers(headers, model, method)

    print(f"[{model['name']}] {method} {url} (stream)  (requested: '{requested_model}' "
          f"→ using: '{model['api_model_name']}')")

    client = httpx.Client(timeout=get_timeout(settings))
    stream_ctx = client.stream(method, url, headers=fwd_headers, content=new_body)
    try:
        resp = stream_ctx.__enter__()  # sends the request, reads status + headers
    except BaseException:
        client.close()
        raise
    return client, stream_ctx, resp


def _close_stream(client, stream_ctx) -> None:
    """Best-effort teardown of an upstream streaming connection."""
    try:
        stream_ctx.__exit__(None, None, None)
    except Exception:
        pass
    try:
        client.close()
    except Exception:
        pass


def _stream_response(client, stream_ctx, up_resp, model: dict,
                     settings: dict, t0: float) -> Response:
    """
    Build a Flask streaming Response that proxies SSE bytes from an already-open
    upstream response, then records usage once the stream completes.

    This is the fix for streaming usage tracking: usage is recorded inside the
    generator's finally block (after the client has consumed the stream), not
    before streaming begins.
    """
    def generate():
        usage = None
        try:
            for line in up_resp.iter_lines():
                # Yield the line back to the client
                yield (line + "\n").encode("utf-8")

                # Try to parse SSE data lines for usage; keep the last one found
                if line.startswith("data: "):
                    sse_data = line[6:]  # strip "data: "
                    if sse_data != "[DONE]":
                        parsed = extract_usage_from_sse_line(sse_data)
                        if parsed:
                            usage = parsed
        finally:
            _close_stream(client, stream_ctx)
            try:
                duration = time.time() - t0
                record_usage_for_model(model, usage, duration, settings)
            except Exception as e:
                print(f"[{model['name']}] failed to record streaming usage: {e}")

    return Response(
        stream_with_context(generate()),
        status=200,
        headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
    )


# ---------------------------------------------------------------------------
# Top-level proxy handler
# ---------------------------------------------------------------------------

def handle_proxy_request(method: str, path: str, headers: dict, body: bytes):
    """
    Full proxy request handler.

    1. Resolve the requested model to a backend config.
    2. Forward the request (with fallback for fast↔smart).
    3. Capture and record usage.
    4. Return a Flask Response.
    """
    # ---- Resolve model ----
    try:
        req_data = json.loads(body)
        requested_model = req_data.get("model")
    except Exception:
        requested_model = None

    model = resolve_model(requested_model)
    if model is None:
        return jsonify({"error": f"Unknown model: '{requested_model}'"}), 400

    if not model.get("enabled", True):
        return jsonify({"error": f"Model '{requested_model}' is disabled."}), 400

    is_streaming = False
    try:
        is_streaming = json.loads(body).get("stream", False)
    except Exception:
        pass

    settings = get_settings()

    # ---- Determine fallback ----
    primary = model
    fallback = None
    primary_label = None
    fallback_label = None

    tags = primary.get("tags", [])
    if "fast" in tags:
        primary_label = "fast"
        fallback = get_model_by_tag("smart")
        fallback_label = "smart"
    elif "smart" in tags:
        primary_label = "smart"
        fallback = get_model_by_tag("fast")
        fallback_label = "fast"

    # Don't fallback to the same model
    if fallback and fallback["name"] == primary["name"]:
        fallback = None

    # ---- Forward to primary ----
    resp, usage, duration, error = _attempt_forward(
        primary, method, path, headers, body, is_streaming, settings
    )

    # ---- Fallback on failure ----
    if error and fallback:
        print(f"[fallback] Primary '{primary['name']}' failed ({error}), "
              f"trying '{fallback['name']}'...")
        resp, usage, duration, error2 = _attempt_forward(
            fallback, method, path, headers, body, is_streaming, settings
        )
        if error2:
            # Both failed — return the primary error
            return _build_error_response(error, resp)
        # Fallback succeeded — record usage under the fallback model
        record_usage_for_model(fallback, usage, duration, settings)
        return resp

    if error:
        return _build_error_response(error, resp)

    # Primary succeeded
    record_usage_for_model(primary, usage, duration, settings)
    return resp


def _attempt_forward_streaming(model: dict, method: str, path: str, headers: dict,
                               body: bytes, settings: dict, t0: float):
    """
    Attempt a streaming forward.

    Returns (response, usage, duration, error) like _attempt_forward. On
    success the response streams the body and records usage itself, so the
    returned usage is always None (record_usage_for_model no-ops on None).

    Because we inspect the upstream status before emitting any bytes, an
    error status can trigger fallback (5xx/429) or be surfaced to the client
    (other 4xx) instead of being masked as a 200 stream.
    """
    attempt_body = body
    for reasoning_retry in (False, True):
        try:
            client, stream_ctx, up = _open_upstream_stream(
                model, method, path, headers, attempt_body, settings
            )
        except httpx.ConnectError as e:
            return None, None, time.time() - t0, f"Connection error: {e}"
        except httpx.ReadTimeout as e:
            return None, None, time.time() - t0, f"Timeout: {e}"
        except httpx.RemoteProtocolError as e:
            return None, None, time.time() - t0, f"Protocol error: {e}"

        status = up.status_code

        if status < 400:
            # Success — stream the body; usage is recorded when it completes.
            resp = _stream_response(client, stream_ctx, up, model, settings, t0)
            return resp, None, time.time() - t0, None

        # Error response — buffer its (small) body so we can inspect/forward it.
        try:
            err_body = up.read()
        except Exception:
            err_body = b""
        err_headers = _filter_response_headers(up.headers)
        _close_stream(client, stream_ctx)

        # Parity with the non-streaming path: one retry without reasoning
        # options if the backend rejected them. Safe — no bytes emitted yet.
        if (not reasoning_retry and status == 400
                and any(m in err_body.decode("utf-8", "replace").lower()
                        for m in UNSUPPORTED_REASONING_ERROR_MARKERS)):
            sanitized_body, changed = _sanitize_reasoning_options(body)
            if changed:
                print(f"[{model['name']}] retrying without unsupported reasoning options (stream)")
                attempt_body = sanitized_body
                continue

        duration = time.time() - t0
        if _should_fallback(status):
            return None, None, duration, f"HTTP {status}"
        # Non-retryable error — surface it to the client (no fallback).
        return (Response(err_body, status=status, headers=err_headers),
                None, duration, None)


def _attempt_forward(model: dict, method: str, path: str, headers: dict,
                     body: bytes, is_streaming: bool, settings: dict):
    """
    Try to forward a request to *model*.

    Returns (response, usage_dict, duration_seconds, error_str).
    Exactly one of (response, usage) or error will be meaningful.
    """
    t0 = time.time()

    try:
        if is_streaming:
            return _attempt_forward_streaming(model, method, path, headers, body, settings, t0)
        else:
            hx_resp, usage = _forward_non_streaming(model, method, path, headers, body, settings)
            if _looks_like_unsupported_reasoning_error(hx_resp):
                sanitized_body, changed = _sanitize_reasoning_options(body)
                if changed:
                    print(f"[{model['name']}] retrying without unsupported reasoning options")
                    hx_resp, usage = _forward_non_streaming(model, method, path, headers, sanitized_body, settings)
            duration = time.time() - t0

            status = hx_resp.status_code
            # Convert httpx.Response → Flask Response
            resp_headers = _filter_response_headers(hx_resp.headers)
            flask_resp = Response(
                hx_resp.content,
                status=status,
                headers=resp_headers,
            )

            if _should_fallback(status):
                return None, usage, duration, f"HTTP {status}"
            return flask_resp, usage, duration, None

    except httpx.ConnectError as e:
        return None, None, time.time() - t0, f"Connection error: {e}"
    except httpx.ReadTimeout as e:
        return None, None, time.time() - t0, f"Timeout: {e}"
    except httpx.RemoteProtocolError as e:
        return None, None, time.time() - t0, f"Protocol error: {e}"
    except Exception as e:
        return None, None, time.time() - t0, f"Unexpected error: {e}"


_HOP_BY_HOP_RESPONSE_HEADERS = (
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-encoding",
    "content-length",
)


def _filter_response_headers(headers) -> dict:
    """Strip hop-by-hop headers that a WSGI app is not allowed to set (PEP 3333)."""
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP_RESPONSE_HEADERS}


def _should_fallback(status_code: int) -> bool:
    return status_code >= 500 or status_code == 429


def _build_error_response(error: str, resp):
    """Build a Flask error response when both primary and fallback fail."""
    if resp is not None:
        return Response(
            resp.content,
            status=resp.status_code,
            headers=_filter_response_headers(resp.headers),
        )
    return jsonify({"error": error}), 502


def record_usage_for_model(model: dict, usage: dict | None,
                           duration_seconds: float, settings: dict):
    """Persist usage data, if available."""
    if usage is None:
        return

    cost = calculate_cost(model, usage, duration_seconds, settings)
    record_usage(
        model_name=model["name"],
        model_display=model.get("display_name", model["name"]),
        model_type=model.get("type", "remote"),
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cached_tokens=usage.get("cached_tokens", 0),
        cost=cost,
        duration_seconds=duration_seconds,
    )
